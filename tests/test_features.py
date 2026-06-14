"""Tests for the declarative feature framework (util/features/).

All local, no network. Covers:
  - feature_id identity invariants (the safety net for the naming refactor)
  - a few pinned golden feature_ids for stable published templates
  - feature-name round-trip (make_feature_name -> parse_feature_name)
  - FeatureBuilder behaviors (dedup, publish filter, missing-dep error, no
    cross-symbol leakage)
  - numeric correctness of custom primitives (returns, tr, zscore, regression,
    tail mean, cs_fraction) including the cs_fraction NaN-handling fix.
  - beta computed offline by injecting the market price series (T5).
"""

import numpy as np
import pandas as pd
import pytest

from util.features.core import (
    FeatureBuilder,
    FeatureNameSpec,
    make_feature_name,
    make_spec,
    col,
    feat,
)
from util.features.transforms import parse_feature_name
from util.features.primitives import (
    prim_ts_return,
    prim_tr,
    prim_zscore,
    prim_cs_fraction,
)
from util.features.primitives_regtrend import prim_ts_reg_slope, prim_ts_reg_r2
import util.features.primitives_regtrend_vectorized  # registers vectorized prims
import util.features.primitives_tailrisk  # registers tail-mean prims
from util.features.primitives import PRIMITIVES

from util.features.families.returns import RETURN_FAMILY
from util.features.families.price import RAWPRICE_FAMILY
from util.features.families.breadth import BREADTH_FAMILY


# ---------------------------------------------------------------------------
# Synthetic panels
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """Deterministic OHLC panel: 3 symbols x ~260 business days."""
    dates = pd.bdate_range("2023-01-02", periods=260)
    rng = np.random.default_rng(42)
    frames = []
    for sym, p0 in [("AAA", 100.0), ("BBB", 50.0), ("CCC", 20.0)]:
        rets = rng.normal(0.0003, 0.012, len(dates))
        close = p0 * np.cumprod(1 + rets)
        high = close * (1 + np.abs(rng.normal(0, 0.004, len(dates))))
        low = close * (1 - np.abs(rng.normal(0, 0.004, len(dates))))
        frames.append(pd.DataFrame({
            "symbol": sym, "date": dates,
            "adjClose": close, "adjHigh": high, "adjLow": low,
        }))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def loglinear_panel() -> pd.DataFrame:
    """One symbol whose log price is exactly linear in t -> known slope/r2."""
    dates = pd.bdate_range("2023-01-02", periods=60)
    t = np.arange(len(dates), dtype=float)
    g = 0.01  # exact log-slope per step
    close = np.exp(2.0 + g * t)
    return pd.DataFrame({"symbol": "LIN", "date": dates, "adjClose": close})


# ---------------------------------------------------------------------------
# Identity invariants  (the safety net for the FeatureNameSpec refactor, T3)
# ---------------------------------------------------------------------------
class TestIdentityInvariants:
    def test_feature_id_independent_of_name(self):
        """Identity is hashed from primitive/inputs/params/post, never the name.

        This guards the T3 naming refactor: switching families to emit
        name_spec must not move any feature_id.
        """
        a = make_spec(
            primitive="ts_return", inputs={"price": col("adjClose")},
            params={"lookback": 5}, name="anything__a__b__none__raw",
        )
        b = make_spec(
            primitive="ts_return", inputs={"price": col("adjClose")},
            params={"lookback": 5}, name="other__x__y__none__raw",
            name_spec=FeatureNameSpec("px", "ret", "logret", {"lb": 5}, "raw"),
        )
        assert a.feature_id == b.feature_id

    def test_raw_string_input_equals_col(self):
        a = make_spec(primitive="ts_return", inputs={"price": "adjClose"},
                      params={"lookback": 5}, name="n__a__b__none__raw")
        b = make_spec(primitive="ts_return", inputs={"price": col("adjClose")},
                      params={"lookback": 5}, name="n__a__b__none__raw")
        assert a.feature_id == b.feature_id

    def test_param_order_does_not_change_id(self):
        a = make_spec(primitive="zscore",
                      inputs={"x": col("c"), "mu": col("m"), "sigma": col("s")},
                      params={"eps": 1e-12}, name="n__a__b__none__raw")
        b = make_spec(primitive="zscore",
                      inputs={"sigma": col("s"), "x": col("c"), "mu": col("m")},
                      params={"eps": 1e-12}, name="n__a__b__none__raw")
        assert a.feature_id == b.feature_id

    @pytest.mark.parametrize("family,key,kwargs,expected_id", [
        (RETURN_FAMILY, "RETURN", dict(price_col="adjClose", lookback=5),
         "fid_beb1fffbcf58c69d0ca9fa983dc81ffe57238f4c35638151663e2a2073cd8881"),
        (RAWPRICE_FAMILY, "DISTANCE_FROM_MOVING_AVERAGE_PRICE",
         dict(price_col="adjClose", window=20),
         "fid_bbfe5d37afa934923eef37cce22d03135a4505416ee2fe1af4341cba127b9e7c"),
        (BREADTH_FAMILY, "POSITIVE_RETURN_FRACTION",
         dict(price_col="adjClose", lookback=1),
         "fid_31ad4f0a6f921752513b528646aad46e842bdae0503017dc24f34ac37ba689b7"),
    ])
    def test_golden_published_feature_ids(self, family, key, kwargs, expected_id):
        """Pinned ids for stable published templates. A change here means a
        computation's identity moved -- intentional only with explicit review."""
        published = [s for s in family[key](**kwargs) if s.publish]
        assert published[-1].feature_id == expected_id


# ---------------------------------------------------------------------------
# Name round-trip  (would have caught the lossy var/beta tags, B3)
# ---------------------------------------------------------------------------
class TestNameRoundTrip:
    @pytest.mark.parametrize("ns", [
        FeatureNameSpec("px", "ret", "logret", {"lb": 5}, "raw"),
        FeatureNameSpec("px", "prc", "dma", {"w": 20}, "raw"),
        FeatureNameSpec("px", "prc", "diff", {"sw": 5, "lw": 20}, "raw"),
        FeatureNameSpec("mkt", "breadth", "retpos", {"lb": 1, "thr": 0}, "raw"),
        FeatureNameSpec("px", "trend", "regbeta", {"w": 10, "ma": 20}, "logma"),
    ])
    def test_int_params_round_trip(self, ns):
        name = make_feature_name(ns.domain, ns.family, ns.signal, ns.params, ns.state)
        parsed = parse_feature_name(name)
        assert parsed.domain == ns.domain
        assert parsed.family == ns.family
        assert parsed.signal == ns.signal
        assert parsed.state == ns.state
        assert parsed.params == ns.params

    def test_empty_params_render_as_none(self):
        name = make_feature_name("px", "tr", "diff", {}, "raw")
        assert name == "px__tr__diff__none__raw"
        assert parse_feature_name(name).params == {}

    @pytest.mark.xfail(reason="Inherent limitation of the rendered-string grammar: "
                              "string-valued params (e.g. drc='gt') have no key/value "
                              "delimiter, so parse_feature_name cannot recover them. "
                              "Use name_spec for lossless structured access "
                              "(see test_string_params_lossless_via_name_spec).",
                       strict=True)
    def test_string_valued_params_round_trip(self):
        ns = FeatureNameSpec("mkt", "breadth", "retpos",
                             {"lb": 1, "thr": 0, "drc": "gt"}, "raw")
        name = make_feature_name(ns.domain, ns.family, ns.signal, ns.params, ns.state)
        assert parse_feature_name(name).params == ns.params


class TestNameSpecIsFirstClass:
    """T3: families emit a structured name_spec, so name info is available
    losslessly without reparsing the rendered string."""

    def test_every_published_family_spec_carries_name_spec(self):
        from util.features.families.returns import RETURN_FAMILY
        from util.features.families.price import RAWPRICE_FAMILY
        from util.features.families.breadth import BREADTH_FAMILY
        cases = [
            (RETURN_FAMILY, "REALIZED_VOLATILITY", dict(price_col="adjClose", window=20)),
            (RAWPRICE_FAMILY, "DISTANCE_FROM_MOVING_AVERAGE_PRICE",
             dict(price_col="adjClose", window=20)),
            (BREADTH_FAMILY, "POSITIVE_RETURN_FRACTION",
             dict(price_col="adjClose", lookback=1)),
        ]
        for family, key, kwargs in cases:
            for spec in family[key](**kwargs):  # incl. intermediates
                assert spec.name_spec is not None, spec.name
                # rendered name is derived from the structured spec
                ns = spec.name_spec
                assert spec.name == make_feature_name(
                    ns.domain, ns.family, ns.signal, ns.params, ns.state)

    def test_string_params_lossless_via_name_spec(self):
        """The drc='gt' that the string parser loses is recoverable structurally."""
        from util.features.families.breadth import BREADTH_FAMILY
        from util.features.transforms import feature_name_spec_from_feature
        published = [s for s in BREADTH_FAMILY["POSITIVE_RETURN_FRACTION"](
            price_col="adjClose", lookback=1) if s.publish][-1]
        ns = feature_name_spec_from_feature(published)
        assert ns.params["drc"] == "gt"
        assert ns.params == {"lb": 1, "thr": 0, "drc": "gt"}


# ---------------------------------------------------------------------------
# Builder behaviors
# ---------------------------------------------------------------------------
class TestBuilder:
    def test_build_published_filters_intermediates(self, panel):
        specs = RAWPRICE_FAMILY["DISTANCE_FROM_MOVING_AVERAGE_PRICE"](
            price_col="adjClose", window=20)
        # template emits maprc(F) + mapd(F) + dma(T)
        assert sum(s.publish for s in specs) == 1
        out = FeatureBuilder(panel).build_published(specs)
        assert list(out) == ["px__prc__dma__w20__raw"]

    def test_build_returns_all_including_intermediates(self, panel):
        specs = RAWPRICE_FAMILY["DISTANCE_FROM_MOVING_AVERAGE_PRICE"](
            price_col="adjClose", window=20)
        out = FeatureBuilder(panel).build(specs)
        assert len(out) == len({s.feature_id for s in specs})

    def test_missing_dependency_raises(self, panel):
        orphan_dep = make_spec(primitive="ts_mean", inputs={"x": col("adjClose")},
                               params={"window": 5}, name="n__a__b__none__raw")
        dependent = make_spec(primitive="scale", inputs={"x": feat(orphan_dep)},
                              params={"scaler": 2.0}, name="n__a__c__none__raw")
        with pytest.raises(ValueError, match="missing feature"):
            FeatureBuilder(panel).build([dependent])  # dep not in graph

    def test_no_cross_symbol_leakage(self, panel):
        """Rolling window must reset at each symbol boundary."""
        spec = make_spec(primitive="ts_mean", inputs={"x": col("adjClose")},
                         params={"window": 3}, name="n__a__b__none__raw")
        out = FeatureBuilder(panel).build([spec])
        s = list(out.values())[0]
        # first 2 rows of every symbol are NaN (min_periods == window == 3)
        first_idx = panel.groupby("symbol").head(2).index
        assert s.loc[first_idx].isna().all()
        # row 3 of BBB equals mean of BBB's first 3 closes only
        bbb = panel.index[panel["symbol"] == "BBB"]
        expected = panel.loc[bbb[:3], "adjClose"].mean()
        assert s.loc[bbb[2]] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Numeric correctness of primitives
# ---------------------------------------------------------------------------
class TestPrimitiveNumerics:
    def test_ts_return_is_log_ratio(self, panel):
        out = prim_ts_return(panel, "symbol", panel["adjClose"], lookback=5)
        aaa = panel.index[panel["symbol"] == "AAA"]
        i = aaa[10]
        expected = np.log(panel.loc[i, "adjClose"] / panel.loc[aaa[5], "adjClose"])
        assert out.loc[i] == pytest.approx(expected)
        assert out.loc[aaa[:5]].isna().all()

    def test_true_range(self, panel):
        out = prim_tr(panel, "symbol", panel["adjHigh"], panel["adjLow"],
                      panel["adjClose"])
        aaa = panel.index[panel["symbol"] == "AAA"]
        i = aaa[7]
        prev_close = panel.loc[aaa[6], "adjClose"]
        hi, lo = panel.loc[i, "adjHigh"], panel.loc[i, "adjLow"]
        expected = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
        assert out.loc[i] == pytest.approx(expected)

    def test_zscore(self):
        x = pd.Series([1.0, 2.0, 3.0])
        mu = pd.Series([0.0, 0.0, 0.0])
        sigma = pd.Series([1.0, 1.0, 1.0])
        out = prim_zscore(x, mu, sigma, eps=0.0)
        assert list(out) == [1.0, 2.0, 3.0]

    def test_regression_slope_and_r2_on_loglinear(self, loglinear_panel):
        df = loglinear_panel
        logp = np.log(df["adjClose"])
        slope = prim_ts_reg_slope(df, "symbol", logp, window=20)
        r2 = prim_ts_reg_r2(df, "symbol", logp, window=20)
        # exact linear -> slope == g (0.01), r2 == 1 on full windows
        assert slope.dropna().iloc[-1] == pytest.approx(0.01, abs=1e-9)
        assert r2.dropna().iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_regtrend_vectorized_matches_simple(self, panel):
        logp = np.log(panel["adjClose"])
        simple = PRIMITIVES["ts_reg_slope"](
            panel, sym_col="symbol", x=logp, window=20)
        vect = PRIMITIVES["ts_reg_slope_vectorized"](
            panel, date_col="date", sym_col="symbol", x=logp, window=20)
        both = pd.concat([simple, vect], axis=1).dropna()
        assert len(both) > 0
        np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=1e-9)

    def test_tail_mean_fast_matches_simple(self, panel):
        ret = prim_ts_return(panel, "symbol", panel["adjClose"], lookback=1)
        simple = PRIMITIVES["ts_tail_mean"](
            panel, sym_col="symbol", x=ret, window=40, q=0.1, tail="lower")
        fast = PRIMITIVES["ts_tail_mean_fast"](
            panel, date_col="date", sym_col="symbol", x=ret,
            window=40, q=0.1, tail="lower")
        both = pd.concat([simple, fast], axis=1).dropna()
        assert len(both) > 0
        np.testing.assert_allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=1e-6)


# ---------------------------------------------------------------------------
# cs_fraction  (regression test for the T1 fix: le branch + NaN handling)
# ---------------------------------------------------------------------------
class TestBeta:
    """T5: beta is computable offline by injecting the market price series."""

    def test_injected_market_gives_known_betas(self):
        from util.features.families.beta import spec_ts_beta_to_market

        dates = pd.bdate_range("2023-01-02", periods=80)
        rng = np.random.default_rng(7)
        mkt = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates))),
                        index=dates)
        # A: identical returns -> beta 1 ; B: 2x log-returns -> beta 2
        a = mkt.to_numpy()
        b = 50.0 * (mkt.to_numpy() / mkt.to_numpy()[0]) ** 2
        panel = pd.concat([
            pd.DataFrame({"symbol": "A", "date": dates, "adjClose": a}),
            pd.DataFrame({"symbol": "B", "date": dates, "adjClose": b}),
        ], ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)

        spec = spec_ts_beta_to_market("adjClose", window=20, market_price=mkt)
        out = FeatureBuilder(panel).build_published([spec])
        beta = list(out.values())[0]

        a_last = panel.index[panel["symbol"] == "A"][-1]
        b_last = panel.index[panel["symbol"] == "B"][-1]
        assert beta.loc[a_last] == pytest.approx(1.0, abs=1e-9)
        assert beta.loc[b_last] == pytest.approx(2.0, abs=1e-9)

    def test_injected_market_input_is_in_identity(self):
        """Injecting the series makes it part of the spec inputs (and id)."""
        from util.features.families.beta import spec_ts_beta_to_market
        idx = pd.bdate_range("2023-01-02", periods=5)
        m1 = pd.Series([1.0, 2, 3, 4, 5], index=idx)
        m2 = pd.Series([1.0, 2, 3, 4, 6], index=idx)
        s1 = spec_ts_beta_to_market("adjClose", window=3, market_price=m1)
        s2 = spec_ts_beta_to_market("adjClose", window=3, market_price=m2)
        assert s1.feature_id != s2.feature_id
        assert s1.name == s2.name  # same rendered name, different provenance


class TestCsFraction:
    @pytest.fixture
    def mini(self):
        df = pd.DataFrame({
            "date": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "symbol": ["A", "B", "C", "A", "B", "C"],
        })
        x = pd.Series([0.01, -0.02, np.nan, 0.0, 0.05, -0.01], index=df.index)
        return df, x

    def test_le_branch_does_not_raise(self, mini):
        df, x = mini
        out = prim_cs_fraction(df, "date", x, threshold=0.0, direction="le")
        assert out.notna().any()

    def test_nan_excluded_from_denominator(self, mini):
        df, x = mini
        out = prim_cs_fraction(df, "date", x, threshold=0.0, direction="gt")
        d1 = out[df["date"] == "d1"]
        # valid on d1: A=0.01(>0), B=-0.02 -> 1/2, NOT 1/3 with C(NaN) counted
        assert d1.iloc[0] == pytest.approx(0.5)

    @pytest.mark.parametrize("drc,expected_d2", [
        ("gt", 1 / 3), ("ge", 2 / 3), ("lt", 1 / 3), ("le", 2 / 3),
    ])
    def test_directions(self, mini, drc, expected_d2):
        df, x = mini
        out = prim_cs_fraction(df, "date", x, threshold=0.0, direction=drc)
        assert out[df["date"] == "d2"].iloc[0] == pytest.approx(expected_d2)

    def test_bad_direction_raises(self, mini):
        df, x = mini
        with pytest.raises(ValueError, match="Unsupported direction"):
            prim_cs_fraction(df, "date", x, threshold=0.0, direction="zz")

    def test_all_nan_date_yields_nan(self):
        df = pd.DataFrame({"date": ["d0", "d0"], "symbol": ["A", "B"]})
        x = pd.Series([np.nan, np.nan], index=df.index)
        out = prim_cs_fraction(df, "date", x, threshold=0.0, direction="gt")
        assert out.isna().all()
