"""Breadth entry-point tests: active-universe filter, liquidity panel built
through the shared merge chain, and qualification logic — all local, no
network (fake fetch + local filesystem)."""

import pandas as pd
import pyarrow.fs as pafs
import pytest

from GetFMPData.breadth_data_pull import (
    COMPONENTS, build_liquidity_panel, filter_active_universe, qualify_symbols,
)
from tests.synthetic import UNIVERSE, make_fake_fetch
from util.data_pull.pull import run_pull
from util.dataset_builder.merge_core import BuildStats


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    fs = pafs.LocalFileSystem()
    base = str(tmp_path_factory.mktemp("breadth") / "raw" / "20260612-breadth")
    # Full UNIVERSE (incl. the inactive penny stock CCC) so qualification has
    # a symbol to reject on the liquidity flag itself.
    run_pull(universe=UNIVERSE, components=COMPONENTS,
             start="2025-01-01", end="2025-12-31", label="20260612-breadth",
             base_path=base, fs=fs, fetch_fn=make_fake_fetch(), verbose=False)
    return fs, base


@pytest.fixture(scope="module")
def panel(snapshot):
    fs, base = snapshot
    return build_liquidity_panel(fs, base, BuildStats())


class TestActiveFilter:
    def test_drops_inactive_and_delisted(self):
        active = filter_active_universe(UNIVERSE)
        assert set(active["symbol"]) == {"AAA", "BBB"}


class TestLiquidityPanel:
    def test_pull_writes_only_breadth_components(self, snapshot):
        fs, base = snapshot
        names = {i.base_name for i in fs.get_file_info(pafs.FileSelector(base))}
        assert {"data_adjusteddailyprice_tk0_pd1.parquet",
                "data_unadjusteddailyprice_tk0_pd1.parquet",
                "data_enterprisevalues_tk0_pd1.parquet",
                "data_tickerprofile.parquet"} <= names
        assert not any("incomestatement" in n for n in names)

    def test_panel_has_liquidity_columns(self, panel):
        assert {"adjClose", "rawClose", "exchange", "numberOfShares",
                "marketCapitalization", "price_tr20", "dollarVolume_tr20",
                "turnOver_tr20", "lowLiquidity"} <= set(panel.columns)

    def test_rolling_is_per_symbol(self, panel):
        p = panel.sort_values(["symbol", "date"])
        head = p.groupby("symbol")["price_tr20"].apply(
            lambda s: s.head(19).isna().all())
        assert head.all()


class TestQualification:
    def test_liquid_symbols_qualify_penny_stock_rejected(self, panel):
        qualified, diag = qualify_symbols(panel, qualify_days=5)
        syms = set(qualified["symbol"])
        assert {"AAA", "BBB"} <= syms        # large, liquid
        assert "CCC" not in syms             # $1.5 stock -> price_tr20 < 2
        assert diag["n_qualified"] == len(qualified)
        assert diag["n_failed_flagged"] >= 1

    def test_missing_recent_days_disqualifies(self, panel):
        last_date = panel["date"].max()
        truncated = panel[~((panel["symbol"] == "AAA")
                            & (panel["date"] == last_date))]
        qualified, diag = qualify_symbols(truncated, qualify_days=5)
        assert "AAA" not in set(qualified["symbol"])
        assert "BBB" in set(qualified["symbol"])
        assert diag["n_failed_missing_days"] >= 1

    def test_qualified_carries_latest_metrics(self, panel):
        qualified, _ = qualify_symbols(panel, qualify_days=5)
        row = qualified[qualified["symbol"] == "AAA"].iloc[0]
        assert row["asOfDate"] == panel["date"].max()
        assert row["dollarVolume_tr20"] > 1_000_000
        assert row["marketCapitalization"] > 200_000_000
        assert pd.notna(row["turnOver_tr20"])
