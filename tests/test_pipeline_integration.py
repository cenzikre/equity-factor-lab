"""End-to-end pipeline test on synthetic data, entirely local:

    universe (fixture) -> run_pull (fake fetch) -> dated snapshot
    -> prepare_quarterly_table + build_batch_panel (construction chain)
    -> PIT and integrity assertions -> report renders

This is the construction path construct_full_data.py runs per ticker batch,
pointed at a pull snapshot directory instead of the S3 root.
"""

from collections import OrderedDict

import pandas as pd
import pyarrow.fs as pafs
import pytest

from tests.synthetic import UNIVERSE, make_fake_fetch
from util.data_pull.pull import run_pull
from util.dataset_builder.column_specs import MERGE_SPECS, PROFILE_COLS
from util.dataset_builder.merge_core import (
    BuildStats, build_batch_panel, prepare_quarterly_table, summarize_batch,
)
from util.dataset_builder.report import render_report
from util.dataset_builder.s3_io import load_datatype, load_single_parquet


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    fs = pafs.LocalFileSystem()
    base = str(tmp_path_factory.mktemp("pipeline") / "raw" / "20260611")
    manifest = run_pull(
        universe=UNIVERSE, components=None,           # all 7 components
        start="2024-01-01", end="2025-12-31",
        label="20260611", base_path=base, fs=fs,
        fetch_fn=make_fake_fetch(), verbose=False,
    )

    stats = BuildStats()
    profile_full = load_single_parquet(fs, "data_tickerprofile.parquet",
                                       base_path=base)
    incm_raw = load_datatype(fs, "incomestatement", base_path=base)
    prepared = OrderedDict()
    for spec in MERGE_SPECS:
        raw = load_datatype(fs, spec.datatype, base_path=base)
        prepared[spec.datatype] = prepare_quarterly_table(
            raw, spec, stats,
            external_avail=incm_raw[["symbol", "date", "filingDate"]]
            if spec.borrow_pit else None,
        )

    adj = load_datatype(fs, "adjusteddailyprice", base_path=base)
    unadj = load_datatype(fs, "unadjusteddailyprice", base_path=base)
    panel = build_batch_panel(adj, unadj, profile_full[PROFILE_COLS],
                              prepared, MERGE_SPECS, stats)
    return {"panel": panel, "stats": stats, "manifest": manifest,
            "profile": profile_full, "n_price_rows": len(adj)}


class TestPanelIntegrity:
    def test_row_count_preserved_through_all_merges(self, pipeline):
        assert len(pipeline["panel"]) == pipeline["n_price_rows"]

    def test_all_validation_checks_passed(self, pipeline):
        assert all(c.startswith("[PASS]")
                   for c in pipeline["stats"].checks_unique())

    def test_key_columns_present(self, pipeline):
        cols = set(pipeline["panel"].columns)
        assert {"adjClose", "rawClose", "exchange", "numberOfShares",
                "revenue", "totalAssets", "freeCashFlow", "returnOnEquity",
                "evPeriodEnd", "incmPeriodEnd", "balPeriodEnd", "cfPeriodEnd",
                "kmPeriodEnd", "kmFilingDate", "incmFilingDate",
                "lowLiquidity"} <= cols

    def test_collision_renames_applied(self, pipeline):
        cols = set(pipeline["panel"].columns)
        assert {"incmNetIncome", "cfNetIncome", "balAccountsReceivables",
                "cfAccountsReceivables", "evEnterpriseValue",
                "kmEnterpriseValue"} <= cols
        assert "netIncome" not in cols and "enterpriseValue" not in cols


class TestPointInTime:
    def test_no_lookahead_on_filing_date(self, pipeline):
        """Statement data must never appear before its filing date."""
        p = pipeline["panel"]
        for col in ("incmFilingDate", "balFilingDate", "cfFilingDate",
                    "kmFilingDate"):
            rows = p.dropna(subset=[col])
            assert (rows["date"] >= rows[col].dt.normalize()).all(), col

    def test_period_end_never_after_daily_date(self, pipeline):
        p = pipeline["panel"]
        for col in ("evPeriodEnd", "incmPeriodEnd", "balPeriodEnd",
                    "cfPeriodEnd", "kmPeriodEnd"):
            rows = p.dropna(subset=[col])
            assert (rows[col] <= rows["date"]).all(), col

    def test_statement_unavailable_during_filing_lag(self, pipeline):
        """With a 40-day synthetic filing lag, a daily row 10 days after a
        quarter end must still carry the PREVIOUS quarter."""
        p = pipeline["panel"]
        sym = p[p["symbol"] == "AAA"]
        probe = sym[sym["date"] == pd.Timestamp("2025-07-10")]  # Q2 end + 9d
        assert len(probe) == 1
        assert probe["incmPeriodEnd"].iloc[0] == pd.Timestamp("2025-03-31")
        # ...and 50 days after quarter end the new quarter is visible
        probe2 = sym[sym["date"] == pd.Timestamp("2025-08-20")]
        assert probe2["incmPeriodEnd"].iloc[0] == pd.Timestamp("2025-06-30")

    def test_ev_uses_period_end_not_filing(self, pipeline):
        """Enterprise values (market-observable) attach at period end."""
        p = pipeline["panel"]
        sym = p[p["symbol"] == "AAA"]
        probe = sym[sym["date"] == pd.Timestamp("2025-07-10")]
        assert probe["evPeriodEnd"].iloc[0] == pd.Timestamp("2025-06-30")


class TestLiquidityFlags:
    def test_rolling_is_per_symbol(self, pipeline):
        p = pipeline["panel"].sort_values(["symbol", "date"])
        head = p.groupby("symbol")["price_tr20"].apply(
            lambda s: s.head(19).isna().all())
        assert head.all()

    def test_penny_stock_flagged(self, pipeline):
        p = pipeline["panel"]
        ccc = p[(p["symbol"] == "CCC") & p["price_tr20"].notna()]
        assert (ccc["lowLiquidity"] == 1).all()  # $1.5 stock -> price_tr20 < 2
        aaa = p[(p["symbol"] == "AAA") & p["dollarVolume_tr20"].notna()
                & p["marketCapitalization"].notna()]
        assert (aaa["lowLiquidity"] == 0).any()


class TestReport:
    def test_report_renders_from_pipeline_stats(self, pipeline):
        summaries = summarize_batch(pipeline["panel"])
        report = render_report(
            build_label="test", config={"n_batches": 1, "fallback_lag_days": 90,
                                        "raw_path": "tmp/raw/20260611"},
            stats=pipeline["stats"],
            per_symbol=summaries["per_symbol"], per_date=summaries["per_date"],
            profile=pipeline["profile"], final_info={
                "s3_path": "local://test", "n_columns":
                    pipeline["panel"].shape[1], "file_size_mb": 1.0,
                "km_borrow_pct": 100.0},
        )
        assert "## Methodology" in report
        assert "raw/20260611" in report
        assert "3" in report  # symbols
        for check in pipeline["stats"].checks_unique():
            assert check in report
