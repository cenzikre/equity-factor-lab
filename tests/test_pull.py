"""Tests for util.data_pull.pull (orchestration, normalization, snapshot
layout) using a fake fetch layer — no network, local filesystem."""

import json

import pandas as pd
import pyarrow.fs as pafs
import pytest

from tests.synthetic import UNIVERSE, make_fake_fetch
from util.data_pull.pull import run_pull, universe_to_profile
from util.dataset_builder.s3_io import list_datatype_files, load_datatype


@pytest.fixture
def local_fs():
    return pafs.LocalFileSystem()


@pytest.fixture
def snapshot(tmp_path, local_fs):
    """Run a standard 2-component pull into a tmp snapshot dir."""
    base = str(tmp_path / "raw" / "20260611")
    manifest = run_pull(
        universe=UNIVERSE,
        components=["adjusteddailyprice", "incomestatement"],
        start="2024-01-01", end="2025-12-31",
        label="20260611", base_path=base, fs=local_fs,
        fetch_fn=make_fake_fetch(error_ticker="BBB",
                                 error_component="incomestatement",
                                 call_log=None),
        ticker_batch_size=2, verbose=False,
    )
    return base, manifest


class TestRunPull:
    def test_writes_expected_files(self, snapshot, tmp_path):
        base, manifest = snapshot
        names = {f.base_name for f in
                 pafs.LocalFileSystem().get_file_info(pafs.FileSelector(base))}
        assert names == {
            "data_adjusteddailyprice_tk0_pd1.parquet",
            "data_incomestatement_tk0_pd1.parquet",
            "data_tickerprofile.parquet",
            "error_log.json", "pull_manifest.json",
        }

    def test_price_normalized_dtypes(self, snapshot, local_fs):
        base, _ = snapshot
        df = load_datatype(local_fs, "adjusteddailyprice", base_path=base)
        assert str(df["date"].dtype).startswith("datetime64")
        assert df["volume"].dtype == "float64"  # came in as strings
        assert set(df["symbol"].unique()) == {"AAA", "BBB", "CCC"}
        # business days in 2024-01-01..2025-12-31
        assert len(df) == 3 * len(pd.bdate_range("2024-01-01", "2025-12-31"))

    def test_error_isolated_and_logged(self, snapshot, local_fs):
        base, manifest = snapshot
        with open(f"{base}/error_log.json") as f:
            errors = json.load(f)
        assert errors == {"BBB": {"incomestatement": "FMPError('429 ...')"}}
        assert manifest["n_errors"] == 1
        # BBB's failed component is absent but its other component survived
        incm = load_datatype(local_fs, "incomestatement", base_path=base)
        assert "BBB" not in set(incm["symbol"])
        px = load_datatype(local_fs, "adjusteddailyprice", base_path=base)
        assert "BBB" in set(px["symbol"])

    def test_quarterly_dates_normalized(self, snapshot, local_fs):
        base, _ = snapshot
        incm = load_datatype(local_fs, "incomestatement", base_path=base)
        for col in ("date", "filingDate", "acceptedDate"):
            assert str(incm[col].dtype).startswith("datetime64"), col
        assert incm["revenue"].dtype == "float64"

    def test_manifest_contents(self, snapshot):
        base, manifest = snapshot
        assert manifest["label"] == "20260611"
        assert manifest["n_tickers"] == 3
        assert manifest["components"] == ["adjusteddailyprice", "incomestatement"]
        assert manifest["files"]["data_tickerprofile.parquet"]["rows"] == 3
        assert manifest["request_plan"][0]["period_index"] == 1
        # 3 tickers x 2 components in a single period window
        assert manifest["n_requests"] == 6

    def test_profile_typed(self, snapshot, local_fs):
        base, _ = snapshot
        prof = pd.read_parquet(f"{base}/data_tickerprofile.parquet")
        assert str(prof["ipoDate"].dtype).startswith("datetime64")
        assert str(prof["delistedDate"].dtype).startswith("datetime64")
        assert prof["symbol"].dtype == "string"

    def test_construction_loader_reads_snapshot(self, snapshot, local_fs):
        """list_datatype_files must resolve the dated layout (path change
        reflected through to the construction step)."""
        base, _ = snapshot
        files = list_datatype_files(local_fs, "adjusteddailyprice", base_path=base)
        assert len(files) == 1 and files[0].endswith("_tk0_pd1.parquet")


class TestPeriodChunking:
    def test_multi_period_price_files(self, tmp_path, local_fs):
        base = str(tmp_path / "raw" / "chunked")
        run_pull(
            universe=UNIVERSE, components=["adjusteddailyprice"],
            start="2018-01-01", end="2025-12-31",
            label="chunked", base_path=base, fs=local_fs,
            fetch_fn=make_fake_fetch(), chunk_years=4, verbose=False,
        )
        files = list_datatype_files(local_fs, "adjusteddailyprice", base_path=base)
        assert [f.split("_")[-1] for f in files] == ["pd1.parquet", "pd2.parquet"]
        df = load_datatype(local_fs, "adjusteddailyprice", base_path=base)
        # windows must not overlap: concat has unique (symbol, date)
        assert not df.duplicated(["symbol", "date"]).any()
        assert df["date"].min() == pd.Timestamp("2018-01-01")
        assert df["date"].max() == pd.Timestamp("2025-12-31")

    def test_ticker_batching_covers_all_tickers(self, tmp_path, local_fs):
        call_log = []
        base = str(tmp_path / "raw" / "batched")
        run_pull(
            universe=UNIVERSE, components=["adjusteddailyprice"],
            start="2025-01-01", end="2025-12-31",
            label="batched", base_path=base, fs=local_fs,
            fetch_fn=make_fake_fetch(call_log=call_log),
            ticker_batch_size=2, verbose=False,
        )
        assert [c["tickers"] for c in call_log] == [["AAA", "BBB"], ["CCC"]]


class TestUniverseToProfile:
    def test_types(self):
        prof = universe_to_profile(UNIVERSE)
        assert prof["exchange"].dtype == "string"
        assert prof["delistedDate"].notna().sum() == 1

    def test_max_symbols_restricts(self, tmp_path, local_fs):
        base = str(tmp_path / "raw" / "limited")
        manifest = run_pull(
            universe=UNIVERSE, components=["adjusteddailyprice"],
            start="2025-01-01", end="2025-06-30",
            label="limited", base_path=base, fs=local_fs,
            fetch_fn=make_fake_fetch(), max_symbols=1, verbose=False,
        )
        assert manifest["n_tickers"] == 1
