"""Tests for util.dataset_builder.s3_io path handling and batching."""

import pandas as pd
import pyarrow.fs as pafs
import pytest

from util.dataset_builder.s3_io import (
    dataset_base_path, list_datatype_files, make_symbol_batches,
    raw_snapshot_path, symbol_row_counts, write_parquet_df,
)


class TestPaths:
    def test_raw_snapshot_under_dataset_root(self):
        assert raw_snapshot_path("20260611") == \
            f"{dataset_base_path()}/raw/20260611"

    def test_list_orders_by_period(self, tmp_path):
        fs = pafs.LocalFileSystem()
        df = pd.DataFrame({"symbol": ["A"], "date": [pd.Timestamp("2024-01-02")]})
        for pd_i in (2, 1, 10):
            write_parquet_df(fs, df, str(tmp_path / f"data_x_tk0_pd{pd_i}.parquet"))
        write_parquet_df(fs, df, str(tmp_path / "data_y_tk0_pd1.parquet"))
        files = list_datatype_files(fs, "x", base_path=str(tmp_path))
        assert [f.rsplit("_", 1)[1] for f in files] == \
            ["pd1.parquet", "pd2.parquet", "pd10.parquet"]

    def test_list_missing_datatype_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            list_datatype_files(pafs.LocalFileSystem(), "zzz",
                                base_path=str(tmp_path))


class TestBatching:
    def test_symbol_row_counts_across_period_files(self, tmp_path):
        fs = pafs.LocalFileSystem()
        d1 = pd.DataFrame({"symbol": ["A"] * 3 + ["B"] * 1})
        d2 = pd.DataFrame({"symbol": ["A"] * 2 + ["C"] * 5})
        write_parquet_df(fs, d1, str(tmp_path / "data_px_tk0_pd1.parquet"))
        write_parquet_df(fs, d2, str(tmp_path / "data_px_tk0_pd2.parquet"))
        counts = symbol_row_counts(fs, ["px"], base_path=str(tmp_path))
        assert counts.to_dict() == {"A": 5, "C": 5, "B": 1}

    def test_make_symbol_batches_balances_rows(self):
        counts = pd.Series({"A": 100, "B": 60, "C": 50, "D": 40, "E": 10})
        batches = make_symbol_batches(counts, 2)
        # greedy descending packing: A->b0; B,C->b1 (110); D->b0 (140); E->b1
        totals = sorted(int(sum(counts[s] for s in b)) for b in batches)
        assert totals == [120, 140]
        assert sorted(s for b in batches for s in b) == list("ABCDE")

    def test_more_batches_than_symbols(self):
        counts = pd.Series({"A": 5, "B": 3})
        batches = make_symbol_batches(counts, 10)
        assert sorted(s for b in batches for s in b) == ["A", "B"]
        assert all(b for b in batches)
