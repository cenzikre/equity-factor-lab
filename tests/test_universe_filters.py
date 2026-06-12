"""Tests for the final universe filters in GetFMPData/build_stock_universe.py."""

import numpy as np
import pandas as pd

from GetFMPData.build_stock_universe import _apply_final_filters


def _sample_df():
    return pd.DataFrame(
        {
            "symbol": ["AAPL", None, "SPAC1", "MSFT", None, "SPAC2"],
            "exchange": ["NASDAQ"] * 6,
            "industry": [
                "Consumer Electronics",
                "Banks",
                "Shell Companies",
                "Software - Infrastructure",
                "Shell Companies",
                "Shell Companies",
            ],
        }
    )


def test_drops_null_symbols_and_shell_companies():
    result = _apply_final_filters(_sample_df())
    assert list(result["symbol"]) == ["AAPL", "MSFT"]
    assert result.index.tolist() == [0, 1]  # reset_index applied


def test_null_symbol_shell_row_counted_once_as_null(tmp_path):
    # Row 4 is both null-symbol and Shell Companies; null_symbol takes priority.
    audit_path = tmp_path / "dropped.csv"
    _apply_final_filters(_sample_df(), audit_path=audit_path)
    audit = pd.read_csv(audit_path)
    assert len(audit) == 4
    assert audit["dropReason"].value_counts().to_dict() == {
        "null_symbol": 2,
        "shell_company": 2,
    }
    assert set(audit.loc[audit["dropReason"] == "shell_company", "symbol"]) == {
        "SPAC1",
        "SPAC2",
    }


def test_no_audit_file_when_nothing_dropped(tmp_path):
    df = pd.DataFrame(
        {"symbol": ["AAPL"], "exchange": ["NASDAQ"], "industry": ["Consumer Electronics"]}
    )
    audit_path = tmp_path / "dropped.csv"
    result = _apply_final_filters(df, audit_path=audit_path)
    assert len(result) == 1
    assert not audit_path.exists()


def test_missing_industry_column_only_drops_nulls():
    df = pd.DataFrame({"symbol": ["AAPL", np.nan]})
    result = _apply_final_filters(df)
    assert list(result["symbol"]) == ["AAPL"]
