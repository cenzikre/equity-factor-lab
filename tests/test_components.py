"""Unit tests for util.data_pull.components (request planning)."""

from datetime import date

import pytest

from util.data_pull.components import (
    COMPONENTS, DEFAULT_COMPONENTS, DataComponent, build_request_plan,
    date_chunks, quarterly_limit, resolve_components,
)


class TestDateChunks:
    def test_decade_split_matches_legacy_layout(self):
        chunks = date_chunks("2000-01-01", "2025-12-31", chunk_years=10)
        assert chunks == [
            (date(2000, 1, 1), date(2009, 12, 31)),
            (date(2010, 1, 1), date(2019, 12, 31)),
            (date(2020, 1, 1), date(2025, 12, 31)),
        ]

    def test_short_range_single_chunk(self):
        chunks = date_chunks("2024-06-01", "2026-06-01", chunk_years=10)
        assert chunks == [(date(2024, 6, 1), date(2026, 6, 1))]

    def test_exact_boundary(self):
        chunks = date_chunks("2010-01-01", "2019-12-31", chunk_years=10)
        assert chunks == [(date(2010, 1, 1), date(2019, 12, 31))]

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            date_chunks("2025-01-01", "2024-01-01")


class TestQuarterlyLimit:
    def test_two_year_monitoring_pull(self):
        # 8 quarters + 4 buffer
        assert quarterly_limit("2024-06-01", "2026-06-01") == 12

    def test_twenty_year_backtest_pull(self):
        lim = quarterly_limit("2006-01-01", "2026-01-01")
        assert 80 <= lim <= 95

    def test_capped_at_legacy_150(self):
        assert quarterly_limit("1980-01-01", "2026-01-01") == 150


class TestResolveComponents:
    def test_default_is_all_seven(self):
        assert len(resolve_components(None)) == 7
        assert [c.key for c in resolve_components(None)] == DEFAULT_COMPONENTS

    def test_subset_by_key(self):
        comps = resolve_components(["adjusteddailyprice", "keymetrics"])
        assert [c.key for c in comps] == ["adjusteddailyprice", "keymetrics"]

    def test_custom_component_object(self):
        custom = DataComponent("ratios", "ratios", "quarterly")
        assert resolve_components([custom]) == [custom]

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            resolve_components(["nope"])


class TestBuildRequestPlan:
    def test_long_pull_structure(self):
        plan = build_request_plan(None, "2006-01-01", "2026-06-10")
        assert [p["period_index"] for p in plan] == [1, 2, 3]
        # quarterly components only in period 1
        assert set(plan[0]["request_specs"]) == set(DEFAULT_COMPONENTS)
        for p in plan[1:]:
            assert set(p["request_specs"]) == {
                "adjusteddailyprice", "unadjusteddailyprice"}

    def test_daily_params_have_window(self):
        plan = build_request_plan(["adjusteddailyprice"], "2024-06-01", "2026-06-10")
        spec = plan[0]["request_specs"]["adjusteddailyprice"]
        assert spec["endpoint"] == "historical-price-eod/dividend-adjusted"
        assert spec["params"] == {"symbol": "{ticker}", "from": "2024-06-01",
                                  "to": "2026-06-10"}

    def test_quarterly_params_have_range_sized_limit(self):
        plan = build_request_plan(["incomestatement"], "2024-06-01", "2026-06-10")
        spec = plan[0]["request_specs"]["incomestatement"]
        assert spec["params"]["period"] == "quarter"
        assert spec["params"]["limit"] == quarterly_limit("2024-06-01", "2026-06-10")
        assert spec["params"]["symbol"] == "{ticker}"

    def test_quarterly_only_pull_has_single_period(self):
        plan = build_request_plan(["balancesheet", "cashflow"],
                                  "2006-01-01", "2026-06-10")
        assert len(plan) == 1
        assert set(plan[0]["request_specs"]) == {"balancesheet", "cashflow"}

    def test_endpoints_match_legacy_pull(self):
        legacy = {
            "adjusteddailyprice": "historical-price-eod/dividend-adjusted",
            "unadjusteddailyprice": "historical-price-eod/non-split-adjusted",
            "incomestatement": "income-statement",
            "balancesheet": "balance-sheet-statement",
            "cashflow": "cash-flow-statement",
            "keymetrics": "key-metrics",
            "enterprisevalues": "enterprise-values",
        }
        assert {k: c.endpoint for k, c in COMPONENTS.items()} == legacy
