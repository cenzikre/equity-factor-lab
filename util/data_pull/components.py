"""Data component registry and FMP request-spec builders.

A *component* is one raw datatype written as `data_{key}_tk0_pd{P}.parquet`
(same naming as the original FMPDataPull.py, so the construction step reads
new and legacy pulls identically).

Daily components are chunked into consecutive date windows (pd1, pd2, ...,
default 10 years each, mirroring the original decade splits) to bound
per-request payloads and in-memory batch size. Quarterly components are a
single pd1 file; their `limit` is derived from the requested date range so a
~2-year monitoring pull doesn't fetch 150 quarters per symbol.
"""

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple, Union

DateLike = Union[str, date]

QUARTERLY_LIMIT_CAP = 150       # original full-history setting
QUARTERLY_BUFFER = 4            # extra quarters before the range start so the
                                # as-of merge has a statement to back-fill from


@dataclass(frozen=True)
class DataComponent:
    """One FMP datatype the pull knows how to fetch."""
    key: str                    # parquet name part, e.g. 'adjusteddailyprice'
    endpoint: str
    granularity: str            # 'daily' | 'quarterly'
    extra_params: Dict = field(default_factory=dict)


COMPONENTS: Dict[str, DataComponent] = {c.key: c for c in [
    DataComponent("adjusteddailyprice", "historical-price-eod/dividend-adjusted", "daily"),
    DataComponent("unadjusteddailyprice", "historical-price-eod/non-split-adjusted", "daily"),
    DataComponent("incomestatement", "income-statement", "quarterly"),
    DataComponent("balancesheet", "balance-sheet-statement", "quarterly"),
    DataComponent("cashflow", "cash-flow-statement", "quarterly"),
    DataComponent("keymetrics", "key-metrics", "quarterly"),
    DataComponent("enterprisevalues", "enterprise-values", "quarterly"),
]}

DEFAULT_COMPONENTS: List[str] = list(COMPONENTS)


def _as_date(d: DateLike) -> date:
    return date.fromisoformat(d) if isinstance(d, str) else d


def date_chunks(start: DateLike, end: DateLike,
                chunk_years: int = 10) -> List[Tuple[date, date]]:
    """Split [start, end] into consecutive windows of at most chunk_years.

    Windows are anchored on the start date's month/day, e.g.
    2000-01-01..2025-12-31 with 10-year chunks ->
    (2000-01-01, 2009-12-31), (2010-01-01, 2019-12-31), (2020-01-01, 2025-12-31).
    """
    start, end = _as_date(start), _as_date(end)
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    chunks = []
    lo = start
    while lo <= end:
        nxt = lo.replace(year=lo.year + chunk_years)
        hi = min(nxt - timedelta(days=1), end)
        chunks.append((lo, hi))
        lo = nxt
    return chunks


def quarterly_limit(start: DateLike, end: DateLike,
                    buffer_quarters: int = QUARTERLY_BUFFER,
                    cap: int = QUARTERLY_LIMIT_CAP) -> int:
    """Number of most-recent quarters needed to cover [start, end]."""
    start, end = _as_date(start), _as_date(end)
    quarters = math.ceil((end - start).days / 91.3)
    return min(cap, max(1, quarters) + buffer_quarters)


def resolve_components(components: Optional[Sequence[Union[str, DataComponent]]]
                       ) -> List[DataComponent]:
    """Accept component keys, DataComponent objects (for custom endpoints),
    or None for the full default set."""
    if components is None:
        components = DEFAULT_COMPONENTS
    out = []
    for c in components:
        if isinstance(c, DataComponent):
            out.append(c)
        elif c in COMPONENTS:
            out.append(COMPONENTS[c])
        else:
            raise KeyError(f"unknown component '{c}'; known: {sorted(COMPONENTS)}")
    return out


def build_request_plan(components: Optional[Sequence[Union[str, DataComponent]]],
                       start: DateLike, end: DateLike,
                       chunk_years: int = 10) -> List[Dict]:
    """Expand components x date windows into per-period request specs.

    Returns a list of period entries:
        {"period_index": 1,            # -> pd1 in the parquet filename
         "from": date, "to": date,     # window (daily components only)
         "request_specs": {key: {"endpoint": ..., "params": {...}}, ...}}

    Daily components appear in every period window; quarterly components only
    in period 1 (single fetch sized by `quarterly_limit`). The specs dict is
    directly consumable by `fetch_fmp_for_tickers` ({ticker} substitution).
    """
    comps = resolve_components(components)
    start, end = _as_date(start), _as_date(end)
    windows = date_chunks(start, end, chunk_years)
    qlimit = quarterly_limit(start, end)

    plan = []
    for i, (lo, hi) in enumerate(windows, start=1):
        specs = {}
        for comp in comps:
            if comp.granularity == "daily":
                specs[comp.key] = {
                    "endpoint": comp.endpoint,
                    "params": {"symbol": "{ticker}", "from": lo.isoformat(),
                               "to": hi.isoformat(), **comp.extra_params},
                }
            elif comp.granularity == "quarterly":
                if i == 1:
                    specs[comp.key] = {
                        "endpoint": comp.endpoint,
                        "params": {"symbol": "{ticker}", "period": "quarter",
                                   "limit": qlimit, **comp.extra_params},
                    }
            else:
                raise ValueError(f"unknown granularity '{comp.granularity}'")
        if specs:
            plan.append({"period_index": i, "from": lo, "to": hi,
                         "request_specs": specs})
    return plan
