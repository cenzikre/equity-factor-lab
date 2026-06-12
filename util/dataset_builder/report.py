"""Markdown summary report for a dataset build.

The renderer is build-agnostic: every number comes from the BuildStats /
config / summary frames passed in, and round-specific narrative (what changed
this time, caveats, discussion) is injected by the caller via `description`
and `extra_sections` rather than living in this module.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .merge_core import BuildStats

DEFAULT_DESCRIPTION = (
    "Daily price + fundamentals panel built by `GetFMPData/construct_full_data.py` "
    "(helpers in `util/dataset_builder/`) from the raw FMP parquet files on S3."
)


def _fmt(v) -> str:
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9 and abs(v) < 1e15:
            return f"{int(round(v)):,}"
        return f"{v:,.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _md_table(header: List[str], rows: List[List]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return "\n".join(lines)


def _methodology_section(config: Dict, stats: BuildStats, final_info: Dict) -> str:
    """Point-in-time conventions implemented by merge_core, with the
    round's actual numbers filled in from the build statistics."""
    lags = [n['lag_days_p50'] for n in stats.table_notes.values()
            if n.get('lag_days_p50')]
    lag_txt = f" (median filing lag ~{lags[0]:.0f} days)" if lags else ""
    borrow = final_info.get('km_borrow_pct')
    borrow_txt = f" ({borrow:.1f}% matched)" if borrow is not None else ""
    fallback = config.get('fallback_lag_days', 90)
    return f"""## Methodology (point-in-time conventions)

1. **As-of merge on filing date.** Quarterly statements (income statement,
   balance sheet, cash flow, key metrics) are merged onto the daily spine as
   of their public availability date — `filingDate`, normalized to midnight
   and clamped to ≥ period end — not the report period end date{lag_txt}.
   Key metrics has no filingDate of its own and borrows the income statement
   filingDate by `(symbol, period end)`{borrow_txt}; rows without one fall
   back to period end + {fallback} days. Enterprise values (shares
   outstanding, market cap, EV) is merged on period end, as these are
   market-observable quantities rather than filing disclosures.
2. **Period end dates retained.** Each quarterly table keeps its report
   period as `evPeriodEnd` / `incmPeriodEnd` / `balPeriodEnd` / `cfPeriodEnd`
   / `kmPeriodEnd` (plus the borrowed `kmFilingDate`).
3. **Per-symbol rolling windows.** The 20-day liquidity metrics are computed
   within each symbol, never across ticker boundaries.
4. **Tie/duplicate handling.** Quarterly tables are deduplicated to one row
   per (symbol, period end) keeping the latest filing, and one row per
   (symbol, availability date) keeping the latest period end (late filers can
   release several quarters on the same day).

Availability convention: a statement is treated as usable **on** its filing
date (`allow_exact_matches=True`). For strict next-day-open semantics,
filter on the retained `*FilingDate` / `*AcceptedDate` columns downstream."""


def render_report(*, build_label: str, config: Dict, stats: BuildStats,
                  per_symbol: pd.DataFrame, per_date: pd.DataFrame,
                  profile: pd.DataFrame, final_info: Dict,
                  description: Optional[str] = None,
                  extra_sections: Iterable[Tuple[str, str]] = ()) -> str:
    """Render the build summary markdown.

    description:    paragraph under the title; defaults to a generic one.
    extra_sections: (heading, markdown body) pairs injected verbatim after
                    the Output section — use for round-specific discussion,
                    comparisons with previous builds, caveats, etc.
    """
    total_rows = int(per_symbol['n_days'].sum())
    n_symbols = len(per_symbol)

    # --- merge step table ---
    step_rows = []
    for step, c in stats.steps.items():
        rows_in = c.get('rows_in', c.get('adj_rows', c.get('rows', 0)))
        rows_out = c.get('rows_out', c.get('rows', 0))
        matched = c.get('rows_matched')
        coverage = f"{matched / rows_out * 100:.1f}%" if matched and rows_out else "—"
        step_rows.append([step, int(rows_in), int(rows_out),
                          int(c.get('cols_added', 0) / c.get('_calls', 1)), coverage])
    steps_md = _md_table(
        ["Step", "Rows in", "Rows out", "Cols added", "Coverage (rows with data)"],
        step_rows)

    # --- source table prep ---
    prep_rows = []
    for name, n in stats.table_notes.items():
        prep_rows.append([
            name, n.get('raw_rows', 0), n.get('final_rows', 0), n.get('symbols', 0),
            n.get('dup_period_end_dropped', 0) + n.get('dup_avail_date_dropped', 0),
            n.get('null_avail_filled', 0), n.get('negative_lag_clamped', 0),
            f"{n.get('lag_days_p50', 0):.0f} / {n.get('lag_days_p95', 0):.0f}",
            f"{n.get('period_end_min', '')} → {n.get('period_end_max', '')}",
        ])
    prep_md = _md_table(
        ["Table", "Raw rows", "Used rows", "Symbols", "Dups dropped",
         "Null filing→fallback", "Neg. lag clamped", "Filing lag p50/p95 (d)",
         "Period end range"],
        prep_rows)

    # --- universe ---
    uni = per_symbol.join(profile.set_index('symbol'), how='left')
    active = uni['isActivelyTrading'].fillna(False).astype(bool)
    exch = uni['exchange'].fillna('(missing)').value_counts().head(8)
    sect = uni['sector'].fillna('(missing)').value_counts().head(12)
    exch_md = _md_table(["Exchange", "Symbols"], [[k, int(v)] for k, v in exch.items()])
    sect_md = _md_table(["Sector", "Symbols"], [[k, int(v)] for k, v in sect.items()])

    yearly = per_date.copy()
    yearly['year'] = yearly.index.year
    ytab = yearly.groupby('year').agg(
        avg_daily_tickers=('n_tickers', 'mean'),
        avg_daily_qualified=('qualified', 'mean'))
    ytab['qualified_share'] = ytab['avg_daily_qualified'] / ytab['avg_daily_tickers']
    year_rows = [[str(int(y)), f"{r.avg_daily_tickers:,.0f}", f"{r.avg_daily_qualified:,.0f}",
                  f"{r.qualified_share * 100:.0f}%"] for y, r in ytab.iterrows()]
    years_md = _md_table(
        ["Year", "Avg tickers/day", "Avg qualified/day (lowLiquidity=0)", "Qualified share"],
        year_rows)

    last_date = per_date.index.max()
    last = per_date.loc[last_date]
    checks_md = "\n".join(f"- {c}" for c in stats.checks_unique())

    incm_cov = per_symbol['incm_covered_days'].sum() / max(total_rows, 1)

    extras_md = "\n".join(
        f"## {heading}\n\n{body.strip()}\n" for heading, body in extra_sections)

    return f"""# Full Dataset Construction Summary — {build_label}

{description or DEFAULT_DESCRIPTION}

## Output

| | |
|---|---|
| S3 path | `{final_info['s3_path']}` |
| Raw inputs | {f"`s3://{config['raw_path']}`" if config.get('raw_path') else "—"} |
| Rows | {total_rows:,} |
| Columns | {final_info['n_columns']:,} |
| Symbols | {n_symbols:,} |
| Date range | {per_symbol['first_date'].min().date()} → {per_symbol['last_date'].max().date()} |
| File size | {final_info['file_size_mb']:,.0f} MB (zstd) |
| Ticker batches | {config['n_batches']} (streamed row groups, ~{total_rows // config['n_batches']:,} rows each) |

{extras_md}{_methodology_section(config, stats, final_info)}

## Source tables (after prep)

{prep_md}

"Dups dropped" = duplicate (symbol, period end) + duplicate (symbol,
availability date) rows removed. Filing lag = filingDate − period end.

## Merge steps

{steps_md}

Row counts are summed across the {config['n_batches']} ticker batches. Every as-of merge
preserves the row count of the daily spine (validated per batch) — coverage
below 100% means early daily rows that predate a symbol's first available
statement, not lost rows.
Income statement fields cover {incm_cov * 100:.1f}% of all daily rows.

## Validation checks

{checks_md}

## Stock universe

- **{n_symbols:,} symbols** with daily prices; {int(active.sum()):,} actively trading,
  {int((~active).sum()):,} delisted/inactive per the ticker profile (survivorship-bias-aware universe).
- Median listing span in panel: {per_symbol['n_days'].median():,.0f} trading days.
- On the last panel date ({last_date.date()}): {int(last['n_tickers']):,} tickers,
  {int(last['qualified']):,} qualified (lowLiquidity = 0).

### By exchange

{exch_md}

### By sector

{sect_md}

### Universe breadth by year

{years_md}

`qualified` excludes rows flagged lowLiquidity = 1, i.e. any of:
market cap < $200M, 20d avg dollar volume < $1M, 20d avg price < $2,
20d avg turnover < 0.05%.

## Reuse

```python
from util.dataset_builder.column_specs import MERGE_SPECS, PROFILE_COLS
from util.dataset_builder.s3_io import (get_s3_filesystem, load_datatype,
                                        symbol_row_counts, make_symbol_batches,
                                        StreamingParquetWriter)
from util.dataset_builder.merge_core import (BuildStats, prepare_quarterly_table,
                                             build_batch_panel)
```

`prepare_quarterly_table` + `build_batch_panel` run the whole chain for any
ticker subset; `StreamingParquetWriter` appends batches into one S3 parquet.
See `GetFMPData/construct_full_data.py` for the orchestration.
"""
