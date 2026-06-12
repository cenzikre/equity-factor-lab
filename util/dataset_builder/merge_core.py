"""Point-in-time merge logic and build statistics.

Differences vs. the original ConstructFullData.ipynb logic, all of which are
reported in the build summary:

1. PIT merge key fix — quarterly statements are as-of merged on their public
   availability date (`filingDate`) instead of the report period end date,
   removing the look-ahead window (median ~38 days) the original join had.
   Key metrics borrows the income statement filingDate per (symbol, period
   end); enterprise values stays on period end (market-observable fields).
2. The report period end date is retained as `{prefix}PeriodEnd` (the
   original merge consumed it as the join key and dropped it).
3. Rolling liquidity metrics are computed per symbol. The original applied
   `df.rolling(20)` on the full frame, letting 20-day windows bleed across
   ticker boundaries.
"""

import gc
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .column_specs import FundamentalMergeSpec


class BuildStats:
    """Accumulates per-table prep notes and per-merge-step counters.

    Merge-step counters are summed across symbol batches so the final report
    reflects the whole panel.
    """

    def __init__(self) -> None:
        self.table_notes: "OrderedDict[str, dict]" = OrderedDict()
        self.steps: "OrderedDict[str, dict]" = OrderedDict()
        self.checks: List[str] = []

    def note_table(self, name: str, **kwargs) -> None:
        self.table_notes.setdefault(name, {}).update(kwargs)

    def add_step(self, step: str, **counts) -> None:
        entry = self.steps.setdefault(step, defaultdict(float))
        entry['_calls'] += 1
        for k, v in counts.items():
            entry[k] += v

    def check(self, ok: bool, message: str) -> None:
        status = "PASS" if ok else "FAIL"
        self.checks.append(f"[{status}] {message}")
        if not ok:
            raise AssertionError(f"build validation failed: {message}")

    def checks_unique(self) -> List[str]:
        """Checks run once per batch; report each distinct check once."""
        return list(dict.fromkeys(self.checks))


def prepare_quarterly_table(df: pd.DataFrame, spec: FundamentalMergeSpec,
                            stats: BuildStats,
                            external_avail: Optional[pd.DataFrame] = None,
                            fallback_lag_days: int = 90) -> pd.DataFrame:
    """Turn a raw quarterly table into an as-of merge right side.

    Output frame:
    - `date` holds the public availability date (PIT key), normalized to
      midnight and never earlier than the period end
    - `{prefix}PeriodEnd` holds the report period end date
    - deduplicated on (symbol, period end) and (symbol, availability date),
      keeping the latest filing / latest period respectively
    - globally sorted by the availability date (merge_asof requirement)
    """
    cols = [c for c in spec.columns if c in df.columns]
    d = df[cols].copy()
    raw_rows = len(d)

    if spec.borrow_pit:
        if external_avail is None:
            raise ValueError(f"{spec.name}: borrow_pit requires external_avail")
        avail_map = (
            external_avail[['symbol', 'date', 'filingDate']]
            .sort_values(['symbol', 'date', 'filingDate'])
            .drop_duplicates(['symbol', 'date'], keep='last')
        )
        d = d.merge(avail_map, on=['symbol', 'date'], how='left')
        borrowed = d['filingDate'].notna().sum()
        stats.note_table(spec.name, borrowed_filing_dates=int(borrowed),
                         borrowed_coverage=borrowed / max(len(d), 1))

    # one row per (symbol, period end): keep the latest filing
    sort_cols = ['symbol', 'date'] + (['filingDate'] if 'filingDate' in d.columns else [])
    d = d.sort_values(sort_cols)
    n0 = len(d)
    d = d.drop_duplicates(['symbol', 'date'], keep='last')
    dup_period = n0 - len(d)

    if spec.pit_key is None and not spec.borrow_pit:
        avail = d['date'].copy()
        null_avail = 0
    else:
        avail = d['filingDate'].dt.normalize()
        null_avail = int(avail.isna().sum())
        avail = avail.fillna(d['date'] + pd.Timedelta(days=fallback_lag_days))

    # results cannot be public before the period ends; clamp bad filing dates
    neg_lag = int((avail < d['date']).sum())
    avail = avail.where(avail >= d['date'], d['date'])
    d['_avail'] = avail

    # one row per (symbol, availability date): keep the latest period end
    # (handles late filers releasing several quarters on the same day)
    d = d.sort_values(['symbol', '_avail', 'date'])
    n1 = len(d)
    d = d.drop_duplicates(['symbol', '_avail'], keep='last')
    dup_avail = n1 - len(d)

    lag = (d['_avail'] - d['date']).dt.days
    stats.note_table(
        spec.name,
        raw_rows=raw_rows, final_rows=len(d), symbols=int(d['symbol'].nunique()),
        dup_period_end_dropped=int(dup_period), dup_avail_date_dropped=int(dup_avail),
        null_avail_filled=null_avail, negative_lag_clamped=neg_lag,
        lag_days_p5=float(lag.quantile(0.05)), lag_days_p50=float(lag.quantile(0.50)),
        lag_days_p95=float(lag.quantile(0.95)),
        period_end_min=str(d['date'].min().date()), period_end_max=str(d['date'].max().date()),
    )

    renames = {'date': f'{spec.prefix}PeriodEnd'}
    renames.update(spec.right_renames)
    d = d.rename(columns=renames).rename(columns={'_avail': 'date'})
    d = d.sort_values(['date', f'{spec.prefix}PeriodEnd']).reset_index(drop=True)
    return d


def merge_price_tables(adj: pd.DataFrame, unadj: pd.DataFrame,
                       stats: BuildStats) -> pd.DataFrame:
    """Outer-join adjusted and unadjusted daily prices into the price spine."""
    a = adj[['symbol', 'date', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'volume']] \
        .rename(columns={'volume': 'adjVolume'})
    u = unadj[['symbol', 'date', 'adjClose', 'volume']] \
        .rename(columns={'adjClose': 'rawClose', 'volume': 'rawVolume'})
    stats.check(not a.duplicated(['symbol', 'date']).any(),
                "adjusted price has unique (symbol, date)")
    stats.check(not u.duplicated(['symbol', 'date']).any(),
                "unadjusted price has unique (symbol, date)")
    out = a.merge(u, on=['symbol', 'date'], how='outer')
    stats.add_step('1. price spine (adj ⟕⟖ unadj)',
                   adj_rows=len(a), unadj_rows=len(u), rows_out=len(out),
                   adj_only=out['rawClose'].isna().sum(),
                   unadj_only=out['adjClose'].isna().sum())
    return out.sort_values(['symbol', 'date']).reset_index(drop=True)


def merge_profile(panel: pd.DataFrame, profile: pd.DataFrame,
                  stats: BuildStats) -> pd.DataFrame:
    """Left-join static ticker profile fields on symbol."""
    stats.check(profile['symbol'].is_unique, "profile has unique symbols")
    rows_in = len(panel)
    out = panel.merge(profile, how='left', on='symbol')
    stats.check(len(out) == rows_in, "profile join preserved row count")
    stats.add_step('2. + ticker profile',
                   rows_in=rows_in, rows_out=len(out),
                   rows_without_profile=out['exchange'].isna().sum())
    return out


def asof_merge_step(panel: pd.DataFrame, right: pd.DataFrame,
                    spec: FundamentalMergeSpec, step_label: str,
                    stats: BuildStats) -> pd.DataFrame:
    """One as-of merge of a prepared quarterly table onto the daily panel.

    The panel must be (and stays) sorted by (date, symbol); merge_asof
    preserves left ordering, so consecutive steps need no re-sorting.
    """
    if spec.left_renames:
        panel = panel.rename(columns=spec.left_renames)
    collisions = set(panel.columns) & (set(right.columns) - {'symbol', 'date'})
    stats.check(not collisions,
                f"{spec.name}: no column collisions"
                + (f" ({sorted(collisions)})" if collisions else ""))

    rows_in, cols_in = panel.shape
    out = pd.merge_asof(panel, right, by='symbol', on='date',
                        direction='backward', allow_exact_matches=True)
    stats.check(len(out) == rows_in, f"{spec.name}: merge_asof preserved row count")

    coverage_col = f'{spec.prefix}PeriodEnd'
    staleness = (out['date'] - out[coverage_col]).dt.days
    stats.add_step(step_label,
                   rows_in=rows_in, rows_out=len(out),
                   cols_added=out.shape[1] - cols_in,
                   rows_matched=out[coverage_col].notna().sum(),
                   stale_rows_gt_400d=(staleness > 400).sum())
    return out


def add_liquidity_flags(panel: pd.DataFrame, stats: BuildStats) -> pd.DataFrame:
    """20-day rolling liquidity metrics and the binary lowLiquidity flag.

    Requires the panel sorted by (symbol, date). Windows are computed per
    symbol; NaN metrics (first 19 days, missing inputs) do not trigger the
    flag, matching the original np.where semantics.
    """
    sym = panel['symbol']

    def roll20(s: pd.Series) -> pd.Series:
        return s.groupby(sym.to_numpy()).transform(lambda x: x.rolling(20).mean())

    panel['price_tr20'] = roll20(panel['adjClose'])
    panel['dollarVolume'] = panel['rawClose'] * panel['rawVolume']
    panel['dollarVolume_tr20'] = roll20(panel['dollarVolume'])
    panel['turnOver'] = panel['adjVolume'] / panel['numberOfShares']
    panel['turnOver_tr20'] = roll20(panel['turnOver'])

    panel['lowLiquidity'] = np.where(
        (panel['marketCapitalization'] < 200_000_000)
        | (panel['dollarVolume_tr20'] < 1_000_000)
        | (panel['price_tr20'] < 2)
        | (panel['turnOver_tr20'] < 0.0005), 1, 0
    )
    stats.add_step('8. liquidity flags',
                   rows=len(panel),
                   flagged_low_liquidity=int(panel['lowLiquidity'].sum()))
    return panel


def build_batch_panel(adj: pd.DataFrame, unadj: pd.DataFrame,
                      profile: pd.DataFrame,
                      prepared_tables: "OrderedDict[str, pd.DataFrame]",
                      specs: List[FundamentalMergeSpec],
                      stats: BuildStats,
                      symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """Run the full merge chain for one ticker batch.

    prepared_tables maps spec.datatype -> output of prepare_quarterly_table
    (full universe; sliced to the batch here). Returns the batch panel
    sorted by (symbol, date).
    """
    panel = merge_price_tables(adj, unadj, stats)
    del adj, unadj
    panel = merge_profile(panel, profile, stats)

    panel = panel.sort_values(['date', 'symbol']).reset_index(drop=True)
    for i, spec in enumerate(specs):
        right = prepared_tables[spec.datatype]
        if symbols is not None:
            right = right[right['symbol'].isin(symbols)]
        label = f"{i + 3}. + {spec.name} (asof "
        label += "period end)" if (spec.pit_key is None and not spec.borrow_pit) else "filing date)"
        panel = asof_merge_step(panel, right, spec, label, stats)
        gc.collect()

    panel = panel.sort_values(['symbol', 'date']).reset_index(drop=True)
    panel = add_liquidity_flags(panel, stats)
    return panel


def summarize_batch(panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per-symbol and per-date summaries used for the final report."""
    per_symbol = panel.groupby('symbol').agg(
        first_date=('date', 'min'), last_date=('date', 'max'),
        n_days=('date', 'size'),
        qualified_days=('lowLiquidity', lambda s: int((s == 0).sum())),
        incm_covered_days=('incmPeriodEnd', lambda s: int(s.notna().sum())),
    )
    per_date = panel.groupby('date').agg(
        n_tickers=('symbol', 'size'),
        qualified=('lowLiquidity', lambda s: int((s == 0).sum())),
    )
    return {'per_symbol': per_symbol, 'per_date': per_date}
