"""Market-breadth data task: active universe -> price + EV pull -> liquidity
panel -> qualified symbol list.

Per-task entry point for the breadth monitoring report. It reuses the generic
pull (util.data_pull) and the construction merge chain (util.dataset_builder)
end to end; the liquidity definition is merge_core.add_liquidity_flags — the
exact logic the full dataset build applies, so tuning it there changes both.

Steps:
 1. Universe   — default universe CSV (reused within --universe-max-age-days,
                 else rebuilt) filtered to isActivelyTrading and not delisted.
 2. Pull       — adjusted + unadjusted daily prices and enterprise values
                 (numberOfShares / marketCapitalization feed the liquidity
                 logic) for the trailing --months window into raw/<label>/.
 3. Panel      — price spine + profile + EV as-of merge + 20-day rolling
                 liquidity flags via build_batch_panel. Single batch: a
                 15-month active-universe panel (~2M rows) fits in RAM.
 4. Qualify    — a symbol qualifies when it has a row on each of the last
                 --qualify-days panel dates with lowLiquidity == 0 on all of
                 them. The flag's NaN semantics are unchanged (NaN inputs do
                 not flag); qualified symbols with missing market cap are
                 counted in the diagnostics for transparency.

Output:
    s3://<bucket>/<prefix>/raw/<label>/...                            pull snapshot
    s3://<bucket>/<prefix>/data_price-ev-liquidity_<label>.parquet    liquidity panel
    s3://<bucket>/<prefix>/raw/<label>/breadth_qualification.json     config + counts
    s3://<bucket>/<prefix>/raw/<label>/breadth_qualified_symbols.csv
    MarketInternalMonitor/universe/breadth-qualified-<label>.csv      report ticker list

Examples:
    python GetFMPData/breadth_data_pull.py                  # full run, ~18.5k requests
    python GetFMPData/breadth_data_pull.py --skip-pull --label 20260612-breadth \\
        --overwrite                                         # requalify from snapshot
    python GetFMPData/breadth_data_pull.py --max-symbols 25 --label smoketest
"""

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from util.data_pull.pull import (  # noqa: E402
    DEFAULT_CLIENT_KWARGS, get_universe, run_pull,
)
from util.dataset_builder.column_specs import MERGE_SPECS, PROFILE_COLS  # noqa: E402
from util.dataset_builder.merge_core import (  # noqa: E402
    BuildStats, build_batch_panel, prepare_quarterly_table,
)
from util.dataset_builder.s3_io import (  # noqa: E402
    dataset_base_path, get_s3_filesystem, load_datatype, load_single_parquet,
    raw_snapshot_path, write_json, write_parquet_df,
)

COMPONENTS = ["adjusteddailyprice", "unadjusteddailyprice", "enterprisevalues"]
EV_SPEC = next(s for s in MERGE_SPECS if s.datatype == "enterprisevalues")

ADJ_COLS = ["symbol", "date", "adjOpen", "adjHigh", "adjLow", "adjClose", "volume"]
UNADJ_COLS = ["symbol", "date", "adjClose", "volume"]
METRIC_COLS = ["adjClose", "rawClose", "price_tr20", "dollarVolume_tr20",
               "turnOver_tr20", "marketCapitalization", "numberOfShares"]

DEFAULT_QUALIFIED_DIR = REPO_ROOT / "MarketInternalMonitor" / "universe"


def filter_active_universe(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Keep symbols that are actively trading and have no delisted date."""
    active = universe_df[
        universe_df["isActivelyTrading"].astype(bool)
        & universe_df["delistedDate"].isna()
    ].reset_index(drop=True)
    print(f"universe: {len(universe_df):,} rows -> "
          f"{len(active):,} actively trading", flush=True)
    return active


def build_liquidity_panel(fs, raw_path: str, stats: BuildStats) -> pd.DataFrame:
    """Snapshot -> price spine + profile + EV as-of merge + liquidity flags.

    Same merge chain construct_full_data.py runs per ticker batch, restricted
    to the enterprise values spec (the only fundamentals input that
    add_liquidity_flags needs).
    """
    profile = load_single_parquet(fs, "data_tickerprofile.parquet",
                                  base_path=raw_path)
    ev_raw = load_datatype(fs, EV_SPEC.datatype, columns=EV_SPEC.columns,
                           base_path=raw_path)
    prepared = {EV_SPEC.datatype: prepare_quarterly_table(ev_raw, EV_SPEC, stats)}
    adj = load_datatype(fs, "adjusteddailyprice", columns=ADJ_COLS,
                        base_path=raw_path)
    unadj = load_datatype(fs, "unadjusteddailyprice", columns=UNADJ_COLS,
                          base_path=raw_path)
    return build_batch_panel(adj, unadj, profile[PROFILE_COLS], prepared,
                             [EV_SPEC], stats)


def qualify_symbols(panel: pd.DataFrame,
                    qualify_days: int = 5) -> Tuple[pd.DataFrame, Dict]:
    """Qualified = present on each of the last `qualify_days` panel dates
    with lowLiquidity == 0 on all of them.

    Returns (qualified_df, diagnostics). The qualified frame carries the
    profile fields and the latest-day liquidity metrics per symbol.
    """
    qdates = np.sort(panel["date"].unique())[-qualify_days:]
    recent = panel[panel["date"].isin(qdates)]

    per_sym = recent.groupby("symbol").agg(
        days_present=("date", "nunique"),
        flagged_days=("lowLiquidity", "sum"),
    )
    full = per_sym["days_present"] == len(qdates)
    clean = per_sym["flagged_days"] == 0
    qualified_syms = per_sym.index[full & clean]

    latest = (recent.sort_values(["symbol", "date"])
              .drop_duplicates("symbol", keep="last")
              .set_index("symbol"))
    keep = (["date", "exchange", "industry", "sector"]
            + [c for c in METRIC_COLS if c in latest.columns])
    qualified = (latest.loc[qualified_syms, keep]
                 .rename(columns={"date": "asOfDate"})
                 .reset_index())

    diagnostics = {
        "qualify_days": int(qualify_days),
        "qualify_dates": [str(pd.Timestamp(d).date()) for d in qdates],
        "n_symbols_in_panel": int(panel["symbol"].nunique()),
        "n_symbols_recent": int(len(per_sym)),
        "n_qualified": int(len(qualified)),
        "n_failed_flagged": int((full & ~clean).sum()),
        "n_failed_missing_days": int((~full).sum()),
        "n_qualified_missing_marketcap":
            int(qualified["marketCapitalization"].isna().sum()),
    }
    return qualified, diagnostics


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=15,
                    help="trailing window length (ignored when --start given)")
    ap.add_argument("--start", default=None,
                    help="window start YYYY-MM-DD (default: end - months)")
    ap.add_argument("--end", default=None, help="window end (default today)")
    ap.add_argument("--label", default=f"{date.today():%Y%m%d}-breadth",
                    help="snapshot label -> s3 .../raw/<label>/")
    ap.add_argument("--universe-csv", default=None,
                    help="existing universe CSV (default: reuse/rebuild per max age)")
    ap.add_argument("--universe-max-age-days", type=float, default=60,
                    help="reuse the default universe CSV if its recorded build "
                         "age is within N days; 0 forces a rebuild")
    ap.add_argument("--qualify-days", type=int, default=5,
                    help="symbol must be present and unflagged on each of the "
                         "last N panel dates")
    ap.add_argument("--skip-pull", action="store_true",
                    help="reuse the existing raw/<label>/ snapshot and only "
                         "re-run panel construction + qualification")
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="restrict pull to first N symbols (trial runs)")
    ap.add_argument("--ticker-batch-size", type=int, default=500)
    ap.add_argument("--calls-per-second", type=int,
                    default=DEFAULT_CLIENT_KWARGS["calls_per_second"])
    ap.add_argument("--concurrency", type=int,
                    default=DEFAULT_CLIENT_KWARGS["concurrency"])
    ap.add_argument("--qualified-csv", default=None,
                    help="local qualified-list path (default MarketInternal"
                         "Monitor/universe/breadth-qualified-<label>.csv)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing liquidity panel on S3")
    args = ap.parse_args()

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp(date.today())
    start = pd.Timestamp(args.start) if args.start \
        else end - pd.DateOffset(months=args.months)

    fs = get_s3_filesystem()
    raw_path = raw_snapshot_path(args.label)
    panel_path = f"{dataset_base_path()}/data_price-ev-liquidity_{args.label}.parquet"
    if not args.overwrite and fs.get_file_info(panel_path).type.name != "NotFound":
        raise SystemExit(
            f"refusing to overwrite existing s3://{panel_path} (use --overwrite)")

    client_kwargs = dict(DEFAULT_CLIENT_KWARGS,
                         calls_per_second=args.calls_per_second,
                         concurrency=args.concurrency)

    if args.skip_pull:
        print(f"--skip-pull: using existing snapshot s3://{raw_path}", flush=True)
    else:
        universe = get_universe(args.universe_csv, client_kwargs,
                                max_age_days=args.universe_max_age_days)
        active = filter_active_universe(universe)
        run_pull(universe=active, components=COMPONENTS,
                 start=start.date(), end=end.date(), label=args.label,
                 client_kwargs=client_kwargs,
                 ticker_batch_size=args.ticker_batch_size,
                 max_symbols=args.max_symbols)

    print("building liquidity panel "
          "(price spine + EV as-of + 20d rolling flags)", flush=True)
    stats = BuildStats()
    panel = build_liquidity_panel(fs, raw_path, stats)
    print(f"panel: {len(panel):,} rows, {panel['symbol'].nunique():,} symbols, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}",
          flush=True)
    for check in stats.checks_unique():
        print(f"  {check}", flush=True)

    qualified, diag = qualify_symbols(panel, args.qualify_days)
    print(f"qualified: {diag['n_qualified']:,} of {diag['n_symbols_recent']:,} "
          f"symbols with recent data "
          f"(flagged: {diag['n_failed_flagged']:,}, "
          f"missing recent days: {diag['n_failed_missing_days']:,}, "
          f"qualified w/o market cap: {diag['n_qualified_missing_marketcap']:,})",
          flush=True)

    write_parquet_df(fs, panel, panel_path)
    qualification = {
        "label": args.label,
        "start": str(start.date()), "end": str(end.date()),
        "components": COMPONENTS,
        "liquidity_logic": "util.dataset_builder.merge_core.add_liquidity_flags",
        **diag,
    }
    write_json(fs, qualification, f"{raw_path}/breadth_qualification.json")
    with fs.open_output_stream(f"{raw_path}/breadth_qualified_symbols.csv") as f:
        f.write(qualified.to_csv(index=False).encode("utf-8"))

    out_csv = Path(args.qualified_csv) if args.qualified_csv \
        else DEFAULT_QUALIFIED_DIR / f"breadth-qualified-{args.label}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    qualified.to_csv(out_csv, index=False)

    print(f"done:\n  s3://{panel_path}\n"
          f"  s3://{raw_path}/breadth_qualification.json\n"
          f"  s3://{raw_path}/breadth_qualified_symbols.csv\n"
          f"  {out_csv}", flush=True)


if __name__ == "__main__":
    main()
