"""Pull FMP data components for the stock universe into a dated S3 snapshot.

Generic runner for both pipeline modes (no per-task entry points yet):
  - long pull for modeling/backtesting:   --start 2006-01-01
  - short pull for monitoring/reporting:  --start <~2 years ago>   (weekly)

By default the stock universe CSV is reused when its recorded build age
(sidecar .meta.json) is within --universe-max-age-days (60); otherwise it
is rebuilt fresh from FMP (build_stock_universe, ~27k requests / ~37 min).
Pass --universe-csv to use a specific file, or --universe-max-age-days 0
to force a rebuild.

Output: s3://<bucket>/<prefix>/raw/<label>/data_{component}_tk0_pd{P}.parquet
plus data_tickerprofile.parquet, error_log.json, pull_manifest.json.
Consume with:  python GetFMPData/construct_full_data.py --raw-date <label>

Examples:
    python GetFMPData/fmp_data_pull.py --start 2006-01-01
    python GetFMPData/fmp_data_pull.py --start 2024-06-01 \\
        --components adjusteddailyprice unadjusteddailyprice keymetrics
    python GetFMPData/fmp_data_pull.py --start 2024-06-01 --max-symbols 20 \\
        --label smoketest   # small trial run
"""

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from util.data_pull.components import DEFAULT_COMPONENTS  # noqa: E402
from util.data_pull.pull import DEFAULT_CLIENT_KWARGS, run_pull  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="pull window start (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="pull window end (default today)")
    ap.add_argument("--components", nargs="+", default=None,
                    choices=DEFAULT_COMPONENTS, metavar="COMPONENT",
                    help=f"subset of {DEFAULT_COMPONENTS} (default all)")
    ap.add_argument("--universe-csv", default=None,
                    help="existing universe CSV (default: reuse/rebuild per max age)")
    ap.add_argument("--universe-max-age-days", type=float, default=60,
                    help="reuse the default universe CSV if its recorded build "
                         "age is within N days; 0 forces a rebuild "
                         "(ignored when --universe-csv is given)")
    ap.add_argument("--label", default=date.today().strftime("%Y%m%d"),
                    help="snapshot label -> s3 .../raw/<label>/")
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="restrict to first N symbols (trial runs)")
    ap.add_argument("--ticker-batch-size", type=int, default=500)
    ap.add_argument("--chunk-years", type=int, default=10,
                    help="daily-price period window size (pd1, pd2, ...)")
    ap.add_argument("--calls-per-second", type=int,
                    default=DEFAULT_CLIENT_KWARGS["calls_per_second"])
    ap.add_argument("--concurrency", type=int,
                    default=DEFAULT_CLIENT_KWARGS["concurrency"])
    args = ap.parse_args()

    client_kwargs = dict(DEFAULT_CLIENT_KWARGS,
                         calls_per_second=args.calls_per_second,
                         concurrency=args.concurrency)
    run_pull(
        universe=args.universe_csv,
        components=args.components,
        start=args.start,
        end=args.end,
        label=args.label,
        client_kwargs=client_kwargs,
        ticker_batch_size=args.ticker_batch_size,
        chunk_years=args.chunk_years,
        max_symbols=args.max_symbols,
        universe_max_age_days=args.universe_max_age_days,
    )


if __name__ == "__main__":
    main()
