"""Reconstruct the full daily price + fundamentals panel with PIT-correct merges.

Streams the build in balanced ticker batches so the whole panel (~26M rows x
~200 cols, far larger than this instance's RAM) never has to be materialized
at once: each batch runs the complete merge chain and is appended as row
groups to a single parquet file on S3.

Usage:
    python GetFMPData/construct_full_data.py                  # full build
    python GetFMPData/construct_full_data.py --max-symbols 30 --out-name data_smoketest.parquet

Output (defaults):
    s3://<bucket>/<prefix>/data_price-profile-ev-incm-bal-cf-km_<label>.parquet
    documentation/dataset-construction-summary-<label>.md
"""

import argparse
import gc
import resource
import sys
import time
from collections import OrderedDict
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from util.dataset_builder.column_specs import (  # noqa: E402
    MERGE_SPECS, PRICE_DATATYPES, PROFILE_COLS, S3_REGION,
)
from util.dataset_builder.s3_io import (  # noqa: E402
    StreamingParquetWriter, dataset_base_path, get_s3_filesystem,
    load_datatype, load_single_parquet, make_symbol_batches,
    raw_snapshot_path, symbol_row_counts,
)
from util.dataset_builder.merge_core import (  # noqa: E402
    BuildStats, build_batch_panel, prepare_quarterly_table, summarize_batch,
)
from util.dataset_builder.report import render_report  # noqa: E402


def log(msg: str) -> None:
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"[{time.strftime('%H:%M:%S')}] (peak rss {rss_gb:.1f}G) {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=date.today().strftime("%Y%m%d"),
                    help="date label appended to the output filename")
    ap.add_argument("--out-name", default=None,
                    help="override output filename (default uses --label)")
    ap.add_argument("--n-batches", type=int, default=24)
    ap.add_argument("--fallback-lag-days", type=int, default=90)
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="restrict to N symbols (smoke testing)")
    ap.add_argument("--report-path", default=None,
                    help="markdown summary path (default documentation/...)")
    ap.add_argument("--description", default=None,
                    help="paragraph under the report title (default: generic)")
    ap.add_argument("--notes-file", default=None,
                    help="markdown file injected into the report as a "
                         "round-specific section (discussion, caveats, ...)")
    ap.add_argument("--notes-title", default="Build notes",
                    help="heading for the --notes-file section")
    ap.add_argument("--raw-date", default=None,
                    help="read raw inputs from the dated pull snapshot "
                         "raw/<RAW_DATE>/ (default: legacy dataset root)")
    args = ap.parse_args()

    extra_sections = []
    if args.notes_file:
        extra_sections.append((args.notes_title, Path(args.notes_file).read_text()))

    out_name = args.out_name or f"data_price-profile-ev-incm-bal-cf-km_{args.label}.parquet"
    base_path = dataset_base_path()
    raw_path = raw_snapshot_path(args.raw_date) if args.raw_date else base_path
    out_path = f"{base_path}/{out_name}"
    report_path = Path(args.report_path) if args.report_path else \
        REPO_ROOT / "documentation" / f"dataset-construction-summary-{args.label}.md"

    fs = get_s3_filesystem(S3_REGION)
    existing = fs.get_file_info(out_path)
    if existing.type.name != "NotFound":
        raise SystemExit(f"refusing to overwrite existing s3://{out_path}")

    stats = BuildStats()
    config = {"n_batches": args.n_batches, "fallback_lag_days": args.fallback_lag_days,
              "label": args.label, "raw_path": raw_path}

    log(f"loading ticker profile and quarterly tables from s3://{raw_path}")
    profile_full = load_single_parquet(fs, "data_tickerprofile.parquet",
                                       base_path=raw_path)
    profile = profile_full[PROFILE_COLS]

    raw = {spec.datatype: load_datatype(fs, spec.datatype, columns=spec.columns,
                                        base_path=raw_path)
           for spec in MERGE_SPECS}
    incm_avail = raw["incomestatement"][["symbol", "date", "filingDate"]]

    log("preparing quarterly tables (dedupe, PIT availability dates)")
    prepared: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
    for spec in MERGE_SPECS:
        prepared[spec.datatype] = prepare_quarterly_table(
            raw[spec.datatype], spec, stats,
            external_avail=incm_avail if spec.borrow_pit else None,
            fallback_lag_days=args.fallback_lag_days,
        )
        log(f"  {spec.datatype}: {len(prepared[spec.datatype]):,} rows")
    km_note = stats.table_notes.get("key metrics", {})
    km_borrow_pct = km_note.get("borrowed_coverage", 1.0) * 100
    del raw, incm_avail
    gc.collect()

    log("counting daily rows per symbol for batch balancing")
    counts = symbol_row_counts(fs, PRICE_DATATYPES, base_path=raw_path)
    if args.max_symbols:
        counts = counts.sample(args.max_symbols, random_state=7)
    batches = make_symbol_batches(counts, args.n_batches)
    log(f"{len(counts):,} symbols, {int(counts.sum()):,} price rows "
        f"-> {len(batches)} batches")

    writer = StreamingParquetWriter(fs, out_path)
    per_symbol_parts, per_date_acc = [], None
    try:
        for i, syms in enumerate(batches, 1):
            symset = set(syms)
            adj = load_datatype(fs, "adjusteddailyprice", symbols=syms,
                                base_path=raw_path)
            unadj = load_datatype(fs, "unadjusteddailyprice", symbols=syms,
                                  base_path=raw_path)
            panel = build_batch_panel(adj, unadj, profile, prepared,
                                      MERGE_SPECS, stats, symbols=symset)
            del adj, unadj

            summaries = summarize_batch(panel)
            per_symbol_parts.append(summaries["per_symbol"])
            pdte = summaries["per_date"]
            per_date_acc = pdte if per_date_acc is None else \
                per_date_acc.add(pdte, fill_value=0)

            writer.write(panel)
            log(f"batch {i}/{len(batches)}: {len(syms)} symbols, "
                f"{len(panel):,} rows written (total {writer.rows_written:,})")
            del panel, summaries
            gc.collect()
    finally:
        writer.close()

    per_symbol = pd.concat(per_symbol_parts).sort_index()
    stats.check(per_symbol.index.is_unique, "no symbol appears in two batches")

    out_info = fs.get_file_info(out_path)
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(fs.open_input_file(out_path)).metadata
    stats.check(meta.num_rows == writer.rows_written,
                f"final file row count matches rows written ({meta.num_rows:,})")

    final_info = {
        "s3_path": f"s3://{out_path}",
        "n_columns": meta.num_columns,
        "file_size_mb": out_info.size / 1024**2,
        "km_borrow_pct": km_borrow_pct,
    }
    report = render_report(build_label=args.label, config=config, stats=stats,
                           per_symbol=per_symbol, per_date=per_date_acc,
                           profile=profile_full, final_info=final_info,
                           description=args.description,
                           extra_sections=extra_sections)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    log(f"done: s3://{out_path} ({final_info['file_size_mb']:.0f} MB, "
        f"{meta.num_rows:,} rows, {meta.num_columns} cols)")
    log(f"report: {report_path}")


if __name__ == "__main__":
    main()
