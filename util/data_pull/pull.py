"""Pull FMP data components for a stock universe into a dated S3 snapshot.

Reuses the hardened request layer from `util.data_client.FMPDataClient`
(token-bucket rate limit ~12 req/s, bounded concurrency, tenacity retries
with exponential backoff on 429/5xx/parse errors; failures recorded inline
as {"__error__": ...} so nothing is lost silently).

Differences vs. the original GetFMPData/FMPDataPull.py:
- universe comes in as a DataFrame / CSV path / fresh build, not a hardcoded CSV
- components and date range are parameters (long backtest pull vs. short
  monitoring pull use the same code path)
- tickers are fetched in batches and normalized incrementally, so a full
  20-year pull never holds the whole raw JSON in memory (no raw_data_temp.json
  checkpoint; tenacity retries + the error log replace it)
- output goes to a dated snapshot directory on S3 (raw/<label>/) together
  with the universe profile parquet, an error log, and a pull manifest

Output files under <base_path>:
    data_{component}_tk0_pd{P}.parquet   one per component x period window
    data_tickerprofile.parquet           typed universe profile
    error_log.json                       {ticker: {component: error}}
    pull_manifest.json                   config + per-file row counts
"""

import asyncio
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
import pyarrow.fs as pafs

from util.data_client.FMPDataClient import FMPClient, fetch_fmp_for_tickers
from util.data_client.dataPullHelpers import normalize_df_for_parquet
from util.dataset_builder.s3_io import (
    get_s3_filesystem, raw_snapshot_path, write_json, write_parquet_df,
)
from .components import DataComponent, build_request_plan

DEFAULT_CLIENT_KWARGS: Dict[str, Any] = dict(
    calls_per_second=12,
    safety_margin=0,
    concurrency=25,
    timeout_s=30,
)

# matches the original FMPDataPull.py normalization settings
NORMALIZE_KWARGS = dict(
    datetime_success_ratio=0.9,
    numeric_success_ratio=0.9,
    treat_huge_int_as_string=False,
)

PROFILE_DTYPES = {
    "symbol": "string", "currency": "string", "exchange": "string",
    "industry": "string", "sector": "string",
}
PROFILE_DATE_COLS = ["ipoDate", "delistedDate"]


def get_universe(universe: Union[None, str, Path, pd.DataFrame] = None,
                 client_kwargs: Optional[Dict] = None,
                 max_age_days: Optional[float] = None) -> pd.DataFrame:
    """Resolve the stock universe for a pull.

    - DataFrame: used as-is
    - str/Path: read as the universe CSV written by build_stock_universe
    - None: reuse the default universe CSV when its recorded build age
      (sidecar .meta.json) is within max_age_days, otherwise build a fresh
      universe directly from FMP (network). max_age_days None/0 always
      rebuilds; unknown age (missing sidecar) counts as stale.
    """
    if isinstance(universe, pd.DataFrame):
        return universe
    if universe is not None:
        return pd.read_csv(universe)
    from GetFMPData.build_stock_universe import (
        DEFAULT_OUT, build_stock_universe, universe_age_days)
    if max_age_days:
        age = universe_age_days(DEFAULT_OUT)
        if age is not None and age <= max_age_days:
            print(f"reusing universe {DEFAULT_OUT.name} "
                  f"(built {age:.1f} days ago, max age {max_age_days:g})",
                  flush=True)
            return pd.read_csv(DEFAULT_OUT)
        print(f"universe is stale or age unknown "
              f"(age {'unknown' if age is None else f'{age:.1f}d'}, "
              f"max age {max_age_days:g}) — rebuilding", flush=True)
    return build_stock_universe(client_kwargs=client_kwargs or DEFAULT_CLIENT_KWARGS)


def universe_to_profile(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Type the universe frame into the data_tickerprofile.parquet schema."""
    df = universe_df.copy()
    for col, dtype in PROFILE_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    for col in PROFILE_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _normalize_payload(payload: Any) -> Optional[pd.DataFrame]:
    """Raw JSON list -> normalized DataFrame (None when empty/no columns)."""
    df = pd.DataFrame(payload)
    if df.empty:
        return None
    df = normalize_df_for_parquet(df, **NORMALIZE_KWARGS)
    df = df.dropna(axis=1, how="all")
    return None if df.empty else df


async def _pull_period(tickers: Sequence[str], request_specs: Dict,
                       client: Optional[FMPClient], fetch_fn: Callable,
                       ticker_batch_size: int,
                       collected: Dict[str, List[pd.DataFrame]],
                       error_log: Dict[str, Dict[str, str]],
                       verbose: bool) -> None:
    """Fetch one period window in ticker batches, normalizing incrementally."""
    n_batches = (len(tickers) + ticker_batch_size - 1) // ticker_batch_size
    for b in range(n_batches):
        batch = list(tickers[b * ticker_batch_size:(b + 1) * ticker_batch_size])
        data = await fetch_fn(batch, request_specs, client=client)
        for ticker, per_type in data.items():
            for key, payload in per_type.items():
                if isinstance(payload, dict) and "__error__" in payload:
                    error_log[ticker][key] = payload["__error__"]
                    continue
                df = _normalize_payload(payload)
                if df is not None:
                    collected[key].append(df)
        if verbose:
            print(f"    ticker batch {b + 1}/{n_batches} "
                  f"({len(batch)} tickers) done", flush=True)


def run_pull(*,
             universe: Union[None, str, Path, pd.DataFrame] = None,
             components: Optional[Sequence[Union[str, DataComponent]]] = None,
             start: Union[str, date],
             end: Union[None, str, date] = None,
             label: Optional[str] = None,
             base_path: Optional[str] = None,
             fs: Optional[pafs.FileSystem] = None,
             client_kwargs: Optional[Dict] = None,
             ticker_batch_size: int = 500,
             chunk_years: int = 10,
             max_symbols: Optional[int] = None,
             universe_max_age_days: Optional[float] = 60,
             fetch_fn: Optional[Callable] = None,
             verbose: bool = True) -> Dict:
    """Run a full data pull and write a dated snapshot.

    universe / components / start / end are the flexible inputs; everything
    else has stable defaults. `fetch_fn` swaps the network layer out for
    testing. Returns the pull manifest (also written as pull_manifest.json).
    """
    end = end or date.today()
    label = label or date.today().strftime("%Y%m%d")
    client_kwargs = client_kwargs or DEFAULT_CLIENT_KWARGS
    fs = fs or get_s3_filesystem()
    base_path = base_path or raw_snapshot_path(label)
    fs.create_dir(base_path, recursive=True)

    universe_df = get_universe(universe, client_kwargs,
                               max_age_days=universe_max_age_days)
    tickers = universe_df["symbol"].dropna().astype(str).unique().tolist()
    if max_symbols:
        tickers = tickers[:max_symbols]

    plan = build_request_plan(components, start, end, chunk_years)
    n_requests = sum(len(p["request_specs"]) for p in plan) * len(tickers)
    if verbose:
        print(f"pull '{label}': {len(tickers):,} tickers, "
              f"{len(plan)} period window(s), ~{n_requests:,} requests "
              f"-> {base_path}", flush=True)

    error_log: Dict[str, Dict[str, str]] = defaultdict(dict)
    files: Dict[str, Dict] = {}
    t0 = time.time()

    for period in plan:
        pd_idx = period["period_index"]
        if verbose:
            print(f"  period pd{pd_idx}: {period['from']} -> {period['to']} "
                  f"({sorted(period['request_specs'])})", flush=True)
        collected: Dict[str, List[pd.DataFrame]] = defaultdict(list)

        async def _run(period=period, collected=collected):
            client = FMPClient(**client_kwargs) if fetch_fn is None else None
            await _pull_period(tickers, period["request_specs"], client,
                               fetch_fn or fetch_fmp_for_tickers,
                               ticker_batch_size, collected, error_log, verbose)

        asyncio.run(_run())

        for key, dfs in collected.items():
            df = pd.concat(dfs, ignore_index=True)
            # second pass: batches can disagree on inferred dtypes; renormalize
            # the concatenated frame so the parquet schema is consistent
            df = normalize_df_for_parquet(df, **NORMALIZE_KWARGS)
            path = f"{base_path}/data_{key}_tk0_pd{pd_idx}.parquet"
            write_parquet_df(fs, df, path)
            files[f"data_{key}_tk0_pd{pd_idx}.parquet"] = {
                "rows": len(df),
                "symbols": int(df["symbol"].nunique()) if "symbol" in df else None,
                "columns": len(df.columns),
            }
            if verbose:
                print(f"    wrote data_{key}_tk0_pd{pd_idx}.parquet "
                      f"({len(df):,} rows)", flush=True)
        del collected

    profile = universe_to_profile(universe_df)
    write_parquet_df(fs, profile, f"{base_path}/data_tickerprofile.parquet")
    files["data_tickerprofile.parquet"] = {
        "rows": len(profile), "symbols": int(profile["symbol"].nunique()),
        "columns": len(profile.columns),
    }

    manifest = {
        "label": label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_path": base_path,
        "start": str(start), "end": str(end),
        "chunk_years": chunk_years,
        "components": sorted({k for p in plan for k in p["request_specs"]}),
        "request_plan": [
            {"period_index": p["period_index"], "from": str(p["from"]),
             "to": str(p["to"]), "components": sorted(p["request_specs"])}
            for p in plan
        ],
        "client_kwargs": client_kwargs,
        "ticker_batch_size": ticker_batch_size,
        "universe_max_age_days": universe_max_age_days,
        "n_tickers": len(tickers),
        "n_requests": n_requests,
        "elapsed_seconds": round(time.time() - t0, 1),
        "files": files,
        "n_error_tickers": len(error_log),
        "n_errors": sum(len(v) for v in error_log.values()),
    }
    write_json(fs, dict(error_log), f"{base_path}/error_log.json")
    write_json(fs, manifest, f"{base_path}/pull_manifest.json")
    if verbose:
        print(f"pull '{label}' complete: {len(files)} files, "
              f"{manifest['n_errors']} errors, "
              f"{manifest['elapsed_seconds']}s", flush=True)
    return manifest
