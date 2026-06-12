"""
build_stock_universe.py

Build the stock universe file used as the master ticker list for all FMP data pulls.

Output (default):
    GetFMPData/universe/info-stock-universe-usTrading-delistedIncl.csv

Pipeline:
    1. Seed list  — financial-statement-symbol-list filtered to USD trading currency
    2. Profiles   — async concurrent fetch of /profile for every seed symbol
    3. US filter  — keep NYSE / NASDAQ / AMEX, exclude ETFs and Funds
    4. Delisted   — paginated fetch of delisted-companies, joined by symbol
    5. Finalize   — slim to info columns, drop nulls and Shell Companies

Usage:
    python GetFMPData/build_stock_universe.py
    python GetFMPData/build_stock_universe.py --out path/to/custom.csv
    python GetFMPData/build_stock_universe.py --calls-per-second 10 --concurrency 20

Environment:
    FMP_API_KEY must be set (or present in a .env file at the repo root).
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Make util importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.data_client.FMPDataClient import FMPClient, fetch_fmp_all_pages

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX"}

# Default FMPClient constructor kwargs — passed through so every asyncio.run()
# creates a fresh limiter/semaphore bound to its own event loop.
DEFAULT_CLIENT_KWARGS: Dict[str, Any] = dict(
    calls_per_second=12,
    safety_margin=0,
    concurrency=25,
    timeout_s=30,
)

PROFILE_COLS = [
    "symbol", "price", "marketCap", "beta", "changePercentage", "volume",
    "currency", "exchange", "industry", "sector", "country",
    "ipoDate", "isEtf", "isFund", "isActivelyTrading",
]

OUTPUT_COLS = [
    "symbol", "currency", "exchange", "industry", "sector",
    "isActivelyTrading", "ipoDate", "delistedDate",
]

DEFAULT_OUT = Path(__file__).parent / "universe" / "info-stock-universe-usTrading-delistedIncl.csv"


# ---------------------------------------------------------------------------
# Build-timestamp sidecar (file mtime is unreliable — git checkout resets it)
# ---------------------------------------------------------------------------

def _meta_path(csv_path: Path) -> Path:
    return Path(csv_path).with_suffix(".meta.json")


def write_universe_meta(csv_path: Path, n_rows: int) -> None:
    """Record the build timestamp next to the universe CSV."""
    meta = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": n_rows,
        "generator": "build_stock_universe",
    }
    _meta_path(csv_path).write_text(json.dumps(meta, indent=2) + "\n")


def universe_age_days(csv_path: Path = DEFAULT_OUT) -> Optional[float]:
    """Age of the universe CSV in days, from its sidecar build timestamp.

    Returns None when the CSV or sidecar is missing/unreadable — callers
    should treat unknown age as stale.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    try:
        meta = json.loads(_meta_path(csv_path).read_text())
        built_at = datetime.fromisoformat(meta["built_at_utc"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (datetime.now(timezone.utc) - built_at).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# Step 1: Seed symbol list
# ---------------------------------------------------------------------------

async def _fetch_usd_symbols_async(client_kwargs: Dict[str, Any]) -> List[str]:
    client = FMPClient(**client_kwargs)
    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        raw = await client.fetch(session, "financial-statement-symbol-list", {})
    df = pd.DataFrame(raw)
    usd = df.loc[df["tradingCurrency"] == "USD", "symbol"].dropna().unique().tolist()
    print(f"  {len(raw):,} total symbols → {len(usd):,} with USD trading currency")
    return usd


def fetch_usd_symbols(client_kwargs: Dict[str, Any]) -> List[str]:
    """Return all FMP symbols that have financial statements and trade in USD."""
    print("Step 1: Fetching financial-statement-symbol-list...")
    return asyncio.run(_fetch_usd_symbols_async(client_kwargs))


# ---------------------------------------------------------------------------
# Step 2: Company profiles (async, rate-limited, concurrent)
# ---------------------------------------------------------------------------

async def _fetch_one_profile(
    session: aiohttp.ClientSession,
    client: FMPClient,
    symbol: str,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Fetch /profile for one symbol.

    Returns (symbol, record_dict) on success, or (symbol, None) when the
    response is empty / multi-record (e.g. SPAC units), or on error.
    Errors are printed but do not propagate — the symbol is silently dropped.
    """
    try:
        result = await client.fetch(session, "profile", {"symbol": symbol})
    except Exception as e:
        return symbol, {"__error__": repr(e)}

    if not isinstance(result, list) or len(result) == 0:
        return symbol, None  # no data for this symbol

    if len(result) > 1:
        # Multi-record edge cases (e.g. SPAC units like IPOCU)
        return symbol, None

    return symbol, result[0]


async def _fetch_profiles_async(
    symbols: List[str],
    client_kwargs: Dict[str, Any],
) -> Tuple[List[Dict], List[str], int]:
    """
    Fan out /profile requests for all symbols concurrently.

    A fresh FMPClient is created inside this coroutine so its AsyncLimiter and
    Semaphore are always bound to the current event loop (avoids the cross-loop
    reuse warning from aiolimiter when asyncio.run() is called multiple times).

    Results are collected via asyncio.as_completed so the tqdm progress bar
    updates in real time rather than only at the end.

    Returns:
        profiles      — list of raw profile dicts
        error_symbols — symbols that raised exceptions after all retries
        n_skipped     — count of symbols with no usable response (empty/multi)
    """
    client = FMPClient(**client_kwargs)
    profiles: List[Dict] = []
    error_symbols: List[str] = []
    n_skipped = 0

    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        # Create all tasks upfront — the semaphore + limiter inside FMPClient
        # ensure we never exceed API rate or concurrency limits regardless of
        # how many tasks are created.
        tasks = [
            asyncio.create_task(_fetch_one_profile(session, client, sym))
            for sym in symbols
        ]

        with tqdm(total=len(tasks), desc="  Fetching profiles", unit="sym") as pbar:
            for coro in asyncio.as_completed(tasks):
                sym, record = await coro
                if isinstance(record, dict) and "__error__" in record:
                    error_symbols.append(sym)
                elif record is None:
                    n_skipped += 1
                else:
                    profiles.append(record)
                pbar.update(1)

    return profiles, error_symbols, n_skipped


def fetch_company_profiles(symbols: List[str], client_kwargs: Dict[str, Any]) -> pd.DataFrame:
    """Fetch /profile for all symbols and return as a DataFrame."""
    print(f"Step 2: Fetching profiles for {len(symbols):,} symbols...")
    profiles, error_syms, n_skipped = asyncio.run(_fetch_profiles_async(symbols, client_kwargs))
    print(
        f"  {len(profiles):,} valid  |  "
        f"{n_skipped:,} skipped (empty/multi-record)  |  "
        f"{len(error_syms):,} errors"
    )
    if error_syms:
        print(f"  Error symbols (first 10): {error_syms[:10]}")

    df = pd.DataFrame(profiles)
    cols = [c for c in PROFILE_COLS if c in df.columns]
    return df[cols]


# ---------------------------------------------------------------------------
# Step 3: Filter to US equities
# ---------------------------------------------------------------------------

def filter_us_equities(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Keep NYSE / NASDAQ / AMEX non-ETF non-Fund stocks only."""
    mask = (
        profiles_df["exchange"].isin(US_EXCHANGES)
        & (~profiles_df["isEtf"].astype(bool))
        & (~profiles_df["isFund"].astype(bool))
    )
    result = profiles_df.loc[mask].copy()
    print(f"Step 3: US equity filter: {len(profiles_df):,} → {len(result):,} rows")
    return result


# ---------------------------------------------------------------------------
# Step 4: Delisted companies (async, paginated)
# ---------------------------------------------------------------------------

def fetch_delisted_companies(client_kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Paginate through delisted-companies until an empty page is returned.
    Returns a deduplicated DataFrame of US-exchange symbols with delistedDate.
    """
    print("Step 4: Fetching delisted companies (paginated)...")
    records = asyncio.run(
        fetch_fmp_all_pages(
            "delisted-companies",
            client=FMPClient(**client_kwargs),
            limit=10000,
            verbose=True,
        )
    )
    df = pd.DataFrame(records)
    df = (
        df[df["exchange"].isin(US_EXCHANGES)]
        .drop_duplicates(subset=["symbol"])
        .reset_index(drop=True)
    )
    print(f"  {len(records):,} total records → {len(df):,} unique US-exchange delisted symbols")
    return df[["symbol", "delistedDate"]]


# ---------------------------------------------------------------------------
# Step 5: Finalize and export
# ---------------------------------------------------------------------------

def _apply_final_filters(
    df: pd.DataFrame,
    audit_path: Optional[Path] = None,
) -> pd.DataFrame:
    null_mask = df["symbol"].isna()
    if "industry" in df.columns:
        shell_mask = ~null_mask & (df["industry"] == "Shell Companies")
    else:
        shell_mask = pd.Series(False, index=df.index)
    drop_mask = null_mask | shell_mask
    if drop_mask.any():
        print(
            f"  Dropped {drop_mask.sum():,} rows "
            f"({null_mask.sum():,} null symbol, {shell_mask.sum():,} Shell Companies)"
        )
        if audit_path is not None:
            audit = df[drop_mask].copy()
            audit["dropReason"] = null_mask.map({True: "null_symbol"}).fillna("shell_company")
            audit_path = Path(audit_path)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit.to_csv(audit_path, index=False)
            print(f"  Dropped-row audit written to {audit_path}")
    return df[~drop_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_stock_universe(
    out_path: Optional[Path] = None,
    client_kwargs: Optional[Dict[str, Any]] = None,
    audit_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run the full pipeline and write the universe CSV.

    A fresh FMPClient is constructed per pipeline step so each asyncio.run()
    gets its own event-loop-bound limiter and semaphore.

    Can be called programmatically:
        from GetFMPData.build_stock_universe import build_stock_universe
        df = build_stock_universe()

    Returns the final DataFrame (also written to out_path).
    """
    if out_path is None:
        out_path = DEFAULT_OUT
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if client_kwargs is None:
        client_kwargs = DEFAULT_CLIENT_KWARGS

    # --- Steps 1–2 ---
    symbols = fetch_usd_symbols(client_kwargs)
    profiles_df = fetch_company_profiles(symbols, client_kwargs)

    # --- Step 3 ---
    us_df = filter_us_equities(profiles_df)

    # --- Step 4 ---
    delisted_df = fetch_delisted_companies(client_kwargs)
    delisted_map = delisted_df.set_index("symbol")["delistedDate"]
    us_df["delistedDate"] = us_df["symbol"].map(delisted_map)
    n_delisted = us_df["delistedDate"].notna().sum()
    print(f"  {n_delisted:,} symbols matched a delisted date")

    # --- Step 5 ---
    out_cols = [c for c in OUTPUT_COLS if c in us_df.columns]
    result = _apply_final_filters(us_df[out_cols].copy(), audit_path=audit_path)

    result.to_csv(out_path, index=False)
    write_universe_meta(out_path, len(result))
    print(f"\nDone — {len(result):,} rows saved to {out_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the FMP US stock universe CSV (active + delisted).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output CSV path",
    )
    parser.add_argument(
        "--calls-per-second",
        type=int,
        default=12,
        help="FMP API calls per second (stay at or below your plan limit)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=25,
        help="Max concurrent in-flight requests",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=None,
        help="Optional path to write rows dropped by the final filter (with dropReason)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    client_kwargs = dict(
        calls_per_second=args.calls_per_second,
        safety_margin=0,
        concurrency=args.concurrency,
        timeout_s=30,
    )
    build_stock_universe(
        out_path=args.out,
        client_kwargs=client_kwargs,
        audit_path=args.audit_csv,
    )


if __name__ == "__main__":
    main()
