"""Deterministic synthetic FMP responses for testing the pull + build pipeline
without any network access."""

from datetime import date, timedelta
from typing import Any, Dict, List, Sequence

import pandas as pd

UNIVERSE = pd.DataFrame({
    "symbol": ["AAA", "BBB", "CCC"],
    "currency": ["USD"] * 3,
    "exchange": ["NYSE", "NASDAQ", "AMEX"],
    "industry": ["Software", "Banks", "Biotechnology"],
    "sector": ["Technology", "Financial Services", "Healthcare"],
    "isActivelyTrading": [True, True, False],
    "ipoDate": ["2010-01-04", "2012-05-01", "2015-09-15"],
    "delistedDate": [None, None, "2025-06-30"],
})

_BASE_PRICE = {"AAA": 100.0, "BBB": 25.0, "CCC": 1.5}  # CCC: penny stock


def quarter_ends(start: date, end: date) -> List[date]:
    qs, d = [], date(start.year, 3, 31)
    while d <= end:
        qs.append(d)
        m = d.month + 3
        y = d.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        day = 31 if m in (3, 12) else 30
        d = date(y, m, day)
    return [q for q in qs if q >= start]


def price_records(symbol: str, lo: date, hi: date) -> List[Dict[str, Any]]:
    days = pd.bdate_range(lo, hi)
    base = _BASE_PRICE.get(symbol, 50.0)
    return [{
        "symbol": symbol, "date": d.date().isoformat(),
        "adjOpen": base + i * 0.01, "adjHigh": base + i * 0.01 + 0.5,
        "adjLow": base + i * 0.01 - 0.5, "adjClose": base + i * 0.01 + 0.1,
        # string-typed volume exercises normalize_df_for_parquet
        "volume": str(1_000_000 + i * 10),
    } for i, d in enumerate(days)]


def quarterly_records(symbol: str, key: str, lo: date, hi: date,
                      filing_lag_days: int = 40) -> List[Dict[str, Any]]:
    # one extra quarter of history before the window, like a real limit-N pull
    out = []
    for i, q in enumerate(quarter_ends(lo - timedelta(days=180), hi)):
        filing = q + timedelta(days=filing_lag_days)
        rec: Dict[str, Any] = {"symbol": symbol, "date": q.isoformat()}
        if key in ("incomestatement", "balancesheet", "cashflow"):
            rec["filingDate"] = filing.isoformat()
            rec["acceptedDate"] = f"{filing.isoformat()} 16:30:00"
        if key == "incomestatement":
            rec.update(revenue=1e9 + i * 1e7, netIncome=1e8 + i * 1e6,
                       depreciationAndAmortization=5e6, eps=1.0 + i * 0.01)
        elif key == "balancesheet":
            rec.update(totalAssets=5e9 + i * 1e7, accountsReceivables=2e8,
                       inventory=1e8, totalDebt=1e9, netDebt=8e8)
        elif key == "cashflow":
            rec.update(netIncome=1e8 + i * 1e6, depreciationAndAmortization=5e6,
                       accountsReceivables=-1e6, inventory=-5e5,
                       operatingCashFlow=2e8, freeCashFlow=1.5e8,
                       capitalExpenditure=-5e7)
        elif key == "keymetrics":
            rec.update(marketCap=1e10, enterpriseValue=1.1e10,
                       returnOnEquity=0.15 + i * 0.001, evToSales=5.0)
        elif key == "enterprisevalues":
            rec.update(stockPrice=_BASE_PRICE.get(symbol, 50.0),
                       numberOfShares=1e8,
                       marketCapitalization=_BASE_PRICE.get(symbol, 50.0) * 1e8,
                       minusCashAndCashEquivalents=-1e8, addTotalDebt=1e9,
                       enterpriseValue=1.1e10)
        out.append(rec)
    out.reverse()  # FMP returns most-recent first
    return out


def make_fake_fetch(error_ticker: str = None, error_component: str = None,
                    empty_ticker: str = None, call_log: list = None):
    """Build an async stand-in for fetch_fmp_for_tickers."""

    async def fake_fetch(tickers: Sequence[str], request_specs: Dict,
                         *, client=None) -> Dict[str, Dict[str, Any]]:
        if call_log is not None:
            call_log.append({"tickers": list(tickers),
                             "specs": sorted(request_specs)})
        results: Dict[str, Dict[str, Any]] = {}
        for t in tickers:
            results[t] = {}
            for key, spec in request_specs.items():
                if t == error_ticker and key == error_component:
                    results[t][key] = {"__error__": "FMPError('429 ...')"}
                    continue
                if t == empty_ticker:
                    results[t][key] = []
                    continue
                params = spec["params"]
                if "from" in params:
                    lo = date.fromisoformat(params["from"])
                    hi = date.fromisoformat(params["to"])
                    results[t][key] = price_records(t, lo, hi)
                else:
                    # quarterly: derive a window from the requested limit
                    hi = date(2025, 12, 31)
                    lo = hi - timedelta(days=int(params["limit"] * 91.3))
                    results[t][key] = quarterly_records(t, key, lo, hi)
        return results

    return fake_fetch
