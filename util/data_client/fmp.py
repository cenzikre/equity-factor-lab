import os
import requests
# import numpy as np
import pandas as pd
from functools import lru_cache


def getFMPData_withkey(endpoint, urlhead=None, apikey=None, **params):
    if urlhead is None:
        urlhead = "https://financialmodelingprep.com/stable/"
    if apikey is None:
        apikey = os.getenv("FMP_API_KEY")
    if apikey is None:
        raise ValueError(
            "FMP API key is missing. Set environment variable FMP_API_KEY "
            "or pass apikey explicitly to getFMPData_withkey()."
        )

    if "from_" in params:
        params["from"] = params.pop("from_")

    url = urlhead + endpoint
    param_lst = [f"apikey={apikey}"] + [str(k) + "=" + str(v) for k, v in params.items()]
    url = "?".join([url, "&".join(param_lst)])
    data = requests.get(url).json()
    return data


def getFMPData(endpoint, urlhead=None, apikey=None, **params):
    """Small FMP client.

    Notes:
    - Prefer setting FMP_API_KEY in the environment instead of hardcoding it.
    - Supports from_ because `from` is a Python keyword.
    """
    if urlhead is None:
        urlhead = "https://financialmodelingprep.com/stable/"
    if apikey is None:
        apikey = os.getenv("FMP_API_KEY")

    if apikey is None:
        raise ValueError(
            "FMP API key is missing. Set environment variable FMP_API_KEY "
            "or pass apikey explicitly to getFMPData()."
        )

    if "from_" in params:
        params["from"] = params.pop("from_")

    url = urlhead + endpoint
    param_lst = [f"apikey={apikey}"] + [
        f"{k}={v}" for k, v in params.items()
    ]
    url = "?".join([url, "&".join(param_lst)])

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


@lru_cache(maxsize=256)
def _fetch_fmp_baseprice_cached(
    symbol: str,
    min_date: str,
    max_date: str,
    endpoint: str = "historical-price-eod/dividend-adjusted",
    urlhead: str = "https://financialmodelingprep.com/stable/",
) -> pd.Series:
    """Fetch and cache market baseline adjusted close by symbol/date window."""
    data = getFMPData(
        endpoint=endpoint,
        urlhead=urlhead,
        symbol=symbol.upper(),
        from_=min_date,
        to=max_date,
    )

    mkt_df = pd.DataFrame(data)
    if mkt_df.empty:
        raise ValueError(
            f"FMP returned no data for market symbol '{symbol}' "
            f"from {min_date} to {max_date}."
        )

    required = {"date", "adjClose"}
    missing = required - set(mkt_df.columns)
    if missing:
        raise ValueError(
            f"FMP response for '{symbol}' is missing columns: {sorted(missing)}"
        )

    mkt_df["date"] = pd.to_datetime(mkt_df["date"])
    mkt_df = mkt_df.sort_values("date").set_index("date")

    out = mkt_df["adjClose"].astype(float).sort_index()
    out.name = f"baseprice_{symbol.lower()}"
    return out