from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from util.data_client.fmp import _fetch_fmp_baseprice_cached
from util.features.primitives import PRIMITIVES


def fetch_market_price(
    market_symbol: str,
    min_date: str,
    max_date: str,
    *,
    endpoint: str = "historical-price-eod/dividend-adjusted",
    urlhead: str = "https://financialmodelingprep.com/stable/",
) -> pd.Series:
    """Fetch the market baseline price series (date-indexed) for beta.

    Exposed so callers can fetch once, outside the builder, and inject the
    result into the beta spec/primitive -- keeping feature construction free of
    network I/O and deterministic/testable.
    """
    return _fetch_fmp_baseprice_cached(
        symbol=market_symbol,
        min_date=min_date,
        max_date=max_date,
        endpoint=endpoint,
        urlhead=urlhead,
    )


def _beta_from_market_price(
    df: pd.DataFrame,
    date_col: str,
    sym_col: str,
    price: pd.Series,
    market_price: pd.Series,
    window: int,
    eps: float = 1e-12,
) -> pd.Series:
    """Pure rolling-beta computation from an asset panel and a market price
    series. No network I/O. Output is aligned back to df row order/index."""
    work = pd.DataFrame({
        "symbol": df[sym_col].to_numpy(),
        "date": pd.to_datetime(df[date_col]).to_numpy(),
        "price": price.to_numpy(),
    }, index=df.index)

    mkt_p = market_price.copy()
    mkt_p.index = pd.to_datetime(mkt_p.index)

    asset_p = (
        work.pivot(index="date", columns="symbol", values="price")
            .sort_index().astype(float)
    )
    beta_keys = work[["symbol", "date"]].copy()

    mkt_r = np.log1p(mkt_p.pct_change(fill_method=None))
    asset_r = np.log1p(asset_p.pct_change(fill_method=None))

    asset_r_aligned, mkt_r_aligned = asset_r.align(mkt_r, join="inner", axis=0)

    cov = asset_r_aligned.rolling(window, min_periods=window).cov(mkt_r_aligned)
    var = mkt_r_aligned.rolling(window, min_periods=window).var()

    beta_wide = cov.div(var.where(var.abs() > eps), axis=0)

    beta_long = (
        beta_wide.stack(future_stack=True)
                 .rename("beta")
                 .reset_index()
    )

    out = beta_keys.merge(
        beta_long,
        on=["symbol", "date"],
        how="left",
        sort=False,
    )["beta"]

    out.index = df.index
    out.name = None
    return out


def prim_ts_beta_to_market(
    df: pd.DataFrame,
    date_col: str,
    sym_col: str,
    price: pd.Series,
    window: int,
    market_price: Optional[pd.Series] = None,
    market_symbol: str = "SPY",
    endpoint: str = "historical-price-eod/dividend-adjusted",
    urlhead: str = "https://financialmodelingprep.com/stable/",
    eps: float = 1e-12,
) -> pd.Series:
    """Rolling beta of each asset's adjusted-close return to the market return.

    Computes:
        beta_i,t = rolling_cov(r_i, r_m, window) / rolling_var(r_m, window)

    The market series can be injected via `market_price` (a date-indexed
    pd.Series), in which case no network call is made -- the preferred path for
    deterministic, offline builds. When `market_price` is None it is fetched for
    the panel's date range using `market_symbol` (legacy convenience).
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    if market_price is None:
        dates = pd.to_datetime(df[date_col])
        market_price = fetch_market_price(
            market_symbol=market_symbol,
            min_date=dates.min().strftime("%Y-%m-%d"),
            max_date=dates.max().strftime("%Y-%m-%d"),
            endpoint=endpoint,
            urlhead=urlhead,
        )

    return _beta_from_market_price(
        df=df,
        date_col=date_col,
        sym_col=sym_col,
        price=price,
        market_price=market_price,
        window=window,
        eps=eps,
    )


def register_beta_primitives() -> None:
    """Register beta primitive into the global primitive registry."""
    PRIMITIVES["ts_beta_to_market"] = prim_ts_beta_to_market


# Register on import so templates work once this module is imported.
register_beta_primitives()
