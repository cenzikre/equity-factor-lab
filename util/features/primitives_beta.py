import numpy as np
import pandas as pd
from util.data_client.fmp import _fetch_fmp_baseprice_cached
from util.features.primitives import PRIMITIVES


def prim_ts_beta_to_market(
    df: pd.DataFrame,
    date_col: str,
    sym_col: str,
    price: pd.Series,
    window: int,
    market_symbol: str = "SPY",
    endpoint: str = "historical-price-eod/dividend-adjusted",
    urlhead: str = "https://financialmodelingprep.com/stable/",
    eps: float = 1e-12,
) -> pd.Series:
    """Rolling beta of each asset's adjusted close return to market return.

    Computes:
        beta_i,t = rolling_cov(r_i, r_m, window) / rolling_var(r_m, window)

    Output is aligned back to the original df row order/index.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    work = pd.DataFrame({
        "symbol": df[sym_col].to_numpy(),
        "date": pd.to_datetime(df[date_col]).to_numpy(),
        "price": price.to_numpy(),
    }, index=df.index)

    min_date = work["date"].min().strftime("%Y-%m-%d")
    max_date = work["date"].max().strftime("%Y-%m-%d")

    mkt_p = _fetch_fmp_baseprice_cached(
        symbol=market_symbol,
        min_date=min_date,
        max_date=max_date,
        endpoint=endpoint,
        urlhead=urlhead,
    )

    asset_p = (
        work.pivot(
            index="date",
            columns="symbol",
            values="price"
        ).sort_index().astype(float)
    )
    beta_keys = work[["symbol", "date"]].copy()

    mkt_r = np.log1p(mkt_p.pct_change(fill_method=None))
    asset_r = np.log1p(asset_p.pct_change(fill_method=None))

    asset_r_aligned, mkt_r_aligned = asset_r.align(
        mkt_r,
        join="inner",
        axis=0,
    )

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


def register_beta_primitives() -> None:
    """Register beta primitive into the global primitive registry."""
    PRIMITIVES["ts_beta_to_market"] = prim_ts_beta_to_market


# Register on import so templates work once this module is imported.
register_beta_primitives()