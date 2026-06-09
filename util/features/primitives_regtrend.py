import numpy as np
import pandas as pd
from typing import Optional, Literal
from util.features.primitives import PRIMITIVES


RegStat = Literal[
    "slope",
    "intercept",
    "r2",
    "resid_std",
    "slope_se",
    "tstat",
]


def _regression_stat_1d(y: np.ndarray, stat: RegStat) -> float:
    """Compute one rolling linear-regression statistic on y ~ 1 + t.

    t is always 0, 1, ..., n-1 inside the rolling window.
    NaNs are removed pairwise.
    """
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]

    n = arr.size
    if n < 2:
        return np.nan

    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    y_mean = arr.mean()

    tc = t - t_mean
    yc = arr - y_mean

    sxx = np.sum(tc * tc)
    if sxx <= 0:
        return np.nan

    beta = np.sum(tc * yc) / sxx
    alpha = y_mean - beta * t_mean

    if stat == "slope":
        return float(beta)

    if stat == "intercept":
        return float(alpha)

    y_hat = alpha + beta * t
    resid = arr - y_hat

    ss_res = np.sum(resid * resid)
    ss_tot = np.sum(yc * yc)

    if stat == "r2":
        if ss_tot <= 0:
            return np.nan
        return float(1.0 - ss_res / ss_tot)

    if n <= 2:
        return np.nan

    resid_var = ss_res / (n - 2)
    resid_std = np.sqrt(resid_var)

    if stat == "resid_std":
        return float(resid_std)

    slope_se = resid_std / np.sqrt(sxx)

    if stat == "slope_se":
        return float(slope_se)

    if stat == "tstat":
        if slope_se <= 0 or not np.isfinite(slope_se):
            return np.nan
        return float(beta / slope_se)

    raise ValueError(f"Unknown regression stat: {stat!r}")


def prim_ts_reg_stat(
    df: pd.DataFrame,
    sym_col: str,
    x: pd.Series,
    window: int,
    stat: RegStat,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Grouped rolling linear-regression statistic for x ~ 1 + t."""
    if window <= 1:
        raise ValueError(f"window must be greater than 1, got {window}")

    if min_periods is None:
        min_periods = window

    if min_periods < 2:
        raise ValueError(f"min_periods must be at least 2, got {min_periods}")

    out = x.groupby(df[sym_col], sort=False).transform(
        lambda s: s.rolling(window, min_periods=min_periods).apply(
            lambda a: _regression_stat_1d(a, stat=stat),
            raw=True,
        )
    )
    out.name = None
    return out


def prim_ts_reg_slope(
    df: pd.DataFrame,
    sym_col: str,
    x: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    return prim_ts_reg_stat(df, sym_col, x, window, "slope", min_periods)


def prim_ts_reg_intercept(
    df: pd.DataFrame,
    sym_col: str,
    x: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    return prim_ts_reg_stat(df, sym_col, x, window, "intercept", min_periods)


def prim_ts_reg_r2(
    df: pd.DataFrame,
    sym_col: str,
    x: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    return prim_ts_reg_stat(df, sym_col, x, window, "r2", min_periods)


def prim_ts_reg_resid_std(
    df: pd.DataFrame,
    sym_col: str,
    x: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    return prim_ts_reg_stat(df, sym_col, x, window, "resid_std", min_periods)


def prim_ts_reg_slope_se(
    df: pd.DataFrame,
    sym_col: str,
    x: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    return prim_ts_reg_stat(df, sym_col, x, window, "slope_se", min_periods)


def prim_ts_reg_tstat(
    df: pd.DataFrame,
    sym_col: str,
    x: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    return prim_ts_reg_stat(df, sym_col, x, window, "tstat", min_periods)


def register_regtrend_primitives() -> None:
    PRIMITIVES["ts_reg_slope"] = prim_ts_reg_slope
    PRIMITIVES["ts_reg_intercept"] = prim_ts_reg_intercept
    PRIMITIVES["ts_reg_r2"] = prim_ts_reg_r2
    PRIMITIVES["ts_reg_resid_std"] = prim_ts_reg_resid_std
    PRIMITIVES["ts_reg_slope_se"] = prim_ts_reg_slope_se
    PRIMITIVES["ts_reg_tstat"] = prim_ts_reg_tstat


register_regtrend_primitives()