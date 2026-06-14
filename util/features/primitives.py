from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd


# ---------- Primitive helpers ----------
def _gb(df: pd.DataFrame, sym_col: str):
    return df.groupby(sym_col, sort=False)


def prim_ts_return(df, sym_col, price: pd.Series, lookback: int) -> pd.Series:
    g = price.groupby(df[sym_col], sort=False)
    out = np.log1p(g.pct_change(lookback, fill_method=None))
    out.name = None
    return out


def prim_ts_mean(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    out.name = None
    return out


def prim_ts_std(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.rolling(window, min_periods=min_periods).std())
    out.name = None
    return out


def prim_ts_max(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.rolling(window, min_periods=min_periods).max())
    out.name = None
    return out


def prim_ts_min(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.rolling(window, min_periods=min_periods).min())
    out.name = None
    return out


def prim_ts_quantile(
    df: pd.DataFrame,
    sym_col: str,
    x: pd.Series,
    window: int,
    q: float,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Grouped rolling quantile."""
    if not 0 < q < 1:
        raise ValueError(f"q must be between 0 and 1, got {q}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    if min_periods is None:
        min_periods = window

    out = x.groupby(df[sym_col], sort=False).transform(
        lambda s: s.rolling(window, min_periods=min_periods).quantile(q)
    )
    out.name = None
    return out


def prim_ts_ewm_mean(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.ewm(alpha=1/window, adjust=False, min_periods=min_periods).mean())
    out.name = None
    return out


def prim_ts_ewm_span_mean(df, sym_col, x: pd.Series, span: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = span
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.ewm(span=span, adjust=False, min_periods=min_periods).mean())
    out.name = None
    return out


def prim_ts_diff(df, sym_col, x: pd.Series, lookback: int) -> pd.Series:
    g = x.groupby(df[sym_col], sort=False)
    out = g.diff(lookback)
    out.name = None
    return out


def prim_ts_slope(df, sym_col, x: pd.Series, lookback: int) -> pd.Series:
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")

    g = x.groupby(df[sym_col], sort=False)
    out = g.diff(lookback) / lookback
    out.name = None
    return out


def prim_ts_pctslope(df, sym_col, x: pd.Series, lookback: int) -> pd.Series:
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")

    g = x.groupby(df[sym_col], sort=False)
    out = g.pct_change(lookback) / lookback
    out.name = None
    return out


def prim_ts_logslope(df, sym_col, x: pd.Series, lookback: int) -> pd.Series:
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")

    g = x.groupby(df[sym_col], sort=False)

    out = np.log(
        g.shift(0) / g.shift(lookback)
    ) / lookback

    out.name = None
    return out


def prim_cs_fraction(
    df: pd.DataFrame,
    date_col: str,
    x: pd.Series,
    threshold: float,
    direction: str = "gt"
):
    if direction == "gt":
        s = x > threshold
    elif direction == "ge":
        s = x >= threshold
    elif direction == "lt":
        s = x < threshold
    elif direction == "le":
        s = x <= threshold
    else:
        raise ValueError("Unsupported direction, please provide supported directions (gt, ge, lt, le)")

    # Comparisons against NaN return False (not NaN), which would otherwise be
    # counted in the cross-sectional denominator. Mask NaN inputs so the fraction
    # is computed only over symbols with a defined value on that date.
    s = s.astype(float).where(x.notna())
    out = s.groupby(df[date_col]).transform("mean")
    out.name = None
    return out


def prim_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return a - b


def prim_rdiff(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return (a / (b + eps)) - 1


def prim_zscore(x: pd.Series, mu: pd.Series, sigma: pd.Series, eps: float = 1e-12) -> pd.Series:
    return (x - mu) / (sigma + eps)


def prim_ratio(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return a / (b + eps)


def prim_abs_ratio(
    a: pd.Series,
    b: pd.Series,
    eps: float = 1e-12,
) -> pd.Series:
    """abs(a) / (abs(b) + eps)."""
    return a.abs() / (b.abs() + eps)


def prim_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(np.maximum(x, eps))


def prim_scale(x: pd.Series, scaler: float) -> pd.Series:
    return x * scaler


def prim_tr(df, sym_col, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.groupby(df[sym_col], sort=False).shift(1)
    hl = (high - low).to_numpy()
    hc = (high - prev_close).abs().to_numpy()
    lc = (low - prev_close).abs().to_numpy()
    out = np.maximum.reduce([hl, hc, lc])
    out = pd.Series(out, index=df.index)
    out.name = None
    return out


def post_clip(x: pd.Series, lo: Optional[float] = None, hi: Optional[float] = None) -> pd.Series:
    return x.clip(lo, hi)


def post_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(np.maximum(x, eps))


def post_index_scale(x: pd.Series, offset: float = 100.0) -> pd.Series:
    return offset - (offset / (1 + x))


def post_scale(x: pd.Series, scaler: float) -> pd.Series:
    return x * scaler


def post_delog(x: pd.Series) -> pd.Series:
    return np.expm1(x)


def post_annualize_simple_return(x: pd.Series, w: int) -> pd.Series:
    return np.power(1 + x, 252 / w) - 1


def post_annualize_log_return(x: pd.Series, w: int) -> pd.Series:
    return x * (252 / w)


def post_annualize_volatility(x: pd.Series, w: int) -> pd.Series:
    return x * np.sqrt(252 / w)


def post_cs_rank(df: pd.DataFrame, date_col: str, x: pd.Series) -> pd.Series:
    return x.groupby(df[date_col], sort=False).rank(pct=True)


def post_cs_zscore(df: pd.DataFrame, date_col: str, x: pd.Series, eps: float = 1e-12) -> pd.Series:
    g = x.groupby(df[date_col], sort=False)
    mu = g.transform("mean")
    sd = g.transform("std")
    return (x - mu) / (sd + eps)


PRIMITIVES: Dict[str, Callable[..., pd.Series]] = {
    "ts_return": prim_ts_return,
    "ts_mean": prim_ts_mean,
    "ts_std": prim_ts_std,
    "ts_max": prim_ts_max,
    "ts_min": prim_ts_min,
    "ts_quantile": prim_ts_quantile,
    "ts_ewm_mean": prim_ts_ewm_mean,
    "ts_ewm_span_mean": prim_ts_ewm_span_mean,
    "ts_diff": prim_ts_diff,
    "ts_slope": prim_ts_slope,
    "ts_pctslope": prim_ts_pctslope,
    "ts_logslope": prim_ts_logslope,
    "cs_fraction": prim_cs_fraction,
    "diff": prim_diff,
    "rdiff": prim_rdiff,
    "scale": prim_scale,
    "zscore": prim_zscore,
    "ratio": prim_ratio,
    "abs_ratio": prim_abs_ratio,
    "log": prim_log,
    "tr": prim_tr,
}

POSTS: Dict[str, Callable[..., pd.Series]] = {
    "clip": post_clip,
    "log": post_log,
    "index_scale": post_index_scale,
    "scale": post_scale,
    "delog": post_delog,
    "annualize_ret": post_annualize_simple_return,
    "annualize_logret": post_annualize_log_return,
    "annualize_vol": post_annualize_volatility,
    "cs_rank": post_cs_rank,
    "cs_zscore": post_cs_zscore,
}