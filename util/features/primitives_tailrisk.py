import numpy as np
import pandas as pd
from typing import Optional
from util.features.primitives import PRIMITIVES
from numpy.lib.stride_tricks import sliding_window_view


def _tail_mean_1d(a: np.ndarray, q: float, tail: str) -> float:
    arr = np.asarray(a, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan

    threshold = np.quantile(arr, q)
    if tail == "lower":
        mask = arr < threshold
    elif tail == "upper":
        mask = arr > threshold
    else:
        raise ValueError(f"tail must be 'lower' or 'upper', got {tail!r}")

    if not mask.any():
        return np.nan
    return float(arr[mask].mean())


def prim_ts_tail_mean(
    df: pd.DataFrame,
    sym_col: str,
    x: pd.Series,
    window: int,
    q: float,
    tail: str = "lower",
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Simple grouped rolling conditional tail mean."""
    if not 0 < q < 1:
        raise ValueError(f"q must be between 0 and 1, got {q}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")

    if min_periods is None:
        min_periods = window

    out = x.groupby(df[sym_col], sort=False).transform(
        lambda s: s.rolling(window, min_periods=min_periods).apply(
            lambda a: _tail_mean_1d(a, q=q, tail=tail),
            raw=True,
        )
    )
    out.name = None
    return out


def prim_ts_tail_mean_fast(
    df: pd.DataFrame,
    date_col: str,
    sym_col: str,
    x: pd.Series,
    window: int,
    q: float,
    tail: str = "lower",
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Fast vectorized rolling conditional tail mean.

    Mirrors the original CVaR logic:
    pivot long returns to wide date x symbol, compute rolling quantile, create
    rolling windows with sliding_window_view, mask lower/upper tail, average,
    then merge back to original row order.
    """
    if not 0 < q < 1:
        raise ValueError(f"q must be between 0 and 1, got {q}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if min_periods is None:
        min_periods = window
    if min_periods != window:
        raise ValueError("fast tail mean currently requires min_periods == window")
    if tail not in {"lower", "upper"}:
        raise ValueError(f"tail must be 'lower' or 'upper', got {tail!r}")

    work = pd.DataFrame(
        {
            "symbol": df[sym_col].to_numpy(),
            "date": pd.to_datetime(df[date_col]).to_numpy(),
            "x": x.to_numpy(),
            "__row_id__": np.arange(len(df)),
        },
        index=df.index,
    )
    keys = work[["__row_id__", "symbol", "date"]].copy()

    wide = work.pivot(index="date", columns="symbol", values="x").sort_index().astype(float)
    if len(wide) < window:
        out = pd.Series(np.nan, index=df.index)
        out.name = None
        return out

    q_wide = wide.rolling(window=window, min_periods=window).quantile(q)
    q_arr = q_wide.iloc[(window - 1):].to_numpy(dtype=float)

    x_arr = wide.to_numpy(dtype=float)
    roll = sliding_window_view(x_arr, window_shape=window, axis=0)

    if tail == "lower":
        mask = roll < q_arr[:, :, None]
    else:
        mask = roll > q_arr[:, :, None]

    mask = mask & np.isfinite(roll)

    sum_ = np.sum(np.where(mask, roll, 0.0), axis=2)
    cnt = np.sum(mask, axis=2)

    out_arr = np.divide(
        sum_,
        cnt,
        out=np.full_like(sum_, np.nan, dtype=float),
        where=cnt > 0,
    )

    out_wide = pd.DataFrame(
        out_arr,
        index=wide.index[(window - 1):],
        columns=wide.columns,
    ).reindex(wide.index)

    out_long = out_wide.reset_index().melt(
        id_vars=out_wide.index.name or "date",
        var_name=out_wide.columns.name or "symbol",
        value_name="tail_mean",
    )
    cols = list(out_long.columns)
    out_long = out_long.rename(columns={cols[0]: "date", cols[1]: "symbol", cols[2]: "tail_mean"})

    merged = keys.merge(out_long, on=["symbol", "date"], how="left", sort=False).sort_values("__row_id__")
    out = pd.Series(merged["tail_mean"].to_numpy(), index=df.index)
    out.name = None
    return out


def register_tailrisk_primitives() -> None:
    PRIMITIVES["ts_tail_mean"] = prim_ts_tail_mean
    PRIMITIVES["ts_tail_mean_fast"] = prim_ts_tail_mean_fast


register_tailrisk_primitives()