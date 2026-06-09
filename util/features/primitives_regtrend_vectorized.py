from __future__ import annotations
from typing import Literal, Optional
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from util.features.primitives import PRIMITIVES


RegStat = Literal["slope", "intercept", "r2", "resid_std", "slope_se", "tstat"]


def _reg_stat_from_roll(roll: np.ndarray, stat: RegStat) -> np.ndarray:
    """Vectorized OLS stat for y ~ 1 + t.

    roll shape must be [n_windows, n_symbols, window].
    This version requires full finite windows; any NaN/inf in a window -> NaN.
    """
    if roll.ndim != 3:
        raise ValueError(f"Expected 3D rolling array, got shape={roll.shape}")

    n_windows, n_symbols, window = roll.shape
    valid = np.isfinite(roll).all(axis=2)

    t = np.arange(window, dtype=float)
    t_mean = t.mean()
    tc = t - t_mean
    sxx = np.sum(tc * tc)
    if sxx <= 0:
        return np.full((n_windows, n_symbols), np.nan, dtype=float)

    y = roll.astype(float, copy=False)
    y_mean = np.mean(y, axis=2)

    beta = np.sum(y * tc[None, None, :], axis=2) / sxx
    alpha = y_mean - beta * t_mean

    if stat == "slope":
        out = beta
    elif stat == "intercept":
        out = alpha
    else:
        y_hat = alpha[:, :, None] + beta[:, :, None] * t[None, None, :]
        resid = y - y_hat
        ss_res = np.sum(resid * resid, axis=2)

        yc = y - y_mean[:, :, None]
        ss_tot = np.sum(yc * yc, axis=2)

        if stat == "r2":
            out = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
        else:
            if window <= 2:
                out = np.full((n_windows, n_symbols), np.nan, dtype=float)
            else:
                resid_std = np.sqrt(ss_res / (window - 2))

                if stat == "resid_std":
                    out = resid_std
                else:
                    slope_se = resid_std / np.sqrt(sxx)

                    if stat == "slope_se":
                        out = slope_se
                    elif stat == "tstat":
                        out = np.where(
                            (slope_se > 0) & np.isfinite(slope_se),
                            beta / slope_se,
                            np.nan,
                        )
                    else:
                        raise ValueError(f"Unknown regression stat: {stat!r}")

    return np.where(valid, np.asarray(out, dtype=float), np.nan)


def prim_ts_reg_stat_vectorized(
    df: pd.DataFrame,
    date_col: str,
    sym_col: str,
    x: pd.Series,
    window: int,
    stat: RegStat,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Fast vectorized rolling regression statistic for x ~ 1 + t.

    Steps:
    1. pivot long data into date x symbol matrix
    2. use sliding_window_view over date axis
    3. compute OLS stats for all windows/symbols at once
    4. melt/merge back to original row order

    Constraint:
    - min_periods must equal window.
    - duplicate (date, symbol) rows must be resolved before calling this.
    """
    if window <= 1:
        raise ValueError(f"window must be greater than 1, got {window}")

    if min_periods is None:
        min_periods = window

    if min_periods != window:
        raise ValueError(
            "prim_ts_reg_stat_vectorized requires min_periods == window. "
            f"Got min_periods={min_periods}, window={window}."
        )

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

    arr = wide.to_numpy(dtype=float)
    roll = sliding_window_view(arr, window_shape=window, axis=0)

    stat_arr = _reg_stat_from_roll(roll, stat=stat)

    stat_wide = pd.DataFrame(
        stat_arr,
        index=wide.index[(window - 1):],
        columns=wide.columns,
    ).reindex(wide.index)

    out_long = stat_wide.reset_index().melt(
        id_vars=stat_wide.index.name or "date",
        var_name=stat_wide.columns.name or "symbol",
        value_name="reg_stat",
    )

    cols = list(out_long.columns)
    out_long = out_long.rename(columns={cols[0]: "date", cols[1]: "symbol", cols[2]: "reg_stat"})

    merged = keys.merge(out_long, on=["symbol", "date"], how="left", sort=False).sort_values("__row_id__")

    out = pd.Series(merged["reg_stat"].to_numpy(), index=df.index)
    out.name = None
    return out


def prim_ts_reg_slope_vectorized(df, date_col, sym_col, x, window, min_periods=None):
    return prim_ts_reg_stat_vectorized(df, date_col, sym_col, x, window, "slope", min_periods)


def prim_ts_reg_intercept_vectorized(df, date_col, sym_col, x, window, min_periods=None):
    return prim_ts_reg_stat_vectorized(df, date_col, sym_col, x, window, "intercept", min_periods)


def prim_ts_reg_r2_vectorized(df, date_col, sym_col, x, window, min_periods=None):
    return prim_ts_reg_stat_vectorized(df, date_col, sym_col, x, window, "r2", min_periods)


def prim_ts_reg_resid_std_vectorized(df, date_col, sym_col, x, window, min_periods=None):
    return prim_ts_reg_stat_vectorized(df, date_col, sym_col, x, window, "resid_std", min_periods)


def prim_ts_reg_slope_se_vectorized(df, date_col, sym_col, x, window, min_periods=None):
    return prim_ts_reg_stat_vectorized(df, date_col, sym_col, x, window, "slope_se", min_periods)


def prim_ts_reg_tstat_vectorized(df, date_col, sym_col, x, window, min_periods=None):
    return prim_ts_reg_stat_vectorized(df, date_col, sym_col, x, window, "tstat", min_periods)


def register_regtrend_vectorized_primitives() -> None:
    PRIMITIVES["ts_reg_slope_vectorized"] = prim_ts_reg_slope_vectorized
    PRIMITIVES["ts_reg_intercept_vectorized"] = prim_ts_reg_intercept_vectorized
    PRIMITIVES["ts_reg_r2_vectorized"] = prim_ts_reg_r2_vectorized
    PRIMITIVES["ts_reg_resid_std_vectorized"] = prim_ts_reg_resid_std_vectorized
    PRIMITIVES["ts_reg_slope_se_vectorized"] = prim_ts_reg_slope_se_vectorized
    PRIMITIVES["ts_reg_tstat_vectorized"] = prim_ts_reg_tstat_vectorized


register_regtrend_vectorized_primitives()
