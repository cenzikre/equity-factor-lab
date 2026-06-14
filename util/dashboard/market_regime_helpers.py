from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages


# -----------------------------
# Data loading
# -----------------------------
def get_fmp_api_key() -> str:
    key = os.getenv("FMP_API_KEY")
    if not key:
        raise ValueError(
            "FMP_API_KEY is missing. Set it before running the report, e.g. "
            "export FMP_API_KEY='your_key' or os.environ['FMP_API_KEY']='your_key'."
        )
    return key


def validate_environment():
    if not os.getenv("FMP_API_KEY"):
        raise ValueError(
            "FMP_API_KEY not found. "
            "Create .env or set environment variable."
        )


def fetch_fmp_price_data(
    tickers: Sequence[str],
    *,
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    endpoint: str = "historical-price-eod/dividend-adjusted",
) -> pd.DataFrame:
    """Fetch adjusted EOD price data from FMP."""
    from util.data_client.fmp import getFMPData
    from util.data_client.dataPullHelpers import normalize_df_for_parquet

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    frames: list[pd.DataFrame] = []

    for tk in tickers:
        data = getFMPData(endpoint=endpoint, symbol=tk.upper(), from_=start_date, to=end_date)
        _df = pd.DataFrame(data)
        if _df.empty:
            raise ValueError(f"No FMP data returned for ticker {tk!r}")

        _df = normalize_df_for_parquet(
            _df,
            datetime_success_ratio=0.9,
            numeric_success_ratio=0.9,
            treat_huge_int_as_string=False,
        )
        if "symbol" not in _df.columns:
            _df["symbol"] = tk.upper()
        frames.append(_df)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["symbol", "date"], ignore_index=True)
    return out


# -----------------------------
# Feature building
# -----------------------------
def get_features(
    builder,
    family: dict,
    template: str,
    params_lst: list[dict],
    transform_fn: Optional[Callable] = None,
    transform_params: Optional[dict] = None,
) -> dict[str, pd.Series]:
    specs = []
    for params in params_lst:
        base = family[template](**params)
        if transform_fn is None:
            specs += base
        else:
            specs += transform_fn(base, **(transform_params or {}))
    return builder.build_published(specs)


def get_feature_df(
    source_df: pd.DataFrame,
    feature_dict: dict[str, pd.Series],
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> pd.DataFrame:
    feature_df = pd.DataFrame(feature_dict, index=source_df.index)
    return pd.concat([source_df[[symbol_col, date_col]], feature_df], axis=1)


def build_market_features(
    price_df: pd.DataFrame,
    *,
    price_col: str = "adjClose",
    return_lookbacks: Sequence[int] = (5, 10, 20, 60, 126),
    dma_windows: Sequence[int] = (5, 10, 20, 60, 126),
    vdma_windows: Sequence[int] = (5, 10, 20, 60, 126),
    rv_windows: Sequence[int] = (5, 10, 20, 60, 126),
    mdd_windows: Sequence[int] = (20, 60, 126, 252),
    reg_ma_windows: Sequence[int] = (5, 10, 20, 60, 126),
    reg_window: int = 10,
    z_window: int = 60,
) -> dict[str, pd.DataFrame]:
    """Build the market-regime feature groups used by the report."""
    import util.features.core as f_core
    import util.features.families.returns as f_return
    import util.features.families.price as f_price
    import util.features.families.trend as f_trend
    import util.features.families.drawdown as f_dd
    import util.features.transforms as trsfm

    source_df = price_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    builder = f_core.FeatureBuilder(source_df, date_col="date", sym_col="symbol")

    RETURN_FAMILY = f_return.RETURN_FAMILY
    RAWPRICE_FAMILY = f_price.RAWPRICE_FAMILY
    TREND_FAMILY = f_trend.TREND_FAMILY
    MDD_FAMILY = f_dd.MDD_FAMILY

    out: dict[str, pd.DataFrame] = {}

    ret_params = [{"price_col": price_col, "lookback": lb} for lb in return_lookbacks]
    ret_raw = get_features(builder, RETURN_FAMILY, "RETURN", ret_params)
    ret_z = get_features(
        builder, RETURN_FAMILY, "RETURN", ret_params,
        transform_fn=trsfm.add_ts_zscore,
        transform_params={"z_window": z_window, "include_base": False},
    )
    out["return_raw"] = get_feature_df(source_df, ret_raw)
    out["return_z"] = get_feature_df(source_df, ret_z)

    dma_params = [{"price_col": price_col, "window": w} for w in dma_windows]
    dma_raw = get_features(builder, RAWPRICE_FAMILY, "DISTANCE_FROM_MOVING_AVERAGE_PRICE", dma_params)
    dma_z = get_features(
        builder, RAWPRICE_FAMILY, "DISTANCE_FROM_MOVING_AVERAGE_PRICE", dma_params,
        transform_fn=trsfm.add_ts_zscore,
        transform_params={"z_window": z_window, "include_base": False},
    )
    out["dma_raw"] = get_feature_df(source_df, dma_raw)
    out["dma_z"] = get_feature_df(source_df, dma_z)

    if "DISTANCE_FROM_MOVING_AVERAGE_PRICE_VOLATILITY_NORMALIZED" in RAWPRICE_FAMILY:
        vdma_params = [{"price_col": price_col, "window": w} for w in vdma_windows]
        vdma_raw = get_features(builder, RAWPRICE_FAMILY, "DISTANCE_FROM_MOVING_AVERAGE_PRICE_VOLATILITY_NORMALIZED", vdma_params)
        vdma_z = get_features(
            builder, RAWPRICE_FAMILY, "DISTANCE_FROM_MOVING_AVERAGE_PRICE_VOLATILITY_NORMALIZED", vdma_params,
            transform_fn=trsfm.add_ts_zscore,
            transform_params={"z_window": z_window, "include_base": False},
        )
        out["vdma_raw"] = get_feature_df(source_df, vdma_raw)
        out["vdma_z"] = get_feature_df(source_df, vdma_z)

    if "REALIZED_VOLATILITY" in RETURN_FAMILY:
        rv_params = [{"price_col": price_col, "window": w} for w in rv_windows]
        rv_raw = get_features(builder, RETURN_FAMILY, "REALIZED_VOLATILITY", rv_params)
        rv_z = get_features(
            builder, RETURN_FAMILY, "REALIZED_VOLATILITY", rv_params,
            transform_fn=trsfm.add_ts_zscore,
            transform_params={"z_window": z_window, "include_base": False},
        )
        out["rv_raw"] = get_feature_df(source_df, rv_raw)
        out["rv_z"] = get_feature_df(source_df, rv_z)

    if "MDD" in MDD_FAMILY:
        mdd_params = [{"price_col": price_col, "window": w} for w in mdd_windows]
        mdd_raw = get_features(builder, MDD_FAMILY, "MDD", mdd_params)
        mdd_z = get_features(
            builder, MDD_FAMILY, "MDD", mdd_params,
            transform_fn=trsfm.add_ts_zscore,
            transform_params={"z_window": z_window, "include_base": False},
        )
        out["mdd_raw"] = get_feature_df(source_df, mdd_raw)
        out["mdd_z"] = get_feature_df(source_df, mdd_z)

    if "REGRESSION_TREND_LOG_MA" in TREND_FAMILY:
        reg_params = [
            {"price_col": price_col, "window": reg_window, "ma_window": mw}
            for mw in reg_ma_windows
        ]
        reg_raw = get_features(builder, TREND_FAMILY, "REGRESSION_TREND_LOG_MA", reg_params)
        reg_z = get_features(
            builder, TREND_FAMILY, "REGRESSION_TREND_LOG_MA", reg_params,
            transform_fn=trsfm.add_ts_zscore,
            transform_params={
                "z_window": z_window,
                "include_base": False,
                "target": {"signal": "regbeta-rv"},
            },
        )
        out["regtrend_raw"] = get_feature_df(source_df, reg_raw)
        out["regtrend_z"] = get_feature_df(source_df, reg_z)

    return out


# -----------------------------
# Views / panels
# -----------------------------
def get_latest_panel(
    feature_df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    feature_cols: Optional[Sequence[str]] = None,
    column_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    v = feature_df.groupby(symbol_col).tail(1).set_index(symbol_col)
    if feature_cols is None:
        v = v.drop(columns=[date_col], errors="ignore")
    else:
        v = v[list(feature_cols)]
    if column_names is not None:
        v.columns = list(column_names)
    return v


def get_wide_view(
    feature_df: pd.DataFrame,
    feature_col: str,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    window: Optional[int] = None,
) -> pd.DataFrame:
    df = feature_df.copy()
    if window is not None:
        df = df.groupby(symbol_col).tail(window)
    return df[[symbol_col, date_col, feature_col]].pivot(index=date_col, columns=symbol_col, values=feature_col).sort_index()


def find_feature_cols(
    feature_df: pd.DataFrame,
    *,
    contains: Optional[str] = None,
    contains_all: Optional[Sequence[str]] = None,
    excludes: Optional[Sequence[str]] = None,
) -> list[str]:
    cols = [c for c in feature_df.columns if c not in {"symbol", "date"}]
    if contains is not None:
        cols = [c for c in cols if contains in c]
    if contains_all is not None:
        cols = [c for c in cols if all(x in c for x in contains_all)]
    if excludes is not None:
        cols = [c for c in cols if all(x not in c for x in excludes)]
    return cols


def add_breadth_rows(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    if {"RSP", "SPY", "IWM"}.issubset(out.index):
        out.loc["BRS"] = 0.5 * (out.loc["RSP"] - out.loc["SPY"]) + 0.5 * (out.loc["IWM"] - out.loc["SPY"])
    if {"QQQ", "IWM"}.issubset(out.index):
        out.loc["QQQ-IWM"] = out.loc["QQQ"] - out.loc["IWM"]
    if {"QQQ", "SPY"}.issubset(out.index):
        out.loc["QQQ-SPY"] = out.loc["QQQ"] - out.loc["SPY"]
    return out


def make_snapshot_table(feature_groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    panels = []
    for key, label in [
        ("return_raw", "Return"), ("return_z", "Return Z"),
        ("dma_raw", "DMA"), ("dma_z", "DMA Z"),
        ("vdma_raw", "VDMA"), ("vdma_z", "VDMA Z"),
        ("regtrend_raw", "RegTrend"), ("regtrend_z", "RegTrend Z"),
        ("rv_raw", "RV"), ("rv_z", "RV Z"),
        ("mdd_raw", "MDD"), ("mdd_z", "MDD Z"),
    ]:
        if key not in feature_groups:
            continue
        df = feature_groups[key]
        cols = find_feature_cols(df, contains="regbeta-rv") if key == "regtrend_raw" else find_feature_cols(df)
        if not cols:
            continue
        p = get_latest_panel(df, feature_cols=cols)
        p.columns = [f"{label}: {c}" for c in p.columns]
        panels.append(p)
    return pd.concat(panels, axis=1) if panels else pd.DataFrame()


def make_summary_score(feature_groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trend_components = []
    for key in ["return_z", "dma_z", "vdma_z", "regtrend_z"]:
        if key in feature_groups:
            trend_components.append(get_latest_panel(feature_groups[key]).mean(axis=1).rename(key))
    trend_score = pd.concat(trend_components, axis=1).mean(axis=1) if trend_components else pd.Series(dtype=float)

    risk_components = []
    for key in ["rv_z", "mdd_z"]:
        if key in feature_groups:
            risk_components.append((-get_latest_panel(feature_groups[key]).mean(axis=1)).rename(key))
    risk_score = pd.concat(risk_components, axis=1).mean(axis=1) if risk_components else pd.Series(dtype=float)

    out = pd.DataFrame({"trend_score": trend_score, "risk_score": risk_score})
    out["composite_score"] = out.mean(axis=1, skipna=True)
    if {"RSP", "SPY", "IWM"}.issubset(out.index):
        out.loc["BRS"] = 0.5 * (out.loc["RSP"] - out.loc["SPY"]) + 0.5 * (out.loc["IWM"] - out.loc["SPY"])
    return out



# -----------------------------
# Feature-name display helpers
# -----------------------------
def _parse_feature_name_for_display(name: str) -> dict:
    """Best-effort parser for names like px__ret__logret__lb20_zw60__z.

    Delegates to the single canonical parser (util.features.transforms.
    parse_feature_name) so there is one name grammar in the codebase; falls
    back to a signal-only dict for names that don't fit the 5-part format.
    """
    from util.features.transforms import parse_feature_name

    try:
        ns = parse_feature_name(name)
    except ValueError:
        return {
            "domain": "",
            "family": "",
            "signal": name,
            "params": {},
            "state": "",
            "raw": name,
        }

    return {
        "domain": ns.domain,
        "family": ns.family,
        "signal": ns.signal,
        "params": dict(ns.params or {}),
        "state": ns.state,
        "raw": name,
    }


def _pretty_param_value(key: str, raw: str) -> str:
    if raw is None:
        return ""

    # Your feature names use compact values. Keep them compact but readable.
    try:
        numeric = int(raw)
    except Exception:
        numeric = None

    if key in {"lb", "w"} and numeric is not None:
        return f"{numeric}D"
    if key == "ma" and numeric is not None:
        return f"MA{numeric}"
    if key == "zw" and numeric is not None:
        return f"Z{numeric}"
    if key == "ap" and numeric is not None:
        return f"AP{numeric}"
    if key == "p":
        return f"p{raw}"

    return f"{key}{raw}"


def _pretty_param_token(key: str, raw: str) -> str:
    val = _pretty_param_value(key, raw)
    if key in {"lb", "w", "ma", "zw", "ap"}:
        return val
    return val


def summarize_feature_columns_for_display(
    cols: Sequence[str],
    *,
    base_title: str,
) -> tuple[list[str], str]:
    """Return simplified x labels and keep chart title clean.

    Rules:
    - Chart title remains exactly the provided base_title.
    - X-axis labels show only the main varying parameter.
    - Hide technical/fixed params such as zw and state from the visual.
    """
    cols = list(cols)
    if len(cols) == 0:
        return [], base_title

    parsed = [_parse_feature_name_for_display(c) for c in cols]

    # If parsing failed, keep raw labels.
    if any(p["domain"] == "" for p in parsed):
        return cols, base_title

    # Prefer the most meaningful business parameter for this family.
    # Priority:
    # - ma: regression trend over different MA windows
    # - lb: return lookback
    # - w: window length
    # - p: percentile
    # - ap: acceleration period
    priority = ["ma", "lb", "w", "p", "ap"]

    chosen_key = None
    for key in priority:
        values = [p["params"].get(key) for p in parsed]
        if len(set(values)) > 1:
            chosen_key = key
            break

    # If no param varies, use signal as a compact fallback.
    if chosen_key is None:
        return [p["signal"] for p in parsed], base_title

    labels = [
        _pretty_param_token(chosen_key, p["params"].get(chosen_key))
        for p in parsed
    ]

    return labels, base_title


def prepare_panel_display(
    panel: pd.DataFrame,
    *,
    title: str,
) -> tuple[pd.DataFrame, str]:
    labels, _ = summarize_feature_columns_for_display(list(panel.columns), base_title=title)
    out = panel.copy()
    if len(labels) > 0 and len(labels) == len(out.columns):
        out.columns = labels
    return out, title


# -----------------------------
# Plotting
# -----------------------------
def plot_heatmap(
    panel: pd.DataFrame,
    *,
    title: str,
    ax,
    is_percentage: bool = False,
    cmap: str = "RdYlGn",
    center: float = 0.0,
    good_high: bool = True,
    annot_fontsize: int = 12,
    tick_fontsize: int = 11,
    title_fontsize: int = 13,
    simplify_columns: bool = True,
) -> None:
    """Plot a readable heatmap.

    Parameters
    ----------
    good_high:
        True  -> high values are good/green, low values are bad/red.
        False -> low values are good/green, high values are bad/red.
                 Used for risk metrics like volatility and drawdown.
    simplify_columns:
        If True, parse feature names and show only varying parameters on x-axis.
    """
    df = panel.copy()

    if simplify_columns:
        df, title = prepare_panel_display(df, title=title)

    arr = df.to_numpy(dtype=float)

    if arr.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=annot_fontsize)
        ax.set_title(title, fontsize=title_fontsize, weight="bold", pad=10)
        return

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        vmin, vmax = -1, 1
    else:
        max_abs = max(abs(np.nanmin(finite - center)), abs(np.nanmax(finite - center)), 1e-9)
        vmin, vmax = center - max_abs, center + max_abs

    actual_cmap = cmap
    if not good_high:
        if cmap.endswith("_r"):
            actual_cmap = cmap[:-2]
        else:
            actual_cmap = cmap + "_r"

    im = ax.imshow(arr, aspect="auto", cmap=actual_cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=title_fontsize, weight="bold", pad=10)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(
        df.columns,
        rotation=0,
        ha="center",
        fontsize=tick_fontsize,
        fontweight="bold",
    )
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index, fontsize=tick_fontsize)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                txt = f"{val:.1%}" if is_percentage else f"{val:.2f}"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=annot_fontsize,
                    fontweight="semibold",
                    color="black",
                )

    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_line_panel(wide_df: pd.DataFrame, *, title: str, ax, symbols: Optional[Sequence[str]] = None, is_percentage: bool = False) -> None:
    df = wide_df.copy()
    if symbols is not None:
        df = df[[s for s in symbols if s in df.columns]]
    for c in df.columns:
        ax.plot(df.index, df[c], label=c, linewidth=1.8)
    ax.axhline(0, linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=max(1, min(len(df.columns), 4)))
    if is_percentage:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))


def _first_col(cols: list[str]) -> Optional[str]:
    return cols[0] if cols else None


def save_pdf_report(feature_groups: dict[str, pd.DataFrame], *, output_pdf: Path, title: str = "Market Regime Report", tickers: Sequence[str] = ("SPY", "QQQ", "IWM", "RSP")) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    score = make_summary_score(feature_groups)

    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(16, 10), dpi=120)
        fig.suptitle(title, fontsize=18, weight="bold", y=0.98)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

        ax1 = fig.add_subplot(gs[0, 0])
        plot_heatmap(score, title="Composite Regime Score", ax=ax1, is_percentage=False) if not score.empty else ax1.text(0.5, 0.5, "No score data", ha="center")

        ax2 = fig.add_subplot(gs[0, 1])
        if "return_z" in feature_groups:
            plot_heatmap(add_breadth_rows(get_latest_panel(feature_groups["return_z"])), title="Return Z-Score Snapshot", ax=ax2, is_percentage=False)
        else:
            ax2.text(0.5, 0.5, "No return z-score data", ha="center")

        ax3 = fig.add_subplot(gs[1, 0])
        if "return_raw" in feature_groups:
            cols = find_feature_cols(feature_groups["return_raw"])
            col = next((c for c in cols if "lb20" in c), _first_col(cols))
            if col:
                plot_line_panel(get_wide_view(feature_groups["return_raw"], col, window=126), title="Recent Return", ax=ax3, symbols=tickers, is_percentage=True)

        ax4 = fig.add_subplot(gs[1, 1])
        if "regtrend_raw" in feature_groups:
            cols = find_feature_cols(feature_groups["regtrend_raw"], contains="regbeta-rv")
            col = _first_col(cols)
            if col:
                plot_line_panel(get_wide_view(feature_groups["regtrend_raw"], col, window=126), title="Risk-Adjusted Trend", ax=ax4, symbols=tickers, is_percentage=False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig)
        plt.close(fig)

        # Detail pages
        page_specs = [
            ("Momentum / Trend Details", [("return_raw", "Return Raw", True), ("return_z", "Return Z", False), ("dma_z", "DMA Z", False), ("vdma_z", "VDMA Z", False)]),
            ("Risk Details", [("rv_raw", "Realized Volatility", True), ("rv_z", "RV Z", False), ("mdd_raw", "Max Drawdown", True), ("mdd_z", "MDD Z", False)]),
            ("Regression Trend Details", [("regtrend_raw", "Regression Trend Raw", False), ("regtrend_z", "Regression Trend Z", False)]),
        ]
        for page_title, panels in page_specs:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=120)
            fig.suptitle(page_title, fontsize=16, weight="bold")
            for ax, (key, ttl, pct) in zip(axes.ravel(), panels):
                if key in feature_groups:
                    cols = find_feature_cols(feature_groups[key], contains="regbeta-rv") if key == "regtrend_raw" else find_feature_cols(feature_groups[key])
                    p = get_latest_panel(feature_groups[key], feature_cols=cols)
                    p = add_breadth_rows(p) if key in {"return_raw", "return_z", "dma_z", "vdma_z"} else p
                    is_risk_metric = key in {"rv_raw", "rv_z", "mdd_raw", "mdd_z"}
                    plot_heatmap(
                        p,
                        title=ttl,
                        ax=ax,
                        is_percentage=pct,
                        cmap="RdYlGn",
                        good_high=not is_risk_metric,
                    )
                else:
                    ax.text(0.5, 0.5, f"No {key}", ha="center", va="center")
                    ax.set_title(ttl)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)


def save_snapshot_csvs(feature_groups: dict[str, pd.DataFrame], *, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "latest_feature_snapshot.csv"
    score_path = output_dir / "latest_regime_score.csv"
    make_snapshot_table(feature_groups).to_csv(snapshot_path)
    make_summary_score(feature_groups).to_csv(score_path)
    return {"snapshot": snapshot_path, "score": score_path}
