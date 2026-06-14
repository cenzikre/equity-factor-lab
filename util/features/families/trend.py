from __future__ import annotations
# from dataclasses import replace
from typing import Any, Dict, Literal, Optional
from util.features.core import (
    FeatureNameSpec,
    FeatureSpec,
    FeatureTemplate,
    make_spec,
    col,
    feat,
)
from util.features.families.price import spec_ts_map
from util.features.families.returns import spec_ts_return, spec_ts_mvr
import util.features.primitives_regtrend
import util.features.primitives_regtrend_vectorized


DOMAIN = "px"
FAMILY = "trend"
RegTrendMethod = Literal["simple", "vectorized"]


def trend_feature_name(
    signal: str,
    *,
    w: int,
    ma: Optional[int] = None,
    ap: Optional[int] = None,
    state: str = "logp",
) -> FeatureNameSpec:
    """Deterministic readable name for regression-trend features.

    Parameters:
    - w: regression window
    - ma: optional moving-average window used before log transform
    - ap: optional acceleration period
    - state:
        logp  = regression on log(price)
        logma = regression on log(MA)
    """
    params: Dict[str, Any] = {"w": w}
    if ma is not None:
        params["ma"] = ma
    if ap is not None:
        params["ap"] = ap

    return FeatureNameSpec(
        domain=DOMAIN,
        family=FAMILY,
        signal=signal,
        params=params,
        state=state,
    )


def _reg_primitive(stat: str, method: RegTrendMethod) -> str:
    simple = {
        "slope": "ts_reg_slope",
        "intercept": "ts_reg_intercept",
        "r2": "ts_reg_r2",
        "resid_std": "ts_reg_resid_std",
        "slope_se": "ts_reg_slope_se",
        "tstat": "ts_reg_tstat",
    }
    vectorized = {
        "slope": "ts_reg_slope_vectorized",
        "intercept": "ts_reg_intercept_vectorized",
        "r2": "ts_reg_r2_vectorized",
        "resid_std": "ts_reg_resid_std_vectorized",
        "slope_se": "ts_reg_slope_se_vectorized",
        "tstat": "ts_reg_tstat_vectorized",
    }

    if method == "simple":
        return simple[stat]
    if method == "vectorized":
        return vectorized[stat]
    raise ValueError(f"Unknown regression trend method: {method!r}")


def _validate_trend_args(
    window: int,
    ma_window: Optional[int] = None,
    accel_period: int = 1,
) -> None:
    if window <= 2:
        raise ValueError(f"regression window must be greater than 2, got {window}")
    if ma_window is not None and ma_window <= 0:
        raise ValueError(f"ma_window must be positive, got {ma_window}")
    if accel_period <= 0:
        raise ValueError(f"accel_period must be positive, got {accel_period}")


def spec_log_price(
    price_col: str,
    *,
    publish: bool = False,
) -> FeatureSpec:
    return make_spec(
        name=FeatureNameSpec(
            domain="px",
            family="prc",
            signal="log",
            params={},
            state="raw",
        ),
        primitive="log",
        inputs={"x": col(price_col)},
        params={"eps": 1e-12},
        publish=publish,
    )


def spec_log_ma(
    price_col: str,
    ma_window: int,
    *,
    publish: bool = False,
) -> FeatureSpec:
    # Reuse existing price-family helper:
    # price -> MA -> post log.
    return spec_ts_map(
        price_col=price_col,
        window=ma_window,
        state="log",
        publish=publish,
    )


def _regtrend_specs_from_signal(
    y_spec: FeatureSpec,
    price_col: str,
    *,
    window: int,
    state: str,
    ma_window: Optional[int] = None,
    accel_period: int = 1,
    method: RegTrendMethod = "vectorized",
) -> list[FeatureSpec]:
    """Build stage 1-2 regression-trend specs from a transformed y series.

    y is typically log(price) or log(MA).
    """
    _validate_trend_args(
        window=window,
        ma_window=ma_window,
        accel_period=accel_period,
    )

    slope = make_spec(
        name=trend_feature_name("regbeta", w=window, ma=ma_window, state=state),
        primitive=_reg_primitive("slope", method),
        inputs={"x": feat(y_spec)},
        params={"window": window},
        publish=True,
    )

    intercept = make_spec(
        name=trend_feature_name("regalpha", w=window, ma=ma_window, state=state),
        primitive=_reg_primitive("intercept", method),
        inputs={"x": feat(y_spec)},
        params={"window": window},
        publish=False,
    )

    r2 = make_spec(
        name=trend_feature_name("regr2", w=window, ma=ma_window, state=state),
        primitive=_reg_primitive("r2", method),
        inputs={"x": feat(y_spec)},
        params={"window": window},
        publish=True,
    )

    slope_se = make_spec(
        name=trend_feature_name("regse", w=window, ma=ma_window, state=state),
        primitive=_reg_primitive("slope_se", method),
        inputs={"x": feat(y_spec)},
        params={"window": window},
        publish=True,
    )

    resid_std = make_spec(
        name=trend_feature_name("residstd", w=window, ma=ma_window, state=state),
        primitive=_reg_primitive("resid_std", method),
        inputs={"x": feat(y_spec)},
        params={"window": window},
        publish=True,
    )

    tstat = make_spec(
        name=trend_feature_name("regt", w=window, ma=ma_window, state=state),
        primitive=_reg_primitive("tstat", method),
        inputs={"x": feat(y_spec)},
        params={"window": window},
        publish=True,
    )

    ret = spec_ts_return(price_col=price_col, lookback=1, publish=False)
    rv = spec_ts_mvr(ret_spec=ret, lookback=1, window=window, publish=False)

    beta_sigma = make_spec(
        name=trend_feature_name("regbeta-rv", w=window, ma=ma_window, state=state),
        primitive="ratio",
        inputs={"a": feat(slope), "b": feat(rv)},
        params={"eps": 1e-12},
        publish=True,
    )

    slope_change = make_spec(
        name=trend_feature_name(
            "regbeta-diff",
            w=window,
            ma=ma_window,
            ap=accel_period,
            state=state,
        ),
        primitive="ts_diff",
        inputs={"x": feat(slope)},
        params={"lookback": accel_period},
        publish=False,
    )

    accel = make_spec(
        name=trend_feature_name(
            "regaccel",
            w=window,
            ma=ma_window,
            ap=accel_period,
            state=state,
        ),
        primitive="scale",
        inputs={"x": feat(slope_change)},
        params={"scaler": 1.0 / accel_period},
        publish=True,
    )

    return [
        slope,
        intercept,
        r2,
        slope_se,
        resid_std,
        tstat,
        ret,
        rv,
        beta_sigma,
        slope_change,
        accel,
    ]


def template_ts_logprice_regtrend(
    price_col: str,
    window: int,
    accel_period: int = 1,
    method: RegTrendMethod = "vectorized",
) -> list[FeatureSpec]:
    """Regression trend family on log(price)."""
    y = spec_log_price(price_col, publish=False)

    trend_specs = _regtrend_specs_from_signal(
        y_spec=y,
        price_col=price_col,
        window=window,
        state="logp",
        ma_window=None,
        accel_period=accel_period,
        method=method,
    )

    return [y] + trend_specs


def template_ts_logma_regtrend(
    price_col: str,
    ma_window: int,
    window: int,
    accel_period: int = 1,
    method: RegTrendMethod = "vectorized",
) -> list[FeatureSpec]:
    """Regression trend family on log(MA)."""
    y = spec_log_ma(price_col, ma_window, publish=False)

    trend_specs = _regtrend_specs_from_signal(
        y_spec=y,
        price_col=price_col,
        window=window,
        state="logma",
        ma_window=ma_window,
        accel_period=accel_period,
        method=method,
    )

    return [y] + trend_specs


def template_ts_maslope(
    price_col: str,
    ma_window: int,
    window: int,
) -> list[FeatureSpec]:

    maprc = make_spec(
        name=FeatureNameSpec(
            domain="px",
            family="prc",
            signal="mean",
            params={"w": ma_window},
            state="raw",
        ),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": ma_window},
        publish=False,
    )

    slope = make_spec(
        name=trend_feature_name("trend", w=window, ma=ma_window, state="raw"),
        primitive="ts_slope",
        inputs={"x": feat(maprc)},
        params={"lookback": window},
        publish=True,
    )

    return [maprc, slope]


def template_ts_mapctslope(
    price_col: str,
    ma_window: int,
    window: int,
) -> list[FeatureSpec]:

    maprc = make_spec(
        name=FeatureNameSpec(
            domain="px",
            family="prc",
            signal="mean",
            params={"w": ma_window},
            state="raw",
        ),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": ma_window},
        publish=False,
    )

    slope = make_spec(
        name=trend_feature_name("trend", w=window, ma=ma_window, state="pct"),
        primitive="ts_pctslope",
        inputs={"x": feat(maprc)},
        params={"lookback": window},
        publish=True,
    )

    return [maprc, slope]


def template_ts_malogslope(
    price_col: str,
    ma_window: int,
    window: int,
) -> list[FeatureSpec]:

    maprc = make_spec(
        name=FeatureNameSpec(
            domain="px",
            family="prc",
            signal="mean",
            params={"w": ma_window},
            state="raw",
        ),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": ma_window},
        publish=False,
    )

    slope = make_spec(
        name=trend_feature_name("trend", w=window, ma=ma_window, state="log"),
        primitive="ts_logslope",
        inputs={"x": feat(maprc)},
        params={"lookback": window},
        publish=False,
    )

    ret = spec_ts_return(price_col=price_col, lookback=1, publish=False)
    rv = spec_ts_mvr(ret_spec=ret, lookback=1, window=window, publish=False)

    slope_sigma = make_spec(
        name=trend_feature_name("trend", w=window, ma=ma_window, state="log-rv"),
        primitive="ratio",
        inputs={"a": feat(slope), "b": feat(rv)},
        params={"eps": 1e-12},
        publish=True,
    )

    return [ret, rv, maprc, slope, slope_sigma]


TREND_FAMILY = {
    "REGRESSION_TREND_LOG_PRICE": FeatureTemplate(
        name="ts_logprice_regtrend",
        domain=DOMAIN,
        family=FAMILY,
        signal="regbeta",
        template_fn=template_ts_logprice_regtrend,
        description="rolling linear-regression trend diagnostics on log price",
        tags=("price", "log", "trend", "regression", "slope", "r2", "timeseries"),
    ),
    "REGRESSION_TREND_LOG_MA": FeatureTemplate(
        name="ts_logma_regtrend",
        domain=DOMAIN,
        family=FAMILY,
        signal="regbeta",
        template_fn=template_ts_logma_regtrend,
        description="rolling linear-regression trend diagnostics on log moving average price",
        tags=("price", "log", "moving average", "trend", "regression", "slope", "r2", "timeseries"),
    ),
    "MOVING_AVERAGE_SLOPE": FeatureTemplate(
        name="ts_maslope",
        domain=DOMAIN,
        family=FAMILY,
        signal="trend",
        template_fn=template_ts_maslope,
        description="slope of moving average price over a lookback period",
        tags=("price", "average", "slope", "trend", "timeseries"),
        unitless=False,
    ),
    "MOVING_AVERAGE_PCT_SLOPE": FeatureTemplate(
        name="ts_mapctslope",
        domain=DOMAIN,
        family=FAMILY,
        signal="trend",
        template_fn=template_ts_mapctslope,
        description="percent slope of moving average price over a lookback period",
        tags=("price", "average", "slope", "trend", "timeseries"),
    ),
    "MOVING_AVERAGE_LOG_SLOPE_NORMALIZED_BY_RV": FeatureTemplate(
        name="ts_malogslope",
        domain=DOMAIN,
        family=FAMILY,
        signal="trend",
        template_fn=template_ts_malogslope,
        description="rv normalized log slope of moving average price over a lookback period",
        tags=("price", "average", "slope", "trend", "timeseries", "rv"),
    ),
}