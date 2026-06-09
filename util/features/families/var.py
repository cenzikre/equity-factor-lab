from __future__ import annotations
from typing import Any, Dict, Literal
from util.features.core import (
    FeatureSpec,
    FeatureTemplate,
    make_feature_name,
    make_spec,
    feat,
)
from util.features.families.returns import spec_ts_return
import util.features.primitives_tailrisk


DOMAIN = "px"
FAMILY = "var"
CVARMethod = Literal["simple", "fast"]


def _validate_var_args(lookback: int, window: int, p: float) -> None:
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if lookback >= window:
        raise ValueError(
            f"Expected lookback < window for VaR-style features; "
            f"got lookback={lookback}, window={window}"
        )
    if not 0 < p < 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")


def _p_name_tag(p: float) -> int:
    """Readable compact percentile tag.

    p=0.05 -> 5, p=0.95 -> 95.
    The exact p still lives in FeatureSpec.params and feature_id.
    """
    return int(round(p * 100))


def tailrisk_feature_name(
    family: str,
    signal: str,
    *,
    lb: int,
    w: int,
    p: float,
    state: str = "raw",
) -> str:
    params: Dict[str, Any] = {
        "lb": lb,
        "w": w,
        "p": _p_name_tag(p),
    }
    return make_feature_name(
        domain=DOMAIN,
        family=family,
        signal=signal,
        params=params,
        state=state,
    )


def spec_ts_var(
    ret_spec: FeatureSpec,
    lookback: int,
    window: int,
    p: float,
    *,
    publish: bool = True,
) -> FeatureSpec:
    _validate_var_args(lookback, window, p)

    return make_spec(
        name=tailrisk_feature_name(
            family="var",
            signal="q",
            lb=lookback,
            w=window,
            p=p,
            state="raw",
        ),
        primitive="ts_quantile",
        inputs={"x": feat(ret_spec)},
        params={
            "window": window,
            "q": p,
        },
        publish=publish,
    )


def spec_ts_cvar(
    ret_spec: FeatureSpec,
    lookback: int,
    window: int,
    p: float,
    *,
    method: CVARMethod = "fast",
    publish: bool = True,
) -> FeatureSpec:
    _validate_var_args(lookback, window, p)

    if window * p < 2:
        raise ValueError(
            f"CVaR tail is too small for stable calculation: "
            f"window * p = {window * p:.2f}; expected >= 2"
        )

    if method == "fast":
        primitive = "ts_tail_mean_fast"
        state = "raw"
    elif method == "simple":
        primitive = "ts_tail_mean"
        state = "simple"
    else:
        raise ValueError(f"Unknown CVaR method: {method!r}")

    return make_spec(
        name=tailrisk_feature_name(
            family="cvar",
            signal="mean",
            lb=lookback,
            w=window,
            p=p,
            state=state,
        ),
        primitive=primitive,
        inputs={"x": feat(ret_spec)},
        params={
            "window": window,
            "q": p,
            "tail": "lower",
        },
        publish=publish,
    )


def spec_ts_varr(
    lower_var_spec: FeatureSpec,
    upper_var_spec: FeatureSpec,
    lookback: int,
    window: int,
    p: float,
    *,
    publish: bool = True,
) -> FeatureSpec:
    _validate_var_args(lookback, window, p)

    return make_spec(
        name=tailrisk_feature_name(
            family="varr",
            signal="r",
            lb=lookback,
            w=window,
            p=p,
            state="raw",
        ),
        primitive="abs_ratio",
        inputs={
            "a": feat(upper_var_spec),
            "b": feat(lower_var_spec),
        },
        params={"eps": 1e-12},
        publish=publish,
    )


def template_ts_var(
    price_col: str,
    lookback: int,
    window: int,
    p: float = 0.05,
) -> list[FeatureSpec]:
    ret = spec_ts_return(price_col, lookback, publish=False)
    var = spec_ts_var(ret, lookback, window, p, publish=True)
    return [ret, var]


def template_ts_varr(
    price_col: str,
    lookback: int,
    window: int,
    p: float = 0.05,
) -> list[FeatureSpec]:
    ret = spec_ts_return(price_col, lookback, publish=False)

    lower = spec_ts_var(ret, lookback, window, p, publish=False)
    upper_p = 1.0 - p
    upper = spec_ts_var(ret, lookback, window, upper_p, publish=False)

    varr = spec_ts_varr(
        lower_var_spec=lower,
        upper_var_spec=upper,
        lookback=lookback,
        window=window,
        p=p,
        publish=True,
    )

    return [ret, lower, upper, varr]


def template_ts_cvar(
    price_col: str,
    lookback: int,
    window: int,
    p: float = 0.05,
    method: CVARMethod = "fast",
) -> list[FeatureSpec]:
    ret = spec_ts_return(price_col, lookback, publish=False)
    cvar = spec_ts_cvar(ret, lookback, window, p, method=method, publish=True)
    return [ret, cvar]


VAR_FAMILY = {
    "VALUE_AT_RISK": FeatureTemplate(
        name="ts_var",
        domain=DOMAIN,
        family="var",
        signal="q",
        template_fn=template_ts_var,
        description="rolling historical return quantile, used as VaR-style downside threshold",
        tags=("return", "risk", "var", "quantile", "timeseries"),
    ),
    "VALUE_AT_RISK_RATIO": FeatureTemplate(
        name="ts_varr",
        domain=DOMAIN,
        family="varr",
        signal="r",
        template_fn=template_ts_varr,
        description="ratio of upper-tail absolute quantile to lower-tail absolute quantile",
        tags=("return", "risk", "var", "ratio", "tail", "timeseries"),
    ),
    "CONDITIONAL_VALUE_AT_RISK": FeatureTemplate(
        name="ts_cvar",
        domain=DOMAIN,
        family="cvar",
        signal="mean",
        template_fn=template_ts_cvar,
        description="rolling conditional lower-tail mean below the VaR quantile",
        tags=("return", "risk", "cvar", "tail", "timeseries"),
    ),
}