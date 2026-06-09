from __future__ import annotations

from util.features.core import (
    FeatureSpec,
    FeatureTemplate,
    make_feature_name,
    make_spec,
    col,
    feat,
)


DOMAIN = "px"
FAMILY = "ret"


def ret_feature_name(signal: str, *, lb=None, w=None, p=None, state="raw") -> str:
    params = {}
    if lb is not None:
        params["lb"] = lb
    if w is not None:
        params["w"] = w
    if p is not None:
        params["p"] = p

    return make_feature_name(
        domain=DOMAIN,
        family=FAMILY,
        signal=signal,
        params=params,
        state=state,
    )


def spec_ts_return(price_col: str, lookback: int, *, publish: bool = True) -> FeatureSpec:
    return make_spec(
        name=ret_feature_name("logret", lb=lookback),
        primitive="ts_return",
        inputs={"price": col(price_col)},
        params={"lookback": lookback},
        publish=publish,
    )


def spec_ts_mar(ret_spec: FeatureSpec, lookback: int, window: int, *, publish: bool = True) -> FeatureSpec:
    return make_spec(
        name=ret_feature_name("mean", lb=lookback, w=window),
        primitive="ts_mean",
        inputs={"x": feat(ret_spec)},
        params={"window": window},
        publish=publish,
    )


def spec_ts_mvr(ret_spec: FeatureSpec, lookback: int, window: int, *, publish: bool = True) -> FeatureSpec:
    return make_spec(
        name=ret_feature_name("std", lb=lookback, w=window),
        primitive="ts_std",
        inputs={"x": feat(ret_spec)},
        params={"window": window},
        publish=publish,
    )


def template_ts_return(price_col: str, lookback: int) -> list[FeatureSpec]:
    ret = spec_ts_return(price_col, lookback, publish=True)
    return [ret]


def template_ts_anuret(price_col: str, lookback: int) -> list[FeatureSpec]:
    ret = make_spec(
        name=ret_feature_name("smplret", lb=lookback, state="annual"),
        primitive="ts_return",
        inputs={"price": col(price_col)},
        params={"lookback": lookback},
        post=[
            ("annualize_logret", {"w": lookback}),
            ("delog", {}),
        ],
        publish=True,
    )
    return [ret]


def template_ts_mar(price_col: str, lookback: int, window: int) -> list[FeatureSpec]:
    ret = spec_ts_return(price_col, lookback, publish=False)
    mar = spec_ts_mar(ret, lookback, window, publish=True)
    return [ret, mar]


def template_ts_mvr(price_col: str, lookback: int, window: int) -> list[FeatureSpec]:
    ret = spec_ts_return(price_col, lookback, publish=False)
    mvr = spec_ts_mvr(ret, lookback, window, publish=True)
    return [ret, mvr]


def template_ts_rv(
    price_col: str,
    window: int,
    lookback: int = 1,
    annual: bool = True
) -> list[FeatureSpec]:
    ret = spec_ts_return(price_col, lookback, publish=False)
    rv = make_spec(
        name=ret_feature_name("rv", lb=lookback, w=window, state="annual"),
        primitive="ts_std",
        inputs={"x": feat(ret)},
        params={"window": window},
        post=[
            ("annualize_vol", {"w": lookback}),
        ],
        publish=True,
    )
    return [ret, rv]


def template_ts_mrz(price_col: str, lookback: int, window: int) -> list[FeatureSpec]:
    ret = spec_ts_return(price_col, lookback, publish=False)
    mu = spec_ts_mar(ret, lookback, window, publish=False)
    sd = spec_ts_mvr(ret, lookback, window, publish=False)
    mrz = make_spec(
        name=ret_feature_name("z", lb=lookback, w=window, state="clip"),
        primitive="zscore",
        inputs={"x": feat(ret), "mu": feat(mu), "sigma": feat(sd)},
        params={"eps": 1e-12},
        post=[("clip", {"lo": -5.0, "hi": 5.0})],
        publish=True,
    )
    return [ret, mu, sd, mrz]


def template_ts_mrr(price_col: str, lookback: int, window: int) -> list[FeatureSpec]:
    ret = spec_ts_return(price_col, lookback, publish=False)
    mu = spec_ts_mar(ret, lookback, window, publish=False)
    mrr = make_spec(
        name=ret_feature_name("r", lb=lookback, w=window),
        primitive="ratio",
        inputs={"a": feat(ret), "b": feat(mu)},
        params={"eps": 1e-12},
        publish=True,
    )
    return [ret, mu, mrr]


RETURN_FAMILY = {
    "RETURN": FeatureTemplate(
        name="ts_return",
        domain=DOMAIN,
        family=FAMILY,
        signal="logret",
        template_fn=template_ts_return,
        description="Time-series log return from price over given lookback window.",
        tags=("price", "return", "timeseries"),
    ),
    "ANNUALIZED_SIMPLE_RETURN": FeatureTemplate(
        name="ts_anuret",
        domain=DOMAIN,
        family=FAMILY,
        signal="ret",
        template_fn=template_ts_anuret,
        description="Annualized return from price over given lookback window.",
        tags=("price", "return", "timeseries", "annualize"),
    ),
    "MOVING_AVERAGE_RETURN": FeatureTemplate(
        name="ts_mar",
        domain=DOMAIN,
        family=FAMILY,
        signal="mean",
        template_fn=template_ts_mar,
        description="moving average of log return",
        tags=("price", "return", "average", "timeseries"),
    ),
    "MOVING_VOLATILITY_RETURN": FeatureTemplate(
        name="ts_mvr",
        domain=DOMAIN,
        family=FAMILY,
        signal="std",
        template_fn=template_ts_mvr,
        description="moving std of log return",
        tags=("price", "return", "std", "volatility", "timeseries"),
    ),
    "RETURN_ZSCORE": FeatureTemplate(
        name="ts_mzr",
        domain=DOMAIN,
        family=FAMILY,
        signal="z",
        template_fn=template_ts_mrz,
        description="zscore of return in moving window",
        tags=("price", "return", "zscore", "timeseries"),
    ),
    "RETURN_RATIO": FeatureTemplate(
        name="ts_mrr",
        domain=DOMAIN,
        family=FAMILY,
        signal="r",
        template_fn=template_ts_mrr,
        description="return to its average ratio in moving window",
        tags=("price", "return", "ratio", "timeseries"),
    ),
    "REALIZED_VOLATILITY": FeatureTemplate(
        name="ts_rv",
        domain=DOMAIN,
        family=FAMILY,
        signal="rv",
        template_fn=template_ts_rv,
        description="realized volatility annualized",
        tags=("price", "return", "std", "volatility", "timeseries", "annualized"),
    ),
}
