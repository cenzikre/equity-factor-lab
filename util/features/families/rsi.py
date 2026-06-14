from __future__ import annotations
from dataclasses import replace
from util.features.core import (
    FeatureNameSpec,
    FeatureSpec,
    FeatureTemplate,
    make_spec,
    col,
    feat,
)


DOMAIN = "px"
FAMILY = "rsi"


def rsi_feature_name(signal: str, *, lb=None, w=None, s=None, p=None, state="raw") -> FeatureNameSpec:
    params = {}
    if lb is not None:
        params["lb"] = lb
    if w is not None:
        params["w"] = w
    if s is not None:
        params["s"] = s
    if p is not None:
        params["p"] = p

    return FeatureNameSpec(
        domain=DOMAIN,
        family=FAMILY,
        signal=signal,
        params=params,
        state=state,
    )


def template_ts_rsi(
    price_col: str,
    window: int,
) -> list[FeatureSpec]:

    pdiff = make_spec(
        name=FeatureNameSpec(
            domain="px",
            family="prc",
            signal="tsdiff",
            params={"lb": 1},
            state="raw",
        ),
        primitive="ts_diff",
        inputs={"x": col(price_col)},
        params={"lookback": 1},
        publish=False,
    )

    gain = make_spec(
        name=rsi_feature_name("gain", s=1),
        primitive="scale",
        inputs={"x": feat(pdiff)},
        params={"scaler": 1},
        post=[("clip", {"lo": 0})],
        publish=False,
    )

    loss = make_spec(
        name=rsi_feature_name("loss", s=-1),
        primitive="scale",
        inputs={"x": feat(pdiff)},
        params={"scaler": -1},
        post=[("clip", {"lo": 0})],
        publish=False,
    )

    avg_gain = make_spec(
        name=rsi_feature_name("avg-gain", w=window),
        primitive="ts_ewm_mean",
        inputs={"x": feat(gain)},
        params={"window": window},
        publish=False,
    )

    avg_loss = make_spec(
        name=rsi_feature_name("avg-loss", w=window),
        primitive="ts_ewm_mean",
        inputs={"x": feat(loss)},
        params={"window": window},
        publish=False,
    )

    rsi = make_spec(
        name=rsi_feature_name("r", w=window, state="index"),
        primitive="ratio",
        inputs={"a": feat(avg_gain), "b": feat(avg_loss)},
        params={"eps": 1e-12},
        post=[("index_scale", {"offset": 100.0})],
        publish=True,
    )

    return [pdiff, gain, loss, avg_gain, avg_loss, rsi]


def template_ts_rsiz(
    price_col: str,
    window: int,
    period: int,
) -> list[FeatureSpec]:

    rsi_template = template_ts_rsi(price_col, window)
    rsi = replace(rsi_template[-1], publish=False)

    mu = make_spec(
        name=rsi_feature_name("mean", w=window, p=period),
        primitive="ts_mean",
        inputs={"x": feat(rsi)},
        params={"window": period},
        publish=False,
    )

    sd = make_spec(
        name=rsi_feature_name("std", w=window, p=period),
        primitive="ts_std",
        inputs={"x": feat(rsi)},
        params={"window": period},
        publish=False,
    )

    rsiz = make_spec(
        name=rsi_feature_name("z", w=window, p=period, state="clip"),
        primitive="zscore",
        inputs={"x": feat(rsi), "mu": feat(mu), "sigma": feat(sd)},
        params={"eps": 1e-12},
        post=[("clip", {"lo": -5.0, "hi": 5.0})],
        publish=True,
    )

    return rsi_template[:-1] + [rsi, mu, sd, rsiz]


RSI_FAMILY = {
    'RSI': FeatureTemplate(
        name="ts_rsi",
        domain=DOMAIN,
        family=FAMILY,
        signal="r",
        template_fn=template_ts_rsi,
        description="relative strength index",
        tags=("price", "rsi", "index", "timeseries"),
    ),
    'RSI_ZSCORE': FeatureTemplate(
        name="ts_rsiz",
        domain=DOMAIN,
        family=FAMILY,
        signal="z",
        template_fn=template_ts_rsiz,
        description="zscore of relative strength index",
        tags=("price", "rsi", "index", "zscore", "timeseries"),
    ),
}