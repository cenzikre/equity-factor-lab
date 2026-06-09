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
FAMILY = "tr"


def tr_feature_name(signal: str, *, lb=None, w=None, p=None, state="raw") -> str:
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


def spec_tr(
    high_col: str,
    low_col: str,
    close_col: str,
    *,
    publish: bool = False,
) -> FeatureSpec:
    return make_spec(
        name=tr_feature_name("diff", state="raw"),
        primitive="tr",
        inputs={
            "high": col(high_col),
            "low": col(low_col),
            "close": col(close_col),
        },
        publish=publish,
    )


def template_tr(
    high_col: str,
    low_col: str,
    close_col: str
) -> list[FeatureSpec]:

    tr = make_spec(
        name=tr_feature_name("diff", state="raw"),
        primitive="tr",
        inputs={
            "high": col(high_col),
            "low": col(low_col),
            "close": col(close_col),
        },
        publish=True
    )

    return [tr]


def template_ts_atr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:

    tr = make_spec(
        name=tr_feature_name("diff", state="raw"),
        primitive="tr",
        inputs={
            "high": col(high_col),
            "low": col(low_col),
            "close": col(close_col),
        },
        publish=False,
    )

    atr = make_spec(
        name=tr_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": feat(tr)},
        params={"window": window},
        publish=True,
    )

    return [tr, atr]


def template_ts_vtr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:

    tr = make_spec(
        name=tr_feature_name("diff", state="raw"),
        primitive="tr",
        inputs={
            "high": col(high_col),
            "low": col(low_col),
            "close": col(close_col),
        },
        publish=False,
    )

    vtr = make_spec(
        name=tr_feature_name("std", w=window, state="raw"),
        primitive="ts_std",
        inputs={"x": feat(tr)},
        params={"window": window},
        publish=True,
    )

    return [tr, vtr]


def template_ts_natr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:

    tr = make_spec(
        name=tr_feature_name("diff", state="raw"),
        primitive="tr",
        inputs={
            "high": col(high_col),
            "low": col(low_col),
            "close": col(close_col),
        },
        publish=False,
    )

    atr = make_spec(
        name=tr_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": feat(tr)},
        params={"window": window},
        publish=False,
    )

    maprc = make_spec(
        name=make_feature_name(
            domain="px",
            family="prc",
            signal="mean",
            params={"w": window},
            state="raw",
        ),
        primitive="ts_mean",
        inputs={"x": col(close_col)},
        params={"window": window},
        publish=False,
    )

    natr = make_spec(
        name=tr_feature_name("map-r", w=window, state="raw"),
        primitive="ratio",
        inputs={"a": feat(atr), "b": feat(maprc)},
        params={"eps": 1e-12},
        publish=True,
    )

    return [tr, atr, maprc, natr]


def template_ts_zatr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:

    tr = make_spec(
        name=tr_feature_name("diff", state="raw"),
        primitive="tr",
        inputs={
            "high": col(high_col),
            "low": col(low_col),
            "close": col(close_col),
        },
        publish=False,
    )

    atr = make_spec(
        name=tr_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": feat(tr)},
        params={"window": window},
        publish=False,
    )

    mu = make_spec(
        name=tr_feature_name("atr-mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": feat(atr)},
        params={"window": window},
        publish=False
    )

    sd = make_spec(
        name=tr_feature_name("atr-std", w=window, state="raw"),
        primitive="ts_std",
        inputs={"x": feat(atr)},
        params={"window": window},
        publish=False
    )

    zatr = make_spec(
        name=tr_feature_name("z", w=window, state="clip"),
        primitive="zscore",
        inputs={"x": feat(atr), "mu": feat(mu), "sigma": feat(sd)},
        params={"eps": 1e-12},
        post=[('clip', {'lo': -5.0, 'hi': 5.0})],
        publish=True,
    )

    return [tr, atr, mu, sd, zatr]


def template_ts_ratr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:

    tr = make_spec(
        name=tr_feature_name("diff", state="raw"),
        primitive="tr",
        inputs={
            "high": col(high_col),
            "low": col(low_col),
            "close": col(close_col),
        },
        publish=False,
    )

    atr = make_spec(
        name=tr_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": feat(tr)},
        params={"window": window},
        publish=False,
    )

    mu = make_spec(
        name=tr_feature_name("atr-mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": feat(atr)},
        params={"window": window},
        publish=False
    )

    ratr = make_spec(
        name=tr_feature_name("atr-r", w=window, state="raw"),
        primitive="ratio",
        inputs={"a": feat(atr), "b": feat(mu)},
        params={"eps": 1e-12},
        publish=True,
    )

    return [tr, atr, mu, ratr]


ATR_FAMILY = {
    'NORMALIZED_AVERAGE_TRUERANGE': FeatureTemplate(
        name="ts_natr",
        domain=DOMAIN,
        family=FAMILY,
        signal="map-r",
        template_fn=template_ts_natr,
        description="normalized average true range in moving window",
        tags=("true range", "average", "normalized", "timeseries"),
    ),
    'ZSCORE_AVERAGE_TRUERANGE': FeatureTemplate(
        name="ts_zatr",
        domain=DOMAIN,
        family=FAMILY,
        signal="z",
        template_fn=template_ts_zatr,
        description="zscore of average true range in moving window",
        tags=("true range", "average", "zscore", "timeseries"),
    ),
    'RATIO_AVERAGE_TRUERANGE': FeatureTemplate(
        name="ts_ratr",
        domain=DOMAIN,
        family=FAMILY,
        signal="atr-r",
        template_fn=template_ts_ratr,
        description="average true range to its average ratio in moving window",
        tags=("true range", "average", "ratio", "timeseries"),
    ),
}