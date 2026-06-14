from __future__ import annotations
from dataclasses import replace
from util.features.families.truerange import template_ts_atr
from util.features.core import (
    FeatureNameSpec,
    FeatureSpec,
    FeatureTemplate,
    make_spec,
    col,
    feat,
)


DOMAIN = "px"
FAMILY = "dd"


def dd_feature_name(signal: str, *, lb=None, w=None, state="raw") -> FeatureNameSpec:
    params = {}
    if lb is not None:
        params["lb"] = lb
    if w is not None:
        params["w"] = w

    return FeatureNameSpec(
        domain=DOMAIN,
        family=FAMILY,
        signal=signal,
        params=params,
        state=state,
    )


def template_ts_mdd(
    price_col: str,
    window: int
) -> list[FeatureSpec]:

    peak = make_spec(
        name=dd_feature_name("max", w=window),
        primitive="ts_max",
        inputs={"x": col(price_col)},
        params={"window": window},
        publish=False,
    )

    dd = make_spec(
        name=dd_feature_name("rdiff", w=window),
        primitive="rdiff",
        inputs={"a": col(price_col), "b": feat(peak)},
        params={"eps": 1e-12},
        publish=False,
    )

    mdd = make_spec(
        name=dd_feature_name("min", w=window, state="scale"),
        primitive="ts_min",
        inputs={"x": feat(dd)},
        params={"window": window},
        post=[("scale", {"scaler": -1})],
        publish=True,
    )

    return [peak, dd, mdd]


def template_ts_mddatrnorm(
    price_col: str,
    high_col: str,
    low_col: str,
    window: int
) -> list[FeatureSpec]:

    mdd_template = template_ts_mdd(price_col, window)
    mdd = replace(
        mdd_template[-1],
        publish=False
    )

    atr_template = template_ts_atr(high_col, low_col, price_col, 14)
    atr = replace(
        atr_template[-1],
        publish=False
    )

    atr_base = make_spec(
        name=FeatureNameSpec(
            domain="px",
            family="tr",
            signal="mean",
            params={"w": 14, "p": 252},
            state="raw"
        ),
        primitive="ts_ewm_span_mean",
        inputs={"x": feat(atr)},
        params={"span": 252},
        publish=False,
    )

    mddatrnorm = make_spec(
        name=dd_feature_name("atr-r", w=window),
        primitive="ratio",
        inputs={"a": feat(mdd), "b": feat(atr_base)},
        params={"eps": 1e-12},
        publish=True
    )

    return [*mdd_template[:-1], mdd, *atr_template[:-1], atr, atr_base, mddatrnorm]


MDD_FAMILY = {
    'MDD': FeatureTemplate(
        name="ts_mdd",
        domain=DOMAIN,
        family=FAMILY,
        signal="max",
        template_fn=template_ts_mdd,
        description="maximum drawdown in moving window",
        tags=("price", "mdd", "max", "timeseries"),
    ),
    'MDD_ATR_NORM': FeatureTemplate(
        name="ts_mddatrnorm",
        domain=DOMAIN,
        family=FAMILY,
        signal="atr-r",
        template_fn=template_ts_mddatrnorm,
        description="maximum drawdown in moving window normalized by average true range baseline",
        tags=("price", "mdd", "norm", "atr", "timeseries"),
    ),
}