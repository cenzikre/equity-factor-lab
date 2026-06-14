from __future__ import annotations

from util.features.core import (
    FeatureNameSpec,
    FeatureSpec,
    FeatureTemplate,
    make_spec,
    col,
    feat,
)
from util.features.families.returns import spec_ts_return


DOMAIN = "mkt"
FAMILY = "breadth"


def breadth_feature_name(signal: str, *, thr=None, drc=None, lb=None, state="raw") -> FeatureNameSpec:
    params = {}
    if lb is not None:
        params["lb"] = lb
    if thr is not None:
        params["thr"] = thr
    if drc is not None:
        params["drc"] = drc

    return FeatureNameSpec(
        domain=DOMAIN,
        family=FAMILY,
        signal=signal,
        params=params,
        state=state,
    )


def template_cs_retposfrac(price_col: str, lookback: int, *, publish: bool = True) -> FeatureSpec:
    ret = spec_ts_return(price_col, lookback, publish=False)
    frac = make_spec(
        name=breadth_feature_name("retpos", thr=0, drc="gt", lb=lookback),
        primitive="cs_fraction",
        inputs={"x": feat(ret)},
        params={"threshold": 0, "direction": "gt"},
        publish=True
    )
    return [ret, frac]


BREADTH_FAMILY = {
    "POSITIVE_RETURN_FRACTION": FeatureTemplate(
        name="cs_retposfrac",
        domain=DOMAIN,
        family=FAMILY,
        signal="retpos",
        template_fn=template_cs_retposfrac,
        description="",
        tags=("return", "positive", "breadth", "cross-stock"),
    )
}