from __future__ import annotations
from dataclasses import replace
from util.features.core import (
    FeatureSpec,
    FeatureTemplate,
    make_feature_name,
    make_spec,
    col,
    feat,
)
from util.features.families.returns import template_ts_mvr


DOMAIN = "px"
FAMILY = "prc"


def prc_feature_name(
    signal: str,
    *,
    lb=None,
    w=None,
    sw=None,
    lw=None,
    p=None,
    state="raw",
) -> str:
    params = {}
    if lb is not None:
        params["lb"] = lb
    if w is not None:
        params["w"] = w
    if sw is not None:
        params["sw"] = sw
    if lw is not None:
        params["lw"] = lw
    if p is not None:
        params["p"] = p

    return make_feature_name(
        domain=DOMAIN,
        family=FAMILY,
        signal=signal,
        params=params,
        state=state,
    )


def spec_ts_map(
    price_col: str,
    window: int,
    *,
    state: str = "raw",
    publish: bool = False
) -> FeatureSpec:
    post = None
    if state == "log":
        post = [("log", {})]
    elif state != "raw":
        raise ValueError(f"Unsupported moving-average price state: {state}")

    return make_spec(
        name=prc_feature_name("mean", w=window, state=state),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": window},
        post=post,
        publish=publish,
    )


def spec_ts_mvp(
    price_col: str,
    window: int,
    *,
    publish: bool = False,
) -> FeatureSpec:
    return make_spec(
        name=prc_feature_name("std", w=window, state="raw"),
        primitive="ts_std",
        inputs={"x": col(price_col)},
        params={"window": window},
        publish=publish,
    )


def template_ts_map(price_col: str, window: int) -> list[FeatureSpec]:

    maprc = make_spec(
        name=prc_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": window},
        # post=[("log", {})],
        publish=True,
    )

    return [maprc]


def template_ts_mvp(price_col: str, window: int) -> list[FeatureSpec]:

    mvprc = make_spec(
        name=prc_feature_name("std", w=window, state="raw"),
        primitive="ts_std",
        inputs={"x": col(price_col)},
        params={"window": window},
        # post=[("log", {})],
        publish=True,
    )

    return [mvprc]


def template_ts_blgz(price_col: str, window: int) -> list[FeatureSpec]:

    maprc = make_spec(
        name=prc_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": window},
        post=None,
        publish=False,
    )

    mvprc = make_spec(
        name=prc_feature_name("std", w=window, state="raw"),
        primitive="ts_std",
        inputs={"x": col(price_col)},
        params={"window": window},
        post=None,
        publish=False,
    )

    blgz = make_spec(
        name=prc_feature_name("z", w=window, state="clip"),
        primitive="zscore",
        inputs={"x": col(price_col), "mu": feat(maprc), "sigma": feat(mvprc)},
        params={"eps": 1e-12},
        post=[("clip", {"lo": -5.0, "hi": 5.0})],
        publish=True,
    )

    return [maprc, mvprc, blgz]


def template_ts_mapd(
    price_col: str,
    short_w: int,
    long_w: int,
) -> list[FeatureSpec]:

    short_map = make_spec(
        name=prc_feature_name("mean", w=short_w, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": short_w},
        # post=[("log", {})],
        publish=False,
    )

    long_map = make_spec(
        name=prc_feature_name("mean", w=long_w, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": long_w},
        # post=[("log", {})],
        publish=False,
    )

    mapd = make_spec(
        name=prc_feature_name("diff", sw=short_w, lw=long_w, state="raw"),
        primitive="diff",
        inputs={"a": feat(short_map), "b": feat(long_map)},
        publish=True,
    )

    return [short_map, long_map, mapd]


def template_ts_mapdrtp(
    price_col: str,
    short_w: int,
    long_w: int,
) -> list[FeatureSpec]:

    short_map = make_spec(
        name=prc_feature_name("mean", w=short_w, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": short_w},
        post=None,
        publish=False,
    )

    long_map = make_spec(
        name=prc_feature_name("mean", w=long_w, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": long_w},
        post=None,
        publish=False,
    )

    mapd = make_spec(
        name=prc_feature_name("diff", sw=short_w, lw=long_w, state="raw"),
        primitive="diff",
        inputs={"a": feat(short_map), "b": feat(long_map)},
        publish=False,
    )

    mapdrtp = make_spec(
        name=prc_feature_name("r", sw=short_w, lw=long_w, state="raw"),
        primitive="ratio",
        inputs={"a": feat(mapd), "b": feat(long_map)},
        params={"eps": 1e-12},
        publish=True,
    )

    return [short_map, long_map, mapd, mapdrtp]


def template_ts_mapdrtatr(
    price_col: str,
    high_col: str,
    low_col: str,
    short_w: int,
    long_w: int
) -> list[FeatureSpec]:

    short_map = make_spec(
        name=prc_feature_name("mean", w=short_w, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": short_w},
        post=None,
        publish=False,
    )

    long_map = make_spec(
        name=prc_feature_name("mean", w=long_w, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": long_w},
        post=None,
        publish=False,
    )

    mapd = make_spec(
        name=prc_feature_name("diff", sw=short_w, lw=long_w, state="raw"),
        primitive="diff",
        inputs={"a": feat(short_map), "b": feat(long_map)},
        publish=False,
    )

    tr = make_spec(
        name=make_feature_name(
            domain="px",
            family="tr",
            signal="diff",
            params={},
            state="raw",
        ),
        primitive="tr",
        inputs={
            "high": col(high_col),
            "low": col(low_col),
            "close": col(price_col),
        },
        publish=False,
    )

    atr = make_spec(
        name=make_feature_name(
            domain="px",
            family="tr",
            signal="mean",
            params={"w": long_w},
            state="raw",
        ),
        primitive="ts_mean",
        inputs={"x": feat(tr)},
        params={"window": long_w},
        publish=False,
    )

    mapdrtatr = make_spec(
        name=prc_feature_name("tr-r", sw=short_w, lw=long_w, state="raw"),
        primitive="ratio",
        inputs={"a": feat(mapd), "b": feat(atr)},
        params={"eps": 1e-12},
        publish=True,
    )

    return [short_map, long_map, mapd, tr, atr, mapdrtatr]


def template_ts_mapdrtmvr(
    price_col: str,
    short_w: int,
    long_w: int
) -> list[FeatureSpec]:

    short_map = make_spec(
        name=prc_feature_name("mean", w=short_w, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": short_w},
        # post=[("log", {})],
        publish=False,
    )

    long_map = make_spec(
        name=prc_feature_name("mean", w=long_w, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": long_w},
        # post=[("log", {})],
        publish=False,
    )

    mapd = make_spec(
        name=prc_feature_name("diff", sw=short_w, lw=long_w, state="log"),
        primitive="diff",
        inputs={"a": feat(short_map), "b": feat(long_map)},
        post=[("log", {})],
        publish=False,
    )

    ret = make_spec(
        name=make_feature_name(
            domain="px",
            family="ret",
            signal="logret",
            params={"lb": 1},
            state="raw",
        ),
        primitive="ts_return",
        inputs={"price": col(price_col)},
        params={"lookback": 1},
        publish=False,
    )

    mvr = make_spec(
        name=make_feature_name(
            domain="px",
            family="ret",
            signal="std",
            params={"lb": 1, "w": long_w},
            state="raw",
        ),
        primitive="ts_std",
        inputs={"x": feat(ret)},
        params={"window": long_w},
        publish=False,
    )

    mapdrtmvr = make_spec(
        name=prc_feature_name("mvr-r", sw=short_w, lw=long_w, state="raw"),
        primitive="ratio",
        inputs={"a": feat(mapd), "b": feat(mvr)},
        params={"eps": 1e-12},
        publish=True,
    )

    return [short_map, long_map, mapd, ret, mvr, mapdrtmvr]


def template_ts_dma(
    price_col: str,
    window: int,
) -> list[FeatureSpec]:

    maprc = make_spec(
        name=prc_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": window},
        post=None,
        publish=False,
    )

    mapd = make_spec(
        name=prc_feature_name("diff", w=window, state="raw"),
        primitive="diff",
        inputs={"a": col(price_col), "b": feat(maprc)},
        publish=False,
    )

    dma = make_spec(
        name=prc_feature_name("dma", w=window, state="raw"),
        primitive="ratio",
        inputs={"a": feat(mapd), "b": feat(maprc)},
        params={"eps": 1e-12},
        publish=True,
    )

    return [maprc, mapd, dma]


def template_ts_vdma(
    price_col: str,
    window: int,
) -> list[FeatureSpec]:

    dma_template = template_ts_dma(price_col, window)
    dma = replace(dma_template[-1], publish=False)

    ret = make_spec(
        name=make_feature_name(
            domain="px",
            family="ret",
            signal="logret",
            params={"lb": 1},
            state="raw",
        ),
        primitive="ts_return",
        inputs={"price": col(price_col)},
        params={"lookback": 1},
        publish=False,
    )

    rv = make_spec(
        name=make_feature_name(
            domain="px",
            family="ret",
            signal="std",
            params={"lb": 1, "w": window},
            state="annual",
        ),
        primitive="ts_std",
        inputs={"x": feat(ret)},
        params={"window": window},
        post=[("annualize_vol", {"w": 1})],
        publish=False,
    )

    vdma = make_spec(
        name=prc_feature_name("vdma", w=window, state="raw"),
        primitive="ratio",
        inputs={"a": feat(dma), "b": feat(rv)},
        params={"eps": 1e-12},
        publish=True,
    )

    return dma_template[:-1] + [dma, ret, rv, vdma]


def template_ts_maslope(
    price_col: str,
    window: int,
    period: int,
) -> list[FeatureSpec]:

    maprc = make_spec(
        name=prc_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": window},
        publish=False,
    )

    slope = make_spec(
        name=prc_feature_name("maslope", w=window, lb=period, state="raw"),
        primitive="ts_slope",
        inputs={"x": feat(maprc)},
        params={"lookback": period},
        publish=True,
    )

    return [maprc, slope]


def template_ts_mapctslope(
    price_col: str,
    window: int,
    period: int,
) -> list[FeatureSpec]:

    maprc = make_spec(
        name=prc_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": window},
        publish=False,
    )

    slope = make_spec(
        name=prc_feature_name("maslope", w=window, lb=period, state="pct"),
        primitive="ts_pctslope",
        inputs={"x": feat(maprc)},
        params={"lookback": period},
        publish=True,
    )

    return [maprc, slope]


def template_ts_malogslope(
    price_col: str,
    window: int,
    period: int,
) -> list[FeatureSpec]:

    maprc = make_spec(
        name=prc_feature_name("mean", w=window, state="raw"),
        primitive="ts_mean",
        inputs={"x": col(price_col)},
        params={"window": window},
        publish=False,
    )

    slope = make_spec(
        name=prc_feature_name("maslope", w=window, lb=period, state="log"),
        primitive="ts_logslope",
        inputs={"x": feat(maprc)},
        params={"lookback": period},
        publish=False,
    )

    rv_specs = template_ts_mvr(price_col, lookback=1, window=period)
    rv = replace(rv_specs[-1], publish=False)

    nslope = make_spec(
        name=prc_feature_name("maslope", w=window, lb=period, state="log-rv"),
        primitive="ratio",
        inputs={"a": feat(slope), "b": feat(rv)},
        params={"eps": 1e-12},
        publish=True,
    )

    return rv_specs[:-1] + [rv, maprc, slope, nslope]


RAWPRICE_FAMILY = {
    "MOVING_AVERAGE_PRICE": FeatureTemplate(
        name="ts_map",
        domain=DOMAIN,
        family=FAMILY,
        signal="mean",
        template_fn=template_ts_map,
        description="moving average of log price",
        tags=("price", "average", "timeseries"),
        unitless=False,
    ),
    "MOVING_VOLATILITY_PRICE": FeatureTemplate(
        name="ts_mvp",
        domain=DOMAIN,
        family=FAMILY,
        signal="std",
        template_fn=template_ts_mvp,
        description="moving standard deviation of log price",
        tags=("price", "std", "volatility", "timeseries"),
        unitless=False,
    ),
    "BOLLINGER_ZSCORE": FeatureTemplate(
        name="ts_blgz",
        domain=DOMAIN,
        family=FAMILY,
        signal="z",
        template_fn=template_ts_blgz,
        description="zscore of close price in moving window",
        tags=("price", "zscore", "timeseries"),
    ),
    "MOVING_AVERAGE_PRICE_DIFF": FeatureTemplate(
        name="ts_mapd",
        domain=DOMAIN,
        family=FAMILY,
        signal="diff",
        template_fn=template_ts_mapd,
        description="log difference between average prices of short and long term moving window",
        tags=("price", "average", "diff", "timeseries"),
    ),
    "MOVING_AVERAGE_PRICE_DIFF_TO_LONG_RATIO": FeatureTemplate(
        name="ts_mapdrtp",
        domain=DOMAIN,
        family=FAMILY,
        signal="r",
        template_fn=template_ts_mapdrtp,
        description="short-long average price difference to long term average price ratio",
        tags=("price", "average", "diff", "ratio", "timeseries"),
    ),
    "MOVING_AVERAGE_PRICE_DIFF_TO_ATR_RATIO": FeatureTemplate(
        name="ts_mapdrtatr",
        domain=DOMAIN,
        family=FAMILY,
        signal="tr-r",
        template_fn=template_ts_mapdrtatr,
        description="short-long average price difference to long term average true range ratio",
        tags=("price", "average", "diff", "ratio", "true range", "timeseries"),
    ),
    "MOVING_AVERAGE_PRICE_DIFF_TO_MVR_RATIO": FeatureTemplate(
        name="ts_mapdrtmvr",
        domain=DOMAIN,
        family=FAMILY,
        signal="mvr-r",
        template_fn=template_ts_mapdrtmvr,
        description="short-long average price difference to long term return volatility ratio",
        tags=("price", "average", "diff", "ratio", "volatility", "timeseries"),
    ),
    "DISTANCE_FROM_MOVING_AVERAGE_PRICE": FeatureTemplate(
        name="ts_dma",
        domain=DOMAIN,
        family=FAMILY,
        signal="dma",
        template_fn=template_ts_dma,
        description="price distance between current and moving average normalized by moving average",
        tags=("price", "average", "diff", "ratio", "timeseries"),
    ),
    "DISTANCE_FROM_MOVING_AVERAGE_PRICE_VOLATILITY_NORMALIZED": FeatureTemplate(
        name="ts_vdma",
        domain=DOMAIN,
        family=FAMILY,
        signal="vdma",
        template_fn=template_ts_vdma,
        description="price distance between current and moving average normalized by volatility",
        tags=("price", "average", "diff", "ratio", "timeseries", "volatility"),
    ),
    "MOVING_AVERAGE_SLOPE": FeatureTemplate(
        name="ts_maslope",
        domain=DOMAIN,
        family=FAMILY,
        signal="maslope",
        template_fn=template_ts_maslope,
        description="slope of moving average price over a lookback period",
        tags=("price", "average", "slope", "trend", "timeseries"),
        unitless=False,
    ),
    "MOVING_AVERAGE_PCT_SLOPE": FeatureTemplate(
        name="ts_mapctslope",
        domain=DOMAIN,
        family=FAMILY,
        signal="maslope",
        template_fn=template_ts_mapctslope,
        description="percent slope of moving average price over a lookback period",
        tags=("price", "average", "slope", "trend", "timeseries"),
    ),
    "MOVING_AVERAGE_LOG_SLOPE_NORMALIZED_BY_RV": FeatureTemplate(
        name="ts_malogslope",
        domain=DOMAIN,
        family=FAMILY,
        signal="maslope",
        template_fn=template_ts_malogslope,
        description="rv normalized log slope of moving average price over a lookback period",
        tags=("price", "average", "slope", "trend", "timeseries", "rv"),
    ),
}