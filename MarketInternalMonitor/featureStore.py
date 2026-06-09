from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Hashable, Optional, List, Tuple, Iterable, Union
import numpy as np
import pandas as pd
import inspect
import json


# ---------- Spec ----------
PostOp = Tuple[str, Dict[str, Any]]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    primitive: str
    inputs: Dict[str, Any]            # values can be: df col name, other spec name, literal, Series
    params: Optional[Dict[str, Any]] = None
    post: Optional[List[PostOp]] = None
    publish: bool = True


# ---------- Primitive helpers ----------
def _gb(df: pd.DataFrame, sym_col: str):
    return df.groupby(sym_col, sort=False)


def prim_ts_return(df, sym_col, price: pd.Series, lookback: int) -> pd.Series:
    g = price.groupby(df[sym_col], sort=False)
    out = np.log1p(g.pct_change(lookback, fill_method=None))
    out.name = None
    return out


def prim_ts_mean(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.rolling(window, min_periods=min_periods).mean())
    out.name = None
    return out


def prim_ts_std(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.rolling(window, min_periods=min_periods).std())
    out.name = None
    return out


def prim_ts_max(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.rolling(window, min_periods=min_periods).max())
    out.name = None
    return out


def prim_ts_min(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.rolling(window, min_periods=min_periods).min())
    out.name = None
    return out


def prim_ts_ewm_mean(df, sym_col, x: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = window
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.ewm(alpha=1/window, adjust=False, min_periods=min_periods).mean())
    out.name = None
    return out


def prim_ts_ewm_span_mean(df, sym_col, x: pd.Series, span: int, min_periods: Optional[int] = None) -> pd.Series:
    if min_periods is None:
        min_periods = span
    g = x.groupby(df[sym_col], sort=False)
    out = g.transform(lambda s: s.ewm(span=span, adjust=False, min_periods=min_periods).mean())
    out.name = None
    return out


def prim_ts_diff(df, sym_col, x: pd.Series, lookback: int) -> pd.Series:
    g = x.groupby(df[sym_col], sort=False)
    out = g.diff(lookback)
    out.name = None
    return out


def prim_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return a - b


def prim_rdiff(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return (a / (b + eps)) - 1


def prim_zscore(x: pd.Series, mu: pd.Series, sigma: pd.Series, eps: float = 1e-12) -> pd.Series:
    return (x - mu) / (sigma + eps)


def prim_ratio(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return a / (b + eps) 


def prim_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(np.maximum(x, eps))


def prim_scale(x: pd.Series, scaler: float) -> pd.Series:
    return x * scaler


def prim_tr(df, sym_col, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.groupby(df[sym_col], sort=False).shift(1)
    hl = (high - low).to_numpy()
    hc = (high - prev_close).abs().to_numpy()
    lc = (low - prev_close).abs().to_numpy()
    out = np.maximum.reduce([hl, hc, lc])
    out = pd.Series(out, index=df.index)
    out.name = None
    return out


def post_clip(x: pd.Series, lo: Optional[float] = None, hi: Optional[float] = None) -> pd.Series:
    return x.clip(lo, hi)


def post_log(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    return np.log(np.maximum(x, eps))


def post_index_scale(x: pd.Series, offset: float = 100.0) -> pd.Series:
    return offset - (offset / (1 + x))


def post_scale(x: pd.Series, scaler: float) -> pd.Series:
    return x * scaler


def post_cs_rank(df: pd.DataFrame, date_col: str, x: pd.Series) -> pd.Series:
    return x.groupby(df[date_col], sort=False).rank(pct=True)


def post_cs_zscore(df: pd.DataFrame, date_col: str, x: pd.Series, eps: float = 1e-12) -> pd.Series:
    g = x.groupby(df[date_col], sort=False)
    mu = g.transform("mean")
    sd = g.transform("std")
    return (x - mu) / (sd + eps)


PRIMITIVES: Dict[str, Callable[..., pd.Series]] = {
    "ts_return": prim_ts_return,
    "ts_mean": prim_ts_mean,
    "ts_std": prim_ts_std,
    "ts_max": prim_ts_max,
    "ts_min": prim_ts_min,
    "ts_ewm_mean": prim_ts_ewm_mean,
    "ts_ewm_span_mean": prim_ts_ewm_span_mean,
    "ts_diff": prim_ts_diff,
    "diff": prim_diff,
    "rdiff": prim_rdiff,
    "scale": prim_scale,
    "zscore": prim_zscore,
    "ratio": prim_ratio,
    "log": prim_log,
    "tr": prim_tr,
}

POSTS: Dict[str, Callable[..., pd.Series]] = {
    "clip": post_clip,
    "log": post_log,
    "index_scale": post_index_scale,
    "scale": post_scale,
    "cs_rank": post_cs_rank,
    "cs_zscore": post_cs_zscore,
}


# ---------- Function call helpers ----------
def prepare_call(fn: Callable, available_inputs: dict) -> Tuple[Union[dict, list], int]:
    fn_sig = inspect.signature(fn)
    fn_params = fn_sig.parameters

    call_kwargs = {}
    missing_required = []

    for name, p in fn_params.items():
        if name in available_inputs:
            call_kwargs[name] = available_inputs[name]
        else:
            required = (
                p.default is inspect._empty
                and p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY
                )
            )
            if required:
                missing_required.append(name)

    if len(missing_required) > 0:
        return missing_required, 300

    return call_kwargs, 200


def normalize_spec(spec: FeatureSpec) -> FeatureSpec:
    return FeatureSpec(
        name=spec.name,
        primitive=spec.primitive,
        inputs=spec.inputs or {},
        params=spec.params or {},
        post=spec.post or [],
        publish=spec.publish,
    )


def canonicalize_value(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: canonicalize_value(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [canonicalize_value(x) for x in v]
    if isinstance(v, pd.Series):
        return {
            "__type__": "Series",
            "name": v.name,
            "index_hash": hash(tuple(v.index)),
            "value_hash": hash(tuple(v.fillna("__nan__")))
        }
    return v


def spec_identity(spec: FeatureSpec) -> dict:
    spec = normalize_spec(spec)
    return {
        "primitive": canonicalize_value(spec.primitive),
        "inputs": canonicalize_value(spec.inputs),
        "params": canonicalize_value(spec.params),
        "post": canonicalize_value(spec.post),
    }


def validate_and_index_specs(specs: Iterable[FeatureSpec]) -> Dict[str, FeatureSpec]:
    by_name = defaultdict(list)
    for s in specs:
        by_name[s.name].append(s)

    spec_map = {}
    for name, group in by_name.items():
        base_identity = spec_identity(group[0])
        for s in group[1:]:
            if spec_identity(s) != base_identity:
                raise ValueError(
                    f"Inconsistent duplicate spec name: {name}\n"
                    f"First: {base_identity}\n"
                    f"Later: {spec_identity(s)}"
                )
        spec_map[name] = group[0]

    return spec_map


def build_spec_map(specs: Iterable[FeatureSpec]):
    by_name = {}
    for spec in map(normalize_spec, specs):
        ident = spec_identity(spec)
        if spec.name in by_name:
            prev_spec, prev_ident = by_name[spec.name]
            if ident != prev_ident:
                raise ValueError(
                    f"Conflicting definitions for spec '{spec.name}'"
                )
        else:
            by_name[spec.name] = (spec, ident)
    return {name: spec for name, (spec, _) in by_name.items()}


# ---------- Builder ----------
class FeatureBuilder:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        date_col: str = 'date',
        sym_col: str = 'symbol',
        eps: float = 1e-12,
    ):
        self.df = df
        self.date_col = date_col
        self.sym_col = sym_col
        self.eps = eps
        self._cache: Dict[Tuple, pd.Series] = {}

    def _resolve(self, v: Any, computed: Dict[str, pd.Series]) -> Any:
        # Series literal
        if isinstance(v, pd.Series):
            return v

        # reference string: either computed node or df column
        if isinstance(v, str):
            if v in computed:
                return computed[v]
            if v in self.df.columns:
                s = self.df[v].copy()
                s.name = v
                return s

        # literal (int/float/etc)
        return v

    def build(self, specs: Iterable[FeatureSpec]) -> Dict[str, pd.Series]:
        if not isinstance(specs, Iterable):
            specs = [specs]
        specs = list(specs)
        spec_map = validate_and_index_specs(specs)

        computed: Dict[str, pd.Series] = {}
        remaining = set(spec_map.keys())

        # iterate-until-done dependency resolution
        progressed = True
        while remaining and progressed:
            progressed = False

            for name in list(remaining):
                spec = spec_map[name]

                # resolve inputs; if any input references an unbuilt spec, defer
                resolved_inputs = {}
                blocked = False
                for k, v in (spec.inputs or {}).items():
                    if isinstance(v, str) and (v in spec_map) and (v not in computed):
                        blocked = True
                        break
                    resolved_inputs[k] = self._resolve(v, computed)

                if blocked:
                    continue

                # resolve params
                params = spec.params or {}

                # prepare available inputs
                # priority: builder context < provided params < provided inputs
                available_inputs = {
                    "df": self.df,
                    "sym_col": self.sym_col,
                    "date_col": self.date_col,
                    "eps": self.eps,
                    **params,
                    **resolved_inputs,
                }

                # grab feature if cached, if not build feature then cache
                cache_key = (spec.primitive, spec.name, tuple(sorted(params.items())))
                if cache_key in self._cache:
                    out = self._cache[cache_key]
                else:
                    prim = PRIMITIVES[spec.primitive]

                    # prepare call_kwargs
                    call_kwargs, status = prepare_call(
                        fn=prim,
                        available_inputs=available_inputs
                    )
                    if status == 300:
                        raise ValueError(
                            f"While building feature '{spec.name}', primitive '{spec.primitive}' "
                            f"is missing required inputs {call_kwargs}. "
                            f"Available keys: {sorted(available_inputs.keys())}"
                        )

                    # pass eps only if primitive accepts it
                    # prim_fn_params = inspect.signature(prim).parameters
                    # if "eps" in prim_fn_params and "eps" not in params:
                    #     params = dict(params)
                    #     params["eps"] = self.eps

                    # execute primitive function
                    out = prim(**call_kwargs)
                    if not isinstance(out, pd.Series):
                        out = pd.Series(out, index=self.df.index)
                    out.name = spec.name
                    self._cache[cache_key] = out

                # post transforms
                if spec.post:
                    for post_name, post_params in spec.post:
                        fn = POSTS[post_name]
                        pp = dict(post_params)

                        post_available_inputs = {
                            "df": self.df,
                            "sym_col": self.sym_col,
                            "date_col": self.date_col,
                            "eps": self.eps,
                            "x": out,
                            **pp,
                        }

                        post_call_kwargs, post_status = prepare_call(
                            fn=fn,
                            available_inputs=post_available_inputs
                        )
                        if post_status == 300:
                            raise ValueError(
                                f"While building feature '{spec.name}', post transform '{post_name}' "
                                f"is missing required inputs {post_call_kwargs}. "
                                f"Available keys: {sorted(post_available_inputs.keys())}"
                            )
                        out = fn(**post_call_kwargs)
                    out.name = spec.name

                computed[name] = out
                remaining.remove(name)
                progressed = True

        if remaining:
            raise ValueError(f"Unresolved specs (cycles or missing deps): {sorted(remaining)}")

        return computed

    def build_published(self, specs: Iterable[FeatureSpec]) -> Dict[str, pd.Series]:
        if not isinstance(specs, Iterable):
            specs = [specs]
        computed = self.build(specs)
        return {s.name: computed[s.name] for s in specs if s.publish}


# ---------- Template ----------
@dataclass
class FeatureTemplate:
    name: str
    domain: str
    family: str
    signal: Optional[str]
    template_fn: Callable[..., list]
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    unitless: bool = True

    def __call__(self, *args, **kwargs):
        return self.template_fn(*args, **kwargs)

    @property
    def full_name(self) -> str:
        parts = [self.domain, self.family]
        if self.signal:
            parts.append(self.signal)
        return "__".join(parts)

    @property
    def signature(self):
        return inspect.signature(self.template_fn)

    @property
    def param_names(self) -> list[str]:
        return list(self.signature.parameters.keys())


# ---------- Return Family ----------
def template_ts_return(price_col: str, lookback: int) -> list[FeatureSpec]:
    return [
        FeatureSpec(
            name=f"px__ret__logret__lb{lookback}__raw",
            primitive="ts_return",
            inputs={"price": price_col},
            params={"lookback": lookback},
            publish=True,
        )
    ]


def template_ts_mar(
    price_col: str,
    lookback: int,
    window: int
) -> list[FeatureSpec]:
    ret = f"px__ret__logret__lb{lookback}__raw"
    mar = f"px__ret__mean__lb{lookback}_w{window}__raw"

    return [
        FeatureSpec(
            name=ret,
            primitive="ts_return",
            inputs={"price": price_col},
            params={"lookback": lookback},
            publish=True,
        ),
        FeatureSpec(
            name=mar,
            primitive="ts_mean",
            inputs={"x": ret},
            params={"window": window},
            publish=True,
        )
    ]


def template_ts_mvr(
    price_col: str,
    lookback: int,
    window: int
) -> list[FeatureSpec]:
    ret = f"px__ret__logret__lb{lookback}__raw"
    mvr = f"px__ret__std__lb{lookback}_w{window}__raw"

    return [
        FeatureSpec(
            name=ret,
            primitive="ts_return",
            inputs={"price": price_col},
            params={"lookback": lookback},
            publish=True,
        ),
        FeatureSpec(
            name=mvr,
            primitive="ts_std",
            inputs={"x": ret},
            params={"window": window},
            publish=True,
        )
    ]


def template_ts_mrz(
    price_col: str,
    lookback: int,
    window: int
) -> list[FeatureSpec]:
    ret = f"px__ret__logret__lb{lookback}__raw"
    mu = f"px__ret__mean__lb{lookback}_w{window}__raw"
    sd = f"px__ret__std__lb{lookback}_w{window}__raw"
    mrz = f"px__ret__z__lb{lookback}_w{window}__clip"

    return [
        FeatureSpec(
            name=ret,
            primitive="ts_return",
            inputs={"price": price_col},
            params={"lookback": lookback},
            publish=True,
        ),
        FeatureSpec(
            name=mu,
            primitive="ts_mean",
            inputs={"x": ret},
            params={"window": window},
            publish=True,
        ),
        FeatureSpec(
            name=sd,
            primitive="ts_std",
            inputs={"x": ret},
            params={"window": window},
            publish=True,
        ),
        FeatureSpec(
            name=mrz,
            primitive="zscore",
            inputs={"x": ret, "mu": mu, "sigma": sd},
            params={"eps": 1e-12},
            post=[('clip', {'lo': -5.0, 'hi': 5.0})],
            publish=True,
        ),
    ]


def template_ts_mrr(
    price_col: str,
    lookback: int,
    window: int
) -> list[FeatureSpec]:
    ret = f"px__ret__logret__lb{lookback}__raw"
    mu = f"px__ret__mean__lb{lookback}_w{window}__raw"
    mrr = f"px__ret__r__lb{lookback}_w{window}__raw"

    return [
        FeatureSpec(
            name=ret,
            primitive="ts_return",
            inputs={"price": price_col},
            params={"lookback": lookback},
            publish=True,
        ),
        FeatureSpec(
            name=mu,
            primitive="ts_mean",
            inputs={"x": ret},
            params={"window": window},
            publish=True,
        ),
        FeatureSpec(
            name=mrr,
            primitive="ratio",
            inputs={"a": ret, "b": mu},
            params={"eps": 1e-12},
            publish=True,
        ),
    ]


RETURN_FAMILY = {
    'RETURN': FeatureTemplate(
        name="ts_return",
        domain="px",
        family="ret",
        signal="logret",
        template_fn=template_ts_return,
        description="Time-series log return from price over given lookback window.",
        tags=("price", "return", "timeseries"),
    ),
    'MOVING_AVERAGE_RETURN': FeatureTemplate(
        name="ts_mar",
        domain="px",
        family="ret",
        signal="mean",
        template_fn=template_ts_mar,
        description="moving average of log return",
        tags=("price", "return", "average", "timeseries"),
    ),
    'MOVING_VOLATILITY_RETURN': FeatureTemplate(
        name="ts_mvr",
        domain="px",
        family="ret",
        signal="std",
        template_fn=template_ts_mvr,
        description="moving std of log return",
        tags=("price", "return", "std", "volatility", "timeseries"),
    ),
    'RETURN_ZSCORE': FeatureTemplate(
        name="ts_mzr",
        domain="px",
        family="ret",
        signal="z",
        template_fn=template_ts_mrz,
        description="zscore of return in moving window",
        tags=("price", "return", "zscore", "timeseries"),
    ),
    'RETURN_RATIO': FeatureTemplate(
        name="ts_mrr",
        domain="px",
        family="ret",
        signal="r",
        template_fn=template_ts_mrr,
        description="return to its average ratio in moving window",
        tags=("price", "return", "ratio", "timeseries"),
    ),
}


# ---------- Raw Price Family ----------
def template_ts_map(
    price_col: str,
    window: int
) -> list[FeatureSpec]:
    maprice = f"px__prc__mean__w{window}__raw"

    return [
        FeatureSpec(
            name=maprice,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": window},
            publish=False,
        )
    ]


def template_ts_mvp(
    price_col: str,
    window: int
) -> list[FeatureSpec]:
    mvprice = f"px__prc__std__w{window}__raw"

    return [
        FeatureSpec(
            name=mvprice,
            primitive="ts_std",
            inputs={"x": price_col},
            params={"window": window},
            publish=False,
        )
    ]


def template_ts_blgz(
    price_col: str,
    window: int
) -> list[FeatureSpec]:
    maprice = f"px__prc__mean__w{window}__raw"
    mvprice = f"px__prc__std__w{window}__raw"
    blgz = f"px__prc__z__w{window}__clip"

    return [
        FeatureSpec(
            name=maprice,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": window},
            publish=False,
        ),
        FeatureSpec(
            name=mvprice,
            primitive="ts_std",
            inputs={"x": price_col},
            params={"window": window},
            publish=False,
        ),
        FeatureSpec(
            name=blgz,
            primitive="zscore",
            inputs={"x": price_col, "mu": maprice, "sigma": mvprice},
            params={"eps": 1e-12},
            post=[('clip', {'lo': -5.0, 'hi': 5.0})],
            publish=True,
        )
    ]


def template_ts_mapd(
    price_col: str,
    short_w: int,
    long_w: int,
) -> list[FeatureSpec]:
    short_map = f"px__prc__mean__w{short_w}__log"
    long_map = f"px__prc__mean__w{long_w}__log"
    mapd = f"px__prc__diff__w{short_w}_w{long_w}__log"

    return [
        FeatureSpec(
            name=short_map,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": short_w},
            post=[("log", {})],
            publish=False,
        ),
        FeatureSpec(
            name=long_map,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": long_w},
            post=[("log", {})],
            publish=False,
        ),
        FeatureSpec(
            name=mapd,
            primitive="diff",
            inputs={"a": short_map, "b": long_map},
            publish=True,
        )
    ]


def template_ts_mapdrtp(
    price_col: str,
    short_w: int,
    long_w: int,
) -> list[FeatureSpec]:
    short_map = f"px__prc__mean__w{short_w}__raw"
    long_map = f"px__prc__mean__w{long_w}__raw"
    mapd = f"px__prc__diff__w{short_w}_w{long_w}__raw"
    mapdrtp = f"px__prc__r__w{short_w}_w{long_w}__raw"

    return [
        FeatureSpec(
            name=short_map,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": short_w},
            publish=False,
        ),
        FeatureSpec(
            name=long_map,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": long_w},
            publish=False,
        ),
        FeatureSpec(
            name=mapd,
            primitive="diff",
            inputs={"a": short_map, "b": long_map},
            publish=False,
        ),
        FeatureSpec(
            name=mapdrtp,
            primitive="ratio",
            inputs={"a": mapd, "b": long_map},
            params={"eps": 1e-12},
            publish=True,
        )
    ]


def template_ts_mapdrtatr(
    price_col: str,
    high_col: str,
    low_col: str,
    short_w: int,
    long_w: int
) -> list[FeatureSpec]:
    short_map = f"px__prc__mean__w{short_w}__raw"
    long_map = f"px__prc__mean__w{long_w}__raw"
    mapd = f"px__prc__diff__w{short_w}_w{long_w}__raw"
    tr = "px__tr__diff__none__raw"
    atr = f"px__tr__mean__w{long_w}__raw"
    mapdrtatr = f"px__prc_tr__r__w{short_w}_w{long_w}__raw"

    return [
        FeatureSpec(
            name=short_map,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": short_w},
            publish=False,
        ),
        FeatureSpec(
            name=long_map,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": long_w},
            publish=False,
        ),
        FeatureSpec(
            name=mapd,
            primitive="diff",
            inputs={"a": short_map, "b": long_map},
            publish=False,
        ),
        FeatureSpec(
            name=tr,
            primitive="tr",
            inputs={"high": high_col, "low": low_col, "close": price_col},
            publish=False,
        ),
        FeatureSpec(
            name=atr,
            primitive="ts_mean",
            inputs={"x": tr},
            params={"window": long_w},
            publish=False,
        ),
        FeatureSpec(
            name=mapdrtatr,
            primitive="ratio",
            inputs={"a": mapd, "b": atr},
            params={"eps": 1e-12},
            publish=True,
        )
    ]


def template_ts_mapdrtmvr(
    price_col: str,
    short_w: int,
    long_w: int
) -> list[FeatureSpec]:
    short_map = f"px__prc__mean__w{short_w}__log"
    long_map = f"px__prc__mean__w{long_w}__log"
    mapd = f"px__prc__diff__w{short_w}_w{long_w}__log"
    ret = "px__ret__logret__lb1__raw"
    mvr = f"px__ret__std__lb1_w{long_w}__raw"
    mapdrtmvr = f"px__prc_ret__r__w{short_w}_w{long_w}__log"

    return [
        FeatureSpec(
            name=short_map,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": short_w},
            post=[("log", {})],
            publish=False,
        ),
        FeatureSpec(
            name=long_map,
            primitive="ts_mean",
            inputs={"x": price_col},
            params={"window": long_w},
            post=[("log", {})],
            publish=False,
        ),
        FeatureSpec(
            name=mapd,
            primitive="diff",
            inputs={"a": short_map, "b": long_map},
            publish=False,
        ),
        FeatureSpec(
            name=ret,
            primitive="ts_return",
            inputs={"price": price_col},
            params={"lookback": 1},
            publish=True,
        ),
        FeatureSpec(
            name=mvr,
            primitive="ts_std",
            inputs={"x": ret},
            params={"window": long_w},
            publish=True,
        ),
        FeatureSpec(
            name=mapdrtmvr,
            primitive="ratio",
            inputs={"a": mapd, "b": mvr},
            params={"eps": 1e-12},
            publish=True,
        )
    ]


RAWPRICE_FAMILY = {
    # 'MOVING_AVERAGE_PRICE': FeatureTemplate(
    #     name="ts_map",
    #     domain="px",
    #     family="prc",
    #     signal="mean",
    #     template_fn=template_ts_map,
    #     description="",
    #     tags=("price", "average", "timeseries"),
    #     unitless=False,
    # ),
    # 'MOVING_VOLATILITY_PRICE': FeatureTemplate(
    #     name="ts_mvp",
    #     domain="px",
    #     family="prc",
    #     signal="std",
    #     template_fn=template_ts_mvp,
    #     description="",
    #     tags=("price", "std", "volatility", "timeseries"),
    #     unitless=False,
    # ),
    'BOLLINGER_ZSCORE': FeatureTemplate(
        name="ts_blgz",
        domain="px",
        family="prc",
        signal="z",
        template_fn=template_ts_blgz,
        description="zscore of close price in moving window",
        tags=("price", "zscore", "timeseries"),
    ),
    'MOVING_AVERAGE_PRICE_DIFF': FeatureTemplate(
        name="ts_mapd",
        domain="px",
        family="prc",
        signal="diff",
        template_fn=template_ts_mapd,
        description="log difference between average prices of short and long term moving window",
        tags=("price", "average", "diff", "timeseries"),
    ),
    'MOVING_AVERAGE_PRICE_DIFF_TO_LONG_RATIO': FeatureTemplate(
        name="ts_mapdrtp",
        domain="px",
        family="prc",
        signal="r",
        template_fn=template_ts_mapdrtp,
        description="short-long average price difference to long term average price ratio",
        tags=("price", "average", "diff", "ratio", "timeseries"),
    ),
    'MOVING_AVERAGE_PRICE_DIFF_TO_ATR_RATIO': FeatureTemplate(
        name="ts_mapdrtatr",
        domain="px",
        family="prc_tr",
        signal="r",
        template_fn=template_ts_mapdrtatr,
        description="short-long average price difference to long term average true range ratio",
        tags=("price", "average", "diff", "ratio", "true range", "timeseries"),
    ),
    'MOVING_AVERAGE_PRICE_DIFF_TO_MVR_RATIO': FeatureTemplate(
        name="ts_mapdrtmvr",
        domain="px",
        family="prc_ret",
        signal="r",
        template_fn=template_ts_mapdrtmvr,
        description="short-long average price difference to long term return volatility ratio",
        tags=("price", "average", "diff", "ratio", "volatility", "timeseries"),
    ),
}


# ---------- True Range Family ----------
def template_tr(
    high_col: str,
    low_col: str,
    close_col: str
) -> list[FeatureSpec]:
    tr = "px__tr__diff__none__raw"

    return [
        FeatureSpec(
            name=tr,
            primitive="tr",
            inputs={"high": high_col, "low": low_col, "close": close_col},
            publish=False,
        )
    ]


def template_ts_atr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:
    tr = "px__tr__diff__none__raw"
    atr = f"px__tr__mean__w{window}__raw"

    return [
        FeatureSpec(
            name=tr,
            primitive="tr",
            inputs={"high": high_col, "low": low_col, "close": close_col},
            publish=False,
        ),
        FeatureSpec(
            name=atr,
            primitive="ts_mean",
            inputs={"x": tr},
            params={"window": window},
            publish=False,
        ),
    ]


def template_ts_vtr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:
    tr = "px__tr__diff__none__raw"
    vtr = f"px__tr__std__w{window}__raw"

    return [
        FeatureSpec(
            name=tr,
            primitive="tr",
            inputs={"high": high_col, "low": low_col, "close": close_col},
            publish=False,
        ),
        FeatureSpec(
            name=vtr,
            primitive="ts_std",
            inputs={"x": tr},
            params={"window": window},
            publish=False,
        ),
    ]


def template_ts_natr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:
    tr = "px__tr__diff__none__raw"
    atr = f"px__tr__mean__w{window}__raw"
    maprice = f"px__prc__mean__w{window}__raw"
    natr = f"px__tr_prc__norm__w{window}__raw"

    return [
        FeatureSpec(
            name=tr,
            primitive="tr",
            inputs={"high": high_col, "low": low_col, "close": close_col},
            publish=False,
        ),
        FeatureSpec(
            name=atr,
            primitive="ts_mean",
            inputs={"x": tr},
            params={"window": window},
            publish=False,
        ),
        FeatureSpec(
            name=maprice,
            primitive="ts_mean",
            inputs={"x": close_col},
            params={"window": window},
            publish=False,
        ),
        FeatureSpec(
            name=natr,
            primitive="ratio",
            inputs={"a": atr, "b": maprice},
            params={"eps": 1e-12},
            publish=True,
        ),
    ]


def template_ts_zatr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:
    tr = "px__tr__diff__none__raw"
    atr = f"px__tr__mean__w{window}__raw"
    mu = f"px__tr__mean2__w{window}__raw"
    sd = f"px__tr__sd__w{window}__raw"
    zatr = f"px__tr__z__w{window}__clip"

    return [
        FeatureSpec(
            name=tr,
            primitive="tr",
            inputs={"high": high_col, "low": low_col, "close": close_col},
            publish=False,
        ),
        FeatureSpec(
            name=atr,
            primitive="ts_mean",
            inputs={"x": tr},
            params={"window": window},
            publish=False,
        ),
        FeatureSpec(
            name=mu,
            primitive="ts_mean",
            inputs={"x": atr},
            params={"window": window},
            publish=True,
        ),
        FeatureSpec(
            name=sd,
            primitive="ts_std",
            inputs={"x": atr},
            params={"window": window, "eps": 1e-12},
            publish=True,
        ),
        FeatureSpec(
            name=zatr,
            primitive="zscore",
            inputs={"x": atr, "mu": mu, "sigma": sd},
            params={"eps": 1e-12},
            post=[('clip', {'lo': -5.0, 'hi': 5.0})],
            publish=True,
        )
    ]


def template_ts_ratr(
    high_col: str,
    low_col: str,
    close_col: str,
    window: int
) -> list[FeatureSpec]:
    tr = "px__tr__diff__none__raw"
    atr = f"px__tr__mean__w{window}__raw"
    mu = f"px__tr__mean2__w{window}__raw"
    ratr = f"px__tr__r__w{window}__raw"

    return [
        FeatureSpec(
            name=tr,
            primitive="tr",
            inputs={"high": high_col, "low": low_col, "close": close_col},
            publish=False,
        ),
        FeatureSpec(
            name=atr,
            primitive="ts_mean",
            inputs={"x": tr},
            params={"window": window},
            publish=False,
        ),
        FeatureSpec(
            name=mu,
            primitive="ts_mean",
            inputs={"x": atr},
            params={"window": window},
            publish=False,
        ),
        FeatureSpec(
            name=ratr,
            primitive="ratio",
            inputs={"a": atr, "b": mu},
            params={"eps": 1e-12},
            publish=True,
        ),
    ]


ATR_FAMILY = {
    'NORMALIZED_AVERAGE_TRUERANGE': FeatureTemplate(
        name="ts_natr",
        domain="px",
        family="tr_prc",
        signal="norm",
        template_fn=template_ts_natr,
        description="normalized average true range in moving window",
        tags=("true range", "average", "normalized", "timeseries"),
    ),
    'ZSCORE_AVERAGE_TRUERANGE': FeatureTemplate(
        name="ts_zatr",
        domain="px",
        family="tr",
        signal="z",
        template_fn=template_ts_zatr,
        description="zscore of average true range in moving window",
        tags=("true range", "average", "zscore", "timeseries"),
    ),
    'RATIO_AVERAGE_TRUERANGE': FeatureTemplate(
        name="ts_ratr",
        domain="px",
        family="tr",
        signal="r",
        template_fn=template_ts_ratr,
        description="average true range to its average ratio in moving window",
        tags=("true range", "average", "ratio", "timeseries"),
    ),
}


# ---------- RSI Family ----------
def template_ts_rsi(
    price_col: str,
    window: int,
) -> list[FeatureSpec]:
    pdiff = "px__prc__tsdiff__lb1__raw"
    gain = "px__rsi__gain__s1__clip"
    loss = "px__rsi__loss__sn1__clip"
    avg_gain = f"px__rsi__gain_mean__w{window}__raw"
    avg_loss = f"px__rsi__loss_mean__w{window}__raw"
    rsi = f"px__rsi__r__w{window}__index"

    return [
        FeatureSpec(
            name=pdiff,
            primitive="ts_diff",
            inputs={"x": price_col},
            params={"lookback": 1},
            publish=False
        ),
        FeatureSpec(
            name=gain,
            primitive="scale",
            inputs={"x": pdiff},
            params={"scaler": 1},
            post=[("clip", {"lo": 0})],
            publish=False
        ),
        FeatureSpec(
            name=loss,
            primitive="scale",
            inputs={"x": pdiff},
            params={"scaler": -1},
            post=[("clip", {"lo": 0})],
            publish=False
        ),
        FeatureSpec(
            name=avg_gain,
            primitive="ts_ewm_mean",
            inputs={"x": gain},
            params={"window": window},
            publish=False
        ),
        FeatureSpec(
            name=avg_loss,
            primitive="ts_ewm_mean",
            inputs={"x": loss},
            params={"window": window},
            publish=False
        ),
        FeatureSpec(
            name=rsi,
            primitive="ratio",
            inputs={"a": avg_gain, "b": avg_loss},
            params={"eps": 1e-12},
            post=[("index_scale", {"offset": 100.0})],
            publish=True,
        ),
    ]


def template_ts_rsiz(
    price_col: str,
    window: int,
    period: int,
) -> list[FeatureSpec]:
    pdiff = "px__prc__tsdiff__lb1__raw"
    gain = "px__rsi__gain__s1__clip"
    loss = "px__rsi__loss__sn1__clip"
    avg_gain = f"px__rsi__gain_mean__w{window}__raw"
    avg_loss = f"px__rsi__loss_mean__w{window}__raw"
    rsi = f"px__rsi__r__w{window}__index"
    mu = f"px__rsi__mean__w{window}_p{period}__index"
    sd = f"px__rsi__std__w{window}_p{period}__index"
    rsiz = f"px__rsi__z__w{window}_p{period}__index"

    return [
        FeatureSpec(
            name=pdiff,
            primitive="ts_diff",
            inputs={"x": price_col},
            params={"lookback": 1},
            publish=False
        ),
        FeatureSpec(
            name=gain,
            primitive="scale",
            inputs={"x": pdiff},
            params={"scaler": 1},
            post=[("clip", {"lo": 0})],
            publish=False
        ),
        FeatureSpec(
            name=loss,
            primitive="scale",
            inputs={"x": pdiff},
            params={"scaler": -1},
            post=[("clip", {"lo": 0})],
            publish=False
        ),
        FeatureSpec(
            name=avg_gain,
            primitive="ts_ewm_mean",
            inputs={"x": gain},
            params={"window": window},
            publish=False
        ),
        FeatureSpec(
            name=avg_loss,
            primitive="ts_ewm_mean",
            inputs={"x": loss},
            params={"window": window},
            publish=False
        ),
        FeatureSpec(
            name=rsi,
            primitive="ratio",
            inputs={"a": avg_gain, "b": avg_loss},
            params={"eps": 1e-12},
            post=[("index_scale", {"offset": 100.0})],
            publish=True,
        ),
        FeatureSpec(
            name=mu,
            primitive="ts_mean",
            inputs={"x": rsi},
            params={"window": period},
            publish=True
        ),
        FeatureSpec(
            name=sd,
            primitive="ts_std",
            inputs={"x": rsi},
            params={"window": period},
            publish=True
        ),
        FeatureSpec(
            name=rsiz,
            primitive="zscore",
            inputs={"x": rsi, "mu": mu, "sigma": sd},
            params={"eps": 1e-12},
            post=[('clip', {'lo': -5.0, 'hi': 5.0})],
            publish=True,
        )
    ]


RSI_FAMILY = {
    'RSI': FeatureTemplate(
        name="ts_rsi",
        domain="px",
        family="rsi",
        signal="r",
        template_fn=template_ts_rsi,
        description="relative strength index",
        tags=("price", "rsi", "index", "timeseries"),
    ),
    'RSI_ZSCORE': FeatureTemplate(
        name="ts_rsiz",
        domain="px",
        family="rsi",
        signal="z",
        template_fn=template_ts_rsiz,
        description="zscore of relative strength index",
        tags=("price", "rsi", "index", "zscore", "timeseries"),
    ),
}


# ---------- Draw Down Family ----------
def template_ts_mdd(
    price_col: str,
    window: int
) -> list[FeatureSpec]:
    peak = f"px__mdd__max__w{window}__raw"
    dd = f"px__mdd__rdiff__w{window}__raw"
    mdd = f"px__mdd__min__w{window}__scale"

    return [
        FeatureSpec(
            name=peak,
            primitive="ts_max",
            inputs={"x": price_col},
            params={"window": window},
            publish=False
        ),
        FeatureSpec(
            name=dd,
            primitive="rdiff",
            inputs={"a": price_col, "b": peak},
            params={"eps": 1e-12},
            publish=False
        ),
        FeatureSpec(
            name=mdd,
            primitive="ts_min",
            inputs={"x": dd},
            params={"window": window},
            post=[("scale", {"scaler": -1})],
            publish=True
        ),
    ]


def template_ts_mddatrnorm(
    price_col: str,
    high_col: str,
    low_col: str,
    window: int
) -> list[FeatureSpec]:
    _mdd = template_ts_mdd(price_col, window)
    _atr = template_ts_atr(high_col, low_col, price_col, 14)

    mdd = _mdd[-1].name
    atr = _atr[-1].name
    atr_base = "px__tr__mean2__w14_p252__raw"
    mddatrnorm = f"px__mdd__atrnorm__w{window}__raw"

    return _mdd + _atr + [
        FeatureSpec(
            name=atr_base,
            primitive="ts_ewm_span_mean",
            inputs={"x": atr},
            params={"span": 252},
            publish=False
        ),
        FeatureSpec(
            name=mddatrnorm,
            primitive="ratio",
            inputs={"a": mdd, "b": atr_base},
            params={"eps": 1e-12},
            publish=True
        ),
    ]


MDD_FAMILY = {
    'MDD': FeatureTemplate(
        name="ts_mdd",
        domain="px",
        family="mdd",
        signal="max",
        template_fn=template_ts_mdd,
        description="maximum drawdown in moving window",
        tags=("price", "mdd", "max", "timeseries"),
    ),
    'MDD_ATR_NORM': FeatureTemplate(
        name="ts_mddatrnorm",
        domain="px",
        family="mdd",
        signal="atrnorm",
        template_fn=template_ts_mddatrnorm,
        description="maximum drawdown in moving window normalized by average true range baseline",
        tags=("price", "mdd", "norm", "atr", "timeseries"),
    ),
}