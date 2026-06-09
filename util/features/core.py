from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, List, Tuple, Iterable, Union, Mapping
import hashlib
import inspect
import json

import numpy as np
import pandas as pd

from util.features.primitives import PRIMITIVES, POSTS


# ---------- Spec / naming ----------
PostOp = Tuple[str, Dict[str, Any]]


@dataclass(frozen=True)
class FeatureNameSpec:
    domain: str
    family: str
    signal: str
    params: Optional[Dict[str, Any]] = None
    state: str = "raw"


PARAM_RENDER_ORDER = ["lb", "w", "sw", "lw", "p", "sp", "s", "off", "zw"]


def _render_name_value(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if float(v).is_integer():
            return str(int(v))
        return str(v).replace("-", "n").replace(".", "p")
    return str(v).replace("-", "n").replace(".", "p").replace(" ", "")


def _render_name_params(params: Optional[Mapping[str, Any]]) -> str:
    params = params or {}
    if not params:
        return "none"
    ordered = [k for k in PARAM_RENDER_ORDER if k in params]
    extras = sorted(k for k in params if k not in PARAM_RENDER_ORDER)
    keys = ordered + extras
    return "_".join(f"{k}{_render_name_value(params[k])}" for k in keys)


def make_feature_name(
    domain: str,
    family: str,
    signal: str,
    params: Optional[Mapping[str, Any]] = None,
    state: str = "raw",
) -> str:
    return f"{domain}__{family}__{signal}__{_render_name_params(params)}__{state}"


# ---------- Explicit input references ----------
@dataclass(frozen=True)
class ColumnRef:
    name: str


@dataclass(frozen=True)
class FeatureRef:
    feature_id: str
    feature_name: Optional[str] = None


@dataclass(frozen=True)
class LiteralRef:
    value: Any


def col(name: str) -> ColumnRef:
    return ColumnRef(name=name)


def feat(spec: "FeatureSpec") -> FeatureRef:
    if spec.feature_id is None:
        raise ValueError(f"Cannot create FeatureRef for spec '{spec.name}' because feature_id is missing")
    return FeatureRef(feature_id=spec.feature_id, feature_name=spec.name)


def lit(value: Any) -> LiteralRef:
    return LiteralRef(value=value)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    primitive: str
    inputs: Dict[str, Any]
    params: Optional[Dict[str, Any]] = None
    post: Optional[List[PostOp]] = None
    publish: bool = True
    feature_id: Optional[str] = None
    column_name: Optional[str] = None
    name_spec: Optional[FeatureNameSpec] = None
    identity_payload: Optional[Dict[str, Any]] = None

    @property
    def feature_name(self) -> str:
        return self.name


@dataclass(frozen=True)
class BuiltFeature:
    feature_name: str
    feature_id: str
    column_name: str
    series: pd.Series
    spec: FeatureSpec


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


# ---------- Canonical identity ----------
def _stable_series_payload(v: pd.Series) -> Dict[str, Any]:
    filled = v.astype(object).where(~v.isna(), "__nan__")
    return {
        "__type__": "Series",
        "name": v.name,
        "index": [repr(x) for x in v.index.tolist()],
        "values": [repr(x) for x in filled.tolist()],
    }


def canonicalize_value(v: Any):
    if isinstance(v, ColumnRef):
        return {"kind": "column", "name": v.name}
    if isinstance(v, FeatureRef):
        return {"kind": "feature", "feature_id": v.feature_id}
    if isinstance(v, LiteralRef):
        return {"kind": "literal", "value": canonicalize_value(v.value)}
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: canonicalize_value(v[k]) for k in sorted(v)}
    if isinstance(v, (list, tuple)):
        return [canonicalize_value(x) for x in v]
    if isinstance(v, pd.Series):
        return {"kind": "series", "payload": _stable_series_payload(v)}
    if isinstance(v, np.generic):
        return v.item()
    return v


def _normalize_input_value(v: Any) -> Any:
    """Normalize template inputs into explicit reference/value objects.

    Rule: raw strings are treated as dataframe columns. Feature dependencies should be
    passed explicitly via feat(upstream_spec) / FeatureRef.
    """
    if isinstance(v, (ColumnRef, FeatureRef, LiteralRef, pd.Series)):
        return v
    if isinstance(v, str):
        return ColumnRef(v)
    return v


def normalize_inputs(inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {k: _normalize_input_value(v) for k, v in (inputs or {}).items()}


def _feature_name_from_spec(name: Optional[str], name_spec: Optional[FeatureNameSpec]) -> str:
    if name_spec is not None:
        return make_feature_name(
            domain=name_spec.domain,
            family=name_spec.family,
            signal=name_spec.signal,
            params=name_spec.params,
            state=name_spec.state,
        )
    if name is None:
        raise ValueError("Either name or name_spec must be provided")
    return name


def make_identity_payload(
    *,
    primitive: str,
    inputs: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    post: Optional[List[PostOp]] = None,
) -> Dict[str, Any]:
    return {
        "primitive": canonicalize_value(primitive),
        "inputs": canonicalize_value(normalize_inputs(inputs)),
        "params": canonicalize_value(params or {}),
        "post": canonicalize_value(post or []),
    }


def _hash_identity_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "fid_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def make_feature_id(
    *,
    primitive: str,
    inputs: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    post: Optional[List[PostOp]] = None,
) -> str:
    return _hash_identity_payload(
        make_identity_payload(primitive=primitive, inputs=inputs, params=params, post=post)
    )


def make_spec(
    *,
    primitive: str,
    inputs: Dict[str, Any],
    name: Optional[str] = None,
    name_spec: Optional[FeatureNameSpec] = None,
    params: Optional[Dict[str, Any]] = None,
    post: Optional[List[PostOp]] = None,
    publish: bool = True,
    column_name: Optional[str] = None,
) -> FeatureSpec:
    """Create a FeatureSpec and compute feature_id immediately.

    Use ColumnRef/col(...) for source columns, FeatureRef/feat(...) for upstream
    feature dependencies, and raw literals for scalar literal values. Raw strings
    in inputs are normalized to ColumnRef for convenience.
    """
    normalized_inputs = normalize_inputs(inputs)
    feature_name = _feature_name_from_spec(name=name, name_spec=name_spec)
    normalized_params = params or {}
    normalized_post = post or []
    payload = make_identity_payload(
        primitive=primitive,
        inputs=normalized_inputs,
        params=normalized_params,
        post=normalized_post,
    )
    fid = _hash_identity_payload(payload)
    return FeatureSpec(
        name=feature_name,
        primitive=primitive,
        inputs=normalized_inputs,
        params=normalized_params,
        post=normalized_post,
        publish=publish,
        feature_id=fid,
        column_name=column_name,
        name_spec=name_spec,
        identity_payload=payload,
    )


def normalize_spec(spec: FeatureSpec) -> FeatureSpec:
    feature_name = _feature_name_from_spec(name=spec.name, name_spec=spec.name_spec)
    inputs = normalize_inputs(spec.inputs)
    params = spec.params or {}
    post = spec.post or []
    payload = spec.identity_payload or make_identity_payload(
        primitive=spec.primitive,
        inputs=inputs,
        params=params,
        post=post,
    )
    fid = spec.feature_id or _hash_identity_payload(payload)
    return FeatureSpec(
        name=feature_name,
        primitive=spec.primitive,
        inputs=inputs,
        params=params,
        post=post,
        publish=spec.publish,
        feature_id=fid,
        column_name=spec.column_name,
        name_spec=spec.name_spec,
        identity_payload=payload,
    )


def local_spec_identity(spec: FeatureSpec) -> dict:
    spec = normalize_spec(spec)
    return {
        "primitive": canonicalize_value(spec.primitive),
        "inputs": canonicalize_value(spec.inputs),
        "params": canonicalize_value(spec.params),
        "post": canonicalize_value(spec.post),
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
                    inspect.Parameter.KEYWORD_ONLY,
                )
            )
            if required:
                missing_required.append(name)

    if missing_required:
        return missing_required, 300
    return call_kwargs, 200


# ---------- Spec indexing / validation ----------
def build_spec_maps(specs: Iterable[FeatureSpec]) -> Dict[str, FeatureSpec]:
    by_id: Dict[str, FeatureSpec] = {}
    for raw_spec in specs:
        spec = normalize_spec(raw_spec)
        assert spec.feature_id is not None
        if spec.feature_id in by_id:
            existing = by_id[spec.feature_id]
            if local_spec_identity(existing) != local_spec_identity(spec):
                raise ValueError(f"Conflicting definitions for feature_id '{spec.feature_id}'")
            # Same computation can appear multiple times. Merge publish intent and keep
            # the first readable name unless the duplicate is publish=True and first is not.
            if spec.publish and not existing.publish:
                by_id[spec.feature_id] = replace(existing, publish=True)
        else:
            by_id[spec.feature_id] = spec
    return by_id


def _dependency_ids(spec: FeatureSpec) -> set[str]:
    deps = set()
    for v in (spec.inputs or {}).values():
        if isinstance(v, FeatureRef):
            deps.add(v.feature_id)
    return deps


# ---------- Builder ----------
class FeatureBuilder:
    """Materialize FeatureSpec objects into pandas Series.

    Core rules:
    - FeatureSpec.feature_id is the only identity key for building/reuse.
    - FeatureSpec.name is human-facing and may collide.
    - FeatureRef dependencies are allowed to be delayed if the dependency is still
      in the current build graph.
    - Missing FeatureRef dependencies raise a clear error.
    - Output column names are collision-resolved only at export/build output time.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        date_col: str = "date",
        sym_col: str = "symbol",
        eps: float = 1e-12,
    ):
        self.df = df
        self.date_col = date_col
        self.sym_col = sym_col
        self.eps = eps
        self._cache: Dict[str, BuiltFeature] = {}

    # ----- dependency readiness -----
    def _feature_ref_status(self, v: Any, remaining_ids: set[str]) -> str:
        """Return readiness status for a FeatureRef input.

        Values:
        - "ready": dependency is already materialized in cache, or v is not FeatureRef
        - "blocked": dependency is valid but not built yet; wait for another iteration
        - "missing": dependency is neither built nor present in current build graph
        """
        if not isinstance(v, FeatureRef):
            return "ready"
        if v.feature_id in self._cache:
            return "ready"
        if v.feature_id in remaining_ids:
            return "blocked"
        return "missing"

    def _check_dependency_readiness(
        self,
        spec: FeatureSpec,
        remaining_ids: set[str],
    ) -> Tuple[bool, List[FeatureRef]]:
        """Check whether a spec can be built in the current iteration.

        Returns:
        - blocked: True when at least one valid dependency is not built yet
        - missing: FeatureRef dependencies absent from both cache and remaining graph
        """
        blocked = False
        missing: List[FeatureRef] = []

        for v in (spec.inputs or {}).values():
            status = self._feature_ref_status(v, remaining_ids)
            if status == "blocked":
                blocked = True
            elif status == "missing":
                assert isinstance(v, FeatureRef)
                missing.append(v)

        return blocked, missing

    @staticmethod
    def _format_feature_refs(refs: List[FeatureRef]) -> List[str]:
        out = []
        for ref in refs:
            label = f" ({ref.feature_name})" if ref.feature_name else ""
            out.append(f"{ref.feature_id}{label}")
        return out

    # ----- input resolution -----
    def _resolve_input(self, v: Any) -> Any:
        """Resolve a normalized spec input into a runtime object.

        FeatureRef should only reach this method after readiness has been checked
        by the build loop. If a FeatureRef is missing here, that is an internal
        ordering/readiness bug rather than a normal delayed dependency.
        """
        if isinstance(v, ColumnRef):
            if v.name not in self.df.columns:
                raise ValueError(f"Input column '{v.name}' not found in dataframe")
            s = self.df[v.name].copy()
            s.name = v.name
            return s

        if isinstance(v, FeatureRef):
            if v.feature_id not in self._cache:
                label = f" ({v.feature_name})" if v.feature_name else ""
                raise RuntimeError(
                    f"Internal builder error: feature dependency '{v.feature_id}'{label} "
                    f"was expected to exist in cache before input resolution."
                )
            return self._cache[v.feature_id].series

        if isinstance(v, LiteralRef):
            return v.value

        if isinstance(v, pd.Series):
            return v

        # non-string raw literals are allowed; raw strings should have been
        # normalized to ColumnRef by make_spec/normalize_spec.
        return v

    # ----- execution -----
    def _make_built_feature(self, spec: FeatureSpec, out: pd.Series) -> BuiltFeature:
        assert spec.feature_id is not None
        out = out.copy()
        out.name = spec.name
        return BuiltFeature(
            feature_name=spec.name,
            feature_id=spec.feature_id,
            column_name=spec.column_name or spec.name,
            series=out,
            spec=spec,
        )

    def _build_one(self, spec: FeatureSpec) -> BuiltFeature:
        """Build one spec whose FeatureRef dependencies are already ready."""
        assert spec.feature_id is not None

        if spec.feature_id in self._cache:
            return self._cache[spec.feature_id]

        resolved_inputs = {
            k: self._resolve_input(v)
            for k, v in (spec.inputs or {}).items()
        }
        params = spec.params or {}
        available_inputs = {
            "df": self.df,
            "sym_col": self.sym_col,
            "date_col": self.date_col,
            "eps": self.eps,
            **params,
            **resolved_inputs,
        }

        if spec.primitive not in PRIMITIVES:
            raise ValueError(f"Unknown primitive '{spec.primitive}' for feature '{spec.name}'")

        prim = PRIMITIVES[spec.primitive]
        call_kwargs, status = prepare_call(fn=prim, available_inputs=available_inputs)
        if status == 300:
            raise ValueError(
                f"While building feature '{spec.name}', primitive '{spec.primitive}' "
                f"is missing required inputs {call_kwargs}. "
                f"Available keys: {sorted(available_inputs.keys())}"
            )

        out = prim(**call_kwargs)
        if not isinstance(out, pd.Series):
            out = pd.Series(out, index=self.df.index)

        if spec.post:
            for post_name, post_params in spec.post:
                if post_name not in POSTS:
                    raise ValueError(f"Unknown post transform '{post_name}' for feature '{spec.name}'")

                fn = POSTS[post_name]
                post_available_inputs = {
                    "df": self.df,
                    "sym_col": self.sym_col,
                    "date_col": self.date_col,
                    "eps": self.eps,
                    "x": out,
                    **dict(post_params),
                }
                post_call_kwargs, post_status = prepare_call(
                    fn=fn,
                    available_inputs=post_available_inputs,
                )
                if post_status == 300:
                    raise ValueError(
                        f"While building feature '{spec.name}', post transform '{post_name}' "
                        f"is missing required inputs {post_call_kwargs}. "
                        f"Available keys: {sorted(post_available_inputs.keys())}"
                    )
                out = fn(**post_call_kwargs)

        built = self._make_built_feature(spec, out)
        self._cache[spec.feature_id] = built
        return built

    def build_records(self, specs: Iterable[FeatureSpec]) -> Dict[str, BuiltFeature]:
        """Build all specs and return BuiltFeature records keyed by feature_id."""
        if not isinstance(specs, Iterable):
            specs = [specs]

        spec_by_id = build_spec_maps(specs)
        remaining_ids = set(spec_by_id.keys())
        requested_ids = set(spec_by_id.keys())

        progressed = True
        while remaining_ids and progressed:
            progressed = False

            for fid in list(remaining_ids):
                spec = spec_by_id[fid]

                blocked, missing = self._check_dependency_readiness(
                    spec=spec,
                    remaining_ids=remaining_ids,
                )

                if missing:
                    raise ValueError(
                        f"Feature '{spec.name}' ({fid}) depends on missing feature ids: "
                        f"{self._format_feature_refs(missing)}"
                    )

                if blocked:
                    continue

                self._build_one(spec)
                remaining_ids.remove(fid)
                progressed = True

        if remaining_ids:
            unresolved = [spec_by_id[fid].name for fid in sorted(remaining_ids)]
            dependency_report: Dict[str, List[str]] = {}
            for fid in sorted(remaining_ids):
                spec = spec_by_id[fid]
                refs = [
                    v for v in (spec.inputs or {}).values()
                    if isinstance(v, FeatureRef)
                    and v.feature_id not in self._cache
                ]
                dependency_report[f"{spec.name} ({fid})"] = self._format_feature_refs(refs)

            raise ValueError(
                f"Unresolved specs (possible cycle or unresolved dependency chain): {unresolved}; "
                f"unresolved dependencies: {dependency_report}"
            )

        return {fid: self._cache[fid] for fid in requested_ids}

    # ----- output naming -----
    def _build_column_names(self, records: List[BuiltFeature]) -> List[BuiltFeature]:
        counts = defaultdict(int)
        for rec in records:
            counts[rec.feature_name] += 1

        out: List[BuiltFeature] = []
        for rec in records:
            if counts[rec.feature_name] == 1:
                column_name = rec.feature_name
            else:
                column_name = f"{rec.feature_name}__{rec.feature_id[:12]}"

            series = rec.series.copy()
            series.name = column_name
            out.append(
                BuiltFeature(
                    feature_name=rec.feature_name,
                    feature_id=rec.feature_id,
                    column_name=column_name,
                    series=series,
                    spec=replace(rec.spec, column_name=column_name),
                )
            )
        return out

    def build(self, specs: Iterable[FeatureSpec]) -> Dict[str, pd.Series]:
        """Build specs and return all requested features keyed by export-safe column name."""
        records_by_id = self.build_records(specs)
        records = self._build_column_names(list(records_by_id.values()))
        return {rec.column_name: rec.series for rec in records}

    def build_published(self, specs: Iterable[FeatureSpec]) -> Dict[str, pd.Series]:
        """Build specs and return only publish=True features keyed by export-safe column name."""
        if not isinstance(specs, Iterable):
            specs = [specs]

        specs = [normalize_spec(s) for s in specs]
        self.build_records(specs)

        seen_ids: set[str] = set()
        published_records: List[BuiltFeature] = []

        for spec in specs:
            if not spec.publish:
                continue
            assert spec.feature_id is not None

            # Publish a computation once, even if the same FeatureSpec appears more
            # than once because multiple templates reused it.
            if spec.feature_id in seen_ids:
                continue
            seen_ids.add(spec.feature_id)

            built = self._cache[spec.feature_id]
            series = built.series.copy()
            series.name = spec.name
            published_records.append(
                BuiltFeature(
                    feature_name=spec.name,
                    feature_id=spec.feature_id,
                    column_name=spec.column_name or spec.name,
                    series=series,
                    spec=spec,
                )
            )

        published_records = self._build_column_names(published_records)
        return {rec.column_name: rec.series for rec in published_records}
