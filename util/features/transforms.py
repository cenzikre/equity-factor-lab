from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Union

from util.features.core import (
    FeatureNameSpec,
    FeatureSpec,
    make_feature_name,
    make_spec,
    feat,
)


TargetSelector = Union[
    Literal["last", "last_published"],
    FeatureSpec,
    str,
    dict[str, Any],
]

_ALLOWED_NAME_SPEC_KEYS = {"domain", "family", "signal", "state", "params"}


def _parse_name_value(raw: str) -> Any:
    """Parse compact name value token back into int/float/string.

    This is only a fallback for old specs without FeatureNameSpec.
    """
    if raw == "":
        return raw

    if raw.startswith("n"):
        restored = "-" + raw[1:]
    else:
        restored = raw

    restored = restored.replace("p", ".")

    try:
        return int(restored)
    except ValueError:
        pass

    try:
        return float(restored)
    except ValueError:
        return raw


def parse_feature_name(name: str) -> FeatureNameSpec:
    """Parse a generated feature name into FeatureNameSpec.

    Expected format:
        {domain}__{family}__{signal}__{params}__{state}

    Example:
        px__prc__dma__w5__raw
    """
    parts = name.split("__")
    if len(parts) != 5:
        raise ValueError(
            f"Cannot parse feature name {name!r}. Expected 5 components: "
            "{domain}__{family}__{signal}__{params}__{state}."
        )

    domain, family, signal, param_str, state = parts
    params: Dict[str, Any] = {}

    if param_str != "none":
        for token in param_str.split("_"):
            if not token:
                continue

            key_chars = []
            val_chars = []
            hit_value = False

            for ch in token:
                if not hit_value and ch.isalpha():
                    key_chars.append(ch)
                else:
                    hit_value = True
                    val_chars.append(ch)

            key = "".join(key_chars)
            raw_val = "".join(val_chars)

            if not key:
                raise ValueError(
                    f"Cannot parse parameter token {token!r} in feature name {name!r}"
                )

            params[key] = _parse_name_value(raw_val)

    return FeatureNameSpec(
        domain=domain,
        family=family,
        signal=signal,
        params=params,
        state=state,
    )


def feature_name_spec_from_feature(spec: FeatureSpec) -> FeatureNameSpec:
    """Get a robust FeatureNameSpec from a feature.

    Priority:
    1. spec.name_spec, if provided
    2. parse spec.name

    If both name and name_spec are provided, this intentionally uses name_spec,
    because it is structured and less lossy.
    """
    if spec.name_spec is not None:
        return spec.name_spec

    return parse_feature_name(spec.name)


def extend_feature_name(
    spec: FeatureSpec,
    *,
    extra_params: Optional[Dict[str, Any]] = None,
    state: Optional[str] = None,
    signal: Optional[str] = None,
    family: Optional[str] = None,
    domain: Optional[str] = None,
) -> tuple[str, FeatureNameSpec]:
    """Create an extended deterministic feature name from an existing spec.

    If spec.name_spec exists, use that structured path. Otherwise parse spec.name.

    Example:
        base: px__prc__dma__w5__raw
        extra_params={"zw": 60}, state="z"
        -> px__prc__dma__w5_zw60__z
    """
    base = feature_name_spec_from_feature(spec)

    params = dict(base.params or {})
    if extra_params:
        params.update(extra_params)

    new_name_spec = FeatureNameSpec(
        domain=domain or base.domain,
        family=family or base.family,
        signal=signal or base.signal,
        params=params,
        state=state or base.state,
    )

    name = make_feature_name(
        domain=new_name_spec.domain,
        family=new_name_spec.family,
        signal=new_name_spec.signal,
        params=new_name_spec.params,
        state=new_name_spec.state,
    )

    return name, new_name_spec


def _name_spec_matches(
    name_spec: FeatureNameSpec,
    target: dict[str, Any],
) -> bool:
    invalid_keys = set(target) - _ALLOWED_NAME_SPEC_KEYS
    if invalid_keys:
        raise ValueError(
            f"Invalid dict target keys: {sorted(invalid_keys)}. "
            f"Allowed keys: {sorted(_ALLOWED_NAME_SPEC_KEYS)}"
        )

    for key, expected in target.items():
        if key == "params":
            actual_params = name_spec.params or {}
            if not isinstance(expected, dict):
                raise ValueError("target['params'] must be a dict")

            for param_key, param_expected in expected.items():
                if actual_params.get(param_key) != param_expected:
                    return False
        else:
            if getattr(name_spec, key) != expected:
                return False

    return True


def _select_target_spec(
    specs: Sequence[FeatureSpec],
    target: TargetSelector = "last_published",
) -> FeatureSpec:
    """Select the feature spec to transform."""
    if not specs:
        raise ValueError("Cannot select target from empty specs")

    if isinstance(target, FeatureSpec):
        return target

    if target == "last":
        return specs[-1]

    if target == "last_published":
        published = [s for s in specs if s.publish]
        if not published:
            raise ValueError("No published spec found in specs")
        return published[-1]

    if isinstance(target, str):
        matches = [
            s for s in specs
            if s.name == target or s.feature_id == target or s.column_name == target
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Target {target!r} matched {len(matches)} specs; expected exactly 1"
            )
        return matches[0]

    if isinstance(target, dict):
        invalid_keys = set(target) - _ALLOWED_NAME_SPEC_KEYS
        if invalid_keys:
            raise ValueError(
                f"Invalid dict target keys: {sorted(invalid_keys)}. "
                f"Allowed keys: {sorted(_ALLOWED_NAME_SPEC_KEYS)}"
            )

        matches = []
        parse_errors = []

        for s in specs:
            try:
                ns = feature_name_spec_from_feature(s)
            except Exception as e:
                parse_errors.append((s.name, str(e)))
                continue

            if _name_spec_matches(ns, target):
                matches.append(s)

        if len(matches) != 1:
            detail = [s.name for s in matches]
            raise ValueError(
                f"Dict target {target!r} matched {len(matches)} specs; expected exactly 1. "
                f"Matches: {detail}. Parse errors: {parse_errors[:5]}"
            )

        return matches[0]

    raise TypeError(f"Unsupported target selector: {target!r}")


def to_zscore(
    specs: Sequence[FeatureSpec],
    *,
    z_window: int,
    target: TargetSelector = "last_published",
    clip: Optional[tuple[float, float]] = (-5.0, 5.0),
    publish: bool = True,
    eps: float = 1e-12,
    z_param_name: str = "zw",
) -> list[FeatureSpec]:
    """Create time-series z-score specs for a target feature.

    Returns only derived specs:
        [mu, sd, z]

    Usage:
        base = SOME_FAMILY["FEATURE"](...)
        specs += base
        specs += to_zscore(base, z_window=60)

    Naming:
        base: px__prc__dma__w5__raw
        mu:   px__prc__dma__w5_zw60__zmu
        sd:   px__prc__dma__w5_zw60__zsd
        z:    px__prc__dma__w5_zw60__z
    """
    if z_window <= 0:
        raise ValueError(f"z_window must be positive, got {z_window}")

    specs = list(specs)
    target_spec = _select_target_spec(specs, target=target)

    extra_params = {z_param_name: z_window}

    mu_name, mu_name_spec = extend_feature_name(
        target_spec,
        extra_params=extra_params,
        state="zmu",
    )
    sd_name, sd_name_spec = extend_feature_name(
        target_spec,
        extra_params=extra_params,
        state="zsd",
    )
    z_name, z_name_spec = extend_feature_name(
        target_spec,
        extra_params=extra_params,
        state="z",
    )

    mu = make_spec(
        name=mu_name,
        name_spec=mu_name_spec,
        primitive="ts_mean",
        inputs={"x": feat(target_spec)},
        params={"window": z_window},
        publish=False,
    )

    sd = make_spec(
        name=sd_name,
        name_spec=sd_name_spec,
        primitive="ts_std",
        inputs={"x": feat(target_spec)},
        params={"window": z_window},
        publish=False,
    )

    post = []
    if clip is not None:
        lo, hi = clip
        post = [("clip", {"lo": lo, "hi": hi})]

    z = make_spec(
        name=z_name,
        name_spec=z_name_spec,
        primitive="zscore",
        inputs={
            "x": feat(target_spec),
            "mu": feat(mu),
            "sigma": feat(sd),
        },
        params={"eps": eps},
        post=post,
        publish=publish,
    )

    return [mu, sd, z]


def add_ts_zscore(
    specs: Sequence[FeatureSpec],
    *,
    z_window: int,
    target: TargetSelector = "last_published",
    clip: Optional[tuple[float, float]] = (-5.0, 5.0),
    publish: bool = True,
    eps: float = 1e-12,
    z_param_name: str = "zw",
    include_base: bool = True,
) -> list[FeatureSpec]:
    """Return specs optionally including base specs plus z-score-derived specs.

    If include_base=True:
        return base + [mu, sd, z]

    If include_base=False:
        return [mu, sd, z]
    """
    specs = list(specs)
    derived = to_zscore(
        specs,
        z_window=z_window,
        target=target,
        clip=clip,
        publish=publish,
        eps=eps,
        z_param_name=z_param_name,
    )

    if include_base:
        return specs + derived

    return derived
