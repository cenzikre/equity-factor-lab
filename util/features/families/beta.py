from __future__ import annotations
from typing import Any, Dict, Optional
from util.features.core import (
    FeatureSpec,
    FeatureTemplate,
    make_feature_name,
    make_spec,
    col,
    # feat,
)
import util.features.primitives_beta


DOMAIN = "px"
FAMILY = "beta"


def beta_feature_name(
    signal: str,
    *,
    w: Optional[int] = None,
    mkt: Optional[str] = None,
    state: str = "raw",
) -> str:
    """Deterministic readable beta feature name.

    Example:
        px__beta__r__w252_mktspy__raw
    """
    params: Dict[str, Any] = {}
    if w is not None:
        params["w"] = w
    if mkt is not None:
        params["mkt"] = mkt.lower()

    return make_feature_name(
        domain=DOMAIN,
        family=FAMILY,
        signal=signal,
        params=params,
        state=state,
    )


def spec_ts_beta_to_market(
    price_col: str,
    window: int,
    *,
    market_symbol: str = "SPY",
    publish: bool = True,
) -> FeatureSpec:
    """Build one rolling market beta spec."""
    return make_spec(
        name=beta_feature_name(
            "r",
            w=window,
            mkt=market_symbol,
            state="raw",
        ),
        primitive="ts_beta_to_market",
        inputs={"price": col(price_col)},
        params={
            "window": window,
            "market_symbol": market_symbol.upper(),
        },
        publish=publish,
    )


def template_ts_beta_to_market(
    price_col: str,
    window: int,
    market_symbol: str = "SPY",
) -> list[FeatureSpec]:
    beta = spec_ts_beta_to_market(
        price_col=price_col,
        window=window,
        market_symbol=market_symbol,
        publish=True,
    )
    return [beta]


BETA_FAMILY = {
    "MARKET_BETA": FeatureTemplate(
        name="ts_beta_to_market",
        domain=DOMAIN,
        family=FAMILY,
        signal="r",
        template_fn=template_ts_beta_to_market,
        description="rolling beta of asset returns to a market baseline symbol",
        tags=("price", "return", "beta", "market", "timeseries"),
    ),
}