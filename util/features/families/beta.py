from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
from util.features.core import (
    FeatureNameSpec,
    FeatureSpec,
    FeatureTemplate,
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
) -> FeatureNameSpec:
    """Deterministic readable beta feature name.

    Example:
        px__beta__r__w252_mktspy__raw
    """
    params: Dict[str, Any] = {}
    if w is not None:
        params["w"] = w
    if mkt is not None:
        params["mkt"] = mkt.lower()

    return FeatureNameSpec(
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
    market_price: Optional[pd.Series] = None,
    publish: bool = True,
) -> FeatureSpec:
    """Build one rolling market beta spec.

    Pass `market_price` (a date-indexed series, e.g. from
    primitives_beta.fetch_market_price) to inject the market baseline and keep
    the build free of network I/O. When omitted, the primitive fetches it for
    the panel's date range (legacy convenience).
    """
    inputs: Dict[str, Any] = {"price": col(price_col)}
    if market_price is not None:
        inputs["market_price"] = market_price

    return make_spec(
        name=beta_feature_name(
            "r",
            w=window,
            mkt=market_symbol,
            state="raw",
        ),
        primitive="ts_beta_to_market",
        inputs=inputs,
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
    market_price: Optional[pd.Series] = None,
) -> list[FeatureSpec]:
    beta = spec_ts_beta_to_market(
        price_col=price_col,
        window=window,
        market_symbol=market_symbol,
        market_price=market_price,
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