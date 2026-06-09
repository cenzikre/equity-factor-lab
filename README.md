# equity-factor-lab

Quantitative equity research stack built on AWS SageMaker. Covers the full pipeline from market data ingestion to factor engineering and market regime reporting.

## What's inside

| Layer | Location | Description |
|---|---|---|
| Data client | `util/data_client/` | Async + sync FMP API clients with rate limiting and retries |
| Feature engine | `util/features/` | Declarative factor framework — returns, vol, beta, trend, RSI, drawdown, tail risk |
| Regime report | `util/dashboard/` | Market regime heatmaps and PDF report generation |
| Notebooks | `*/` | Exploratory research, data QA, and feature development |

## Setup

```bash
bash setup_fmp_env.sh          # create conda env (Python 3.10)
source activate_fmp_env.sh     # activate it

cp .env.example .env           # or run: python check_env.py
# edit .env and set FMP_API_KEY=<your_key>
```

## Quick start

**Pull market data (async, bulk):**
```python
from util.data_client.FMPDataClient import FMPClient, fetch_fmp_for_tickers
import asyncio

client = FMPClient()  # reads FMP_API_KEY from .env
data = asyncio.run(fetch_fmp_for_tickers(tickers, request_specs, client=client))
```

**Build factors:**
```python
from util.features.core import FeatureBuilder
from util.features.families.returns import RETURN_FAMILY

specs = RETURN_FAMILY["REALIZED_VOLATILITY"](price_col="adjClose", window=21)
result = FeatureBuilder(panel_df).build_published(specs)
```

**Run the market regime report:**
```bash
python market_regime_report.py
# output → reports/market_regime/<date>/
```

## Data source

[Financial Modeling Prep (FMP)](https://financialmodelingprep.com/) — requires an API key set in `.env` as `FMP_API_KEY`.
