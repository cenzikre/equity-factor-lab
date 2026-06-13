# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

All code runs in the `fmp_data` conda env (`~/.conda/envs/fmp_data`, Python 3.10). The base `/opt/conda` python lacks the async/pull dependencies — don't use it for pulls or tests.

```bash
# Create and configure the conda environment (one-time; also registers the
# "Python (fmp_data)" Jupyter kernel)
bash setup_fmp_env.sh

# Activate the environment in subsequent sessions
source activate_fmp_env.sh   # activates 'fmp_data' env
# or: conda activate fmp_data
# or call the interpreter directly: ~/.conda/envs/fmp_data/bin/python
```

Required packages: `requests`, `aiohttp`, `aiolimiter`, `tenacity`, `tqdm`, `python-dotenv`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `pytest` (dev), `ipykernel` (notebooks)

Set the FMP API key before running any data pull code:
```bash
export FMP_API_KEY=<your_key>
```

## Project Architecture

This is a quantitative trading research codebase running on AWS SageMaker. Work is primarily done in Jupyter notebooks with supporting Python libraries under `util/`.

### Data Pipeline

**`util/data_client/`** — FMP (Financial Modeling Prep) API clients:
- `FMPDataClient.py` — async batch client (`FMPClient` + `fetch_fmp_for_tickers`). Handles rate limiting (token-bucket, ~12 req/s), concurrency (semaphore, 25 in-flight), exponential-backoff retries via `tenacity`. Use this for bulk pulls across many tickers.
- `fmp.py` — sync single-request helpers (`getFMPData`, `getFMPData_withkey`) and an `lru_cache`-backed baseline price fetcher. Use for ad-hoc/notebook queries.
- `dataPullHelpers.py` — `normalize_df_for_parquet()`: dtype normalization for mixed-schema API responses before writing parquet. Handles numeric/datetime inference, huge-int-as-string edge cases.

**`util/data_pull/`** — bulk pull orchestration (supersedes the legacy `FMPDataPull.py`, removed 2026-06):
- `components.py` — data component registry (7 FMP datatypes), date-window chunking for daily prices (pd1, pd2, ...), range-sized quarterly limits.
- `pull.py` — `run_pull()`: universe → batched fetch → normalization → dated S3 snapshot `raw/<label>/` with `data_{component}_tk0_pd{P}.parquet`, `data_tickerprofile.parquet`, `error_log.json`, `pull_manifest.json`. Injectable `fetch_fn` for testing.

**`util/dataset_builder/`** — panel dataset construction (supersedes `ConstructFullData.ipynb`): column specs / merge specs, S3 parquet I/O with symbol predicate pushdown + `StreamingParquetWriter`, point-in-time merge core (as-of `filingDate`, per-symbol liquidity rollings, validation stats), markdown build report renderer.

**Pipeline** (full regeneration; instance RAM is ~15GB so construction streams in ticker batches):
```bash
python GetFMPData/build_stock_universe.py                          # optional; pull reuses the universe CSV if its
                                                                   # sidecar age <= 60d (--universe-max-age-days), else rebuilds
python GetFMPData/fmp_data_pull.py --start 2006-01-01 --label <L>  # long pull (or --start ~2y ago, weekly)
python GetFMPData/construct_full_data.py --raw-date <L> --label <L>
```

**Breadth task** (per-task entry point): `python GetFMPData/breadth_data_pull.py` — active universe → 15-month price + enterprise-values pull (~18.5k requests) → liquidity panel via the same `merge_core` chain (`add_liquidity_flags` is the single source of the liquidity definition) → qualified symbol list in `MarketInternalMonitor/universe/breadth-qualified-<label>.csv`. `--skip-pull --overwrite` requalifies from an existing snapshot without API calls.

Tests: `python -m pytest tests/` (no network; fake fetch + local filesystem).

**`GetFMPData/`** — entry-point scripts above, plus the universe CSVs and legacy notebooks. The canonical data-client implementations are in `util/data_client/`.

**`MarketInternalMonitor/universe/`** — CSV stock universe files used as the ticker list for bulk pulls.

### Feature Engineering

**`util/features/`** — declarative feature computation framework:

- **`core.py`** — the engine. Key types:
  - `FeatureSpec`: declarative description of one feature (primitive, inputs, params, post-transforms). Identity is deterministic: `feature_id` is a SHA-256 hash of the canonicalized spec, so identical computations are deduplicated automatically.
  - `FeatureBuilder`: materializes `FeatureSpec` objects into `pd.Series` against a panel DataFrame. Handles dependency ordering (topological sort via iterative unblocking), caching by `feature_id`, and output column name deduplication.
  - `ColumnRef` / `col()`: reference a DataFrame column as input.
  - `FeatureRef` / `feat()`: reference another `FeatureSpec` as upstream input (creates a dependency edge).
  - `LiteralRef` / `lit()`: embed a scalar literal in a spec.
  - `make_spec()`: primary factory — computes `feature_id` immediately and normalizes inputs.

- **`primitives.py`** — the `PRIMITIVES` registry mapping string names to functions. All primitives receive `(df, sym_col, ...)` and return a `pd.Series`. Also defines `POSTS` (post-transform registry: `clip`, `log`, `cs_rank`, annualization helpers, etc.).

- **`primitives_beta.py`** — `prim_ts_beta_to_market()`: rolling OLS beta of each asset to a market benchmark (default SPY). Fetches market prices via `_fetch_fmp_baseprice_cached`. Registers into `PRIMITIVES`.

- **`primitives_regtrend.py`** — rolling OLS regression stats (`slope`, `intercept`, `r2`, `resid_std`, `slope_se`, `tstat`) for y ~ 1 + t. Window-based, NaN-tolerant (row-by-row). Registers into `PRIMITIVES`.

- **`primitives_regtrend_vectorized.py`** — vectorized version of the same regression stats using `sliding_window_view`; requires fully finite windows. Faster for large panels. Registers into `PRIMITIVES`.

- **`primitives_tailrisk.py`** — tail risk primitives (CVaR / expected shortfall) via `sliding_window_view`. Supports `lower`/`upper` tail. Registers into `PRIMITIVES`.

- **`transforms.py`** — higher-level spec transformations, `parse_feature_name()`, and `TargetSelector` logic for selecting features from a built set.

- **`families/`** — organized groups of feature specs/templates:
  - `returns.py` — log return, annualized return, moving average/vol of returns, return z-score, realized vol (`RETURN_FAMILY`)
  - `price.py` — price-level features
  - `rsi.py` — RSI
  - `truerange.py` — true range / ATR
  - `drawdown.py` — drawdown features
  - `beta.py` — market beta
  - `var.py` — value at risk / tail risk
  - `trend.py` — trend/slope features

Each family file exports a `*_FAMILY` dict of `FeatureTemplate` objects. `FeatureTemplate` wraps a `template_fn` that returns `List[FeatureSpec]` (including intermediate publish=False specs needed to compute the published output).

**`MarketInternalMonitor/featureStore.py`** — older monolithic feature store; the canonical implementation is now `util/features/`.

### Dashboard / Reporting

**`util/dashboard/market_regime_helpers.py`** — helpers for the market regime report (~630 lines):
- `validate_environment()` / `get_fmp_api_key()` — env checks
- `fetch_fmp_price_data()` — pulls adjusted EOD prices for a ticker list via `fmp.py`
- `build_market_features()` — runs the full feature pipeline (returns, DMAs, realized vol, drawdown, regression trend, beta) and returns a `dict[str, pd.DataFrame]` of feature groups
- `save_pdf_report()` — renders multi-page heatmap PDF via matplotlib
- `save_snapshot_csvs()` — writes `latest_feature_snapshot.csv` and `latest_regime_score.csv`

**`market_regime_report.py`** — runnable entry point that wires the above into a complete report. Outputs to `reports/market_regime/<date>/`. Run with:
```bash
python market_regime_report.py
```

### Utilities

**`nb_code_parser/nb_code_extract.py`** — CLI that extracts all code cells from a `.ipynb` notebook into a plain `.txt` file (one `# ===== Code Cell N =====` block per cell). Usage:
```bash
python nb_code_parser/nb_code_extract.py --name <notebook.ipynb>
# writes nb_code_parser/<stem>_code.txt
```

**`check_env.py`** — one-shot setup helper: creates `.env` with a default `FMP_API_KEY` if missing, loads it, and prints a masked key diagnostic. Run once after cloning on a fresh instance.

### Feature Naming Convention

Feature names follow the format:
```
{domain}__{family}__{signal}__{params}__{state}
```
Example: `px__ret__logret__lb21__raw`

Param rendering: `lb` (lookback), `w` (window), `sw`, `lw`, `p`, `sp`, `s`, `off`, `zw` are rendered in order; floats use `p` for `.` and `n` for `-`.

### DataFrame Contract

The panel DataFrame passed to `FeatureBuilder` must have:
- A `symbol` column (configurable via `sym_col`)
- A `date` column (configurable via `date_col`)
- Rows sorted by `(symbol, date)` within each group (group-wise operations use `groupby(sym_col)`)

### Notebooks

- `marketReporting.ipynb` — market reporting
- `MarketInternalMonitor/SP500Monitor.ipynb` — S&P 500 market internals monitoring
- `MarketInternalMonitor/QuantFeatures.ipynb` — feature engineering exploration
- `MarketInternalMonitor/multiStockDayModelDatePrepare.ipynb` — multi-stock dataset preparation
- `GetFMPData/ConstructFullData.ipynb` — deprecated full data construction pipeline (superseded by `construct_full_data.py`; kept for reference)
- `GetFMPData/DataEval.ipynb` — data evaluation / QA notebook
- `BuildQuantFeatures/BuidlFeatures.ipynb` — feature build pipeline

## Key Patterns

**Building features:**
```python
from util.features.core import FeatureBuilder
from util.features.families.returns import RETURN_FAMILY

specs = RETURN_FAMILY["REALIZED_VOLATILITY"](price_col="adjClose", window=21)
builder = FeatureBuilder(panel_df)
result = builder.build_published(specs)  # {col_name: pd.Series}
```

**Async FMP bulk pull:**
```python
from util.data_client.FMPDataClient import FMPClient, fetch_fmp_for_tickers
import asyncio

client = FMPClient()  # reads FMP_API_KEY from env
data = asyncio.run(fetch_fmp_for_tickers(tickers, request_specs, client=client))
# data["AAPL"]["daily price"] -> raw JSON; errors stored as {"__error__": ...}
```

**Sync FMP single call:**
```python
from util.data_client.fmp import getFMPData
data = getFMPData("income-statement", symbol="AAPL", period="annual", limit=5)
```
