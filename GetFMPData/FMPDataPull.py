import json
import asyncio
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from FMPDataClient import FMPClient, fetch_fmp_for_tickers
from dataPullHelpers import normalize_df_for_parquet

load_dotenv()


async def main():
    # tickers = [
    #     "AAPL",
    #     "MSFT",
    # ]

    path = 'MarketInternalMonitor/universe/info-stock-universe-usTrading-delistedIncl.csv'
    stocks_df = pd.read_csv(path)
    tickers = stocks_df['symbol'].tolist()
    print(f"{len(tickers)} tickers read, data pull start")

    request_specs = {
        # "adjusted daily price": {
        #     "endpoint": "historical-price-eod/dividend-adjusted",
        #     "params": {
        #         "symbol": "{ticker}",
        #         "from": "2020-01-01",
        #         "to": "2025-12-31",
        #     },
        # },
        # "unadjusted daily price": {
        #     "endpoint": "historical-price-eod/non-split-adjusted",
        #     "params": {
        #         "symbol": "{ticker}",
        #         "from": "2020-01-01",
        #         "to": "2025-12-31",
        #     },
        # },
        # "income statement": {
        #     "endpoint": "income-statement",
        #     "params": {"symbol": "{ticker}", "period": "quarter", "limit": 150},
        # },
        "balance sheet": {
            "endpoint": "balance-sheet-statement",
            "params": {"symbol": "{ticker}", "period": "quarter", "limit": 150},
        },
        "cash flow": {
            "endpoint": "cash-flow-statement",
            "params": {"symbol": "{ticker}", "period": "quarter", "limit": 150},
        },
        "key metrics": {
            "endpoint": "key-metrics",
            "params": {"symbol": "{ticker}", "period": "quarter", "limit": 150},
        },
        "enterprise values": {
            "endpoint": "enterprise-values",
            "params": {"symbol": "{ticker}", "period": "quarter", "limit": 150},
        },
    }

    client = FMPClient(
        calls_per_second=12,
        safety_margin=0,
        concurrency=25,
        timeout_s=30,
    )

    data = await fetch_fmp_for_tickers(tickers, request_specs, client=client)
    with open("GetFMPData/raw_data/raw_data_temp.json", "w", encoding='utf-8') as f:
        json.dump(data, f)
    print(f"{len(data)} tickers pulled, normalization start")

    _datasets, errorLog = defaultdict(list), defaultdict(dict)
    for tk, _val in data.items():
        for info, _data in _val.items():
            if isinstance(_data, dict) and '__error__' in _data:
                errorLog[tk][info] = _data['__error__']
            else:
                df_new = pd.DataFrame(_data)
                df_new = normalize_df_for_parquet(
                    df_new,
                    datetime_success_ratio=0.9,
                    numeric_success_ratio=0.9,
                    treat_huge_int_as_string=False
                )
                df_new = df_new.dropna(axis=1, how="all")
                if df_new.empty:
                    continue
                _datasets[info].append(df_new)

    _datasets_final = {}
    for info, dfs in _datasets.items():
        if not dfs:
            continue
        _datasets_final[info] = pd.concat(dfs, ignore_index=True)
    print(f"{len(_datasets)} tables created, serialization start")

    for info, _dataset in _datasets_final.items():
        _dataset.to_parquet(
            f"GetFMPData/data/data_{info.replace(' ', '')}_tk0_pd1.parquet",
            compression='zstd',
            index=False
        )

    with open("GetFMPData/error_log.json", "w", encoding='utf-8') as f:
        json.dump(errorLog, f)
    print("all tasks completed")

asyncio.run(main())



