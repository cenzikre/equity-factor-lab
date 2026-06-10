import os
import asyncio
import aiohttp
from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from typing import Any, Dict, List, Optional, Tuple


class FMPError(Exception):
    pass


class FMPClient:
    """
    Robust async FMP client:
    - Rate limit: safe under 750 calls/min (use safety margin)
    - Concurrency: bounded to stay efficient
    - Retries: exponential backoff + jitter; respects Retry-After for 429 when present
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://financialmodelingprep.com/stable/",
        calls_per_second: int = 12,
        safety_margin: int = 0,          # keep below plan limit for extra safety
        concurrency: int = 25,
        timeout_s: float = 30.0,
    ):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP API key missing. Pass api_key=... or set env var FMP_API_KEY")

        self.base_url = base_url.rstrip("/") + "/"

        # aiolimiter is token-bucket-ish; we set max rate slightly below limit to be safe.
        self.limiter = AsyncLimiter(max_rate=max(1, calls_per_second - safety_margin), time_period=1)

        # Concurrency controls in-flight requests to avoid resource blowups
        self.sem = asyncio.Semaphore(concurrency)

        self.timeout = aiohttp.ClientTimeout(total=timeout_s)

    @staticmethod
    def _build_url(base_url: str, endpoint: str) -> str:
        endpoint = endpoint.lstrip("/")
        return base_url + endpoint

    @staticmethod
    def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        # allow from_ in python signature
        if "from_" in params:
            params = dict(params)
            params["from"] = params.pop("from_")
        # drop None values to keep URL clean
        return {k: v for k, v in params.items() if v is not None}

    async def _request_json(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        params: Dict[str, Any],
    ) -> Any:
        """
        One request (rate-limited + concurrency-limited).
        Raises for retry-worthy situations.
        """
        url = self._build_url(self.base_url, endpoint)
        params = self._normalize_params(params)
        params["apikey"] = self.api_key

        async with self.sem:
            # Rate limit on request start
            async with self.limiter:
                async with session.get(url, params=params) as resp:
                    # 429: too many requests
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After")
                        # If API tells us how long to wait, do it, then raise to retry
                        if retry_after:
                            try:
                                await asyncio.sleep(float(retry_after))
                            except ValueError:
                                # if header is weird, fall back to retry/backoff
                                pass
                        raise FMPError(f"429 Too Many Requests for {endpoint} params={params}")

                    # Retry on transient 5xx
                    if 500 <= resp.status < 600:
                        text = await resp.text()
                        raise FMPError(f"{resp.status} Server error: {text[:200]}")

                    # Hard fail on other 4xx (usually bad params)
                    if 400 <= resp.status < 500:
                        text = await resp.text()
                        raise ValueError(f"{resp.status} Client error: {text[:200]}")

                    # Parse JSON
                    try:
                        return await resp.json(content_type=None)
                    except Exception as e:
                        text = await resp.text()
                        raise FMPError(f"JSON parse error: {e}. Body starts: {text[:200]}")

    # Tenacity retry wrapper: retries only on our FMPError (429/5xx/parse issues)
    @retry(
        retry=retry_if_exception_type(FMPError),
        wait=wait_exponential_jitter(initial=1, max=30),  # backoff + jitter
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async def fetch(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        params: Dict[str, Any],
    ) -> Any:
        return await self._request_json(session, endpoint, params)


async def fetch_fmp_for_tickers(
    tickers: List[str],
    request_specs: Dict[str, Dict[str, Any]],
    *,
    client: Optional[FMPClient] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    tickers: ["AAPL", "MSFT", ...]
    request_specs:
      {
        "daily price": {"endpoint": "historical-price-full", "params": {"symbol": "{ticker}", "from_": "2024-01-01"}},
        "income statement": {"endpoint": "income-statement", "params": {"symbol": "{ticker}", "period": "annual"}}
      }

    Returns:
      { "AAPL": {"daily price": <json>, "income statement": <json>}, ... }
    """
    if client is None:
        client = FMPClient()

    # Pre-build tasks with metadata so we can place results correctly
    tasks: List[Tuple[str, str, asyncio.Task]] = []

    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        for ticker in tickers:
            t = ticker.strip().upper()
            for info_type, spec in request_specs.items():
                endpoint = spec["endpoint"]
                params = dict(spec.get("params", {}))

                # allow "{ticker}" substitution in any param value
                for k, v in list(params.items()):
                    if isinstance(v, str) and "{ticker}" in v:
                        params[k] = v.replace("{ticker}", t)

                # common FMP convention is 'symbol='
                # If user didn't specify symbol, try to set it.
                if "symbol" not in params and "ticker" not in params:
                    params["symbol"] = t

                task = asyncio.create_task(client.fetch(session, endpoint, params))
                tasks.append((t, info_type, task))

        results: Dict[str, Dict[str, Any]] = {ticker.strip().upper(): {} for ticker in tickers}

        # Gather but keep failures attached to keys (no silent data loss)
        gathered = await asyncio.gather(*(t[2] for t in tasks), return_exceptions=True)

        for (ticker, info_type, _task), outcome in zip(tasks, gathered):
            if isinstance(outcome, Exception):
                # Save error so you never "lose" information silently
                results[ticker][info_type] = {"__error__": repr(outcome)}
            else:
                results[ticker][info_type] = outcome

        return results


async def fetch_fmp_all_pages(
    endpoint: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    client: Optional[FMPClient] = None,
    page_param: str = "page",
    start_page: int = 0,
    limit: int = 10000,
    limit_param: str = "limit",
    max_pages: int = 1000,
    verbose: bool = False,
) -> List[Any]:
    """
    Collect all records from a paginated FMP list endpoint.

    Fetches pages sequentially (each page depends on the previous not being
    empty), using the client's rate limiter and retry logic on every request.
    Stops when a page returns an empty response or max_pages is reached.

    Args:
        endpoint:    FMP endpoint, e.g. "delisted-companies"
        params:      Extra query params (besides page/limit/apikey)
        client:      FMPClient instance; created from env if omitted
        page_param:  Query param name for page number (default "page")
        start_page:  First page index (default 0)
        limit:       Records per page
        limit_param: Query param name for page size (default "limit")
        max_pages:   Safety cap on total pages (default 1000)
        verbose:     Print per-page progress when True

    Returns:
        Flat list of all records across all pages.
    """
    if client is None:
        client = FMPClient()
    if params is None:
        params = {}

    all_records: List[Any] = []

    async with aiohttp.ClientSession(timeout=client.timeout) as session:
        for page in range(start_page, start_page + max_pages):
            page_params = {**params, page_param: page, limit_param: limit}
            result = await client.fetch(session, endpoint, page_params)

            if not result:
                break

            all_records.extend(result)
            if verbose:
                print(f"  [{endpoint}] page {page}: +{len(result):,} records (total {len(all_records):,})")

    return all_records


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    async def main():
        tickers = ["AAPL", "MSFT"]

        request_specs = {
            "daily price": {
                "endpoint": "historical-price-full",
                "params": {
                    "symbol": "{ticker}",
                    "from_": "2024-01-01",
                    "to": "2024-12-31",
                },
            },
            "income statement": {
                "endpoint": "income-statement",
                "params": {"symbol": "{ticker}", "period": "annual", "limit": 5},
            },
        }

        client = FMPClient(
            api_key="YOUR_KEY_HERE",    # better: set env var FMP_API_KEY
            calls_per_second=12,
            safety_margin=1,            # request starts capped at 740/min
            concurrency=25,             # tune based on latency & CPU/network
            timeout_s=30,
        )

        data = await fetch_fmp_for_tickers(tickers, request_specs, client=client)
        print(data["AAPL"].keys())
        # print(data)

    asyncio.run(main())
