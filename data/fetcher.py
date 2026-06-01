"""
data/fetcher.py — Async market data fetcher with per-symbol timeout + diagnostic logging
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# ── LRU cache ──────────────────────────────────────────────────────────────────
_CACHE: OrderedDict[str, Tuple[pd.DataFrame, float]] = OrderedDict()
_CACHE_TTL  = 55
_CACHE_MAX  = 200
_cache_lock = threading.Lock()

FETCH_TIMEOUT_SECONDS = 30   # hard per-symbol timeout


def _cache_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}::{timeframe}"


def _get_cache(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    key = _cache_key(symbol, timeframe)
    with _cache_lock:
        if key in _CACHE:
            df, ts = _CACHE[key]
            if time.time() - ts < _CACHE_TTL:
                _CACHE.move_to_end(key)
                return df.copy()
    return None


def _set_cache(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    key = _cache_key(symbol, timeframe)
    with _cache_lock:
        _CACHE[key] = (df.copy(), time.time())
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


# ── Thread-local exchange ─────────────────────────────────────────────────────
_thread_local = threading.local()


def _get_exchange(exchange_id: str = "binance") -> ccxt.Exchange:
    attr = f"exchange_{exchange_id}"
    exc  = getattr(_thread_local, attr, None)
    if exc is None:
        cls = getattr(ccxt, exchange_id)
        exc = cls({
            "enableRateLimit": True,
            "timeout": 20000,        # 20s socket timeout (ms) — ccxt native
        })
        setattr(_thread_local, attr, exc)
    return exc


# ── Bounded thread pool ───────────────────────────────────────────────────────
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=6,
    thread_name_prefix="fetcher",
)


def _fetch_sync(symbol: str, timeframe: str, limit: int, exchange_id: str) -> Optional[pd.DataFrame]:
    exc   = _get_exchange(exchange_id)
    ohlcv = exc.fetch_ohlcv(symbol, timeframe, limit=limit)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv, columns=["Timestamp", "O", "H", "L", "C", "V"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df.set_index("Timestamp").sort_index()
    return df.iloc[:-1]


# ── Core fetch with hard timeout ──────────────────────────────────────────────
async def fetch_candles(
    symbol: str,
    timeframe: str,
    limit: int = 350,
    exchange_id: str = "binance",
    use_cache: bool = True,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    if use_cache:
        cached = _get_cache(symbol, timeframe)
        if cached is not None:
            return cached

    loop      = asyncio.get_event_loop()
    last_exc: Optional[Exception] = None

    for attempt in range(retries):
        try:
            # Hard timeout per attempt — never hangs forever
            df = await asyncio.wait_for(
                loop.run_in_executor(
                    _EXECUTOR,
                    _fetch_sync, symbol, timeframe, limit, exchange_id,
                ),
                timeout=FETCH_TIMEOUT_SECONDS,
            )
            if df is None:
                return None
            if use_cache:
                _set_cache(symbol, timeframe, df)
            return df

        except asyncio.TimeoutError:
            logger.warning("⏱ TIMEOUT (%ds) fetching %s %s (attempt %d/%d)",
                           FETCH_TIMEOUT_SECONDS, symbol, timeframe, attempt + 1, retries)
            last_exc = TimeoutError(f"fetch timeout {symbol} {timeframe}")
            await asyncio.sleep(2.0 ** attempt)

        except ccxt.NetworkError as e:
            last_exc = e
            wait = 2.0 ** attempt
            logger.warning("🌐 Network error %s %s (attempt %d/%d) retry in %.0fs: %s",
                           symbol, timeframe, attempt + 1, retries, wait, e)
            await asyncio.sleep(wait)

        except ccxt.RateLimitExceeded as e:
            wait = 5.0 * (attempt + 1)
            logger.warning("🚦 Rate limit %s — waiting %.0fs", symbol, wait)
            await asyncio.sleep(wait)
            last_exc = e

        except ccxt.ExchangeError as e:
            logger.error("❌ Exchange error %s %s: %s", symbol, timeframe, e)
            return None

    logger.error("💀 Gave up fetching %s %s after %d attempts: %s",
                 symbol, timeframe, retries, last_exc)
    return None


# ── Batch fetch with per-symbol timeout + progress ───────────────────────────
async def fetch_batch(
    symbols: List[str],
    timeframe: str,
    limit: int = 350,
    max_concurrent: int = 6,
) -> Dict[str, Optional[pd.DataFrame]]:
    sem   = asyncio.Semaphore(max_concurrent)
    total = len(symbols)
    done  = 0
    t0    = time.time()

    async def _safe(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
        nonlocal done
        async with sem:
            df = await fetch_candles(sym, timeframe, limit=limit)
            done += 1
            elapsed = time.time() - t0
            logger.info(
                "  📥 [%d/%d] %-12s %s — %s in %.1fs",
                done, total, sym, timeframe,
                f"{len(df)} candles" if df is not None else "FAILED",
                elapsed,
            )
            return sym, df

    results = await asyncio.gather(*[_safe(s) for s in symbols])
    ok  = sum(1 for _, df in results if df is not None)
    logger.info("  ✅ fetch_batch done: %d/%d symbols OK | %.1fs", ok, total, time.time() - t0)
    return {sym: df for sym, df in results}
