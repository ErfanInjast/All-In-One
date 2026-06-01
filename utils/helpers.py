"""
utils/helpers.py — Logging setup, rate limiter, retry decorator
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from functools import wraps
from typing import Callable


def setup_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log", encoding="utf-8"),
        ],
    )
    for noisy in ("httpx", "telegram", "ccxt", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class RateLimiter:
    def __init__(self, rate: float, per: float = 1.0):
        self._rate = rate; self._per = per
        self._tokens = rate; self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._tokens = min(self._rate, self._tokens + (now - self._last) * (self._rate / self._per))
            self._last = now
            if self._tokens < 1.0:
                await asyncio.sleep((1.0 - self._tokens) / (self._rate / self._per))
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


def async_retry(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    def decorator(fn: Callable):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(retries):
                try:
                    return await fn(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    logging.getLogger(__name__).warning(
                        "Retry %d/%d for %s: %s", attempt + 1, retries, fn.__name__, e)
                    await asyncio.sleep(wait); wait *= backoff
        return wrapper
    return decorator


def seconds_until(target_ts: float) -> str:
    secs = max(0, int(target_ts - time.time()))
    h, rem = divmod(secs, 3600); m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
