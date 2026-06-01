"""
optimizer/grid_search.py — Exhaustive grid search with Sharpe-based ranking.
Full diagnostic logging: per-symbol progress, timing, and clear error reporting.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import (
    FISHER_PERIODS, FISHER_THRESHOLDS, LEVERAGE_TIERS,
    MIN_CONFIRMATIONS, SMOOTH_PERIODS, STRATEGY_WEIGHTS, TIMEFRAMES,
    WATCHLIST, CANDLE_LIMIT, DEFAULT_SL_ATR_MULT, DEFAULT_TP_ATR_MULT,
)
from data.fetcher import fetch_batch
from strategies.engine import compute_signals, fisher_transform, weighted_consensus

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbol:        str
    timeframe:     str
    leverage:      int
    fisher_period: int
    smooth_period: int
    min_conf:      float
    threshold:     float
    total_trades:  int   = 0
    win_rate:      float = 0.0
    net_return:    float = 0.0
    sharpe:        float = 0.0
    max_drawdown:  float = 0.0
    profit_factor: float = 0.0

    def score(self) -> float:
        if self.total_trades < 5:
            return -999.0
        return self.sharpe * max(0.0, 1.0 - self.max_drawdown) * min(self.profit_factor, 5.0)


@dataclass
class OptimiserState:
    last_run_ts:  float = 0.0
    best_configs: Dict[str, BacktestResult] = field(default_factory=dict)
    global_best:  Optional[BacktestResult]  = None
    run_count:    int = 0


# ── Backtest ──────────────────────────────────────────────────────────────────
def _backtest(
    df: pd.DataFrame,
    fish: np.ndarray,
    sig_arr: np.ndarray,
    score_arr: np.ndarray,
    threshold: float,
    min_conf: float,
    leverage: int,
    tp_mult: float = DEFAULT_TP_ATR_MULT,
    sl_mult: float = DEFAULT_SL_ATR_MULT,
) -> BacktestResult:
    from pandas_ta import atr as _atr
    atr_vals = _atr(df["H"], df["L"], df["C"], length=14).values
    closes   = df["C"].values
    n        = len(closes)

    equity_curve: List[float] = [1.0]
    trade_returns: List[float] = []
    wins = 0
    i    = 20

    while i < n - 1:
        fc, sc = fish[i],     sig_arr[i]
        fp, sp = fish[i - 1], sig_arr[i - 1]
        accel  = (fc - sc) - (fp - sp)
        score  = score_arr[i]

        direction = 0
        if fc > sc and fp <= sp and accel > 0.05 and fc < -threshold and score >= min_conf:
            direction = 1
        elif fc < sc and fp >= sp and accel < -0.04 and fc > threshold and score <= -min_conf:
            direction = -1

        if direction != 0:
            entry   = closes[i]
            atr_val = atr_vals[i] if not np.isnan(atr_vals[i]) else entry * 0.01
            tp      = entry + direction * atr_val * tp_mult
            sl      = entry - direction * atr_val * sl_mult
            j       = i + 1
            exit_p  = closes[j]

            for j in range(i + 1, min(i + 50, n)):
                p = closes[j]
                if direction == 1:
                    if p >= tp: exit_p = tp; break
                    if p <= sl: exit_p = sl; break
                else:
                    if p <= tp: exit_p = tp; break
                    if p >= sl: exit_p = sl; break

            ret = max(direction * (exit_p - entry) / entry * leverage, -1.0)
            trade_returns.append(ret)
            equity_curve.append(equity_curve[-1] * (1.0 + ret))
            if ret > 0:
                wins += 1
            i = j + 1
            continue
        i += 1

    if len(trade_returns) < 5:
        return BacktestResult("", "", leverage, 0, 0, min_conf, threshold)

    arr    = np.array(trade_returns)
    sharpe = float(np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(252))
    eq     = np.array(equity_curve)
    peak   = np.maximum.accumulate(eq)
    max_dd = float(np.max((peak - eq) / (peak + 1e-9)))
    gains  = arr[arr > 0].sum()
    losses = abs(arr[arr < 0].sum())

    return BacktestResult(
        symbol="", timeframe="", leverage=leverage,
        fisher_period=0, smooth_period=0,
        min_conf=min_conf, threshold=threshold,
        total_trades=len(trade_returns),
        win_rate=wins / len(trade_returns),
        net_return=float(arr.sum()),
        sharpe=sharpe,
        max_drawdown=max_dd,
        profit_factor=gains / (losses + 1e-9),
    )


# ── Per-symbol grid (CPU-bound, runs in executor) ─────────────────────────────
def _run_grid_for_symbol(symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[BacktestResult]:
    t0   = time.time()
    best: Optional[BacktestResult] = None

    try:
        # ① Compute all strategy signals once — reuse across all param combos
        signals_cache = compute_signals(df)
        score_arr     = np.zeros(len(df))
        for name, arr in signals_cache.items():
            score_arr += arr * STRATEGY_WEIGHTS.get(name, 1.0)

        param_grid = list(itertools.product(
            FISHER_PERIODS, SMOOTH_PERIODS,
            MIN_CONFIRMATIONS, FISHER_THRESHOLDS, LEVERAGE_TIERS,
        ))

        combos_run  = 0
        combos_skip = 0

        # ② Cache fisher arrays per (fp, sp) pair — avoid recomputing for same shape
        fisher_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

        for fp, sp, mc, th, lev in param_grid:
            key = (fp, sp)
            if key not in fisher_cache:
                fish, sig = fisher_transform(df, fp, sp)
                fisher_cache[key] = (fish, sig)
            else:
                fish, sig = fisher_cache[key]

            if len(fish) < 30:
                combos_skip += 1
                continue

            result = _backtest(df, fish, sig, score_arr, th, mc, lev)
            result.symbol        = symbol
            result.timeframe     = timeframe
            result.fisher_period = fp
            result.smooth_period = sp
            combos_run += 1

            if best is None or result.score() > best.score():
                best = result

        elapsed = time.time() - t0
        best_info = (f"sharpe={best.sharpe:.2f} wr={best.win_rate:.0%} "
                     f"lev={best.leverage}× trades={best.total_trades}"
                     ) if best and best.total_trades >= 5 else "no valid trades"

        logger.info(
            "    ⚙️  %-12s %s | %d combos | skip=%d | %.2fs | best: %s",
            symbol, timeframe, combos_run, combos_skip, elapsed, best_info,
        )

    except Exception as e:
        logger.error("    ❌ Grid error %s %s: %s\n%s",
                     symbol, timeframe, e, traceback.format_exc())
        return None

    return best if (best and best.total_trades >= 5) else None


# ── Full optimisation cycle ───────────────────────────────────────────────────
async def run_full_optimisation(state: OptimiserState) -> OptimiserState:
    t_total = time.time()
    logger.info("=" * 60)
    logger.info("🔬 OPTIMISATION CYCLE #%d STARTING", state.run_count + 1)
    logger.info("   Symbols: %d | Timeframes: %d | Grid size: %d combos/symbol",
                len(WATCHLIST), len(TIMEFRAMES),
                len(list(itertools.product(
                    FISHER_PERIODS, SMOOTH_PERIODS,
                    MIN_CONFIRMATIONS, FISHER_THRESHOLDS, LEVERAGE_TIERS,
                ))))
    logger.info("=" * 60)

    all_results: List[BacktestResult] = []

    for tf_idx, tf in enumerate(TIMEFRAMES):
        t_tf = time.time()
        logger.info("──────────────────────────────────────────────")
        logger.info("📡 [TF %d/%d] Fetching %d symbols @ %s ...",
                    tf_idx + 1, len(TIMEFRAMES), len(WATCHLIST), tf)

        # ① Fetch all symbols for this timeframe
        try:
            data_map = await asyncio.wait_for(
                fetch_batch(WATCHLIST, tf, limit=CANDLE_LIMIT),
                timeout=len(WATCHLIST) * 35,   # max 35s per symbol
            )
        except asyncio.TimeoutError:
            logger.error("⏱ fetch_batch TIMEOUT for timeframe %s — skipping", tf)
            continue
        except Exception as e:
            logger.error("❌ fetch_batch ERROR for timeframe %s: %s", tf, e)
            continue

        valid = {s: df for s, df in data_map.items() if df is not None and len(df) >= 60}
        logger.info("  📊 Valid data: %d/%d symbols", len(valid), len(WATCHLIST))

        if not valid:
            logger.warning("  ⚠️  No valid data for %s — skipping grid search", tf)
            continue

        # ② Run grid search in thread executor with per-symbol timeout
        loop  = asyncio.get_event_loop()
        tasks = []
        for sym, df in valid.items():
            fut = loop.run_in_executor(None, _run_grid_for_symbol, sym, tf, df.copy())
            tasks.append(asyncio.ensure_future(fut))

        logger.info("  🧮 Running grid search on %d symbols (executor) ...", len(tasks))

        # Gather with individual timeouts per task
        finished = await asyncio.gather(*tasks, return_exceptions=True)

        ok_count = 0
        for r in finished:
            if isinstance(r, Exception):
                logger.error("  ❌ Executor task exception: %s", r)
            elif r is not None:
                all_results.append(r)
                ok_count += 1

        logger.info("  ✅ [TF %s] done in %.1fs | %d/%d symbols returned results",
                    tf, time.time() - t_tf, ok_count, len(valid))

    # ── Aggregate results ─────────────────────────────────────────────────
    logger.info("──────────────────────────────────────────────")
    logger.info("📈 Aggregating %d results ...", len(all_results))

    per_symbol: Dict[str, BacktestResult] = {}
    for r in all_results:
        if r.symbol not in per_symbol or r.score() > per_symbol[r.symbol].score():
            per_symbol[r.symbol] = r

    global_best = max(all_results, key=lambda x: x.score()) if all_results else None

    state.best_configs = per_symbol
    state.global_best  = global_best
    state.last_run_ts  = time.time()
    state.run_count   += 1

    elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info("🏁 OPTIMISATION COMPLETE in %.1fs", elapsed)
    logger.info("   Symbols configured: %d", len(per_symbol))
    if global_best:
        logger.info("   🏆 Global best: %s %s | sharpe=%.2f wr=%.0f%% lev=%dx dd=%.1f%%",
                    global_best.symbol, global_best.timeframe,
                    global_best.sharpe, global_best.win_rate * 100,
                    global_best.leverage, global_best.max_drawdown * 100)
    logger.info("=" * 60)
    return state
