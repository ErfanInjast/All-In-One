"""
core/scanner.py — Real-time signal scanner with TP/SL management
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np
import pandas_ta as ta

from config.settings import (
    DEFAULT_SL_ATR_MULT, DEFAULT_TP_ATR_MULT,
    SIGNAL_SCAN_INTERVAL_SECONDS, STRATEGY_WEIGHTS, WATCHLIST,
)
from core.signal_state import Direction, Signal, SignalRegistry
from data.fetcher import fetch_candles
from optimizer.grid_search import BacktestResult, OptimiserState
from strategies.engine import compute_signals, fisher_transform, weighted_consensus

logger = logging.getLogger(__name__)


def _get_config(symbol: str, opt_state: OptimiserState) -> BacktestResult:
    cfg = opt_state.best_configs.get(symbol)
    if cfg:
        return cfg
    return BacktestResult(
        symbol=symbol, timeframe="15m", leverage=1,
        fisher_period=9, smooth_period=5,
        min_conf=3.0, threshold=0.10,
    )


def _calc_tp_sl(entry: float, direction: int, df,
                tp_mult: float = DEFAULT_TP_ATR_MULT,
                sl_mult: float = DEFAULT_SL_ATR_MULT):
    atr = ta.atr(df["H"], df["L"], df["C"], length=14)
    atr_val = float(atr.iloc[-1]) if atr is not None and not atr.empty else entry * 0.01
    tp = entry + direction * atr_val * tp_mult
    sl = entry - direction * atr_val * sl_mult
    return tp, sl, atr_val


async def _evaluate_symbol(
    symbol: str,
    opt_state: OptimiserState,
    registry: SignalRegistry,
) -> Optional[Signal]:
    cfg = _get_config(symbol, opt_state)

    df = await fetch_candles(symbol, cfg.timeframe, limit=250)
    if df is None or len(df) < 50:
        return None

    fish, sig_arr = fisher_transform(df, cfg.fisher_period, cfg.smooth_period)
    if len(fish) < 2:
        return None

    # ── check exit for existing open position ──────────────────────────────
    open_sig = registry.get_open(symbol)
    if open_sig is not None:
        last_close = float(df["C"].iloc[-1])
        hit_tp = (
            (open_sig.direction == Direction.LONG  and last_close >= open_sig.tp_price) or
            (open_sig.direction == Direction.SHORT and last_close <= open_sig.tp_price)
        )
        hit_sl = (
            (open_sig.direction == Direction.LONG  and last_close <= open_sig.sl_price) or
            (open_sig.direction == Direction.SHORT and last_close >= open_sig.sl_price)
        )
        fisher_reversal = (
            (open_sig.direction == Direction.LONG  and fish[-1] < sig_arr[-1] and fish[-2] >= sig_arr[-2]) or
            (open_sig.direction == Direction.SHORT and fish[-1] > sig_arr[-1] and fish[-2] <= sig_arr[-2])
        )
        if hit_tp or hit_sl or fisher_reversal:
            reason = "TP" if hit_tp else ("SL" if hit_sl else "Fisher Reversal")
            closed = registry.close_signal(symbol, last_close)
            if closed:
                logger.info("Closed %s %s @ %.6g (%s) | PnL %+.2f%%",
                            symbol, open_sig.direction.value, last_close,
                            reason, (closed.pnl_pct or 0) * 100)
        return None

    # ── check entry ────────────────────────────────────────────────────────
    accel = (fish[-1] - sig_arr[-1]) - (fish[-2] - sig_arr[-2])

    all_signals = compute_signals(df)
    consensus, max_consensus = weighted_consensus(all_signals, STRATEGY_WEIGHTS)

    long_cond = (
        fish[-1] > sig_arr[-1] and fish[-2] <= sig_arr[-2] and
        accel > 0.05 and fish[-1] < -cfg.threshold and consensus >= cfg.min_conf
    )
    short_cond = (
        fish[-1] < sig_arr[-1] and fish[-2] >= sig_arr[-2] and
        accel < -0.04 and fish[-1] > cfg.threshold and consensus <= -cfg.min_conf
    )

    if not (long_cond or short_cond):
        return None

    direction = Direction.LONG if long_cond else Direction.SHORT
    dir_int   = 1 if direction == Direction.LONG else -1
    entry     = float(df["C"].iloc[-1])
    tp, sl, _ = _calc_tp_sl(entry, dir_int, df)

    new_sig = Signal(
        id="",
        symbol=symbol,
        timeframe=cfg.timeframe,
        direction=direction,
        entry_price=entry,
        tp_price=tp,
        sl_price=sl,
        leverage=cfg.leverage,
        fisher_val=float(fish[-1]),
        accel=float(accel),
        consensus=float(consensus),
        max_consensus=float(max_consensus),
    )
    registry.add_signal(new_sig)
    logger.info("New signal: %s %s x%d @ %.6g | Consensus %.1f/%.1f",
                symbol, direction.value, cfg.leverage, entry,
                consensus, max_consensus)
    return new_sig


async def run_scanner(
    opt_state: OptimiserState,
    registry: SignalRegistry,
    signal_callback,
    stop_event: asyncio.Event,
):
    logger.info("Scanner started — interval %ds", SIGNAL_SCAN_INTERVAL_SECONDS)

    while not stop_event.is_set():
        t0 = time.time()
        # cap matches _EXECUTOR max_workers in fetcher — no point queuing more
        sem = asyncio.Semaphore(6)

        async def _safe_eval(sym: str):
            async with sem:
                return await _evaluate_symbol(sym, opt_state, registry)

        results = await asyncio.gather(*[_safe_eval(s) for s in WATCHLIST])
        new_signals = [r for r in results if r is not None]

        for sig in new_signals:
            try:
                await signal_callback(sig)
            except Exception as e:
                logger.error("Signal callback error: %s", e)

        elapsed = time.time() - t0
        wait    = max(0, SIGNAL_SCAN_INTERVAL_SECONDS - elapsed)
        logger.info("Scan done in %.1fs | %d symbols | %d new signals | sleep %.0fs",
                    elapsed, len(WATCHLIST), len(new_signals), wait)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass

    logger.info("Scanner stopped.")
