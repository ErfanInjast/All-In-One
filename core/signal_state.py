"""
core/signal_state.py — Thread-safe signal registry and performance tracker
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Direction(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


@dataclass
class Signal:
    id:           str
    symbol:       str
    timeframe:    str
    direction:    Direction
    entry_price:  float
    tp_price:     float
    sl_price:     float
    leverage:     int
    fisher_val:   float
    accel:        float
    consensus:    float
    max_consensus: float
    timestamp:    float = field(default_factory=time.time)
    is_open:      bool  = True
    exit_price:   Optional[float] = None
    exit_time:    Optional[float] = None
    pnl_pct:      Optional[float] = None   # levered

    @property
    def risk_reward(self) -> float:
        try:
            reward = abs(self.tp_price - self.entry_price)
            risk   = abs(self.sl_price - self.entry_price)
            return reward / (risk + 1e-9)
        except ZeroDivisionError:
            return 0.0


@dataclass
class PerformanceStats:
    total_signals: int   = 0
    open_signals:  int   = 0
    total_closed:  int   = 0
    wins:          int   = 0
    losses:        int   = 0
    total_pnl:     float = 0.0
    max_drawdown:  float = 0.0
    best_trade:    float = 0.0
    worst_trade:   float = 0.0
    _peak_equity:  float = 1.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_closed if self.total_closed else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total_closed if self.total_closed else 0.0

    @property
    def profit_factor(self) -> float:
        return (self.total_pnl / max(abs(self.total_pnl), 1e-9)
                if self.total_pnl > 0 else 0.0)


class SignalRegistry:
    """In-memory registry — signal history, open positions, performance."""

    def __init__(self, max_history: int = 50):
        self._max_history = max_history
        self._open:   Dict[str, Signal] = {}      # symbol → latest open signal
        self._history: List[Signal]     = []
        self.stats    = PerformanceStats()
        self._counter = 0

    # ── create ──────────────────────────────────────────────────────────────
    def add_signal(self, sig: Signal) -> None:
        sig.id = f"SIG-{self._counter:05d}"
        self._counter += 1
        self._open[sig.symbol] = sig
        self._history.append(sig)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self.stats.total_signals += 1
        self.stats.open_signals  += 1

    # ── close ────────────────────────────────────────────────────────────────
    def close_signal(self, symbol: str, exit_price: float) -> Optional[Signal]:
        sig = self._open.pop(symbol, None)
        if sig is None:
            return None

        sig.is_open    = False
        sig.exit_price = exit_price
        sig.exit_time  = time.time()

        direction_mult = 1 if sig.direction == Direction.LONG else -1
        raw_ret = direction_mult * (exit_price - sig.entry_price) / sig.entry_price
        sig.pnl_pct = max(raw_ret * sig.leverage, -1.0)

        self.stats.total_closed += 1
        self.stats.open_signals  = max(0, self.stats.open_signals - 1)
        self.stats.total_pnl    += sig.pnl_pct

        if sig.pnl_pct > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        self.stats.best_trade  = max(self.stats.best_trade,  sig.pnl_pct)
        self.stats.worst_trade = min(self.stats.worst_trade, sig.pnl_pct)

        # running drawdown
        new_eq = self.stats._peak_equity * (1.0 + sig.pnl_pct)
        if new_eq > self.stats._peak_equity:
            self.stats._peak_equity = new_eq
        else:
            dd = (self.stats._peak_equity - new_eq) / self.stats._peak_equity
            self.stats.max_drawdown = max(self.stats.max_drawdown, dd)

        return sig

    # ── queries ──────────────────────────────────────────────────────────────
    def has_open(self, symbol: str) -> bool:
        return symbol in self._open

    def get_open(self, symbol: str) -> Optional[Signal]:
        return self._open.get(symbol)

    def all_open(self) -> List[Signal]:
        return list(self._open.values())

    def recent_history(self, n: int = 10) -> List[Signal]:
        return list(reversed(self._history[-n:]))

    def symbol_performance(self, symbol: str) -> Dict:
        trades = [s for s in self._history if s.symbol == symbol and not s.is_open]
        if not trades:
            return {}
        pnls = [t.pnl_pct for t in trades if t.pnl_pct is not None]
        wins = sum(1 for p in pnls if p > 0)
        return {
            "trades":     len(trades),
            "win_rate":   wins / len(trades),
            "total_pnl":  sum(pnls),
            "avg_pnl":    sum(pnls) / len(pnls) if pnls else 0,
        }
