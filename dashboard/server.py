"""
dashboard/server.py — FastAPI + WebSocket real-time dashboard server
Runs alongside the trading bot and streams live state to the browser.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from core.signal_state import Direction, SignalRegistry
from optimizer.grid_search import OptimiserState

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket connection manager
# ──────────────────────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self._connections: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        logger.info("WS client connected (%d total)", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        logger.info("WS client disconnected (%d total)", len(self._connections))

    async def broadcast(self, data: Dict[str, Any]) -> None:
        if not self._connections:
            return
        payload = json.dumps(data)
        dead: Set[WebSocket] = set()
        for ws in self._connections.copy():
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections.discard(ws)


_manager = ConnectionManager()


# ──────────────────────────────────────────────────────────────────────────────
# State serializers
# ──────────────────────────────────────────────────────────────────────────────
def _serialize_state(registry: SignalRegistry, opt_state: OptimiserState) -> Dict:
    s = registry.stats

    # open positions
    open_pos = []
    for sig in registry.all_open():
        open_pos.append({
            "id":         sig.id,
            "symbol":     sig.symbol,
            "direction":  sig.direction.value,
            "leverage":   sig.leverage,
            "entry":      sig.entry_price,
            "tp":         sig.tp_price,
            "sl":         sig.sl_price,
            "timeframe":  sig.timeframe,
            "fisher":     round(sig.fisher_val, 4),
            "accel":      round(sig.accel, 4),
            "consensus":  round(sig.consensus, 2),
            "maxConsen":  round(sig.max_consensus, 2),
            "rr":         round(sig.risk_reward, 2),
            "timestamp":  sig.timestamp,
        })

    # recent signals history
    history = []
    for sig in registry.recent_history(50):
        history.append({
            "id":        sig.id,
            "symbol":    sig.symbol,
            "direction": sig.direction.value,
            "leverage":  sig.leverage,
            "entry":     sig.entry_price,
            "exit":      sig.exit_price,
            "pnl":       round(sig.pnl_pct * 100, 2) if sig.pnl_pct is not None else None,
            "rr":        round(sig.risk_reward, 2),
            "timeframe": sig.timeframe,
            "isOpen":    sig.is_open,
            "timestamp": sig.timestamp,
            "exitTime":  sig.exit_time,
        })

    # optimizer top configs
    opt_configs = []
    if opt_state.best_configs:
        for r in sorted(opt_state.best_configs.values(),
                        key=lambda x: x.score(), reverse=True)[:30]:
            opt_configs.append({
                "symbol":    r.symbol,
                "timeframe": r.timeframe,
                "leverage":  r.leverage,
                "fp":        r.fisher_period,
                "sp":        r.smooth_period,
                "sharpe":    round(r.sharpe, 3),
                "winRate":   round(r.win_rate * 100, 1),
                "pf":        round(r.profit_factor, 2),
                "maxDD":     round(r.max_drawdown * 100, 1),
                "trades":    r.total_trades,
                "score":     round(r.score(), 3),
            })

    # pnl curve (from history, closed trades)
    pnl_curve = []
    running = 0.0
    for sig in reversed(registry.recent_history(50)):
        if not sig.is_open and sig.pnl_pct is not None:
            running += sig.pnl_pct * 100
            pnl_curve.append({
                "t":   sig.exit_time or sig.timestamp,
                "pnl": round(running, 2),
                "sym": sig.symbol,
            })

    next_opt = opt_state.last_run_ts + 43200
    secs_left = max(0, int(next_opt - time.time()))

    return {
        "type": "state",
        "ts":   time.time(),
        "stats": {
            "totalSignals": s.total_signals,
            "openSignals":  s.open_signals,
            "totalClosed":  s.total_closed,
            "wins":         s.wins,
            "losses":       s.losses,
            "winRate":      round(s.win_rate * 100, 1),
            "totalPnl":     round(s.total_pnl * 100, 2),
            "bestTrade":    round(s.best_trade * 100, 2),
            "worstTrade":   round(s.worst_trade * 100, 2),
            "maxDrawdown":  round(s.max_drawdown * 100, 2),
            "avgPnl":       round(s.avg_pnl * 100, 2),
        },
        "optimizer": {
            "runCount":    opt_state.run_count,
            "lastRun":     opt_state.last_run_ts,
            "secsUntilNext": secs_left,
            "configCount": len(opt_state.best_configs),
            "globalBest": {
                "symbol":   opt_state.global_best.symbol,
                "tf":       opt_state.global_best.timeframe,
                "sharpe":   round(opt_state.global_best.sharpe, 2),
                "winRate":  round(opt_state.global_best.win_rate * 100, 1),
                "leverage": opt_state.global_best.leverage,
            } if opt_state.global_best else None,
        },
        "openPositions": open_pos,
        "history":       history,
        "optConfigs":    opt_configs,
        "pnlCurve":      pnl_curve,
    }


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────
def create_app(registry: SignalRegistry, opt_state: OptimiserState) -> FastAPI:
    app = FastAPI(title="Trading Bot Dashboard", docs_url=None, redoc_url=None)

    # serve static files
    static_dir = BASE_DIR / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=FileResponse)
    async def root():
        return FileResponse(str(BASE_DIR / "static" / "index.html"))

    @app.get("/api/state")
    async def api_state():
        return _serialize_state(registry, opt_state)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await _manager.connect(ws)
        try:
            # send current state immediately on connect
            await ws.send_text(json.dumps(_serialize_state(registry, opt_state)))
            while True:
                # keep connection alive — client pings us
                data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if data == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            _manager.disconnect(ws)

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Broadcast loop — push updates to all connected clients every 5s
# ──────────────────────────────────────────────────────────────────────────────
async def run_broadcast_loop(
    registry: SignalRegistry,
    opt_state: OptimiserState,
    stop_event: asyncio.Event,
    interval: float = 5.0,
) -> None:
    logger.info("Dashboard broadcast loop started (interval=%.0fs)", interval)
    while not stop_event.is_set():
        try:
            payload = _serialize_state(registry, opt_state)
            await _manager.broadcast(payload)
        except Exception as e:
            logger.error("Broadcast error: %s", e)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
    logger.info("Dashboard broadcast loop stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Push a single signal event to all clients immediately
# ──────────────────────────────────────────────────────────────────────────────
async def push_signal_event(sig_data: Dict) -> None:
    await _manager.broadcast({"type": "signal", "signal": sig_data})
