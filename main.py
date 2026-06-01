"""
main.py — Orchestrator: scanner + optimiser + Telegram panel + Web dashboard
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

import uvicorn

from config.settings import (
    OPTIMISE_INTERVAL_HOURS, TELEGRAM_ADMIN_IDS, TELEGRAM_BOT_TOKEN,
)
from core.scanner import run_scanner
from core.signal_state import SignalRegistry
from dashboard.server import create_app, run_broadcast_loop, push_signal_event
from optimizer.grid_search import OptimiserState, run_full_optimisation
from telegram_bot.bot import TelegramPanel
from utils.helpers import setup_logging

logger = logging.getLogger(__name__)

# ── guards ────────────────────────────────────────────────────────────────────
if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN":
    print("Set TELEGRAM_BOT_TOKEN in config/settings.py before starting!")
    sys.exit(1)
if TELEGRAM_ADMIN_IDS == [123456789]:
    print("Set your real TELEGRAM_ADMIN_IDS in config/settings.py!")
    sys.exit(1)

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8080


# ── optimiser loop ────────────────────────────────────────────────────────────
async def optimiser_loop(
    state: OptimiserState,
    stop_event: asyncio.Event,
    panel: TelegramPanel,
):
    interval = OPTIMISE_INTERVAL_HOURS * 3600
    while not stop_event.is_set():
        logger.info("Optimiser cycle starting ...")
        await panel.broadcast_text("🔬 *Optimisation cycle started.* Running in background ...")
        try:
            await run_full_optimisation(state)
            await panel.broadcast_text(
                f"✅ *Optimisation complete.*\n"
                f"  Symbols configured: `{len(state.best_configs)}`\n"
                f"  Run #{state.run_count}"
            )
        except Exception as e:
            logger.error("Optimiser error: %s", e)
            await panel.broadcast_text(f"❌ *Optimiser error:* `{e}`")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
    logger.info("Optimiser loop stopped.")


# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    setup_logging()
    logger.info("Trading Signal Bot starting ...")

    registry  = SignalRegistry()
    opt_state = OptimiserState()
    stop      = asyncio.Event()

    # ── manual opt trigger ────────────────────────────────────────────────
    async def trigger_opt():
        await run_full_optimisation(opt_state)
        await panel.broadcast_text(
            f"✅ *Manual optimisation complete.* "
            f"Symbols: `{len(opt_state.best_configs)}`"
        )

    # ── telegram panel ────────────────────────────────────────────────────
    panel = TelegramPanel(
        token=TELEGRAM_BOT_TOKEN,
        registry=registry,
        opt_state=opt_state,
        run_opt_cb=trigger_opt,
    )
    tg_app = panel.build()

    # ── signal callback (fires on both Telegram + WS dashboard) ──────────
    async def on_signal(sig):
        await panel.broadcast_signal(sig)
        # push to dashboard WebSocket clients immediately
        sig_data = {
            "id":        sig.id,
            "symbol":    sig.symbol,
            "direction": sig.direction.value,
            "leverage":  sig.leverage,
            "entry":     sig.entry_price,
            "tp":        sig.tp_price,
            "sl":        sig.sl_price,
            "timeframe": sig.timeframe,
            "fisher":    round(sig.fisher_val, 4),
            "accel":     round(sig.accel, 4),
            "consensus": round(sig.consensus, 2),
            "maxConsen": round(sig.max_consensus, 2),
            "rr":        round(sig.risk_reward, 2),
        }
        await push_signal_event(sig_data)

    # ── graceful shutdown ────────────────────────────────────────────────
    def _shutdown(*_):
        logger.info("Shutdown signal received.")
        stop.set()

    for sig_name in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_event_loop().add_signal_handler(sig_name, _shutdown)
        except (NotImplementedError, RuntimeError):
            pass

    # ── start telegram polling ────────────────────────────────────────────
    await tg_app.initialize()
    await tg_app.start()
    if tg_app.updater:
        await tg_app.updater.start_polling(drop_pending_updates=True)

    await panel.broadcast_text(
        f"🟢 *Bot started.*\n"
        f"Dashboard: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}\n"
        f"Use /menu for Telegram panel."
    )

    # ── fastapi / uvicorn (dashboard) ─────────────────────────────────────
    fastapi_app = create_app(registry, opt_state)
    config      = uvicorn.Config(
        fastapi_app,
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    # ── launch all tasks ─────────────────────────────────────────────────
    tasks = [
        asyncio.create_task(optimiser_loop(opt_state, stop, panel), name="optimiser"),
        asyncio.create_task(run_scanner(opt_state, registry, on_signal, stop), name="scanner"),
        asyncio.create_task(run_broadcast_loop(registry, opt_state, stop), name="ws-broadcast"),
        asyncio.create_task(server.serve(), name="dashboard"),
    ]

    logger.info("Dashboard → http://%s:%s", DASHBOARD_HOST, DASHBOARD_PORT)

    try:
        await stop.wait()
    finally:
        server.should_exit = True
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if tg_app.updater:
            await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()
        logger.info("Bot stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
