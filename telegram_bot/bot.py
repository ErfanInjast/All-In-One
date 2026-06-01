"""
telegram_bot/bot.py — Full management panel with inline keyboards and callbacks.
Handles ConnectTimeout / network instability via configurable timeouts,
optional SOCKS5/HTTP proxy, and per-send retry logic.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Optional

from telegram import (
    InlineKeyboardButton, InlineKeyboardMarkup, Update,
)
from telegram.constants import ParseMode
from telegram.error import NetworkError, RetryAfter, TelegramError, TimedOut
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler,
    ContextTypes, MessageHandler, filters,
)
from telegram.request import HTTPXRequest

from config.settings import (
    MAX_PERFORMANCE_ROWS, MAX_SIGNALS_IN_HISTORY, STRATEGY_WEIGHTS,
    TELEGRAM_ADMIN_IDS, TELEGRAM_CONNECT_TIMEOUT, TELEGRAM_POOL_TIMEOUT,
    TELEGRAM_PROXY, TELEGRAM_READ_TIMEOUT, TELEGRAM_SEND_RETRIES,
    TELEGRAM_WRITE_TIMEOUT, TIMEFRAMES, WATCHLIST,
)
from core.signal_state import Direction, Signal, SignalRegistry
from optimizer.grid_search import OptimiserState

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"


def _is_admin(user_id: Optional[int]) -> bool:
    return user_id in TELEGRAM_ADMIN_IDS


def _admin_only(fn):
    """
    Decorator that works for bound methods (self, update, ctx) AND plain
    functions (update, ctx). Forwards *args/**kwargs transparently so the
    method signature is never broken.
    """
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        update_obj = next((a for a in args if isinstance(a, Update)), None)
        uid = (update_obj.effective_user.id
               if update_obj and update_obj.effective_user else None)
        if not _is_admin(uid):
            if update_obj and update_obj.effective_message:
                await update_obj.effective_message.reply_text("⛔ Unauthorized.")
            return
        return await fn(*args, **kwargs)
    return wrapper


# ──────────────────────────────────────────────────────────────────────────────
# Robust send — retries on timeout / network errors
# ──────────────────────────────────────────────────────────────────────────────
async def _safe_send(coro_factory, retries: int = TELEGRAM_SEND_RETRIES) -> bool:
    """
    Call `coro_factory()` (a zero-arg async callable) with exponential backoff.
    Handles TimedOut, NetworkError, RetryAfter automatically.
    Returns True on success, False after all retries exhausted.
    """
    for attempt in range(1, retries + 1):
        try:
            await coro_factory()
            return True
        except RetryAfter as e:
            # Telegram told us exactly how long to wait
            wait = float(e.retry_after) + 1.0
            logger.warning("Telegram rate limit — waiting %.0fs", wait)
            await asyncio.sleep(wait)
        except (TimedOut, NetworkError) as e:
            wait = min(2.0 ** attempt, 60.0)
            logger.warning(
                "Telegram network error (attempt %d/%d) — retry in %.0fs: %s",
                attempt, retries, wait, e,
            )
            await asyncio.sleep(wait)
        except TelegramError as e:
            # non-retryable (bad token, chat not found, etc.)
            logger.error("Telegram error (non-retryable): %s", e)
            return False
    logger.error("Telegram send failed after %d attempts.", retries)
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Message builders
# ──────────────────────────────────────────────────────────────────────────────
def _build_signal_message(sig: Signal) -> str:
    emoji = "🚀" if sig.direction == Direction.LONG else "🔴"
    arrow = "📈" if sig.direction == Direction.LONG else "📉"
    return (
        f"{emoji} *{sig.direction.value} SIGNAL* — `{sig.symbol}`\n"
        f"{'─' * 32}\n"
        f"💰 *Entry:*    `{sig.entry_price:.6g}`\n"
        f"🎯 *TP:*       `{sig.tp_price:.6g}`\n"
        f"🛡 *SL:*       `{sig.sl_price:.6g}`\n"
        f"⚖️ *Leverage:* `{sig.leverage}×`\n"
        f"📐 *R/R:*      `1 : {sig.risk_reward:.2f}`\n"
        f"⏱ *TF:*       `{sig.timeframe}`\n"
        f"{arrow} *Fisher:*  `{sig.fisher_val:.4f}` | Accel `{sig.accel:.4f}`\n"
        f"🧠 *Consensus:* `{sig.consensus:.1f} / {sig.max_consensus:.1f}`\n"
        f"🕐 `{_ts(sig.timestamp)}`\n"
        f"🆔 `{sig.id}`"
    )


def _build_dashboard_text(registry: SignalRegistry, opt_state: OptimiserState) -> str:
    s = registry.stats
    next_opt  = opt_state.last_run_ts + 43200
    secs_left = max(0, next_opt - time.time())
    h, rem    = divmod(int(secs_left), 3600)
    m, _      = divmod(rem, 60)

    open_list = registry.all_open()
    open_str  = "\n".join(
        f"  • `{sig.symbol}` {sig.direction.value} ×{sig.leverage} | "
        f"Entry `{sig.entry_price:.6g}` | TF `{sig.timeframe}`"
        for sig in open_list
    ) or "  _None_"

    best     = opt_state.global_best
    best_str = (
        f"`{best.symbol}` {best.timeframe} | "
        f"Sharpe `{best.sharpe:.2f}` | WR `{best.win_rate:.0%}` | "
        f"Lev `{best.leverage}×`"
    ) if best else "_Not run yet_"

    return (
        f"📊 *TRADING BOT DASHBOARD*\n"
        f"{'━' * 34}\n\n"
        f"*📈 Performance*\n"
        f"  Signals:  `{s.total_signals}` total | `{s.open_signals}` open\n"
        f"  Closed:   `{s.total_closed}` | WR `{s.win_rate:.0%}`\n"
        f"  Net PnL:  `{_pct(s.total_pnl)}`\n"
        f"  Best:     `{_pct(s.best_trade)}` | Worst `{_pct(s.worst_trade)}`\n"
        f"  Max DD:   `{_pct(s.max_drawdown)}`\n\n"
        f"*🔴 Open Positions ({len(open_list)})*\n{open_str}\n\n"
        f"*🏆 Global Best Config*\n  {best_str}\n\n"
        f"*⚙️ Optimiser*\n"
        f"  Runs: `{opt_state.run_count}` | Next in `{h:02d}:{m:02d}`\n"
        f"  Symbols configured: `{len(opt_state.best_configs)}`\n"
    )


def _build_performance_table(registry: SignalRegistry) -> str:
    history = registry.recent_history(MAX_PERFORMANCE_ROWS)
    if not history:
        return "📭 *No closed trades yet.*"
    rows = ["*Recent Trades*\n```",
            f"{'SYM':<10} {'DIR':<6} {'LV':>3} {'PNL':>8} {'RR':>5}",
            "─" * 36]
    for sig in history:
        if sig.pnl_pct is None:
            continue
        rows.append(
            f"{sig.symbol[:10]:<10} {sig.direction.value:<6} "
            f"{sig.leverage:>2}× {sig.pnl_pct * 100:>+6.1f}% "
            f"{sig.risk_reward:>5.1f}"
        )
    rows.append("```")
    return "\n".join(rows)


def _build_optimizer_report(opt_state: OptimiserState) -> str:
    if not opt_state.best_configs:
        return "🔬 *No optimisation run yet.*"
    lines = ["*🔬 Optimiser Results — Top Configs*\n```",
             f"{'SYM':<10} {'TF':<5} {'FP':>3} {'SP':>3} {'LV':>3} "
             f"{'SH':>6} {'WR':>6} {'PF':>5}",
             "─" * 48]
    for r in sorted(opt_state.best_configs.values(),
                    key=lambda x: x.score(), reverse=True)[:MAX_PERFORMANCE_ROWS]:
        lines.append(
            f"{r.symbol[:10]:<10} {r.timeframe:<5} "
            f"{r.fisher_period:>3} {r.smooth_period:>3} {r.leverage:>2}× "
            f"{r.sharpe:>6.2f} {r.win_rate:>5.0%} {r.profit_factor:>5.2f}"
        )
    lines.append("```")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Keyboards
# ──────────────────────────────────────────────────────────────────────────────
def _main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Dashboard",   callback_data="dashboard"),
         InlineKeyboardButton("📈 Performance", callback_data="performance")],
        [InlineKeyboardButton("🔬 Optimiser",   callback_data="optimizer"),
         InlineKeyboardButton("📋 Open Pos.",   callback_data="open_positions")],
        [InlineKeyboardButton("🔔 Signal Log",  callback_data="signal_log"),
         InlineKeyboardButton("⚙️ Settings",    callback_data="settings")],
        [InlineKeyboardButton("🔁 Run Optimiser Now", callback_data="run_optimizer")],
    ])


def _back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("« Back to Menu", callback_data="main_menu"),
    ]])


def _settings_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📋 Watchlist",        callback_data="settings_watchlist")],
        [InlineKeyboardButton("⏱ Timeframes",        callback_data="settings_timeframes")],
        [InlineKeyboardButton("⚖️ Leverage Tiers",   callback_data="settings_leverage")],
        [InlineKeyboardButton("🧠 Strategy Weights", callback_data="settings_weights")],
        [InlineKeyboardButton("🌐 Network Info",     callback_data="settings_network")],
        [InlineKeyboardButton("« Back",              callback_data="main_menu")],
    ])


# ──────────────────────────────────────────────────────────────────────────────
# TelegramPanel
# ──────────────────────────────────────────────────────────────────────────────
class TelegramPanel:
    def __init__(
        self,
        token:     str,
        registry:  SignalRegistry,
        opt_state: OptimiserState,
        run_opt_cb,
    ):
        self._token     = token
        self._registry  = registry
        self._opt_state = opt_state
        self._run_opt   = run_opt_cb
        self._app: Optional[Application] = None

    # ── bootstrap ────────────────────────────────────────────────────────────
    def build(self) -> Application:
        # Build HTTPXRequest with custom timeouts and optional proxy
        request_kwargs = dict(
            connect_timeout = TELEGRAM_CONNECT_TIMEOUT,
            read_timeout    = TELEGRAM_READ_TIMEOUT,
            write_timeout   = TELEGRAM_WRITE_TIMEOUT,
            pool_timeout    = TELEGRAM_POOL_TIMEOUT,
        )
        if TELEGRAM_PROXY:
            request_kwargs["proxy"] = TELEGRAM_PROXY
            logger.info("Telegram using proxy: %s", TELEGRAM_PROXY)

        request = HTTPXRequest(**request_kwargs)

        app = (
            Application.builder()
            .token(self._token)
            .request(request)
            .build()
        )

        app.add_handler(CommandHandler("start",  self._cmd_start))
        app.add_handler(CommandHandler("menu",   self._cmd_menu))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CallbackQueryHandler(self._callback_router))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._unknown))
        self._app = app
        return app

    # ── broadcast helpers ────────────────────────────────────────────────────
    async def broadcast_signal(self, sig: Signal) -> None:
        if self._app is None:
            return
        text = _build_signal_message(sig)
        for admin_id in TELEGRAM_ADMIN_IDS:
            await _safe_send(
                lambda aid=admin_id: self._app.bot.send_message(
                    chat_id=aid, text=text,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=_back_keyboard(),
                )
            )

    async def broadcast_text(self, text: str) -> None:
        if self._app is None:
            return
        for admin_id in TELEGRAM_ADMIN_IDS:
            await _safe_send(
                lambda aid=admin_id: self._app.bot.send_message(
                    chat_id=aid, text=text,
                    parse_mode=ParseMode.MARKDOWN,
                )
            )

    # ── commands ─────────────────────────────────────────────────────────────
    @_admin_only
    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await _safe_send(lambda: update.message.reply_text(
            "🤖 *Trading Signal Bot* is active.\n\nUse /menu for the control panel.",
            parse_mode=ParseMode.MARKDOWN,
        ))

    @_admin_only
    async def _cmd_menu(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await _safe_send(lambda: update.message.reply_text(
            "📋 *Main Menu*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_main_keyboard(),
        ))

    @_admin_only
    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        text = _build_dashboard_text(self._registry, self._opt_state)
        await _safe_send(lambda: update.message.reply_text(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=_main_keyboard(),
        ))

    async def _unknown(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        pass

    # ── callback router ───────────────────────────────────────────────────────
    async def _callback_router(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await _safe_send(lambda: query.answer())

        if not _is_admin(query.from_user.id if query.from_user else None):
            await _safe_send(lambda: query.edit_message_text("⛔ Unauthorized."))
            return

        handlers = {
            "main_menu":           self._cb_main_menu,
            "dashboard":           self._cb_dashboard,
            "performance":         self._cb_performance,
            "optimizer":           self._cb_optimizer,
            "open_positions":      self._cb_open_positions,
            "signal_log":          self._cb_signal_log,
            "settings":            self._cb_settings,
            "run_optimizer":       self._cb_run_optimizer,
            "settings_watchlist":  self._cb_settings_watchlist,
            "settings_timeframes": self._cb_settings_timeframes,
            "settings_leverage":   self._cb_settings_leverage,
            "settings_weights":    self._cb_settings_weights,
            "settings_network":    self._cb_settings_network,
        }
        handler = handlers.get(query.data)
        if handler:
            await handler(query)
        else:
            await _safe_send(lambda: query.edit_message_text(
                "❓ Unknown action.", reply_markup=_back_keyboard()
            ))

    # ── callback pages ────────────────────────────────────────────────────────
    async def _cb_main_menu(self, query):
        await _safe_send(lambda: query.edit_message_text(
            "📋 *Main Menu*", parse_mode=ParseMode.MARKDOWN,
            reply_markup=_main_keyboard(),
        ))

    async def _cb_dashboard(self, query):
        text = _build_dashboard_text(self._registry, self._opt_state)
        await _safe_send(lambda: query.edit_message_text(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=_main_keyboard(),
        ))

    async def _cb_performance(self, query):
        await _safe_send(lambda: query.edit_message_text(
            _build_performance_table(self._registry),
            parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard(),
        ))

    async def _cb_optimizer(self, query):
        await _safe_send(lambda: query.edit_message_text(
            _build_optimizer_report(self._opt_state),
            parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard(),
        ))

    async def _cb_open_positions(self, query):
        opens = self._registry.all_open()
        if not opens:
            text = "📭 *No open positions.*"
        else:
            lines = ["*Open Positions*\n"]
            for sig in opens:
                lines.append(
                    f"• `{sig.symbol}` {sig.direction.value} ×{sig.leverage} | "
                    f"Entry `{sig.entry_price:.6g}` | "
                    f"TP `{sig.tp_price:.6g}` | "
                    f"SL `{sig.sl_price:.6g}` | "
                    f"TF `{sig.timeframe}`"
                )
            text = "\n".join(lines)
        await _safe_send(lambda: query.edit_message_text(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard(),
        ))

    async def _cb_signal_log(self, query):
        history = self._registry.recent_history(MAX_SIGNALS_IN_HISTORY)
        if not history:
            text = "📭 *No signals yet.*"
        else:
            lines = ["*Signal History*\n"]
            for sig in history:
                status  = "🟢" if sig.is_open else ("✅" if (sig.pnl_pct or 0) > 0 else "❌")
                pnl_str = f" | PnL `{_pct(sig.pnl_pct)}`" if sig.pnl_pct is not None else ""
                lines.append(
                    f"{status} `{sig.symbol}` {sig.direction.value} ×{sig.leverage}"
                    f" @ `{sig.entry_price:.6g}`{pnl_str} | `{sig.timeframe}`"
                )
            text = "\n".join(lines)
        await _safe_send(lambda: query.edit_message_text(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard(),
        ))

    async def _cb_settings(self, query):
        await _safe_send(lambda: query.edit_message_text(
            "⚙️ *Settings*\nChoose a category:",
            parse_mode=ParseMode.MARKDOWN, reply_markup=_settings_keyboard(),
        ))

    async def _cb_settings_watchlist(self, query):
        wl = "\n".join(f"  • `{s}`" for s in WATCHLIST)
        await _safe_send(lambda: query.edit_message_text(
            f"📋 *Watchlist* ({len(WATCHLIST)} symbols)\n\n{wl}",
            parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard(),
        ))

    async def _cb_settings_timeframes(self, query):
        tf = " | ".join(f"`{t}`" for t in TIMEFRAMES)
        await _safe_send(lambda: query.edit_message_text(
            f"⏱ *Active Timeframes*\n\n{tf}",
            parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard(),
        ))

    async def _cb_settings_leverage(self, query):
        from config.settings import LEVERAGE_TIERS
        lev = " | ".join(f"`{l}×`" for l in LEVERAGE_TIERS)
        await _safe_send(lambda: query.edit_message_text(
            f"⚖️ *Leverage Tiers Tested*\n\n{lev}",
            parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard(),
        ))

    async def _cb_settings_weights(self, query):
        lines = [f"  `{name}`: `{w}`" for name, w in STRATEGY_WEIGHTS.items()]
        await _safe_send(lambda: query.edit_message_text(
            "🧠 *Strategy Weights*\n\n" + "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard(),
        ))

    async def _cb_settings_network(self, query):
        proxy_str = f"`{TELEGRAM_PROXY}`" if TELEGRAM_PROXY else "_Not set_"
        text = (
            f"🌐 *Network Configuration*\n\n"
            f"  Proxy:    {proxy_str}\n"
            f"  Connect:  `{TELEGRAM_CONNECT_TIMEOUT}s`\n"
            f"  Read:     `{TELEGRAM_READ_TIMEOUT}s`\n"
            f"  Write:    `{TELEGRAM_WRITE_TIMEOUT}s`\n"
            f"  Retries:  `{TELEGRAM_SEND_RETRIES}`\n\n"
            f"_Edit `config/settings.py` → `TELEGRAM\\_PROXY` to set a proxy._"
        )
        await _safe_send(lambda: query.edit_message_text(
            text, parse_mode=ParseMode.MARKDOWN, reply_markup=_back_keyboard(),
        ))

    async def _cb_run_optimizer(self, query):
        await _safe_send(lambda: query.edit_message_text(
            "🔬 *Optimiser started manually...* This may take several minutes.",
            parse_mode=ParseMode.MARKDOWN,
        ))
        asyncio.create_task(self._run_opt())
