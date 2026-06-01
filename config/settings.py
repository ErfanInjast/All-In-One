"""
Global Configuration — Trading Signal Bot
"""
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Telegram
# ──────────────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = "YOUR_BOT_TOKEN"
TELEGRAM_ADMIN_IDS: List[int] = [123456789]

# اگر از ایران / VPN استفاده می‌کنید آدرس پروکسی را اینجا بگذارید.
# مثال socks5:  "socks5://127.0.0.1:10808"
# مثال http:    "http://127.0.0.1:8080"
# برای غیرفعال کردن: None
TELEGRAM_PROXY: Optional[str] = None

# Timeout (ثانیه) — برای اینترنت ناپایدار مقدار بالاتر بهتر است
TELEGRAM_CONNECT_TIMEOUT: float = 30.0
TELEGRAM_READ_TIMEOUT:    float = 30.0
TELEGRAM_WRITE_TIMEOUT:   float = 30.0
TELEGRAM_POOL_TIMEOUT:    float = 10.0

# تلاش مجدد ارسال پیام تلگرام
TELEGRAM_SEND_RETRIES: int = 5

# ──────────────────────────────────────────────────────────────────────────────
# Exchange
# ──────────────────────────────────────────────────────────────────────────────
EXCHANGE_ID: str = "binance"
EXCHANGE_SANDBOX: bool = False

# ──────────────────────────────────────────────────────────────────────────────
# Watchlist
# ──────────────────────────────────────────────────────────────────────────────
WATCHLIST: List[str] = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "DOT/USDT", "MATIC/USDT",
    "LINK/USDT", "LTC/USDT", "UNI/USDT", "ATOM/USDT", "FIL/USDT",
    "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT",
]

# ──────────────────────────────────────────────────────────────────────────────
# Timeframes
# ──────────────────────────────────────────────────────────────────────────────
TIMEFRAMES: List[str] = ["5m", "15m", "30m", "1h", "4h", "1d"]

# ──────────────────────────────────────────────────────────────────────────────
# Leverage tiers
# ──────────────────────────────────────────────────────────────────────────────
LEVERAGE_TIERS: List[int] = [1, 2, 3, 5, 10, 20]

# ──────────────────────────────────────────────────────────────────────────────
# Optimiser search space
# ──────────────────────────────────────────────────────────────────────────────
FISHER_PERIODS:    List[int]   = [8, 9, 10, 12, 14]
SMOOTH_PERIODS:    List[int]   = [3, 4, 5, 7, 8]
MIN_CONFIRMATIONS: List[int]   = [2, 3, 4, 5]
FISHER_THRESHOLDS: List[float] = [0.05, 0.10, 0.15, 0.20]

# ──────────────────────────────────────────────────────────────────────────────
# Risk management
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_RISK_PER_TRADE: float = 0.01
DEFAULT_MAX_DRAWDOWN:   float = 0.15
DEFAULT_TP_ATR_MULT:    float = 2.0
DEFAULT_SL_ATR_MULT:    float = 1.2
TRAILING_STOP_PCT:      float = 0.02

# ──────────────────────────────────────────────────────────────────────────────
# Operational timing
# ──────────────────────────────────────────────────────────────────────────────
SIGNAL_SCAN_INTERVAL_SECONDS: int = 60
OPTIMISE_INTERVAL_HOURS:      int = 12
CANDLE_LIMIT:                 int = 350

# ──────────────────────────────────────────────────────────────────────────────
# Strategy weights
# ──────────────────────────────────────────────────────────────────────────────
STRATEGY_WEIGHTS: dict = {
    "RSI":        1.0,
    "MACD":       1.2,
    "BB":         0.8,
    "EMA_Cross":  1.0,
    "Stoch":      0.9,
    "Ichimoku":   1.3,
    "ATR_Trend":  1.1,
    "OBV":        0.7,
    "CCI":        0.8,
    "ADX":        1.2,
    "VWAP":       1.0,
    "SuperTrend": 1.3,
}

# ──────────────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────────────
MAX_SIGNALS_IN_HISTORY: int = 50
MAX_PERFORMANCE_ROWS:   int = 20
