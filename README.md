# Trading Signal Bot — Professional Grade

## Architecture

```
trading_bot/
├── main.py                    # Orchestrator — entry point
├── requirements.txt
├── config/
│   └── settings.py            # ALL parameters in one place
├── core/
│   ├── scanner.py             # Real-time signal scanner + TP/SL exit
│   └── signal_state.py        # Signal registry + performance tracker
├── strategies/
│   └── engine.py              # Fisher Transform + 12 strategies
├── data/
│   └── fetcher.py             # Async CCXT with LRU cache
├── optimizer/
│   └── grid_search.py         # Exhaustive grid search + Sharpe backtest
├── telegram_bot/
│   └── bot.py                 # Full inline keyboard admin panel
└── utils/
    └── helpers.py             # Logging, rate limiter, retry
```

## Setup

```bash
pip install -r requirements.txt
```

Edit `config/settings.py`:
```python
TELEGRAM_BOT_TOKEN = "your_token_here"
TELEGRAM_ADMIN_IDS = [your_numeric_id]
```

Run:
```bash
python main.py
```

## Telegram Panel Commands

| Command | Action |
|---------|--------|
| `/start` | Activate bot |
| `/menu` | Open inline control panel |
| `/status` | Quick dashboard |

## Panel Buttons

| Button | Shows |
|--------|-------|
| 📊 Dashboard | Live stats, open positions, optimiser status |
| 📈 Performance | Last N closed trades with PnL |
| 🔬 Optimiser | Top configs per symbol (Sharpe, WR, PF) |
| 📋 Open Pos. | All currently open signals |
| 🔔 Signal Log | Full signal history |
| ⚙️ Settings | Watchlist, TFs, leverage tiers, strategy weights |
| 🔁 Run Now | Trigger optimiser manually |

## Signal Message Format

```
🚀 LONG SIGNAL — BTC/USDT
────────────────────────────────
💰 Entry:   43521.50
🎯 TP:      44610.30
🛡 SL:      42980.10
⚖️ Leverage: 3×
📐 R/R:     1 : 2.1
⏱ TF:      15m
📈 Fisher:  -0.3421 | Accel 0.0812
🧠 Consensus: 7.4 / 12.0
🕐 2025-01-15 14:32 UTC
🆔 SIG-00042
```

## Optimisation Cycle (every 12 hours)

Tests all combinations of:
- **20 symbols** × **6 timeframes** × **6 leverage tiers**
- × **5 Fisher periods** × **5 smooth periods**
- × **4 min confirmations** × **4 Fisher thresholds**
- = **~144,000 parameter combinations**

Ranks by: `Sharpe × (1 - MaxDrawdown) × ProfitFactor`

## Strategies Included

| # | Strategy | Weight |
|---|----------|--------|
| 1 | RSI (14) | 1.0 |
| 2 | MACD | 1.2 |
| 3 | Bollinger Bands | 0.8 |
| 4 | EMA Cross 9/21 | 1.0 |
| 5 | Stochastic | 0.9 |
| 6 | Ichimoku Cloud | 1.3 |
| 7 | ATR Trend | 1.1 |
| 8 | OBV | 0.7 |
| 9 | CCI | 0.8 |
| 10 | ADX | 1.2 |
| 11 | VWAP | 1.0 |
| 12 | SuperTrend | 1.3 |

## Risk Management

- ATR-based dynamic TP/SL per symbol
- Trailing stop support (`TRAILING_STOP_PCT`)
- Max drawdown circuit breaker (`DEFAULT_MAX_DRAWDOWN`)
- Per-trade risk sizing (`DEFAULT_RISK_PER_TRADE`)
- Fisher reversal forced exit

## Disclaimer

For educational and research purposes only.
Not financial advice. Use at your own risk.
