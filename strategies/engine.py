"""
strategies/engine.py — Fisher Transform + 12-strategy consensus engine
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Fisher Transform
# ──────────────────────────────────────────────────────────────────────────────
def fisher_transform(
    df: pd.DataFrame,
    period: int = 9,
    smooth_period: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (fish, signal) arrays aligned to df index."""
    try:
        tp = (df["H"] + df["L"] + df["C"]) / 3.0
        hma = ta.hma(tp, length=5)

        rolling_max = hma.rolling(window=period).max()
        rolling_min = hma.rolling(window=period).min()
        denom = (rolling_max - rolling_min).replace(0, 1e-10)

        raw = 2.0 * ((hma - rolling_min) / denom) - 1.0
        raw = np.nan_to_num(raw.values)

        v = pd.Series(raw).ewm(alpha=0.5, adjust=False).mean().values
        v_clip = np.clip(v, -0.999, 0.999)
        fish = 0.5 * np.log((1.0 + v_clip) / (1.0 - v_clip))
        sig = pd.Series(fish).ewm(span=smooth_period, adjust=False).mean().values
        return fish, sig
    except Exception as e:
        logger.error("Fisher error: %s", e)
        return np.array([]), np.array([])


# ──────────────────────────────────────────────────────────────────────────────
# 12 Individual Strategies → each returns np.ndarray of {-1, 0, 1}
# ──────────────────────────────────────────────────────────────────────────────
def _rsi(df: pd.DataFrame) -> np.ndarray:
    rsi = ta.rsi(df["C"], length=14)
    return np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)).astype(float)


def _macd(df: pd.DataFrame) -> np.ndarray:
    m = ta.macd(df["C"])
    if m is None or m.empty:
        return np.zeros(len(df))
    macd_line = m.iloc[:, 0]
    signal_line = m.iloc[:, 1]
    return np.where(macd_line > signal_line, 1, -1).astype(float)


def _bbands(df: pd.DataFrame) -> np.ndarray:
    bb = ta.bbands(df["C"], length=20, std=2)
    if bb is None or bb.empty:
        return np.zeros(len(df))
    lower, upper = bb.iloc[:, 0], bb.iloc[:, 2]
    return np.where(df["C"] < lower, 1, np.where(df["C"] > upper, -1, 0)).astype(float)


def _ema_cross(df: pd.DataFrame) -> np.ndarray:
    fast = ta.ema(df["C"], length=9)
    slow = ta.ema(df["C"], length=21)
    return np.where(fast > slow, 1, -1).astype(float)


def _stoch(df: pd.DataFrame) -> np.ndarray:
    st = ta.stoch(df["H"], df["L"], df["C"])
    if st is None or st.empty:
        return np.zeros(len(df))
    k = st.iloc[:, 0]
    return np.where(k < 20, 1, np.where(k > 80, -1, 0)).astype(float)


def _ichimoku(df: pd.DataFrame) -> np.ndarray:
    try:
        ichi, _ = ta.ichimoku(df["H"], df["L"], df["C"])
        if ichi is None or ichi.empty:
            return np.zeros(len(df))
        span_a = ichi.iloc[:, 0]
        span_b = ichi.iloc[:, 1]
        return np.where(df["C"] > span_a, 1, np.where(df["C"] < span_b, -1, 0)).astype(float)
    except Exception:
        return np.zeros(len(df))


def _atr_trend(df: pd.DataFrame) -> np.ndarray:
    atr = ta.atr(df["H"], df["L"], df["C"], length=14)
    c, cp = df["C"].values, df["C"].shift(1).values
    return np.where(c > cp + atr * 1.5, 1, np.where(c < cp - atr * 1.5, -1, 0)).astype(float)


def _obv(df: pd.DataFrame) -> np.ndarray:
    obv = ta.obv(df["C"], df["V"])
    obv_ema = ta.ema(obv, length=20)
    return np.where(obv > obv_ema, 1, -1).astype(float)


def _cci(df: pd.DataFrame) -> np.ndarray:
    cci = ta.cci(df["H"], df["L"], df["C"], length=20)
    return np.where(cci < -100, 1, np.where(cci > 100, -1, 0)).astype(float)


def _adx(df: pd.DataFrame) -> np.ndarray:
    adx = ta.adx(df["H"], df["L"], df["C"], length=14)
    if adx is None or adx.empty:
        return np.zeros(len(df))
    adx_val = adx.iloc[:, 0]
    dmp = adx.iloc[:, 1]
    dmn = adx.iloc[:, 2]
    return np.where(
        (adx_val > 25) & (dmp > dmn), 1,
        np.where((adx_val > 25) & (dmp < dmn), -1, 0),
    ).astype(float)


def _vwap(df: pd.DataFrame) -> np.ndarray:
    try:
        vwap = ta.vwap(df["H"], df["L"], df["C"], df["V"])
        if vwap is None:
            return np.zeros(len(df))
        return np.where(df["C"] > vwap, 1, -1).astype(float)
    except Exception:
        return np.zeros(len(df))


def _supertrend(df: pd.DataFrame) -> np.ndarray:
    try:
        st = ta.supertrend(df["H"], df["L"], df["C"], length=7, multiplier=3.0)
        if st is None or st.empty:
            return np.zeros(len(df))
        direction_col = [c for c in st.columns if "SUPERTd" in c]
        if not direction_col:
            return np.zeros(len(df))
        direction = st[direction_col[0]]
        return np.where(direction == 1, 1, -1).astype(float)
    except Exception:
        return np.zeros(len(df))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
STRATEGY_MAP = {
    "RSI":        _rsi,
    "MACD":       _macd,
    "BB":         _bbands,
    "EMA_Cross":  _ema_cross,
    "Stoch":      _stoch,
    "Ichimoku":   _ichimoku,
    "ATR_Trend":  _atr_trend,
    "OBV":        _obv,
    "CCI":        _cci,
    "ADX":        _adx,
    "VWAP":       _vwap,
    "SuperTrend": _supertrend,
}


def compute_signals(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Run all strategies, return dict of signal arrays."""
    results: Dict[str, np.ndarray] = {}
    for name, fn in STRATEGY_MAP.items():
        try:
            arr = fn(df)
            # ensure same length as df
            if len(arr) != len(df):
                arr = np.zeros(len(df))
            results[name] = arr
        except Exception as e:
            logger.warning("Strategy %s failed: %s", name, e)
            results[name] = np.zeros(len(df))
    return results


def weighted_consensus(
    signals: Dict[str, np.ndarray],
    weights: Dict[str, float],
    idx: int = -1,
) -> Tuple[float, float]:
    """
    Returns (weighted_score, max_possible_score) at position `idx`.
    Positive = bullish, negative = bearish.
    """
    score = 0.0
    max_score = 0.0
    for name, arr in signals.items():
        w = weights.get(name, 1.0)
        max_score += w
        if len(arr) > 0:
            score += arr[idx] * w
    return score, max_score
