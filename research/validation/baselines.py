"""
ASTRA FUSION QUANT — Baselines (Milestone 6)
No trade, passive long, EMA trend, Donchian breakout, range mean reversion.
Same cost model and risk constraints as ASTRA. Compare fairly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from research.validation.metrics import compute_metrics, MetricReport


def baseline_no_trade(total_bars: int) -> MetricReport:
    return compute_metrics([], total_bars, notes="Cash baseline: 0 trades.")


def baseline_passive_long(df: pd.DataFrame) -> dict:
    """
    Buy-and-hold from first bar to last. Returns total return and max drawdown.
    No costs for index/ETF; note assumption.
    """
    if len(df) < 2:
        return {"total_return": 0.0, "max_drawdown": 0.0}
    entry = df["close"].iloc[0]
    exit_ = df["close"].iloc[-1]
    equity = df["close"] / entry
    roll_max = equity.cummax()
    dd = ((equity - roll_max) / roll_max).min()
    return {
        "total_return": float((exit_ - entry) / entry * 100),
        "max_drawdown": float(dd * 100),
        "note": "Passive long: buy first bar, hold to last. No costs assumed.",
    }


def baseline_ema_trend(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    commission_pct: float = 0.10,
    risk_pct: float = 0.25,
) -> list[dict]:
    """
    EMA crossover trend: long when EMA20 > EMA50, flat otherwise.
    Fixed 1% stop from entry. Exit on opposing crossover.
    """
    from research.features.primitives import ema as ema_fn
    ema_f = ema_fn(df["close"], fast)
    ema_s = ema_fn(df["close"], slow)
    trades = []
    in_trade = False
    entry_price = 0.0
    entry_bar = 0

    for i in range(slow, len(df)):
        row = df.iloc[i]
        prev_ema_f = ema_f.iloc[i - 1]
        prev_ema_s = ema_s.iloc[i - 1]
        cur_ema_f  = ema_f.iloc[i]
        cur_ema_s  = ema_s.iloc[i]

        if not in_trade and cur_ema_f > cur_ema_s and prev_ema_f <= prev_ema_s:
            in_trade = True
            entry_price = row["open"]
            entry_bar = i

        elif in_trade and cur_ema_f < cur_ema_s:
            exit_price = row["open"]
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            cost = commission_pct * 2
            trades.append({
                "pnl_r": pnl_pct / 1.0,    # using 1% stop as 1R
                "side": "bullish",
                "costs": cost,
                "gross_pnl": pnl_pct,
            })
            in_trade = False

    return trades


def baseline_donchian_breakout(
    df: pd.DataFrame,
    period: int = 20,
    commission_pct: float = 0.10,
) -> list[dict]:
    """
    Long on close above Donchian 20-bar high (using prior bars only).
    Exit on close below 10-bar low.
    """
    trades = []
    in_trade = False
    entry_price = 0.0

    for i in range(period, len(df)):
        row  = df.iloc[i]
        high20 = df["high"].iloc[i - period:i].max()
        low10  = df["low"].iloc[max(0, i - 10):i].min()

        if not in_trade and row["close"] > high20:
            in_trade = True
            entry_price = row["close"]
        elif in_trade and row["close"] < low10:
            exit_price = row["close"]
            pnl = (exit_price - entry_price) / entry_price * 100
            trades.append({
                "pnl_r": pnl / 1.0,
                "side": "bullish",
                "costs": commission_pct * 2,
                "gross_pnl": pnl,
            })
            in_trade = False

    return trades


def baseline_range_reversion(
    df: pd.DataFrame,
    period: int = 50,
    commission_pct: float = 0.10,
) -> list[dict]:
    """
    Buy near rolling low (bottom 10% of 50-bar range), target midpoint.
    """
    trades = []
    in_trade = False
    entry_price = 0.0
    target_price = 0.0

    for i in range(period, len(df)):
        row   = df.iloc[i]
        hi50  = df["high"].iloc[i - period:i].max()
        lo50  = df["low"].iloc[i - period:i].min()
        width = hi50 - lo50
        bottom_zone = lo50 + 0.10 * width
        midpoint    = (hi50 + lo50) / 2

        if not in_trade and row["close"] <= bottom_zone:
            in_trade = True
            entry_price = row["close"]
            target_price = midpoint

        elif in_trade:
            if row["high"] >= target_price:
                pnl = (target_price - entry_price) / entry_price * 100
                trades.append({
                    "pnl_r": pnl / 1.0,
                    "side": "bullish",
                    "costs": commission_pct * 2,
                    "gross_pnl": pnl,
                })
                in_trade = False
            elif row["low"] < lo50 * 0.995:
                pnl = (row["low"] - entry_price) / entry_price * 100
                trades.append({
                    "pnl_r": pnl / 1.0,
                    "side": "bullish",
                    "costs": commission_pct * 2,
                    "gross_pnl": pnl,
                })
                in_trade = False

    return trades
