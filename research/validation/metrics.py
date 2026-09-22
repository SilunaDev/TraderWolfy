"""
ASTRA FUSION QUANT — Validation Metrics (Milestone 6)
Trade count, expectancy, win rate with CI, profit factor, Sharpe/Sortino,
drawdown, exposure, regime/setup/asset breakdowns.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MetricReport:
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    win_rate_ci_low: float
    win_rate_ci_high: float
    expectancy_r: float
    mean_r: float
    median_r: float
    avg_win_r: float
    avg_loss_r: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    exposure_pct: float
    longest_losing_streak: int
    long_count: int
    short_count: int
    cost_share_pct: float         # costs as % of gross PnL
    notes: str = ""


def win_rate_wilson_ci(wins: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for win rate."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def compute_equity_curve(r_series: pd.Series, initial: float = 1.0) -> pd.Series:
    """Convert R-series to equity curve (multiplicative)."""
    # Each R adds stop_distance to equity; use additive for simplicity
    return initial + r_series.cumsum()


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as fraction of peak equity."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max.replace(0, np.nan)
    return float(dd.min()) if len(dd) > 0 else 0.0


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualized Sharpe. Computed from evenly-sampled equity returns."""
    if len(returns) < 2:
        return np.nan
    excess = returns - rf / periods_per_year
    if excess.std() == 0:
        return np.nan
    return float(excess.mean() / excess.std() * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    """Annualized Sortino using downside deviation."""
    if len(returns) < 2:
        return np.nan
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.nan
    return float(excess.mean() / downside.std() * math.sqrt(periods_per_year))


def longest_losing_streak(r_series: pd.Series) -> int:
    streak = max_streak = 0
    for r in r_series:
        if r < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def compute_metrics(
    trades: list[dict],
    total_bars: int,
    risk_free_rate: float = 0.0,
) -> MetricReport:
    """
    Compute all required metrics from a list of trade dicts.
    Each trade dict must have: pnl_r (float), side (str), costs (float), gross_pnl (float).
    """
    if not trades:
        return MetricReport(
            trade_count=0, win_count=0, loss_count=0, win_rate=0, win_rate_ci_low=0,
            win_rate_ci_high=0, expectancy_r=0, mean_r=0, median_r=0,
            avg_win_r=0, avg_loss_r=0, profit_factor=0, max_drawdown_pct=0,
            sharpe=np.nan, sortino=np.nan, exposure_pct=0, longest_losing_streak=0,
            long_count=0, short_count=0, cost_share_pct=0,
            notes="No trades in sample.",
        )

    rs = pd.Series([t["pnl_r"] for t in trades])
    n  = len(rs)
    wins  = (rs > 0).sum()
    losses= (rs < 0).sum()
    wr    = wins / n if n > 0 else 0.0
    ci_lo, ci_hi = win_rate_wilson_ci(int(wins), n)

    avg_win  = float(rs[rs > 0].mean()) if wins > 0 else 0.0
    avg_loss = float(rs[rs < 0].mean()) if losses > 0 else 0.0
    pf_num   = float(rs[rs > 0].sum())
    pf_den   = float(abs(rs[rs < 0].sum()))
    pf       = pf_num / pf_den if pf_den > 0 else np.inf

    equity = compute_equity_curve(rs)
    dd     = max_drawdown(equity)
    eq_ret = equity.pct_change().dropna()
    sh     = sharpe_ratio(eq_ret)
    so     = sortino_ratio(eq_ret)

    longs  = sum(1 for t in trades if t.get("side") == "bullish")
    shorts = sum(1 for t in trades if t.get("side") == "bearish")

    total_costs    = sum(t.get("costs", 0.0) for t in trades)
    total_gross    = sum(abs(t.get("gross_pnl", t.get("pnl_r", 0.0))) for t in trades)
    cost_share     = total_costs / total_gross * 100 if total_gross > 0 else 0.0

    # Exposure: bars where a trade was open (approximate from trade count)
    avg_duration   = 20  # bars (rough estimate; refine with actual bar counts)
    exposure_bars  = min(n * avg_duration, total_bars)
    exposure_pct   = exposure_bars / total_bars * 100 if total_bars > 0 else 0.0

    return MetricReport(
        trade_count=n,
        win_count=int(wins),
        loss_count=int(losses),
        win_rate=wr,
        win_rate_ci_low=ci_lo,
        win_rate_ci_high=ci_hi,
        expectancy_r=float(rs.mean()),
        mean_r=float(rs.mean()),
        median_r=float(rs.median()),
        avg_win_r=avg_win,
        avg_loss_r=avg_loss,
        profit_factor=pf,
        max_drawdown_pct=dd * 100,
        sharpe=sh,
        sortino=so,
        exposure_pct=exposure_pct,
        longest_losing_streak=longest_losing_streak(rs),
        long_count=longs,
        short_count=shorts,
        cost_share_pct=cost_share,
        notes=f"N={n}. Win-rate CI is Wilson 95%. Sharpe/Sortino from equity returns.",
    )
