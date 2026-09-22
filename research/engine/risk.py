"""
ASTRA FUSION QUANT — Risk, Targets, Sizing (Milestone 3)
Entry/stop/target construction, net RR, position sizing at 0.25% risk.
Fixed-RR baseline (TP1=2D, TP2=3D, TP3=4D) and structure-mode targets.
All prices rounded to tick; full precision retained for calculations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


@dataclass
class RiskSnapshot:
    """Immutable risk snapshot frozen at signal time."""
    signal_id: str
    side: str
    reference_entry: float       # signal candle close — NOT a fill
    entry_kind: str = "next_available_market"
    structural_stop: float = 0.0
    stop_distance: float = 0.0   # D = abs(entry - stop)
    stop_dist_atr: float = 0.0   # D / ATR
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    gross_rr: float = 0.0        # abs(tp1-entry) / D
    friction_price: float = 0.0  # round-trip friction in price units
    net_rr: float = 0.0          # (abs(tp1-entry) - friction) / (D + friction)
    quantity: float = 0.0
    risk_budget: float = 0.0     # account units at risk
    target_mode: str = "fixed_rr"


def build_structural_stop(
    side: str,
    stop_anchor: float,
    tick_size: float,
    buffer_atr: float = 0.15,
    atr: float = 0.0,
) -> float:
    """
    Structural stop = anchor price +/- (0.15*ATR buffer).
    Long: stop below anchor, rounded DOWN to tick.
    Short: stop above anchor, rounded UP to tick.
    """
    buf = buffer_atr * atr
    if side == "bullish":
        raw_stop = stop_anchor - buf
        return _floor_to_tick(raw_stop, tick_size)
    else:
        raw_stop = stop_anchor + buf
        return _ceil_to_tick(raw_stop, tick_size)


def _floor_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return math.floor(price / tick) * tick


def _ceil_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return math.ceil(price / tick) * tick


def compute_fixed_rr_targets(
    entry: float, stop: float, side: str, tick_size: float
) -> tuple[float, float, float]:
    """TP1=2D, TP2=3D, TP3=4D from entry."""
    d = abs(entry - stop)
    sign = 1.0 if side == "bullish" else -1.0
    tp1 = entry + sign * 2.0 * d
    tp2 = entry + sign * 3.0 * d
    tp3 = entry + sign * 4.0 * d
    return tp1, tp2, tp3


def compute_net_rr(
    entry: float,
    stop: float,
    tp1: float,
    friction_price: float,
) -> float:
    """
    netRR = (abs(TP1 - E) - friction) / (D + friction)
    Reject non-positive numerator (returns 0.0).
    friction_price = estimated round-trip friction in price units.
    """
    D = abs(entry - stop)
    gain = abs(tp1 - entry) - friction_price
    cost = D + friction_price
    if gain <= 0 or cost <= 0:
        return 0.0
    return gain / cost


def compute_position_size(
    entry: float,
    stop: float,
    account_equity: float,
    risk_budget_pct: float,
    point_value: float,
    commission_per_side_pct: float,
    slippage_price: float,
    qty_increment: float,
    min_qty: float,
    max_concentration: float = 0.20,
) -> float:
    """
    Quantity = floor_to_increment(risk_budget / risk_per_unit).
    risk_budget = account_equity * risk_budget_pct / 100
    risk_per_unit = stop_distance * point_value + estimated_losing_costs_per_unit

    Uses ADVERSE entry slippage in sizing estimate.
    Returns 0.0 if below minimum size.
    """
    stop_dist = abs(entry - stop)
    if stop_dist <= 0 or point_value <= 0:
        return 0.0

    risk_budget = account_equity * risk_budget_pct / 100.0

    # Adverse entry: entry is worse by slippage
    adverse_entry = entry + slippage_price  # long: entry higher
    actual_stop_dist = abs(adverse_entry - stop)
    commission_cost = adverse_entry * point_value * (commission_per_side_pct / 100.0) * 2
    risk_per_unit = actual_stop_dist * point_value + commission_cost

    if risk_per_unit <= 0:
        return 0.0

    raw_qty = risk_budget / risk_per_unit
    qty = math.floor(raw_qty / qty_increment) * qty_increment

    # Concentration limit
    max_notional = account_equity * max_concentration
    max_qty_by_notional = math.floor(max_notional / (entry * point_value) / qty_increment) * qty_increment
    qty = min(qty, max_qty_by_notional)

    return qty if qty >= min_qty else 0.0


def build_risk_snapshot(
    signal_id: str,
    side: str,
    entry: float,
    stop_anchor: float,
    atr: float,
    tick_size: float,
    account_equity: float = 10_000.0,
    risk_budget_pct: float = 0.25,
    point_value: float = 1.0,
    commission_per_side_pct: float = 0.10,
    slippage_ticks: int = 1,
    qty_increment: float = 0.001,
    min_qty: float = 0.001,
    target_mode: str = "fixed_rr",
    structure_tp1: Optional[float] = None,
    structure_tp2: Optional[float] = None,
    structure_tp3: Optional[float] = None,
    stop_min_atr: float = 0.5,
    stop_max_atr: float = 3.0,
) -> RiskSnapshot:
    """Build a complete RiskSnapshot. Returns it regardless of validity — caller checks gates."""
    slippage_price = slippage_ticks * tick_size
    friction_price = entry * (commission_per_side_pct / 100.0) * 2 + slippage_price * 2

    stop = build_structural_stop(side, stop_anchor, tick_size, buffer_atr=0.15, atr=atr)
    D = abs(entry - stop)
    stop_dist_atr = D / atr if atr > 0 else 0.0

    if target_mode == "structure" and structure_tp1 is not None:
        tp1 = structure_tp1
        tp2 = structure_tp2
        tp3 = structure_tp3
    else:
        tp1, tp2, tp3 = compute_fixed_rr_targets(entry, stop, side, tick_size)

    gross_rr = abs(tp1 - entry) / D if D > 0 else 0.0
    net_rr   = compute_net_rr(entry, stop, tp1, friction_price)

    qty = compute_position_size(
        entry=entry, stop=stop,
        account_equity=account_equity, risk_budget_pct=risk_budget_pct,
        point_value=point_value, commission_per_side_pct=commission_per_side_pct,
        slippage_price=slippage_price, qty_increment=qty_increment, min_qty=min_qty,
    )

    return RiskSnapshot(
        signal_id=signal_id,
        side=side,
        reference_entry=entry,
        structural_stop=stop,
        stop_distance=D,
        stop_dist_atr=stop_dist_atr,
        tp1=tp1, tp2=tp2, tp3=tp3,
        gross_rr=gross_rr,
        friction_price=friction_price,
        net_rr=net_rr,
        quantity=qty,
        risk_budget=account_equity * risk_budget_pct / 100.0,
        target_mode=target_mode,
    )
