"""
ASTRA FUSION QUANT — Fibonacci Levels (Milestone 2)
Impulse anchor detection, retracement/extension levels, OTE zone (0.618–0.786).
Anchors freeze when the setup is created.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from research.features.pivots import Pivot, PivotStore

FIB_RETRACEMENTS  = [0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 1.0]
FIB_EXTENSIONS    = [1.272, 1.618, 2.0]
OTE_LOW, OTE_HIGH = 0.618, 0.786

@dataclass
class FibAnchor:
    origin_price: float   # impulse start (a)
    end_price: float      # impulse end   (b)
    direction: str        # 'bullish' or 'bearish'
    origin_bar: int
    end_bar: int
    is_valid: bool = True

    def retracement(self, ratio: float) -> float:
        a, b = self.origin_price, self.end_price
        if self.direction == "bullish":
            return b - ratio * (b - a)
        else:
            return b + ratio * (a - b)

    def extension(self, ratio: float) -> float:
        a, b = self.origin_price, self.end_price
        if self.direction == "bullish":
            return a + ratio * (b - a)
        else:
            return a - ratio * (a - b)

    def ote_zone(self) -> tuple[float, float]:
        lo = self.retracement(OTE_HIGH)
        hi = self.retracement(OTE_LOW)
        return (min(lo, hi), max(lo, hi))

    def in_ote(self, price: float) -> bool:
        lo, hi = self.ote_zone()
        return lo <= price <= hi


def find_impulse(
    store: PivotStore,
    direction: str,
    atr: float,
    up_to_bar: int,
    min_height_atr: float = 2.0,
    min_origin_sep_bars: int = 5,
) -> Optional[FibAnchor]:
    """
    Find the latest confirmed alternating external impulse aligned with direction.
    Impulse height >= 2*ATR, origin separation >= 5 bars.
    """
    if atr <= 0:
        return None
    confirmed = [p for p in store.all_pivots if p.confirmation_bar <= up_to_bar and not p.ambiguous]
    if len(confirmed) < 2:
        return None

    if direction == "bullish":
        # Upward leg: low → high
        lows  = [p for p in confirmed if p.side == "low"]
        highs = [p for p in confirmed if p.side == "high"]
        for h in reversed(highs):
            candidates = [l for l in lows if l.origin_bar < h.origin_bar
                          and h.origin_bar - l.origin_bar >= min_origin_sep_bars]
            if not candidates:
                continue
            l = max(candidates, key=lambda x: x.origin_bar)
            height = h.price - l.price
            if height >= min_height_atr * atr:
                return FibAnchor(l.price, h.price, "bullish", l.origin_bar, h.origin_bar)
    else:
        highs = [p for p in confirmed if p.side == "high"]
        lows  = [p for p in confirmed if p.side == "low"]
        for l in reversed(lows):
            candidates = [h for h in highs if h.origin_bar < l.origin_bar
                          and l.origin_bar - h.origin_bar >= min_origin_sep_bars]
            if not candidates:
                continue
            h = max(candidates, key=lambda x: x.origin_bar)
            height = h.price - l.price
            if height >= min_height_atr * atr:
                return FibAnchor(h.price, l.price, "bearish", h.origin_bar, l.origin_bar)
    return None


def fib_confluence_flag(price: float, anchor: Optional[FibAnchor], atr: float) -> bool:
    """True if price is within 0.1*ATR of any Fibonacci retracement level."""
    if anchor is None or atr <= 0:
        return False
    for r in FIB_RETRACEMENTS:
        lvl = anchor.retracement(r)
        if abs(price - lvl) <= 0.1 * atr:
            return True
    return False
