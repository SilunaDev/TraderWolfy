"""
ASTRA FUSION QUANT — Market Structure (Milestone 2)
BOS, CHoCH, MSS detection. Displacement. Sweep detection. Structure state.

All events use confirmed closed bars. Epsilon = max(2 ticks, 0.05*A).
Prior-bar ATR used for thresholds so event candle doesn't change its own threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import pandas as pd

from research.features.pivots import Pivot, PivotStore

logger = logging.getLogger(__name__)

_struct_counter = 0


def _sid() -> str:
    global _struct_counter
    _struct_counter += 1
    return f"SE_{_struct_counter:06d}"


TICK_APPROX = 1e-5  # fallback when tick size unknown


class StructureState(Enum):
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class BreakType(Enum):
    INITIAL = "INITIAL"       # Neutral state, first break
    BOS = "BOS"               # Break in established structure direction
    CHOCH = "CHoCH"           # First opposite break — structure change
    MSS = "MSS"               # CHoCH + displacement — market structure shift


@dataclass
class StructureEvent:
    """Immutable record of a structure break event."""
    id: str
    break_type: BreakType
    side: str                       # 'bullish' (up) or 'bearish' (down)
    broken_level_price: float       # price of the broken pivot
    broken_pivot_id: str
    confirmation_bar: int
    confirmation_time: pd.Timestamp
    close_price: float
    is_displacement: bool = False   # True if displacement candle


@dataclass
class SweepEvent:
    """Liquidity sweep event."""
    id: str
    side: str                       # 'bullish' (swept low) or 'bearish' (swept high)
    level_price: float              # swept level price
    wick_extreme: float             # H or L of sweep candle
    reclaim_close: float
    sweep_bar: int
    sweep_time: pd.Timestamp


class StructureEngine:
    """
    Tracks market structure state and emits BOS/CHoCH/MSS events.

    Two rising external highs and lows = bullish structure.
    Two falling highs and lows = bearish structure.
    Each pivot can be broken once.
    Structure state changes only from observable events.
    """

    def __init__(self, tick_size: float = TICK_APPROX):
        self.tick_size = tick_size
        self.state = StructureState.NEUTRAL
        self.events: list[StructureEvent] = []
        self.sweeps: list[SweepEvent] = []
        self._broken_pivot_ids: set[str] = set()

        # For structure tracking
        self._last_two_highs: list[Pivot] = []
        self._last_two_lows: list[Pivot] = []

    def epsilon(self, atr: float) -> float:
        return max(2 * self.tick_size, 0.05 * atr)

    def is_displacement(
        self,
        body: float, range_: float, close: float, open_: float, atr_prev: float
    ) -> bool:
        """
        Displacement: body >= 0.8*A, range >= 1.2*A, body/range >= 0.65,
        close within directional outer 25% of candle.
        Zero-range candle always fails.
        """
        if range_ <= 0 or atr_prev <= 0:
            return False
        if body < 0.8 * atr_prev:
            return False
        if range_ < 1.2 * atr_prev:
            return False
        if body / range_ < 0.65:
            return False
        # Directional outer 25%: bullish close in top 25%, bearish in bottom 25%
        bullish_candle = close > open_
        if bullish_candle:
            return close >= (open_ + 0.75 * range_) if open_ < close else False
        else:
            return close <= (open_ - 0.75 * range_) if open_ > close else False

    def update_structure_direction(self, ext_store: PivotStore, up_to_bar: int) -> None:
        """
        Update internal structure direction tracking from external pivot store.
        Two rising external H+L = bullish. Two falling = bearish.
        """
        confirmed = ext_store.confirmed_before(up_to_bar)
        highs = sorted([p for p in confirmed if p.side == "high"], key=lambda p: p.confirmation_bar)
        lows  = sorted([p for p in confirmed if p.side == "low"],  key=lambda p: p.confirmation_bar)

        self._last_two_highs = highs[-2:] if len(highs) >= 2 else highs
        self._last_two_lows  = lows[-2:]  if len(lows) >= 2  else lows

    def _eval_structure_state(self, atr: float) -> None:
        """Re-evaluate structure state from last two highs and lows."""
        eps = self.epsilon(atr)
        h = self._last_two_highs
        l = self._last_two_lows
        if len(h) >= 2 and len(l) >= 2:
            higher_highs = h[-1].price - h[-2].price > eps
            higher_lows  = l[-1].price - l[-2].price > eps
            lower_highs  = h[-2].price - h[-1].price > eps
            lower_lows   = l[-2].price - l[-1].price > eps
            if higher_highs and higher_lows:
                self.state = StructureState.BULLISH
            elif lower_highs and lower_lows:
                self.state = StructureState.BEARISH
            else:
                self.state = StructureState.NEUTRAL

    def process_bar(
        self,
        bar_idx: int,
        ts: pd.Timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
        atr_prev: float,                 # prior-bar ATR
        ext_store: PivotStore,           # external pivot store
    ) -> list[StructureEvent]:
        """
        Process one closed bar. Returns new structure events created this bar.
        Updates structure state. Detects BOS/CHoCH/MSS.
        """
        if atr_prev <= 0:
            return []

        eps = self.epsilon(atr_prev)
        body = abs(close - open_)
        range_ = high - low
        disp = self.is_displacement(body, range_, close, open_, atr_prev)

        new_events: list[StructureEvent] = []
        confirmed = ext_store.confirmed_before(bar_idx)

        # Check for bullish break: close above active high by epsilon
        active_highs = [
            p for p in confirmed
            if p.side == "high" and p.id not in self._broken_pivot_ids
        ]
        active_lows = [
            p for p in confirmed
            if p.side == "low" and p.id not in self._broken_pivot_ids
        ]

        # Bullish break
        for pivot in sorted(active_highs, key=lambda p: p.price):
            if close > pivot.price + eps:
                break_type = self._classify_break("bullish", eps)
                self._broken_pivot_ids.add(pivot.id)
                ev = StructureEvent(
                    id=_sid(),
                    break_type=break_type if not disp or break_type == BreakType.CHOCH
                                else BreakType.MSS,
                    side="bullish",
                    broken_level_price=pivot.price,
                    broken_pivot_id=pivot.id,
                    confirmation_bar=bar_idx,
                    confirmation_time=ts,
                    close_price=close,
                    is_displacement=disp,
                )
                if break_type == BreakType.CHOCH and disp:
                    ev.break_type = BreakType.MSS
                new_events.append(ev)
                self.events.append(ev)
                logger.debug("Structure %s at bar %d: %s", ev.break_type.value, bar_idx, ev.id)
                break  # one break per bar per side

        # Bearish break
        for pivot in sorted(active_lows, key=lambda p: p.price, reverse=True):
            if close < pivot.price - eps:
                break_type = self._classify_break("bearish", eps)
                self._broken_pivot_ids.add(pivot.id)
                ev = StructureEvent(
                    id=_sid(),
                    break_type=break_type,
                    side="bearish",
                    broken_level_price=pivot.price,
                    broken_pivot_id=pivot.id,
                    confirmation_bar=bar_idx,
                    confirmation_time=ts,
                    close_price=close,
                    is_displacement=disp,
                )
                if break_type == BreakType.CHOCH and disp:
                    ev.break_type = BreakType.MSS
                new_events.append(ev)
                self.events.append(ev)
                break

        # Update structure direction
        self.update_structure_direction(ext_store, bar_idx)
        self._eval_structure_state(atr_prev)
        return new_events

    def _classify_break(self, direction: str, eps: float) -> BreakType:
        """Classify this break as INITIAL, BOS, or CHoCH based on current state."""
        if self.state == StructureState.NEUTRAL:
            return BreakType.INITIAL
        if direction == "bullish" and self.state == StructureState.BULLISH:
            return BreakType.BOS
        if direction == "bearish" and self.state == StructureState.BEARISH:
            return BreakType.BOS
        return BreakType.CHOCH

    def detect_sweep(
        self,
        bar_idx: int,
        ts: pd.Timestamp,
        high: float,
        low: float,
        close: float,
        open_: float,
        atr_prev: float,
        active_levels: list[dict],   # list of active level dicts with 'price' and 'side'
    ) -> list[SweepEvent]:
        """
        Detect liquidity sweeps against pre-existing active levels.
        Bullish sweep: low < level - eps AND close > level (wick through, reclaim on close).
        Bearish sweep: high > level + eps AND close < level.
        Wick penetration without reclaim is NOT a sweep.
        """
        eps = self.epsilon(atr_prev)
        new_sweeps = []

        for lvl in active_levels:
            price = lvl["price"]
            side = lvl.get("side", "unknown")

            # Bullish sweep of a low/support level
            if low < price - eps and close > price:
                sw = SweepEvent(
                    id=_sid(),
                    side="bullish",
                    level_price=price,
                    wick_extreme=low,
                    reclaim_close=close,
                    sweep_bar=bar_idx,
                    sweep_time=ts,
                )
                new_sweeps.append(sw)
                self.sweeps.append(sw)

            # Bearish sweep of a high/resistance level
            elif high > price + eps and close < price:
                sw = SweepEvent(
                    id=_sid(),
                    side="bearish",
                    level_price=price,
                    wick_extreme=high,
                    reclaim_close=close,
                    sweep_bar=bar_idx,
                    sweep_time=ts,
                )
                new_sweeps.append(sw)
                self.sweeps.append(sw)

        return new_sweeps

    def latest_event(self, side: Optional[str] = None, max_age_bars: int = 10) -> Optional[StructureEvent]:
        """Return most recent event optionally filtered by side."""
        for ev in reversed(self.events):
            if side and ev.side != side:
                continue
            return ev
        return None

    def structure_direction_score(self) -> float:
        """Return +1 for bullish, -1 for bearish, 0 for neutral."""
        if self.state == StructureState.BULLISH:
            return 1.0
        elif self.state == StructureState.BEARISH:
            return -1.0
        return 0.0
