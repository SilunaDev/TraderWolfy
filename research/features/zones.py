"""
ASTRA FUSION QUANT — SMC Zones: FVG + Order Blocks (Milestone 2)
FVG and order block detection with FRESH→TOUCHED→CONSUMED/INVALIDATED/EXPIRED lifecycle.
Breaker and inverse FVG are optional (off by default).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from research.features.structure import StructureEvent, BreakType

logger = logging.getLogger(__name__)

_zone_counter = 0


def _zid() -> str:
    global _zone_counter
    _zone_counter += 1
    return f"ZNE_{_zone_counter:06d}"


class ZoneState(Enum):
    FRESH = "FRESH"
    TOUCHED = "TOUCHED"
    CONSUMED = "CONSUMED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


@dataclass
class Zone:
    """SMC zone (FVG or order block) with frozen bounds and lifecycle state."""
    id: str
    zone_type: str          # 'fvg_bull', 'fvg_bear', 'ob_bull', 'ob_bear'
    direction: str          # 'bullish' or 'bearish'
    lower_bound: float      # frozen at creation
    upper_bound: float      # frozen at creation
    origin_bar: int         # bar that created the zone
    known_at_bar: int       # bar when zone is confirmed (eligible_from = known_at + 1)
    origin_time: pd.Timestamp
    known_at_time: pd.Timestamp
    state: ZoneState = ZoneState.FRESH
    expiry_bar: int = 0     # bar at which zone expires
    touch_count: int = 0
    last_touch_bar: int = -1
    invalidation_bar: Optional[int] = None
    is_displacement_zone: bool = False
    htf_aligned: bool = False
    # Signal tracking: at most one signal per zone/setup combo
    emitted_signal_ids: list[str] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def mid(self) -> float:
        return (self.upper_bound + self.lower_bound) / 2.0

    @property
    def eligible_from_bar(self) -> int:
        return self.known_at_bar + 1

    @property
    def quality(self) -> float:
        """
        Zone quality for location score.
        0.6 base + 0.2 if displacement + 0.2 if HTF aligned, capped at 1.0.
        """
        return min(0.6 + 0.2 * self.is_displacement_zone + 0.2 * self.htf_aligned, 1.0)

    def is_active(self, current_bar: int) -> bool:
        return (
            self.state in (ZoneState.FRESH, ZoneState.TOUCHED)
            and current_bar >= self.eligible_from_bar
            and current_bar < self.expiry_bar
        )


class ZoneEngine:
    """
    Detects and manages FVG and order block zones.
    All creation uses prior-bar ATR for thresholds.
    """

    MIN_FVG_WIDTH_ATR = 0.10
    MAX_OB_WIDTH_ATR = 2.0
    MIN_OB_WIDTH_ATR = 0.10
    OB_LOOKBACK = 10              # bars to look back for qualifying OB candle
    DEFAULT_EXPIRY_BARS = 100

    def __init__(
        self,
        fvg_enabled: bool = True,
        ob_enabled: bool = True,
        breaker_enabled: bool = False,
        inverse_fvg_enabled: bool = False,
        expiry_bars: int = DEFAULT_EXPIRY_BARS,
        tick_size: float = 1e-5,
    ):
        self.fvg_enabled = fvg_enabled
        self.ob_enabled = ob_enabled
        self.breaker_enabled = breaker_enabled
        self.inverse_fvg_enabled = inverse_fvg_enabled
        self.expiry_bars = expiry_bars
        self.tick_size = tick_size
        self._zones: list[Zone] = []
        self._bar_history: list[dict] = []   # rolling window of recent bars

    @property
    def active_zones(self) -> list[Zone]:
        return [z for z in self._zones if z.state in (ZoneState.FRESH, ZoneState.TOUCHED)]

    def _epsilon(self, atr: float) -> float:
        return max(2 * self.tick_size, 0.05 * atr)

    def process_bar(
        self,
        bar_idx: int,
        ts: pd.Timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
        atr_prev: float,
        structure_events: list[StructureEvent],
        htf_bias: str = "neutral",
    ) -> list[Zone]:
        """
        Process one closed bar:
        1. Update existing zone states (touch/consume/invalidate/expire)
        2. Detect new FVGs and order blocks
        Returns newly created zones this bar.
        """
        # Store bar in history
        self._bar_history.append({
            "bar_idx": bar_idx, "ts": ts,
            "open": open_, "high": high, "low": low, "close": close,
            "atr_prev": atr_prev,
        })
        if len(self._bar_history) > self.OB_LOOKBACK + 5:
            self._bar_history.pop(0)

        eps = self._epsilon(atr_prev) if atr_prev > 0 else self.tick_size

        # 1. Update existing zones
        for zone in self._zones:
            if zone.state in (ZoneState.CONSUMED, ZoneState.INVALIDATED, ZoneState.EXPIRED):
                continue
            if bar_idx >= zone.expiry_bar:
                zone.state = ZoneState.EXPIRED
                continue
            if bar_idx < zone.eligible_from_bar:
                continue

            # Touch: bar intersects zone
            if low <= zone.upper_bound and high >= zone.lower_bound:
                if bar_idx != zone.known_at_bar:  # creation candle cannot count
                    zone.touch_count += 1
                    zone.last_touch_bar = bar_idx
                    zone.state = ZoneState.TOUCHED

            # Invalidation
            if zone.direction == "bullish" and close < zone.lower_bound - eps:
                zone.state = ZoneState.INVALIDATED
                zone.invalidation_bar = bar_idx
            elif zone.direction == "bearish" and close > zone.upper_bound + eps:
                zone.state = ZoneState.INVALIDATED
                zone.invalidation_bar = bar_idx

        if atr_prev <= 0:
            return []

        new_zones: list[Zone] = []

        # 2. Detect FVGs (requires 3 bars: t-2, t-1, t)
        if self.fvg_enabled and len(self._bar_history) >= 3:
            fvg = self._detect_fvg(bar_idx, ts, atr_prev, htf_bias)
            if fvg:
                self._zones.append(fvg)
                new_zones.append(fvg)

        # 3. Detect order blocks at structure events this bar
        if self.ob_enabled:
            for ev in structure_events:
                if ev.confirmation_bar == bar_idx:
                    ob = self._detect_ob(bar_idx, ts, ev, atr_prev, htf_bias)
                    if ob:
                        self._zones.append(ob)
                        new_zones.append(ob)

        return new_zones

    def _detect_fvg(
        self, bar_idx: int, ts: pd.Timestamp, atr_prev: float, htf_bias: str
    ) -> Optional[Zone]:
        """
        Bullish FVG at bar t: L[t] > H[t-2] and width >= 0.1*ATR.
        Middle candle (t-1) must satisfy displacement using its own prior ATR.
        Zone is [H[t-2], L[t]]. Eligible from t+1.
        """
        h = self._bar_history
        if len(h) < 3:
            return None

        t   = h[-1]   # current bar
        t_1 = h[-2]   # middle bar
        t_2 = h[-3]   # two bars ago

        # Use prior ATR at t-1 for middle-candle displacement check
        atr_t1 = t_1.get("atr_prev", atr_prev)

        # Bullish FVG
        if t["low"] > t_2["high"]:
            width = t["low"] - t_2["high"]
            if width >= self.MIN_FVG_WIDTH_ATR * atr_prev:
                # Check middle candle displacement
                body_t1 = abs(t_1["close"] - t_1["open"])
                range_t1 = t_1["high"] - t_1["low"]
                disp = (
                    atr_t1 > 0
                    and body_t1 >= 0.8 * atr_t1
                    and range_t1 >= 1.2 * atr_t1
                    and (body_t1 / range_t1 >= 0.65 if range_t1 > 0 else False)
                )
                htf_aligned = htf_bias == "bullish"
                zone = Zone(
                    id=_zid(),
                    zone_type="fvg_bull",
                    direction="bullish",
                    lower_bound=t_2["high"],
                    upper_bound=t["low"],
                    origin_bar=bar_idx,
                    known_at_bar=bar_idx,
                    origin_time=ts,
                    known_at_time=ts,
                    expiry_bar=bar_idx + self.expiry_bars,
                    is_displacement_zone=disp,
                    htf_aligned=htf_aligned,
                )
                logger.debug("Bullish FVG %s at bar %d (width=%.5f)", zone.id, bar_idx, width)
                return zone

        # Bearish FVG
        if t["high"] < t_2["low"]:
            width = t_2["low"] - t["high"]
            if width >= self.MIN_FVG_WIDTH_ATR * atr_prev:
                body_t1 = abs(t_1["close"] - t_1["open"])
                range_t1 = t_1["high"] - t_1["low"]
                disp = (
                    atr_t1 > 0
                    and body_t1 >= 0.8 * atr_t1
                    and range_t1 >= 1.2 * atr_t1
                    and (body_t1 / range_t1 >= 0.65 if range_t1 > 0 else False)
                )
                htf_aligned = htf_bias == "bearish"
                zone = Zone(
                    id=_zid(),
                    zone_type="fvg_bear",
                    direction="bearish",
                    lower_bound=t["high"],
                    upper_bound=t_2["low"],
                    origin_bar=bar_idx,
                    known_at_bar=bar_idx,
                    origin_time=ts,
                    known_at_time=ts,
                    expiry_bar=bar_idx + self.expiry_bars,
                    is_displacement_zone=disp,
                    htf_aligned=htf_aligned,
                )
                logger.debug("Bearish FVG %s at bar %d (width=%.5f)", zone.id, bar_idx, width)
                return zone

        return None

    def _detect_ob(
        self,
        bar_idx: int,
        ts: pd.Timestamp,
        event: StructureEvent,
        atr_prev: float,
        htf_bias: str,
    ) -> Optional[Zone]:
        """
        Order block at a displacement break:
        - Bullish OB: at a bullish displacement break, most recent bearish candle in prior 10 bars
        - Its full high-low range becomes the zone
        - Known at the break confirmation bar
        """
        direction = event.side   # 'bullish' or 'bearish'
        lookback = self._bar_history[:-1]  # exclude current bar

        # Find qualifying OB candle
        ob_bar = None
        if direction == "bullish":
            # Most recent bearish candle (close < open)
            for b in reversed(lookback[-self.OB_LOOKBACK:]):
                if b["close"] < b["open"]:
                    ob_bar = b
                    break
        else:
            # Most recent bullish candle (close > open)
            for b in reversed(lookback[-self.OB_LOOKBACK:]):
                if b["close"] > b["open"]:
                    ob_bar = b
                    break

        if ob_bar is None:
            return None

        lo = ob_bar["low"]
        hi = ob_bar["high"]
        width = hi - lo

        if width < self.MIN_OB_WIDTH_ATR * atr_prev:
            return None
        if width > self.MAX_OB_WIDTH_ATR * atr_prev:
            return None

        zone_type = f"ob_{direction[:4]}"
        htf_aligned = (
            (direction == "bullish" and htf_bias == "bullish")
            or (direction == "bearish" and htf_bias == "bearish")
        )
        zone = Zone(
            id=_zid(),
            zone_type=zone_type,
            direction=direction,
            lower_bound=lo,
            upper_bound=hi,
            origin_bar=ob_bar["bar_idx"],
            known_at_bar=bar_idx,         # known at break confirmation
            origin_time=ob_bar["ts"],
            known_at_time=ts,
            expiry_bar=bar_idx + self.expiry_bars,
            is_displacement_zone=event.is_displacement,
            htf_aligned=htf_aligned,
        )
        logger.debug("OB %s (%s) at bar %d from break event %s", zone.id, direction, bar_idx, event.id)
        return zone

    def get_zones_near(self, price: float, direction: str, atr: float, max_dist_atr: float = 0.25) -> list[Zone]:
        """Return active zones in the given direction within max_dist_atr of price."""
        result = []
        for z in self.active_zones:
            if z.direction != direction:
                continue
            dist = min(abs(price - z.lower_bound), abs(price - z.upper_bound))
            if dist <= max_dist_atr * atr:
                result.append(z)
        return result
