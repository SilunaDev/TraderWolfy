"""
ASTRA FUSION QUANT — Support/Resistance Levels (Milestone 2)
Swing zones with quality scoring, 30-level cap, nearest-3 display.
Level quality = 0.30*src_quality + 0.25*min(touches/3,1) + 0.20*exp(-age/100) + 0.25*min(rejections/3,1)
Source quality: chart pivot 0.5 | day/HTF1 0.75 | week/HTF2 1.0
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from research.features.pivots import Pivot

logger = logging.getLogger(__name__)

_level_counter = 0


def _lid() -> str:
    global _level_counter
    _level_counter += 1
    return f"LVL_{_level_counter:06d}"


SOURCE_QUALITY = {
    "chart_pivot": 0.50,
    "day_htf1":    0.75,
    "week_htf2":   1.00,
}

MAX_ACTIVE_LEVELS = 30
ZONE_HALF_WIDTH_ATR = 0.15    # half-width of swing zone in ATR units
MIN_TOUCH_SEPARATION_BARS = 3
REJECTION_MIN_MOVE_ATR = 0.25 # minimum move from edge to count as rejection


@dataclass
class Level:
    """Support/resistance level with quality tracking."""
    id: str
    role: str                          # 'support' or 'resistance'
    lower_bound: float
    upper_bound: float
    center: float
    source_quality: float
    known_at_bar: int
    known_at_time: pd.Timestamp
    source_pivot_ids: list[str] = field(default_factory=list)
    touches: int = 0
    last_touch_bar: int = -1
    rejections: int = 0
    age_bars: int = 0
    invalidated: bool = False
    invalidation_bar: Optional[int] = None
    role_flipped: bool = False          # True after breakout-retest role flip

    @property
    def quality(self) -> float:
        """
        Current quality score (0–1).
        quality = 0.30*src_q + 0.25*min(touches/3,1) + 0.20*exp(-age/100) + 0.25*min(rejections/3,1)
        """
        src  = 0.30 * self.source_quality
        tch  = 0.25 * min(self.touches / 3.0, 1.0)
        age  = 0.20 * math.exp(-self.age_bars / 100.0)
        rej  = 0.25 * min(self.rejections / 3.0, 1.0)
        return min(src + tch + age + rej, 1.0)

    @property
    def is_major(self) -> bool:
        return self.quality >= 0.6

    def contains(self, price: float) -> bool:
        return self.lower_bound <= price <= self.upper_bound

    def distance_from_center(self, price: float) -> float:
        return abs(price - self.center)


class LevelStore:
    """
    Manages active support/resistance levels.
    Cap at MAX_ACTIVE_LEVELS; display nearest 3 per side.
    """

    def __init__(self, max_levels: int = MAX_ACTIVE_LEVELS, tick_size: float = 1e-5):
        self.max_levels = max_levels
        self.tick_size = tick_size
        self._levels: list[Level] = []

    @property
    def active(self) -> list[Level]:
        return [l for l in self._levels if not l.invalidated]

    def add_from_pivot(
        self,
        pivot: Pivot,
        atr: float,
        current_bar: int,
        source: str = "chart_pivot",
    ) -> Optional[Level]:
        """
        Create or strengthen a level from a confirmed pivot.
        Zone half-width = 0.15*ATR. Match to nearest existing same-role zone
        whose center is within the frozen tolerance; tie-break by oldest ID.
        """
        if atr <= 0:
            return None

        role = "resistance" if pivot.side == "high" else "support"
        half = ZONE_HALF_WIDTH_ATR * atr
        new_center = pivot.price

        # Try to match existing zone
        src_q = SOURCE_QUALITY.get(source, 0.5)
        for existing in self.active:
            if existing.role == role:
                if abs(existing.center - new_center) <= half:
                    # Strengthen existing — do NOT migrate bounds
                    existing.source_quality = max(existing.source_quality, src_q)
                    existing.source_pivot_ids.append(pivot.id)
                    logger.debug("Strengthened level %s with pivot %s", existing.id, pivot.id)
                    return existing

        # Create new level
        lvl = Level(
            id=_lid(),
            role=role,
            lower_bound=new_center - half,
            upper_bound=new_center + half,
            center=new_center,
            source_quality=src_q,
            known_at_bar=pivot.confirmation_bar,
            known_at_time=pivot.confirmation_time,
            source_pivot_ids=[pivot.id],
        )

        self._prune_if_needed()
        self._levels.append(lvl)
        logger.debug("New level %s (role=%s, center=%.5f, q=%.2f)", lvl.id, role, lvl.center, lvl.quality)
        return lvl

    def _prune_if_needed(self) -> None:
        """Remove oldest low-quality levels if over cap."""
        active = self.active
        if len(active) >= self.max_levels:
            # Sort by quality ascending, remove lowest quality first
            to_remove = sorted(active, key=lambda l: (l.quality, l.known_at_bar))
            to_remove[0].invalidated = True
            logger.debug("Pruned level %s (quality=%.2f) to stay under cap", to_remove[0].id, to_remove[0].quality)

    def update_bar(
        self,
        bar_idx: int,
        high: float,
        low: float,
        close: float,
        open_: float,
        atr_prev: float,
    ) -> None:
        """
        Update touch/rejection counts and invalidation for all active levels.
        Called once per confirmed closed bar.
        """
        eps = max(2 * self.tick_size, 0.05 * atr_prev) if atr_prev > 0 else 2 * self.tick_size

        for lvl in self.active:
            lvl.age_bars += 1

            # Touch: bar intersects zone
            if low <= lvl.upper_bound and high >= lvl.lower_bound:
                if bar_idx - lvl.last_touch_bar >= MIN_TOUCH_SEPARATION_BARS:
                    lvl.touches += 1
                    lvl.last_touch_bar = bar_idx

                    # Rejection: close outside zone in expected direction, move >= 0.25*ATR
                    if lvl.role == "support" and close > lvl.upper_bound:
                        move = close - lvl.lower_bound
                        if move >= REJECTION_MIN_MOVE_ATR * atr_prev:
                            lvl.rejections += 1
                    elif lvl.role == "resistance" and close < lvl.lower_bound:
                        move = lvl.upper_bound - close
                        if move >= REJECTION_MIN_MOVE_ATR * atr_prev:
                            lvl.rejections += 1

            # Invalidation: close decisively through level
            if lvl.role == "support" and close < lvl.lower_bound - eps:
                lvl.invalidated = True
                lvl.invalidation_bar = bar_idx
            elif lvl.role == "resistance" and close > lvl.upper_bound + eps:
                lvl.invalidated = True
                lvl.invalidation_bar = bar_idx

    def nearest_levels(self, price: float, n: int = 3) -> dict:
        """Return nearest n support and n resistance levels to price."""
        supports = sorted(
            [l for l in self.active if l.role == "support"],
            key=lambda l: abs(l.center - price)
        )[:n]
        resistances = sorted(
            [l for l in self.active if l.role == "resistance"],
            key=lambda l: abs(l.center - price)
        )[:n]
        return {"support": supports, "resistance": resistances}

    def nearest_obstacle(self, price: float, direction: str) -> Optional[Level]:
        """
        Return nearest level that would be an obstacle in the given direction.
        direction='bullish' → nearest resistance above price
        direction='bearish' → nearest support below price
        """
        if direction == "bullish":
            candidates = [l for l in self.active if l.role == "resistance" and l.center > price]
            if not candidates:
                return None
            return min(candidates, key=lambda l: l.center - price)
        else:
            candidates = [l for l in self.active if l.role == "support" and l.center < price]
            if not candidates:
                return None
            return min(candidates, key=lambda l: price - l.center)

    def levels_between(self, price_a: float, price_b: float) -> list[Level]:
        """All active levels between two prices (major obstacles, quality >= 0.6)."""
        lo, hi = min(price_a, price_b), max(price_a, price_b)
        return [l for l in self.active if l.is_major and lo < l.center < hi]
