"""
ASTRA FUSION QUANT — Regime Classifier (Milestone 2)
Two-axis regime: Directional (BULL/BEAR/RANGE/TRANSITION) + Volatility (LOW/NORMAL/HIGH/EXTREME).
3-bar confirmation to enter a state. Retention rules per blueprint Section 11.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DirectionalState(Enum):
    BULL_TREND  = "BULL_TREND"
    BEAR_TREND  = "BEAR_TREND"
    RANGE       = "RANGE"
    TRANSITION  = "TRANSITION"


class VolatilityState(Enum):
    LOW     = "LOW"      # NATR percentile < 20
    NORMAL  = "NORMAL"   # 20-80
    HIGH    = "HIGH"     # 80-95
    EXTREME = "EXTREME"  # > 95


@dataclass
class RegimeSnapshot:
    directional: DirectionalState
    volatility: VolatilityState
    trend_t: float
    adx: float
    er20: float
    natr_pct: float


class RegimeClassifier:
    """
    Classifies market regime on each confirmed closed bar.

    Directional classification (all thresholds are UNVALIDATED hypotheses):
    - Bull candidate:  T >= 0.35, ADX >= 23, ER20 >= 0.30, no bearish external structure
    - Bear candidate:  mirrored
    - Range candidate: ADX <= 18, ER20 <= 0.25, valid range (2 touches each bound, width 2-8*A)
    - Otherwise:       Transition

    3 consecutive candidate closes required to enter a state.
    Retention: bull while ADX>=20 and T>=0.20; range while ADX<=21 and no boundary break.
    On retention failure → TRANSITION immediately, then 3 closes for new state.

    Volatility: NATR percentile thresholds (UNVALIDATED): 20/80/95.
    """

    def __init__(
        self,
        adx_trend_min: float = 23.0,
        adx_range_max: float = 18.0,
        adx_retain_trend: float = 20.0,
        adx_retain_range: float = 21.0,
        er_trend_min: float = 0.30,
        er_range_max: float = 0.25,
        trend_t_min: float = 0.35,
        trend_t_retain: float = 0.20,
        vol_low_pct: float = 20.0,
        vol_high_pct: float = 80.0,
        vol_extreme_pct: float = 95.0,
        confirm_bars: int = 3,
    ):
        self.adx_trend_min    = adx_trend_min
        self.adx_range_max    = adx_range_max
        self.adx_retain_trend = adx_retain_trend
        self.adx_retain_range = adx_retain_range
        self.er_trend_min     = er_trend_min
        self.er_range_max     = er_range_max
        self.trend_t_min      = trend_t_min
        self.trend_t_retain   = trend_t_retain
        self.vol_low_pct      = vol_low_pct
        self.vol_high_pct     = vol_high_pct
        self.vol_extreme_pct  = vol_extreme_pct
        self.confirm_bars     = confirm_bars

        self._directional = DirectionalState.TRANSITION
        self._volatility  = VolatilityState.NORMAL

        # Candidate tracking
        self._candidate: Optional[DirectionalState] = None
        self._candidate_count: int = 0

        # Range boundary tracking
        self._range_low: Optional[float] = None
        self._range_high: Optional[float] = None

        self._history: list[RegimeSnapshot] = []

    @property
    def directional(self) -> DirectionalState:
        return self._directional

    @property
    def volatility(self) -> VolatilityState:
        return self._volatility

    def update(
        self,
        trend_t: float,
        adx: float,
        er20: float,
        natr_pct: float,
        structure_dir: float,         # +1 bull, -1 bear, 0 neutral
        has_bearish_ext_structure: bool,
        has_bullish_ext_structure: bool,
        range_low: Optional[float],
        range_high: Optional[float],
        atr: float,
        close: float,
    ) -> RegimeSnapshot:
        """
        Update regime for one closed bar.
        Returns the current RegimeSnapshot.
        """
        # ── Volatility state (independent of directional) ────────────────────
        if not np.isnan(natr_pct):
            if natr_pct > self.vol_extreme_pct:
                self._volatility = VolatilityState.EXTREME
            elif natr_pct > self.vol_high_pct:
                self._volatility = VolatilityState.HIGH
            elif natr_pct < self.vol_low_pct:
                self._volatility = VolatilityState.LOW
            else:
                self._volatility = VolatilityState.NORMAL

        # ── Directional state ────────────────────────────────────────────────
        candidate = self._classify_candidate(
            trend_t, adx, er20,
            has_bearish_ext_structure, has_bullish_ext_structure,
            range_low, range_high, atr
        )

        current = self._directional

        # Check retention of current state
        retained = self._check_retention(current, trend_t, adx, er20, close, range_low, range_high, atr)

        if not retained:
            # Fail retention → immediately TRANSITION
            self._directional = DirectionalState.TRANSITION
            self._candidate = None
            self._candidate_count = 0
        else:
            # Accumulate candidate confirmation
            if candidate == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = candidate
                self._candidate_count = 1

            # Enter new state after 3 consecutive candidate closes
            if (
                self._candidate_count >= self.confirm_bars
                and self._candidate != current
                and self._candidate is not None
            ):
                self._directional = self._candidate
                self._candidate_count = 0
                logger.debug("Regime → %s", self._directional.value)

        # Update range bounds if in range mode
        if self._directional == DirectionalState.RANGE and range_low and range_high:
            self._range_low  = range_low
            self._range_high = range_high

        snap = RegimeSnapshot(
            directional=self._directional,
            volatility=self._volatility,
            trend_t=trend_t,
            adx=adx,
            er20=er20,
            natr_pct=natr_pct,
        )
        self._history.append(snap)
        return snap

    def _classify_candidate(
        self,
        trend_t: float,
        adx: float,
        er20: float,
        has_bearish_ext: bool,
        has_bullish_ext: bool,
        range_low: Optional[float],
        range_high: Optional[float],
        atr: float,
    ) -> DirectionalState:
        """Classify this bar's candidate state (before confirmation count)."""
        if np.isnan(trend_t) or np.isnan(adx) or np.isnan(er20):
            return DirectionalState.TRANSITION

        # Bull trend candidate
        if (
            trend_t >= self.trend_t_min
            and adx >= self.adx_trend_min
            and er20 >= self.er_trend_min
            and not has_bearish_ext
        ):
            return DirectionalState.BULL_TREND

        # Bear trend candidate
        if (
            trend_t <= -self.trend_t_min
            and adx >= self.adx_trend_min
            and er20 >= self.er_trend_min
            and not has_bullish_ext
        ):
            return DirectionalState.BEAR_TREND

        # Range candidate
        if (
            adx <= self.adx_range_max
            and er20 <= self.er_range_max
            and range_low is not None
            and range_high is not None
            and atr > 0
        ):
            width = range_high - range_low
            if 2 * atr <= width <= 8 * atr:
                return DirectionalState.RANGE

        return DirectionalState.TRANSITION

    def _check_retention(
        self,
        current: DirectionalState,
        trend_t: float,
        adx: float,
        er20: float,
        close: float,
        range_low: Optional[float],
        range_high: Optional[float],
        atr: float,
    ) -> bool:
        """Return True if current state is retained, False if retention fails."""
        if current in (DirectionalState.BULL_TREND, DirectionalState.BEAR_TREND):
            t_sign = 1.0 if current == DirectionalState.BULL_TREND else -1.0
            return (
                not np.isnan(adx) and adx >= self.adx_retain_trend
                and not np.isnan(trend_t) and trend_t * t_sign >= self.trend_t_retain
            )
        elif current == DirectionalState.RANGE:
            if np.isnan(adx) or adx > self.adx_retain_range:
                return False
            if range_low and range_high and atr > 0:
                eps = max(2e-5, 0.05 * atr)
                if close < range_low - eps or close > range_high + eps:
                    return False
            return True
        return True   # TRANSITION always retains (waiting for 3-bar confirmation)

    def allows_new_setups(self, setup_type: str) -> bool:
        """Return True if the current regime allows new setups of this type."""
        if self._volatility == VolatilityState.EXTREME:
            return False
        if self._directional == DirectionalState.TRANSITION:
            # Only sweep-reversal or breakout-retest can qualify through complete sequence
            return setup_type in ("sweep_reversal", "breakout_retest")
        if self._volatility == VolatilityState.LOW:
            return setup_type == "breakout_retest"
        return True

    def is_range_valid(self) -> bool:
        return self._directional == DirectionalState.RANGE

    def range_position(self, close: float) -> float:
        """Return 0-1 position within range. 0=bottom, 1=top. NaN if not in range."""
        if self._range_low is None or self._range_high is None:
            return np.nan
        width = self._range_high - self._range_low
        if width <= 0:
            return np.nan
        return max(0.0, min(1.0, (close - self._range_low) / width))
