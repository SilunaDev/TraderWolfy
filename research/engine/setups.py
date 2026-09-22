"""
ASTRA FUSION QUANT — Four Setup State Machines (Milestone 2)
A. Trend Pullback  B. Sweep Reversal  C. Breakout Retest  D. Range Mean Reversion

States: IDLE → ARMED → WAIT_CONFIRMATION → WAIT_RETEST → TRIGGERED
Terminal: INVALIDATED | EXPIRED
All state transitions use confirmed closed bars.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_setup_counter = 0

def _setup_id(type_code: str) -> str:
    global _setup_counter
    _setup_counter += 1
    return f"{type_code}_{_setup_counter:06d}"

class SetupState(Enum):
    IDLE             = "IDLE"
    ARMED            = "ARMED"
    WAIT_CONFIRMATION= "WAIT_CONFIRMATION"
    WAIT_RETEST      = "WAIT_RETEST"
    TRIGGERED        = "TRIGGERED"
    INVALIDATED      = "INVALIDATED"
    EXPIRED          = "EXPIRED"

class SetupType(Enum):
    TREND_PULLBACK   = "trend_pullback"
    SWEEP_REVERSAL   = "sweep_reversal"
    BREAKOUT_RETEST  = "breakout_retest"
    RANGE_REVERSION  = "range_reversion"

@dataclass
class SetupRecord:
    """Immutable setup record created when a candidate arms."""
    id: str
    type: SetupType
    side: str                        # 'bullish' or 'bearish'
    state: SetupState
    arm_bar: int
    arm_time: pd.Timestamp
    # Frozen anchor prices at arming
    anchor_price: float              # key reference (zone center, sweep low, boundary)
    stop_anchor: float               # used to build structural stop
    # State transition timestamps
    state_bars: dict = field(default_factory=dict)
    expiry_bar: int = 0
    invalidation_reason: str = ""
    triggered_bar: Optional[int] = None
    triggered_time: Optional[pd.Timestamp] = None
    # Sub-anchors
    zone_id: Optional[str] = None
    sweep_extreme: Optional[float] = None
    mss_bar: Optional[int] = None
    retest_low: Optional[float] = None
    boundary_price: Optional[float] = None
    internal_high_at_sweep: Optional[float] = None


def _generic_bullish_trigger(
    open_: float, high: float, low: float, close: float,
    prev_high: float, prev_open: float, prev_close: float,
    zone_upper: Optional[float], atr: float
) -> bool:
    """
    Generic bullish trigger:
    C>O, close in upper 30% of candle, body>=0.25*ATR, PLUS at least one of:
    - close above previous high
    - bullish body engulfs previous bearish body
    - lower wick >= 2*body at locked support zone
    Zero-range bars fail.
    """
    range_ = high - low
    if range_ <= 0 or atr <= 0:
        return False
    body = close - open_
    if body <= 0:
        return False
    if body < 0.25 * atr:
        return False
    # Upper 30% of candle
    if close < low + 0.70 * range_:
        return False
    # At least one confirmatory condition
    cond1 = close > prev_high
    cond2 = (open_ < prev_open and close > prev_close and prev_close < prev_open)
    lower_wick = open_ - low
    cond3 = (zone_upper is not None and lower_wick >= 2 * body)
    return cond1 or cond2 or cond3


def _generic_bearish_trigger(
    open_: float, high: float, low: float, close: float,
    prev_low: float, prev_open: float, prev_close: float,
    zone_lower: Optional[float], atr: float
) -> bool:
    range_ = high - low
    if range_ <= 0 or atr <= 0:
        return False
    body = open_ - close
    if body <= 0:
        return False
    if body < 0.25 * atr:
        return False
    if close > high - 0.70 * range_:
        return False
    cond1 = close < prev_low
    cond2 = (open_ > prev_open and close < prev_close and prev_close > prev_open)
    upper_wick = high - open_
    cond3 = (zone_lower is not None and upper_wick >= 2 * body)
    return cond1 or cond2 or cond3


class TrendPullbackSetup:
    """
    Setup A: Trend Pullback
    Bull conditions:
    1. Bull-trend regime; HTF1 bullish; HTF2 not bearish
    2. Most recent bullish BOS within 20 bars; lock impulse + support/OB/FVG anchors
    3. Within 15 bars, price touches zone and stays above impulse low. Dist <= 0.25*ATR
    4. On touch bar or within 3 bars, bullish trigger closes above zone upper edge
    5. Invalidate on close below anchor-eps, HTF1 bearish, new opposing break, or expiry
    """
    TYPE = SetupType.TREND_PULLBACK
    BOS_LOOKBACK   = 20
    TOUCH_WINDOW   = 15
    TRIGGER_WINDOW = 3

    def __init__(self):
        self._candidates: list[SetupRecord] = []

    def try_arm(self, bar_idx: int, ts: pd.Timestamp, ctx: dict) -> Optional[SetupRecord]:
        """Try to arm a new candidate. ctx contains all market state."""
        regime   = ctx.get("regime_directional", "")
        htf1_bias= ctx.get("htf1_bias", "")
        htf2_bias= ctx.get("htf2_bias", "")
        recent_bos_bar = ctx.get("recent_bull_bos_bar")
        zone       = ctx.get("nearest_bull_zone")
        impulse_low= ctx.get("impulse_low")
        atr        = ctx.get("atr_prev", 0.0)

        if regime != "BULL_TREND":
            return None
        if htf1_bias != "bullish":
            return None
        if htf2_bias == "bearish":
            return None
        if recent_bos_bar is None or bar_idx - recent_bos_bar > self.BOS_LOOKBACK:
            return None
        if zone is None or impulse_low is None or atr <= 0:
            return None

        rec = SetupRecord(
            id=_setup_id("TP"),
            type=self.TYPE,
            side="bullish",
            state=SetupState.ARMED,
            arm_bar=bar_idx,
            arm_time=ts,
            anchor_price=zone["center"],
            stop_anchor=impulse_low,
            expiry_bar=bar_idx + self.TOUCH_WINDOW,
            zone_id=zone.get("id"),
        )
        rec.state_bars["armed"] = bar_idx
        self._candidates.append(rec)
        logger.debug("TP setup armed: %s at bar %d", rec.id, bar_idx)
        return rec

    def update(self, rec: SetupRecord, bar_idx: int, ts: pd.Timestamp, ctx: dict) -> SetupRecord:
        """Advance state machine for one bar."""
        if rec.state in (SetupState.TRIGGERED, SetupState.INVALIDATED, SetupState.EXPIRED):
            return rec

        atr       = ctx.get("atr_prev", 0.0)
        close     = ctx.get("close", 0.0)
        open_     = ctx.get("open", 0.0)
        high      = ctx.get("high", 0.0)
        low       = ctx.get("low", 0.0)
        prev_high = ctx.get("prev_high", 0.0)
        prev_open = ctx.get("prev_open", 0.0)
        prev_close= ctx.get("prev_close", 0.0)
        htf1_bias = ctx.get("htf1_bias", "")
        eps = max(2e-5, 0.05 * atr) if atr > 0 else 2e-5

        # Expiry
        if bar_idx >= rec.expiry_bar:
            rec.state = SetupState.EXPIRED
            return rec

        # Hard invalidation
        if close < rec.stop_anchor - eps:
            rec.state = SetupState.INVALIDATED
            rec.invalidation_reason = "CLOSE_BELOW_IMPULSE_LOW"
            return rec
        if htf1_bias == "bearish":
            rec.state = SetupState.INVALIDATED
            rec.invalidation_reason = "HTF1_TURNED_BEARISH"
            return rec

        zone_center = rec.anchor_price
        zone_upper  = zone_center + 0.15 * atr
        zone_lower  = zone_center - 0.15 * atr

        if rec.state == SetupState.ARMED:
            # Check for touch
            dist = abs(close - zone_center)
            touched = (low <= zone_upper and high >= zone_lower)
            if touched and dist <= 0.25 * atr and close > rec.stop_anchor:
                rec.state = SetupState.WAIT_CONFIRMATION
                rec.state_bars["touched"] = bar_idx
                rec.expiry_bar = bar_idx + self.TRIGGER_WINDOW

        elif rec.state == SetupState.WAIT_CONFIRMATION:
            trig = _generic_bullish_trigger(
                open_, high, low, close,
                prev_high, prev_open, prev_close,
                zone_upper, atr
            )
            if trig and close > zone_upper:
                rec.state = SetupState.TRIGGERED
                rec.triggered_bar = bar_idx
                rec.triggered_time = ts
                rec.state_bars["triggered"] = bar_idx
                logger.debug("TP triggered: %s at bar %d", rec.id, bar_idx)

        return rec


class SweepReversalSetup:
    """
    Setup B: Sweep Reversal
    1. Sweep pre-existing low/liquidity + reclaim. Lock sweep extreme + recent internal high.
    2. Within 10 bars, close above that internal high with displacement → bullish MSS.
    3. Select FVG from displacement, else its OB. Neither → expire.
    4. Within 10 bars of MSS, retest and reclaim zone with bullish trigger.
    5. Stop below min low since sweep.
    All four stages must be identifiable in the event log.
    """
    TYPE = SetupType.SWEEP_REVERSAL
    MSS_WINDOW    = 10
    RETEST_WINDOW = 10

    def __init__(self):
        self._candidates: list[SetupRecord] = []

    def try_arm_on_sweep(self, bar_idx: int, ts: pd.Timestamp, ctx: dict) -> Optional[SetupRecord]:
        sweep_event  = ctx.get("latest_bull_sweep")
        internal_high= ctx.get("nearest_internal_high_before_sweep")
        htf1_bias    = ctx.get("htf1_bias", "")
        htf2_bias    = ctx.get("htf2_bias", "")
        if sweep_event is None or internal_high is None:
            return None
        if htf1_bias == "bearish" and htf2_bias == "bearish":
            return None

        rec = SetupRecord(
            id=_setup_id("SR"),
            type=self.TYPE,
            side="bullish",
            state=SetupState.ARMED,
            arm_bar=bar_idx,
            arm_time=ts,
            anchor_price=sweep_event.get("level_price", 0.0),
            stop_anchor=sweep_event.get("wick_extreme", 0.0),
            expiry_bar=bar_idx + self.MSS_WINDOW,
            sweep_extreme=sweep_event.get("wick_extreme"),
            internal_high_at_sweep=internal_high,
        )
        rec.state_bars["armed"] = bar_idx
        self._candidates.append(rec)
        return rec

    def update(self, rec: SetupRecord, bar_idx: int, ts: pd.Timestamp, ctx: dict) -> SetupRecord:
        if rec.state in (SetupState.TRIGGERED, SetupState.INVALIDATED, SetupState.EXPIRED):
            return rec

        atr    = ctx.get("atr_prev", 0.0)
        close  = ctx.get("close", 0.0)
        open_  = ctx.get("open", 0.0)
        high   = ctx.get("high", 0.0)
        low    = ctx.get("low", 0.0)
        prev_high = ctx.get("prev_high", 0.0)
        prev_open = ctx.get("prev_open", 0.0)
        prev_close= ctx.get("prev_close", 0.0)
        htf1_bias = ctx.get("htf1_bias", "")
        htf2_bias = ctx.get("htf2_bias", "")
        fvg_after_mss = ctx.get("fvg_after_mss")
        ob_after_mss  = ctx.get("ob_after_mss")
        eps = max(2e-5, 0.05 * atr) if atr > 0 else 2e-5

        if bar_idx >= rec.expiry_bar:
            rec.state = SetupState.EXPIRED
            return rec

        sweep_eps = rec.sweep_extreme - eps if rec.sweep_extreme else 0.0
        if close < sweep_eps:
            rec.state = SetupState.INVALIDATED
            rec.invalidation_reason = "CLOSE_BELOW_SWEEP_EXTREME"
            return rec
        if htf1_bias == "bearish" and htf2_bias == "bearish":
            rec.state = SetupState.INVALIDATED
            rec.invalidation_reason = "HTF_BOTH_BEARISH"
            return rec

        if rec.state == SetupState.ARMED:
            # Wait for MSS: close above internal high with displacement
            int_high = rec.internal_high_at_sweep or 0.0
            body_ = abs(close - open_)
            range_ = high - low
            is_disp = (
                atr > 0 and body_ >= 0.8 * atr and range_ >= 1.2 * atr
                and (body_ / range_ >= 0.65 if range_ > 0 else False)
                and close > int_high + eps
            )
            if is_disp:
                rec.state = SetupState.WAIT_RETEST
                rec.mss_bar = bar_idx
                rec.state_bars["mss"] = bar_idx
                rec.expiry_bar = bar_idx + self.RETEST_WINDOW
                # Select zone
                zone = fvg_after_mss or ob_after_mss
                if zone is None:
                    rec.state = SetupState.EXPIRED
                    rec.invalidation_reason = "NO_FVG_OR_OB_AFTER_MSS"
                else:
                    rec.zone_id = zone.get("id")
                    rec.anchor_price = zone.get("center", rec.anchor_price)

        elif rec.state == SetupState.WAIT_RETEST:
            zone_upper = rec.anchor_price + 0.15 * atr
            zone_lower = rec.anchor_price - 0.15 * atr
            in_zone = low <= zone_upper and high >= zone_lower
            if in_zone:
                trig = _generic_bullish_trigger(
                    open_, high, low, close,
                    prev_high, prev_open, prev_close,
                    zone_upper, atr
                )
                if trig and close > zone_upper:
                    rec.state = SetupState.TRIGGERED
                    rec.triggered_bar = bar_idx
                    rec.triggered_time = ts
                    rec.state_bars["triggered"] = bar_idx

        return rec


class BreakoutRetestSetup:
    """
    Setup C: Breakout Retest
    1. Arm vs frozen range boundary or 20-bar Donchian high (using bars through t-1).
       Require valid range or >= 5 squeeze bars among prior 10.
    2. Displacement closes above boundary by eps; HTF1 not bearish; both HTFs not bearish.
    3. Within 8 bars, low returns within 0.2*ATR of boundary and reclaims with trigger.
    4. Invalidate on close below boundary - 0.5*ATR.
    5. No chasing when signal close > 1.5*ATR above boundary.
    """
    TYPE = SetupType.BREAKOUT_RETEST
    RETEST_WINDOW = 8

    def __init__(self):
        self._candidates: list[SetupRecord] = []

    def try_arm(self, bar_idx: int, ts: pd.Timestamp, ctx: dict) -> Optional[SetupRecord]:
        boundary    = ctx.get("breakout_boundary")
        is_squeeze  = ctx.get("has_squeeze", False)
        is_range    = ctx.get("regime_directional", "") == "RANGE"
        htf1_bias   = ctx.get("htf1_bias", "")
        htf2_bias   = ctx.get("htf2_bias", "")
        close       = ctx.get("close", 0.0)
        atr         = ctx.get("atr_prev", 0.0)
        body_       = ctx.get("body", 0.0)
        range_bar   = ctx.get("range_bar", 0.0)
        open_       = ctx.get("open", 0.0)

        if boundary is None or atr <= 0:
            return None
        if not (is_range or is_squeeze):
            return None
        if htf1_bias == "bearish":
            return None
        if htf1_bias == "bearish" and htf2_bias == "bearish":
            return None

        eps = max(2e-5, 0.05 * atr)
        is_disp = (
            body_ >= 0.8 * atr and range_bar >= 1.2 * atr
            and (body_ / range_bar >= 0.65 if range_bar > 0 else False)
            and close > boundary + eps
        )
        if not is_disp:
            return None

        rec = SetupRecord(
            id=_setup_id("BR"),
            type=self.TYPE,
            side="bullish",
            state=SetupState.ARMED,
            arm_bar=bar_idx,
            arm_time=ts,
            anchor_price=boundary,
            stop_anchor=boundary,
            expiry_bar=bar_idx + self.RETEST_WINDOW,
            boundary_price=boundary,
        )
        rec.state_bars["armed"] = bar_idx
        self._candidates.append(rec)
        return rec

    def update(self, rec: SetupRecord, bar_idx: int, ts: pd.Timestamp, ctx: dict) -> SetupRecord:
        if rec.state in (SetupState.TRIGGERED, SetupState.INVALIDATED, SetupState.EXPIRED):
            return rec

        atr    = ctx.get("atr_prev", 0.0)
        close  = ctx.get("close", 0.0)
        open_  = ctx.get("open", 0.0)
        high   = ctx.get("high", 0.0)
        low    = ctx.get("low", 0.0)
        prev_high = ctx.get("prev_high", 0.0)
        prev_open = ctx.get("prev_open", 0.0)
        prev_close= ctx.get("prev_close", 0.0)
        boundary  = rec.boundary_price or rec.anchor_price

        if bar_idx >= rec.expiry_bar:
            rec.state = SetupState.EXPIRED
            return rec
        if atr > 0 and close < boundary - 0.5 * atr:
            rec.state = SetupState.INVALIDATED
            rec.invalidation_reason = "CLOSE_BELOW_BOUNDARY_MINUS_HALF_ATR"
            return rec

        if rec.state == SetupState.ARMED:
            near_boundary = low <= boundary + 0.2 * atr
            reclaimed = close > boundary
            if near_boundary and reclaimed:
                trig = _generic_bullish_trigger(
                    open_, high, low, close,
                    prev_high, prev_open, prev_close,
                    boundary, atr
                )
                if trig:
                    # No chasing: signal close must not be > 1.5*ATR above boundary
                    if atr > 0 and close > boundary + 1.5 * atr:
                        rec.state = SetupState.INVALIDATED
                        rec.invalidation_reason = "OVEREXTENDED_ABOVE_BOUNDARY"
                    else:
                        rec.state = SetupState.TRIGGERED
                        rec.triggered_bar = bar_idx
                        rec.triggered_time = ts
                        rec.retest_low = low
                        rec.stop_anchor = min(low, boundary - 0.25 * atr)
                        rec.state_bars["triggered"] = bar_idx
        return rec


class RangeMeanReversionSetup:
    """
    Setup D: Range Mean Reversion
    1. Valid RANGE state. Freeze boundaries. Long in bottom 20%; short in top 20%.
    2. Sweep + reclaim lower boundary, OR touch support + rejection candle.
       Block if both HTFs strongly opposing.
    3. Bullish trigger on that bar or next 2 closes. RSI recovery from <40 over 5 bars is support.
    4. Midpoint = first obstacle/TP1. Trade only if net RR passes minimum.
    5. Stop below sweep/rejection extreme. Expire after 5 bars or displacement close outside range.
    """
    TYPE = SetupType.RANGE_REVERSION
    TRIGGER_WINDOW = 2
    EXPIRY_WINDOW  = 5
    BOTTOM_PCT     = 0.20
    TOP_PCT        = 0.80

    def __init__(self):
        self._candidates: list[SetupRecord] = []

    def try_arm(self, bar_idx: int, ts: pd.Timestamp, ctx: dict) -> Optional[SetupRecord]:
        regime   = ctx.get("regime_directional", "")
        rng_lo   = ctx.get("range_low")
        rng_hi   = ctx.get("range_high")
        pos      = ctx.get("range_position", 0.5)
        close    = ctx.get("close", 0.0)
        low      = ctx.get("low", 0.0)
        atr      = ctx.get("atr_prev", 0.0)
        htf1_bias= ctx.get("htf1_bias", "")
        htf2_bias= ctx.get("htf2_bias", "")

        if regime != "RANGE" or rng_lo is None or rng_hi is None:
            return None
        if htf1_bias == "bearish" and htf2_bias == "bearish":
            return None
        if pos > self.BOTTOM_PCT:
            return None   # not in bottom zone for long

        midpoint = (rng_lo + rng_hi) / 2.0
        rec = SetupRecord(
            id=_setup_id("RR"),
            type=self.TYPE,
            side="bullish",
            state=SetupState.ARMED,
            arm_bar=bar_idx,
            arm_time=ts,
            anchor_price=rng_lo,
            stop_anchor=low,
            expiry_bar=bar_idx + self.EXPIRY_WINDOW,
            boundary_price=rng_lo,
        )
        rec.state_bars["armed"] = bar_idx
        self._candidates.append(rec)
        return rec

    def update(self, rec: SetupRecord, bar_idx: int, ts: pd.Timestamp, ctx: dict) -> SetupRecord:
        if rec.state in (SetupState.TRIGGERED, SetupState.INVALIDATED, SetupState.EXPIRED):
            return rec

        atr    = ctx.get("atr_prev", 0.0)
        close  = ctx.get("close", 0.0)
        open_  = ctx.get("open", 0.0)
        high   = ctx.get("high", 0.0)
        low    = ctx.get("low", 0.0)
        prev_high = ctx.get("prev_high", 0.0)
        prev_open = ctx.get("prev_open", 0.0)
        prev_close= ctx.get("prev_close", 0.0)
        rng_hi = ctx.get("range_high", rec.anchor_price + 10)
        rng_lo = ctx.get("range_low", rec.anchor_price)

        if bar_idx >= rec.expiry_bar:
            rec.state = SetupState.EXPIRED
            return rec

        # Displacement close outside range invalidates
        body_ = abs(close - open_)
        range_ = high - low
        if atr > 0 and range_ >= 1.2 * atr and close < rng_lo:
            rec.state = SetupState.INVALIDATED
            rec.invalidation_reason = "DISPLACEMENT_OUTSIDE_RANGE"
            return rec

        if rec.state == SetupState.ARMED:
            trig = _generic_bullish_trigger(
                open_, high, low, close,
                prev_high, prev_open, prev_close,
                rng_lo, atr
            )
            if trig:
                rec.state = SetupState.TRIGGERED
                rec.triggered_bar = bar_idx
                rec.triggered_time = ts
                rec.stop_anchor = low
                rec.state_bars["triggered"] = bar_idx
        return rec


class SetupManager:
    """
    Manages all four setup state machines.
    Enforces 5-bar cooldown, conflict resolution, and competition rules.
    """
    COOLDOWN_BARS = 5

    def __init__(self, cooldown_bars: int = 5):
        self.cooldown_bars = cooldown_bars
        self._tp = TrendPullbackSetup()
        self._sr = SweepReversalSetup()
        self._br = BreakoutRetestSetup()
        self._rr = RangeMeanReversionSetup()
        self._active: list[SetupRecord] = []
        self._last_signal_bar: int = -999
        self._all_history: list[SetupRecord] = []

    def process_bar(self, bar_idx: int, ts: pd.Timestamp, ctx: dict) -> list[SetupRecord]:
        """
        Process one bar: try to arm new setups, advance existing ones.
        Returns list of newly TRIGGERED setups this bar.
        """
        # Update all active setups
        still_active = []
        triggered_this_bar = []

        for rec in self._active:
            if rec.type == SetupType.TREND_PULLBACK:
                rec = self._tp.update(rec, bar_idx, ts, ctx)
            elif rec.type == SetupType.SWEEP_REVERSAL:
                rec = self._sr.update(rec, bar_idx, ts, ctx)
            elif rec.type == SetupType.BREAKOUT_RETEST:
                rec = self._br.update(rec, bar_idx, ts, ctx)
            elif rec.type == SetupType.RANGE_REVERSION:
                rec = self._rr.update(rec, bar_idx, ts, ctx)

            if rec.state == SetupState.TRIGGERED:
                triggered_this_bar.append(rec)
            elif rec.state in (SetupState.ARMED, SetupState.WAIT_CONFIRMATION, SetupState.WAIT_RETEST):
                still_active.append(rec)
            # else: terminal state, drop from active

        self._active = still_active

        # Try arming new setups (respecting cooldown)
        in_cooldown = (bar_idx - self._last_signal_bar) < self.cooldown_bars
        if not in_cooldown:
            new_tp = self._tp.try_arm(bar_idx, ts, ctx)
            new_br = self._br.try_arm(bar_idx, ts, ctx)
            new_rr = self._rr.try_arm(bar_idx, ts, ctx)
            new_sr_sweep = ctx.get("latest_bull_sweep")
            new_sr = self._sr.try_arm_on_sweep(bar_idx, ts, ctx) if new_sr_sweep else None

            for rec in [new_tp, new_sr, new_br, new_rr]:
                if rec is not None:
                    self._active.append(rec)
                    self._all_history.append(rec)

        # Conflict resolution for multiple triggers
        if len(triggered_this_bar) > 1:
            long_triggers = [r for r in triggered_this_bar if r.side == "bullish"]
            short_triggers = [r for r in triggered_this_bar if r.side == "bearish"]
            if long_triggers and short_triggers:
                # Both sides → NO TRADE with CONFLICTING_SETUPS
                for r in triggered_this_bar:
                    r.state = SetupState.INVALIDATED
                    r.invalidation_reason = "CONFLICTING_SETUPS"
                return []
            # Same side: pick highest priority
            # Priority: sweep_reversal > trend_pullback > breakout_retest > range_reversion
            priority = {
                SetupType.SWEEP_REVERSAL:  1,
                SetupType.TREND_PULLBACK:  2,
                SetupType.BREAKOUT_RETEST: 3,
                SetupType.RANGE_REVERSION: 4,
            }
            triggered_this_bar.sort(key=lambda r: priority.get(r.type, 9))
            for r in triggered_this_bar[1:]:
                r.state = SetupState.INVALIDATED
                r.invalidation_reason = "LOWER_PRIORITY_CONFLICT"
            triggered_this_bar = triggered_this_bar[:1]

        if triggered_this_bar:
            self._last_signal_bar = bar_idx

        return triggered_this_bar
