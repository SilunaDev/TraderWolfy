"""
ASTRA FUSION QUANT — Hard No-Trade Gates (Milestone 3)
All 22 rejection codes from blueprint Section 14.
Applied before signal emission; returns every applicable rejection code.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

ALL_REJECTION_CODES = [
    "DATA_INVALID", "WARMING_UP", "HTF_INVALID", "OUTSIDE_SESSION",
    "EXTREME_VOLATILITY", "TRANSITION_BLOCKED", "SETUP_EXPIRED",
    "ANCHOR_INVALIDATED", "HTF_CONFLICT", "LOW_COVERAGE", "FAMILY_CONFLICT",
    "LOW_SCORE", "NO_TRIGGER", "BAD_LOCATION", "OVEREXTENDED", "RR_TOO_LOW",
    "STOP_TOO_TIGHT", "STOP_TOO_WIDE", "COSTS_TOO_HIGH", "COOLDOWN",
    "DUPLICATE_SETUP", "CONFLICTING_SETUPS",
]


@dataclass
class GateResult:
    passed: bool
    rejection_codes: list[str]


def apply_hard_gates(ctx: dict) -> GateResult:
    """
    Apply all hard no-trade filters. Returns GateResult with all applicable codes.

    ctx keys:
      data_valid (bool), warming_up (bool), htf_valid (bool),
      in_session (bool), volatility_state (str), regime_directional (str),
      setup_expired (bool), anchor_invalidated (bool),
      htf1_bias (str), htf2_bias (str),
      coverage_W (float), conflict (float), side_quality (float),
      has_trigger (bool), in_zone (bool),
      overextended (bool), net_rr (float),
      stop_dist_atr (float), cost_pct_of_stop (float),
      in_cooldown (bool), duplicate_setup (bool), conflicting_setups (bool),
      stop_min_atr (float), stop_max_atr (float),
      max_cost_pct_stop (float), rr_min_net (float),
      score_floor (float),
    """
    codes: list[str] = []

    if not ctx.get("data_valid", True):
        codes.append("DATA_INVALID")
    if ctx.get("warming_up", False):
        codes.append("WARMING_UP")
    if not ctx.get("htf_valid", True):
        codes.append("HTF_INVALID")
    if not ctx.get("in_session", True):
        codes.append("OUTSIDE_SESSION")

    vol_state = ctx.get("volatility_state", "NORMAL")
    if vol_state == "EXTREME":
        codes.append("EXTREME_VOLATILITY")

    regime = ctx.get("regime_directional", "TRANSITION")
    setup_type = ctx.get("setup_type", "")
    if regime == "TRANSITION" and setup_type in ("trend_pullback", "range_reversion"):
        codes.append("TRANSITION_BLOCKED")

    if ctx.get("setup_expired", False):
        codes.append("SETUP_EXPIRED")
    if ctx.get("anchor_invalidated", False):
        codes.append("ANCHOR_INVALIDATED")

    # HTF conflict: HTF1 and HTF2 both strongly opposing
    htf1 = ctx.get("htf1_bias", "neutral")
    htf2 = ctx.get("htf2_bias", "neutral")
    side = ctx.get("side", "bullish")
    opposing = "bearish" if side == "bullish" else "bullish"
    if htf1 == opposing and htf2 == opposing:
        codes.append("HTF_CONFLICT")

    W = ctx.get("coverage_W", 1.0)
    if W < 0.85:
        codes.append("LOW_COVERAGE")

    conflict = ctx.get("conflict", 0.0)
    if conflict > 0.35:
        codes.append("FAMILY_CONFLICT")

    sq = ctx.get("side_quality", 0.0)
    floor = ctx.get("score_floor", 75.0)
    if sq < floor:
        codes.append("LOW_SCORE")

    if not ctx.get("has_trigger", False):
        codes.append("NO_TRIGGER")
    if not ctx.get("in_zone", False):
        codes.append("BAD_LOCATION")

    if ctx.get("overextended", False):
        codes.append("OVEREXTENDED")

    net_rr = ctx.get("net_rr", np.nan)
    rr_min = ctx.get("rr_min_net", 1.5)
    if np.isnan(net_rr) or net_rr < rr_min:
        codes.append("RR_TOO_LOW")

    stop_atr = ctx.get("stop_dist_atr", np.nan)
    stop_min = ctx.get("stop_min_atr", 0.5)
    stop_max = ctx.get("stop_max_atr", 3.0)
    if not np.isnan(stop_atr):
        if stop_atr < stop_min:
            codes.append("STOP_TOO_TIGHT")
        if stop_atr > stop_max:
            codes.append("STOP_TOO_WIDE")

    cost_pct = ctx.get("cost_pct_of_stop", np.nan)
    max_cost = ctx.get("max_cost_pct_stop", 0.20)
    if not np.isnan(cost_pct) and cost_pct > max_cost:
        codes.append("COSTS_TOO_HIGH")

    if ctx.get("in_cooldown", False):
        codes.append("COOLDOWN")
    if ctx.get("duplicate_setup", False):
        codes.append("DUPLICATE_SETUP")
    if ctx.get("conflicting_setups", False):
        codes.append("CONFLICTING_SETUPS")

    return GateResult(passed=len(codes) == 0, rejection_codes=codes)
