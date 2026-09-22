"""
ASTRA FUSION QUANT — Evidence Families + Scoring (Milestone 3)
Six signed families: S (structure), T (trend), M (momentum), L (location), V (volume), P (trigger).
Weight profiles per setup type. Coverage, BullScore, BearScore, NetScore, SideQuality, Conflict.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ── Weight profiles (sum to 1.0) ─────────────────────────────────────────────
WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "trend_pullback":   {"S": 0.25, "T": 0.20, "M": 0.15, "L": 0.20, "V": 0.05, "P": 0.15},
    "sweep_reversal":   {"S": 0.30, "T": 0.05, "M": 0.20, "L": 0.20, "V": 0.05, "P": 0.20},
    "breakout_retest":  {"S": 0.25, "T": 0.20, "M": 0.10, "L": 0.15, "V": 0.10, "P": 0.20},
    "range_reversion":  {"S": 0.25, "T": 0.05, "M": 0.20, "L": 0.25, "V": 0.05, "P": 0.20},
}

COVERAGE_MINIMUM = 0.85   # W >= 0.85 required
MIN_ALIGNED_FAMILIES = 3  # at least 3 non-trigger families with aligned evidence >= 0.25


@dataclass
class FamilyScores:
    """Raw family scores in [-1, 1]. NaN = unavailable."""
    S: float = np.nan   # structure/liquidity
    T: float = np.nan   # trend
    M: float = np.nan   # momentum/divergence
    L: float = np.nan   # location
    V: float = np.nan   # participation/volume
    P: float = np.nan   # trigger

    def as_dict(self) -> dict[str, float]:
        return {"S": self.S, "T": self.T, "M": self.M, "L": self.L, "V": self.V, "P": self.P}


@dataclass
class EnsembleResult:
    setup_type: str
    side: str                        # 'bullish' or 'bearish'
    side_d: float                    # +1 or -1
    family_scores: FamilyScores
    weights: dict[str, float]
    coverage_W: float
    bull_score: float                # 0–100
    bear_score: float                # 0–100
    net_score: float                 # -100 to 100
    conflict: float                  # 0–1
    side_quality: float              # 0–100
    tier: str                        # "A+", "A", "B", or "NO_TRADE"
    rejection_reasons: list[str] = field(default_factory=list)
    aligned_family_count: int = 0


def compute_families(
    side: str,
    ctx: dict,
    setup_type: str,
    zone_quality: float = 0.0,
    fib_confluence: bool = False,
) -> FamilyScores:
    """
    Compute the six family scores for a given side and context.

    ctx keys used:
      structure_dir, recent_break_dir, recent_sweep_dir (for S)
      trend_t, htf1_trend_t, htf2_trend_t (for T)
      rsi, relevant_divergence_dir (for M)
      in_zone, zone_quality, zone_role (for L)
      signed_participation (for V)
      has_trigger (for P)
    """
    d = 1.0 if side == "bullish" else -1.0
    fs = FamilyScores()

    # ── S: Structure/Liquidity ────────────────────────────────────────────────
    struct_dir = ctx.get("structure_dir", 0.0)           # +1/-1/0
    break_dir  = ctx.get("recent_break_dir", 0.0)        # +1/-1/0, 0 if expired
    sweep_dir  = ctx.get("recent_sweep_dir", 0.0)        # +1/-1/0, 0 if expired
    s_raw = (0.4 * struct_dir + 0.3 * break_dir + 0.3 * sweep_dir)
    fs.S = float(np.clip(s_raw, -1.0, 1.0))

    # ── T: Trend ─────────────────────────────────────────────────────────────
    t_chart = ctx.get("trend_t", np.nan)
    if not np.isnan(t_chart):
        fs.T = float(np.clip(t_chart, -1.0, 1.0))

    # ── M: Momentum/Divergence ────────────────────────────────────────────────
    rsi = ctx.get("rsi", np.nan)
    regime = ctx.get("regime_directional", "TRANSITION")
    div_dir = ctx.get("divergence_dir", 0.0)  # +1/-1/0, 0 if absent/expired

    if not np.isnan(rsi):
        if regime in ("BULL_TREND", "BEAR_TREND"):
            rsi_score = float(np.clip((rsi - 50.0) / 20.0, -1.0, 1.0))
        else:
            # Range mode: reversed
            rsi_score = float(np.clip((50.0 - rsi) / 20.0, -1.0, 1.0))
        m_raw = 0.6 * rsi_score + 0.4 * div_dir
        fs.M = float(np.clip(m_raw, -1.0, 1.0))

    # ── L: Location ───────────────────────────────────────────────────────────
    in_zone  = ctx.get("in_zone", False)
    zq       = zone_quality
    z_role   = ctx.get("zone_role", "")    # 'support' or 'resistance'
    fib_flag = fib_confluence

    if in_zone and zq > 0:
        l_mag = 0.9 * zq + 0.1 * float(fib_flag)
        # Sign: support → bullish evidence; resistance → bearish evidence
        if z_role == "support":
            fs.L = float(np.clip(l_mag, -1.0, 1.0))
        elif z_role == "resistance":
            fs.L = float(np.clip(-l_mag, -1.0, 1.0))
        else:
            fs.L = 0.0
    elif in_zone:
        fs.L = 0.0  # available but neutral

    # ── V: Participation ─────────────────────────────────────────────────────
    signed_part = ctx.get("signed_participation", np.nan)
    if not np.isnan(signed_part):
        fs.V = float(np.clip(signed_part, -1.0, 1.0))

    # ── P: Trigger ────────────────────────────────────────────────────────────
    has_bull_trig = ctx.get("has_bull_trigger", False)
    has_bear_trig = ctx.get("has_bear_trigger", False)
    if side == "bullish":
        fs.P = 1.0 if has_bull_trig else (0.0 if not has_bear_trig else -1.0)
    else:
        fs.P = 1.0 if has_bear_trig else (0.0 if not has_bull_trig else -1.0)

    return fs


def compute_ensemble(
    setup_type: str,
    side: str,
    fs: FamilyScores,
    score_floor: float = 75.0,
    emit_tier_b: bool = False,
) -> EnsembleResult:
    """
    Compute ensemble scores from family scores.
    Returns EnsembleResult with tier assignment and rejection reasons.
    """
    d = 1.0 if side == "bullish" else -1.0
    weights = WEIGHT_PROFILES.get(setup_type, WEIGHT_PROFILES["trend_pullback"])
    family_vals = fs.as_dict()

    # Coverage W = sum of weights for available families
    W = sum(w for k, w in weights.items() if not np.isnan(family_vals[k]))
    rejection_reasons: list[str] = []

    if W < COVERAGE_MINIMUM:
        rejection_reasons.append("LOW_COVERAGE")

    # B = sum(w * max(f, 0)) / W  — bullish evidence
    # D = sum(w * max(-f, 0)) / W — bearish evidence
    # Use aligned scores: x = d * f (positive = aligned with side)
    B, D = 0.0, 0.0
    side_pos_sum = 0.0
    side_neg_sum = 0.0

    aligned_non_trigger = 0

    for k, w in weights.items():
        f = family_vals[k]
        if np.isnan(f):
            continue
        B += w * max(f, 0.0)
        D += w * max(-f, 0.0)
        aligned_score = d * f
        side_pos_sum += w * max(aligned_score, 0.0)
        side_neg_sum += w * max(-aligned_score, 0.0)
        # Count aligned non-trigger families
        if k != "P" and aligned_score >= 0.25:
            aligned_non_trigger += 1

    if W > 0:
        B /= W
        D /= W
        side_pos_sum /= W
        side_neg_sum /= W
    else:
        B = D = 0.0

    bull_score = 100.0 * B
    bear_score = 100.0 * D
    net_score  = 100.0 * (B - D)

    # Conflict = 2*min(B,D)/(B+D), or 0 when B+D == 0
    bd_sum = B + D
    conflict = 2.0 * min(B, D) / bd_sum if bd_sum > 0 else 0.0

    # SideQuality = 100 * clip(sum(w*max(d*f,0))/W - 0.5*sum(w*max(-d*f,0))/W, 0, 1) * sqrt(W)
    sq_raw = np.clip(side_pos_sum - 0.5 * side_neg_sum, 0.0, 1.0) * np.sqrt(W)
    side_quality = 100.0 * sq_raw

    # Tier assignment
    p_val = family_vals.get("P", np.nan)
    l_val = family_vals.get("L", np.nan)
    trigger_valid   = not np.isnan(p_val) and (d * p_val) > 0
    location_valid  = not np.isnan(l_val) and (d * l_val) > 0

    if not trigger_valid:
        rejection_reasons.append("NO_TRIGGER")
    if not location_valid:
        rejection_reasons.append("BAD_LOCATION")
    if conflict > 0.35:
        rejection_reasons.append("FAMILY_CONFLICT")
    if aligned_non_trigger < MIN_ALIGNED_FAMILIES:
        rejection_reasons.append("LOW_SCORE")

    if rejection_reasons:
        tier = "NO_TRADE"
    elif side_quality >= 85.0:
        tier = "A+"
    elif side_quality >= 75.0:
        tier = "A"
    elif side_quality >= 65.0 and emit_tier_b:
        tier = "B"
    else:
        tier = "NO_TRADE"
        if not rejection_reasons:
            rejection_reasons.append("LOW_SCORE")

    # Floor check
    if tier != "NO_TRADE" and side_quality < score_floor:
        tier = "NO_TRADE"
        if "LOW_SCORE" not in rejection_reasons:
            rejection_reasons.append("LOW_SCORE")

    return EnsembleResult(
        setup_type=setup_type,
        side=side,
        side_d=d,
        family_scores=fs,
        weights=weights,
        coverage_W=W,
        bull_score=bull_score,
        bear_score=bear_score,
        net_score=net_score,
        conflict=conflict,
        side_quality=side_quality,
        tier=tier,
        rejection_reasons=rejection_reasons,
        aligned_family_count=aligned_non_trigger,
    )
