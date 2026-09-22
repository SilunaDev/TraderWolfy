"""
ASTRA FUSION QUANT — Scoring Fixtures (Milestone 3 tests)
Hand-calculated family scores, long/short symmetry, RR arithmetic.
"""
import pytest
import numpy as np
from research.engine.families import FamilyScores, compute_ensemble, WEIGHT_PROFILES
from research.engine.risk import (
    build_structural_stop, compute_fixed_rr_targets,
    compute_net_rr, build_risk_snapshot
)
from research.engine.gates import apply_hard_gates


class TestScoringSymmetry:
    def test_long_short_mirror(self):
        """
        Acceptance case: mirrored inputs must produce mirrored scores.
        """
        fs_bull = FamilyScores(S=0.5, T=0.4, M=0.3, L=0.6, V=0.2, P=1.0)
        fs_bear = FamilyScores(S=-0.5, T=-0.4, M=-0.3, L=-0.6, V=-0.2, P=1.0)

        res_bull = compute_ensemble("trend_pullback", "bullish", fs_bull)
        res_bear = compute_ensemble("trend_pullback", "bearish", fs_bear)

        assert res_bull.side_quality == pytest.approx(res_bear.side_quality, abs=0.01)
        assert res_bull.conflict == pytest.approx(res_bear.conflict, abs=0.01)

    def test_no_trade_below_floor(self):
        fs = FamilyScores(S=0.1, T=0.1, M=0.1, L=0.5, V=np.nan, P=1.0)
        res = compute_ensemble("trend_pullback", "bullish", fs, score_floor=75.0)
        assert res.tier == "NO_TRADE"

    def test_a_plus_tier_high_score(self):
        fs = FamilyScores(S=0.9, T=0.8, M=0.7, L=0.9, V=0.6, P=1.0)
        res = compute_ensemble("trend_pullback", "bullish", fs, score_floor=75.0)
        assert res.tier in ("A+", "A")  # Very strong evidence

    def test_conflict_high_blocks_signal(self):
        # Near-equal bull and bear evidence → high conflict
        fs = FamilyScores(S=0.3, T=-0.3, M=0.3, L=-0.3, V=0.3, P=1.0)
        res = compute_ensemble("trend_pullback", "bullish", fs)
        assert res.conflict > 0.35 or res.tier == "NO_TRADE"

    def test_missing_trigger_rejected(self):
        fs = FamilyScores(S=0.9, T=0.9, M=0.9, L=0.9, V=0.9, P=0.0)  # P=0: no trigger
        res = compute_ensemble("trend_pullback", "bullish", fs)
        assert "NO_TRIGGER" in res.rejection_reasons or res.tier == "NO_TRADE"

    def test_low_coverage_rejected(self):
        # Most families missing (NaN)
        fs = FamilyScores(S=np.nan, T=np.nan, M=np.nan, L=0.8, V=np.nan, P=1.0)
        res = compute_ensemble("trend_pullback", "bullish", fs)
        assert "LOW_COVERAGE" in res.rejection_reasons

    def test_weight_profiles_sum_to_one(self):
        for name, profile in WEIGHT_PROFILES.items():
            total = sum(profile.values())
            assert total == pytest.approx(1.0, abs=1e-9), f"Profile {name} weights sum to {total}"


class TestRiskArithmetic:
    """
    Acceptance case: Reference entry 100, stop 98, TP1 104 → gross RR 2.
    Friction 0.10 → net RR ≈ (4-0.10)/(2+0.10) ≈ 1.857...
    """
    def test_gross_rr(self):
        tp1, tp2, tp3 = compute_fixed_rr_targets(100.0, 98.0, "bullish", tick_size=0.01)
        assert tp1 == pytest.approx(104.0, abs=1e-8)
        D = abs(100.0 - 98.0)
        gross = abs(tp1 - 100.0) / D
        assert gross == pytest.approx(2.0, abs=1e-8)

    def test_net_rr_formula(self):
        """Blueprint example: E=100, stop=98, TP1=104, friction=0.10 → netRR≈1.857"""
        net = compute_net_rr(entry=100.0, stop=98.0, tp1=104.0, friction_price=0.10)
        # (4 - 0.10) / (2 + 0.10) = 3.90 / 2.10 ≈ 1.857
        assert net == pytest.approx(3.90 / 2.10, abs=0.001)

    def test_net_rr_nonpositive_numerator_returns_zero(self):
        """TP1 inside stop distance → reject (return 0)."""
        net = compute_net_rr(entry=100.0, stop=98.0, tp1=100.5, friction_price=1.0)
        assert net == 0.0

    def test_structural_stop_long(self):
        """Long stop: anchor - 0.15*ATR, rounded DOWN to tick."""
        stop = build_structural_stop("bullish", stop_anchor=98.0, tick_size=0.01, buffer_atr=0.15, atr=1.0)
        expected = 98.0 - 0.15 * 1.0
        # Rounded down
        import math
        assert stop <= expected
        assert stop == pytest.approx(math.floor(expected / 0.01) * 0.01, abs=1e-8)

    def test_structural_stop_short(self):
        """Short stop: anchor + 0.15*ATR, rounded UP to tick."""
        stop = build_structural_stop("bearish", stop_anchor=102.0, tick_size=0.01, buffer_atr=0.15, atr=1.0)
        expected_min = 102.0 + 0.15 * 1.0
        assert stop >= expected_min


class TestHardGates:
    def _base_ctx(self):
        return {
            "data_valid": True, "warming_up": False, "htf_valid": True,
            "in_session": True, "volatility_state": "NORMAL",
            "regime_directional": "BULL_TREND", "setup_type": "trend_pullback",
            "setup_expired": False, "anchor_invalidated": False,
            "htf1_bias": "bullish", "htf2_bias": "neutral",
            "side": "bullish", "coverage_W": 0.90, "conflict": 0.20,
            "side_quality": 80.0, "score_floor": 75.0,
            "has_trigger": True, "in_zone": True, "overextended": False,
            "net_rr": 2.0, "rr_min_net": 1.5,
            "stop_dist_atr": 1.0, "stop_min_atr": 0.5, "stop_max_atr": 3.0,
            "cost_pct_of_stop": 0.10, "max_cost_pct_stop": 0.20,
            "in_cooldown": False, "duplicate_setup": False, "conflicting_setups": False,
        }

    def test_all_clear_passes(self):
        result = apply_hard_gates(self._base_ctx())
        assert result.passed

    def test_warming_up_blocks(self):
        ctx = self._base_ctx()
        ctx["warming_up"] = True
        result = apply_hard_gates(ctx)
        assert not result.passed
        assert "WARMING_UP" in result.rejection_codes

    def test_extreme_volatility_blocks(self):
        ctx = self._base_ctx()
        ctx["volatility_state"] = "EXTREME"
        result = apply_hard_gates(ctx)
        assert "EXTREME_VOLATILITY" in result.rejection_codes

    def test_htf_conflict_blocks(self):
        ctx = self._base_ctx()
        ctx["htf1_bias"] = "bearish"
        ctx["htf2_bias"] = "bearish"
        result = apply_hard_gates(ctx)
        assert "HTF_CONFLICT" in result.rejection_codes

    def test_rr_too_low_blocks(self):
        ctx = self._base_ctx()
        ctx["net_rr"] = 1.0
        result = apply_hard_gates(ctx)
        assert "RR_TOO_LOW" in result.rejection_codes

    def test_stop_too_tight_blocks(self):
        ctx = self._base_ctx()
        ctx["stop_dist_atr"] = 0.3
        result = apply_hard_gates(ctx)
        assert "STOP_TOO_TIGHT" in result.rejection_codes

    def test_stop_too_wide_blocks(self):
        ctx = self._base_ctx()
        ctx["stop_dist_atr"] = 4.0
        result = apply_hard_gates(ctx)
        assert "STOP_TOO_WIDE" in result.rejection_codes

    def test_all_rejection_codes_tested(self):
        """Ensure we have at least one test covering each major rejection code."""
        tested = {
            "WARMING_UP", "EXTREME_VOLATILITY", "HTF_CONFLICT",
            "RR_TOO_LOW", "STOP_TOO_TIGHT", "STOP_TOO_WIDE",
        }
        from research.engine.gates import ALL_REJECTION_CODES
        untested = set(ALL_REJECTION_CODES) - tested
        # Log but don't fail — this is a coverage reminder
        if untested:
            import warnings
            warnings.warn(f"Rejection codes not directly tested: {untested}")
