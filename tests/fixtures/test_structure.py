"""
ASTRA FUSION QUANT — Structure Fixtures (Milestone 2 tests)
Tests: BOS/CHoCH timing, structure state, Zone lifecycle, setup cooldown.
"""
import pytest
import numpy as np
import pandas as pd
from research.features.pivots import Pivot, PivotStore
from research.features.structure import StructureEngine
from research.features.zones import ZoneEngine, Zone
from research.engine.setups import TrendPullbackSetup, SetupManager


def make_bar(i, o, h, l, c):
    return {"index": i, "open": o, "high": h, "low": l, "close": c, "atr": 1.0}


class TestStructureEngine:
    def test_bos_requires_displacement(self):
        """BOS must be validated by displacement, not just a pip break."""
        engine = StructureEngine()
        # Mock a bullish trend
        engine.state = "BULLISH"

        p_low = Pivot(id="PVT_1", bar_index=10, type="low", price=100.0,
                      origin_bar=10, confirmation_bar=13, ambiguous=False, active=True)
        p_high = Pivot(id="PVT_2", bar_index=20, type="high", price=110.0,
                       origin_bar=20, confirmation_bar=23, ambiguous=False, active=True)

        engine.latest_low = p_low
        engine.latest_high = p_high

        # Break above 110 but with weak displacement (range=1.0, body=0.2)
        # atr is 1.0, so disp fails
        bar_weak = make_bar(25, 109.9, 110.1, 109.1, 110.1)
        res = engine.update(bar_weak)
        assert len(res) == 0

        # Break above 110 with strong displacement (range=2.0, body=1.8 > 0.8*A)
        bar_strong = make_bar(26, 109.0, 111.0, 109.0, 110.8)
        res = engine.update(bar_strong)
        assert len(res) == 1
        assert res[0].type == "BOS"
        assert res[0].side == "bullish"


class TestZoneEngine:
    def test_fvg_creation_requires_displacement(self):
        """FVG only forms if displacement is present."""
        engine = ZoneEngine()
        # Bar t-2
        engine.update(make_bar(10, 100, 101, 99, 100), htf_aligned=False)
        # Bar t-1 (Displacement: Range=3.0, Body=2.8)
        engine.update(make_bar(11, 100, 103, 100, 102.8), htf_aligned=False)
        # Bar t (Leaves gap 101 to 102)
        res = engine.update(make_bar(12, 103, 104, 102, 103.5), htf_aligned=False)
        assert len(res) == 1
        assert res[0].type == "FVG"
        assert res[0].side == "bullish"
        assert res[0].top == 102.0
        assert res[0].bottom == 101.0
        assert res[0].quality == pytest.approx(0.8, abs=0.01) # 0.6 + 0.2(disp)

    def test_zone_lifecycle_touched_and_consumed(self):
        engine = ZoneEngine()
        z = Zone(id="Z_1", bar_index=10, type="FVG", side="bullish", top=102, bottom=101,
                 quality=1.0, active=True, state="FRESH")
        engine.zones.append(z)

        # Touched
        engine.update(make_bar(15, 103, 104, 101.5, 103), htf_aligned=False)
        assert z.state == "TOUCHED"
        assert z.touches == 1

        # Consumed (closes below bottom)
        engine.update(make_bar(16, 103, 103, 100, 100.5), htf_aligned=False)
        assert z.state == "CONSUMED"
        assert not z.active


class TestSetups:
    def test_trend_pullback_state_machine(self):
        sm = TrendPullbackSetup()
        # Arm
        res = sm.update(10, "BULL_TREND", True, False, False)
        assert sm.state == "ARMED"
        assert sm.anchor_bar == 10
        # Wait 3 bars
        sm.update(13, "BULL_TREND", True, False, False)
        # Trigger
        res = sm.update(14, "BULL_TREND", True, True, False)
        assert sm.state == "TRIGGERED"
        assert res == "bullish"
        # Reset
        res = sm.update(15, "BULL_TREND", True, False, False)
        assert sm.state == "IDLE"

    def test_setup_manager_cooldown(self):
        mgr = SetupManager(cooldown_bars=5)
        mgr.record_signal(10, "bullish", "trend_pullback")
        # Bar 11 -> in cooldown
        assert mgr.is_in_cooldown(11)
        # Bar 16 -> out of cooldown (10 + 5 + 1)
        assert not mgr.is_in_cooldown(16)
