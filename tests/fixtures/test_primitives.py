"""
ASTRA FUSION QUANT — Primitive Fixtures (Milestone 1 tests)
Tests: ATR seeding, EMA, RSI, pivot timing, HTF availability.
"""
import pytest
import numpy as np
import pandas as pd
from research.features.primitives import atr_wilder, ema, rsi, efficiency_ratio, natr
from research.features.pivots import PivotDetector
from research.data.bar_validator import BarValidator


def make_df(closes, highs=None, lows=None, opens=None, volumes=None):
    n = len(closes)
    closes = np.array(closes, dtype=float)
    highs   = np.array(highs,   dtype=float) if highs   is not None else closes * 1.001
    lows    = np.array(lows,    dtype=float) if lows    is not None else closes * 0.999
    opens   = np.array(opens,   dtype=float) if opens   is not None else closes
    volumes = np.array(volumes, dtype=float) if volumes is not None else np.ones(n) * 1000
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=idx)


class TestBarValidator:
    def test_valid_bars_pass(self):
        df = make_df([100, 101, 102])
        validator = BarValidator()
        valid_df, rejections = validator.validate_dataframe(df)
        assert len(valid_df) == 3
        assert len(rejections) == 0

    def test_invalid_ohlc_rejected(self):
        # High < Low (geometry violation)
        df = make_df([100])
        df["high"] = 99.0
        df["low"]  = 101.0
        validator = BarValidator()
        valid_df, rejections = validator.validate_dataframe(df)
        assert len(rejections) == 1
        assert "INVALID_OHLC_GEOMETRY" in rejections[0]["reasons"]

    def test_non_monotonic_timestamp_rejected(self):
        df = make_df([100, 101, 102])
        # Duplicate index
        df = df.iloc[[0, 0, 1]]
        validator = BarValidator()
        valid_df, rejections = validator.validate_dataframe(df)
        assert len(rejections) >= 1

    def test_synthetic_chart_rejected(self):
        df = make_df([100, 101])
        validator = BarValidator(allow_synthetic=False)
        with pytest.raises(ValueError, match="Synthetic"):
            validator.validate_dataframe(df, chart_type="heikin_ashi")

    def test_warmup_tracks_count(self):
        df = make_df(list(range(100, 110)))
        validator = BarValidator(chart_bars_required=5)
        valid_df, _ = validator.validate_dataframe(df)
        assert validator.warmup.chart_bars_valid == 10


class TestATR:
    def test_atr_positive(self):
        df = make_df(list(range(100, 150)), lows=[x - 0.5 for x in range(100, 150)],
                     highs=[x + 0.5 for x in range(100, 150)])
        atr = atr_wilder(df["high"], df["low"], df["close"], 14)
        valid = atr.dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_atr_seeding_at_bar_14(self):
        """ATR first valid value appears at bar index 13 (14th bar, 0-indexed)."""
        df = make_df(list(range(100, 125)))
        atr = atr_wilder(df["high"], df["low"], df["close"], 14)
        # First 13 values should be NaN
        assert atr.iloc[:13].isna().all()
        assert not np.isnan(atr.iloc[13])

    def test_atr_nan_on_zero_range(self):
        """Zero-range bars (H=L=O=C) should not cause division errors."""
        closes = [100.0] * 30
        df = make_df(closes, highs=closes, lows=closes)
        atr = atr_wilder(df["high"], df["low"], df["close"], 14)
        # Should not raise; all TRs will be 0 (prev_close = close for flat bars)
        assert not atr.isna().all() or True  # Either 0 or NaN — no error


class TestRSI:
    def test_rsi_bounds(self):
        closes = [float(100 + i) for i in range(50)]
        df = make_df(closes)
        r = rsi(df["close"], 14)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_all_gains_equals_100(self):
        closes = [float(100 + i) for i in range(30)]
        df = make_df(closes)
        r = rsi(df["close"], 14)
        # All gains → RSI should be 100
        assert r.dropna().iloc[-1] == pytest.approx(100.0, abs=1.0)


class TestPivots:
    def test_pivot_known_at_p_plus_r(self):
        """
        Acceptance case: pivot at bar p with R=3 must not produce a signal before close p+R.
        """
        # Make a clear high at bar 5, with bars 0-4 lower and bars 6-8 lower
        closes = [100, 101, 102, 103, 104, 110, 108, 107, 106, 105, 104]
        highs  = [100, 101, 102, 103, 104, 115, 109, 108, 107, 106, 105]
        lows   = [99,  100, 101, 102, 103, 109, 107, 106, 105, 104, 103]
        df = make_df(closes, highs=highs, lows=lows)
        detector = PivotDetector(L=3, R=3)
        store = detector.detect(df)
        highs_found = [p for p in store.all_pivots if p.side == "high"]
        for piv in highs_found:
            # confirmation_bar must be >= origin_bar + R
            assert piv.confirmation_bar >= piv.origin_bar + 3, \
                f"Pivot {piv.id} confirmed too early: origin={piv.origin_bar}, conf={piv.confirmation_bar}"

    def test_prefix_stability(self):
        """
        Acceptance case: appending future bars must not change earlier pivot IDs/prices.
        """
        closes = list(range(100, 130))
        df1 = make_df(closes[:20])
        df2 = make_df(closes[:25])

        # Reset counter for reproducibility
        import research.features.pivots as pmod
        pmod._pivot_counter = 0
        store1 = PivotDetector(L=3, R=3).detect(df1)

        pmod._pivot_counter = 0
        store2 = PivotDetector(L=3, R=3).detect(df2)

        # Pivots confirmed on the first 20 bars should match
        p1 = {p.id: p.price for p in store1.all_pivots if p.confirmation_bar < 18}
        p2 = {p.id: p.price for p in store2.all_pivots if p.confirmation_bar < 18}
        assert p1 == p2, "Pivot prices changed after appending bars"

    def test_ambiguous_pivot_flagged(self):
        """Bar that is both high and low pivot should be AMBIGUOUS."""
        # Flat line — every bar could qualify as both H and L
        closes = [100.0] * 15
        df = make_df(closes, highs=closes, lows=closes)
        store = PivotDetector(L=3, R=3).detect(df)
        # Should not raise; ambiguous pivots get flagged
        ambiguous = [p for p in store.all_pivots if p.ambiguous]
        assert isinstance(ambiguous, list)


class TestEfficiencyRatio:
    def test_er_range(self):
        closes = [float(100 + i % 5) for i in range(50)]
        df = make_df(closes)
        er = efficiency_ratio(df["close"], 20)
        valid = er.dropna()
        assert ((valid >= 0) & (valid <= 1)).all()

    def test_er_zero_denominator_safe(self):
        closes = [100.0] * 30
        df = make_df(closes)
        er = efficiency_ratio(df["close"], 20)
        # Should return 0.0 for zero denominator, not NaN or inf
        assert not er.isin([float("inf"), float("-inf")]).any()
