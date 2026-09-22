"""
ASTRA FUSION QUANT — HTF Feature Computation (Milestone 1)
Aggregates chart-timeframe OHLCV to higher timeframes.
Provides confirmed-bar-only availability, matching Pine's request.security pattern.

Key rule: at chart bar close t, only HTF bars whose close_time <= t are usable.
Never use the developing (not-yet-closed) HTF bar.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from research.features.primitives import (
    atr_wilder,
    ema,
    trend_score,
)

logger = logging.getLogger(__name__)

# Timeframe string → approximate minutes
TF_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1H": 60, "2H": 120, "4H": 240,
    "1D": 1440, "1W": 10080, "1M": 43200,
}


def tf_to_minutes(tf: str) -> int:
    v = TF_MINUTES.get(tf)
    if v is None:
        raise ValueError(f"Unknown timeframe '{tf}'. Add to TF_MINUTES.")
    return v


def validate_htf_order(chart_tf: str, htf1: str, htf2: str) -> None:
    """Reject invalid timeframe ordering: chart < HTF1 < HTF2 required."""
    c = tf_to_minutes(chart_tf)
    h1 = tf_to_minutes(htf1)
    h2 = tf_to_minutes(htf2)
    if not (c < h1 < h2):
        raise ValueError(
            f"Invalid timeframe ordering: chart={chart_tf} ({c}m), "
            f"HTF1={htf1} ({h1}m), HTF2={htf2} ({h2}m). "
            f"Require chart < HTF1 < HTF2."
        )


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample chart OHLCV to a higher timeframe.
    rule: pandas offset alias e.g. '1H', '4H', '1D', '1W'
    Returns DataFrame with HTF OHLCV indexed by bar OPEN time (UTC).
    """
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    htf = df.resample(rule, closed="left", label="left").agg(agg)
    htf = htf.dropna(how="all")
    return htf


def _tf_to_pandas_rule(tf: str) -> str:
    """Convert ASTRA TF string to pandas resample rule."""
    mapping = {
        "1m": "1min", "3m": "3min", "5m": "5min",
        "15m": "15min", "30m": "30min",
        "1H": "1h",  "2H": "2h",  "4H": "4h",
        "1D": "1D",  "1W": "1W",  "1M": "1ME",
    }
    r = mapping.get(tf)
    if r is None:
        raise ValueError(f"Cannot map timeframe '{tf}' to pandas rule.")
    return r


class HTFContext:
    """
    Provides confirmed HTF OHLCV and derived features for each chart bar.

    Availability rule (matching Pine request.security with lookahead_on + offset):
    At chart bar close time t, the latest available HTF bar is the one whose
    CLOSE time <= t. This is equivalent to using the previous confirmed HTF value.
    """

    def __init__(
        self,
        chart_df: pd.DataFrame,
        chart_tf: str,
        htf1: str,
        htf2: str,
    ):
        validate_htf_order(chart_tf, htf1, htf2)
        self.chart_tf = chart_tf
        self.htf1 = htf1
        self.htf2 = htf2

        rule1 = _tf_to_pandas_rule(htf1)
        rule2 = _tf_to_pandas_rule(htf2)

        self._htf1_raw = resample_ohlcv(chart_df, rule1)
        self._htf2_raw = resample_ohlcv(chart_df, rule2)

        self._htf1_features = self._compute_htf_features(self._htf1_raw, htf1)
        self._htf2_features = self._compute_htf_features(self._htf2_raw, htf2)

        # For each chart bar, find the confirmed (previous close) HTF bar index
        # HTF bar at index i has close_time = index[i+1] - 1 tick
        # So at chart bar close t, we use HTF bars with open_time < t
        self._chart_index = chart_df.index

    @staticmethod
    def _compute_htf_features(htf_df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Compute EMA, ATR, trend score for a HTF DataFrame."""
        if len(htf_df) < 5:
            return htf_df.copy()

        df = htf_df.copy()
        df["ema20"]  = ema(df["close"], 20)
        df["ema50"]  = ema(df["close"], 50)
        df["ema200"] = ema(df["close"], 200)
        df["atr14"]  = atr_wilder(df["high"], df["low"], df["close"], 14)

        df["trend_t"] = trend_score(
            df["close"], df["ema20"], df["ema50"], df["ema200"], df["atr14"]
        )
        return df

    def _get_confirmed_htf_value(
        self, htf_features: pd.DataFrame, chart_ts: pd.Timestamp, col: str
    ) -> float:
        """
        Get the latest confirmed HTF value available at chart bar close time chart_ts.
        A HTF bar is confirmed only when its CLOSE time <= chart_ts.
        HTF bar open at t_htf closes at t_htf + htf_duration - 1 tick.
        We conservatively use: HTF bars with open_time < chart_ts (previous bar).
        """
        # Bars with open time strictly before chart_ts are confirmed
        eligible = htf_features[htf_features.index < chart_ts]
        if eligible.empty:
            return np.nan
        val = eligible[col].dropna()
        if val.empty:
            return np.nan
        return float(val.iloc[-1])

    def get_htf_snapshot(self, chart_ts: pd.Timestamp) -> dict:
        """
        Return a dict of confirmed HTF values available at chart_ts.
        All values are from the previous confirmed HTF bar only.
        """
        snap: dict = {}

        for prefix, feats in [("htf1", self._htf1_features), ("htf2", self._htf2_features)]:
            for col in ["open", "high", "low", "close", "volume",
                        "ema20", "ema50", "ema200", "atr14", "trend_t"]:
                if col in feats.columns:
                    snap[f"{prefix}_{col}"] = self._get_confirmed_htf_value(feats, chart_ts, col)
                else:
                    snap[f"{prefix}_{col}"] = np.nan

        # HTF bias: bullish T>=0.35, bearish T<=-0.35, neutral otherwise
        for prefix in ["htf1", "htf2"]:
            t = snap.get(f"{prefix}_trend_t", np.nan)
            if np.isnan(t):
                snap[f"{prefix}_bias"] = "unknown"
            elif t >= 0.35:
                snap[f"{prefix}_bias"] = "bullish"
            elif t <= -0.35:
                snap[f"{prefix}_bias"] = "bearish"
            else:
                snap[f"{prefix}_bias"] = "neutral"

        return snap

    def build_htf_series(self) -> pd.DataFrame:
        """
        Build a DataFrame aligned to the chart index with confirmed HTF values
        for every chart bar. Useful for vectorized research.
        """
        rows = []
        for ts in self._chart_index:
            snap = self.get_htf_snapshot(ts)
            snap["timestamp"] = ts
            rows.append(snap)

        result = pd.DataFrame(rows).set_index("timestamp")
        return result
