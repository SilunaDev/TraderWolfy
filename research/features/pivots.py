"""
ASTRA FUSION QUANT — Pivot Detection (Milestone 1)
High/low pivot detection with L/R lookback, alternating swing sequence,
confirmation timing, and ambiguous pivot handling.

Key availability rule:
  A pivot at bar p with R right-hand confirmation bars is KNOWN at close of bar p+R.
  No trading signal can use this pivot before bar p+R.

Internal pivots:  L=3, R=3
External pivots:  L=5, R=5 (structure)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_pivot_counter = 0


def _next_id() -> str:
    global _pivot_counter
    _pivot_counter += 1
    return f"PVT_{_pivot_counter:06d}"


@dataclass
class Pivot:
    """Immutable pivot record."""
    id: str
    side: str                     # 'high' or 'low'
    price: float
    origin_bar: int               # bar index where pivot originates
    confirmation_bar: int         # bar index when pivot is KNOWN (origin + R)
    origin_time: pd.Timestamp
    confirmation_time: pd.Timestamp
    strength: int                 # number of L+R bars used
    is_external: bool = False     # True if from external (structural) pivot set
    ambiguous: bool = False       # True if bar qualified as both H and L


@dataclass
class PivotStore:
    """Stores all confirmed pivots with alternating swing tracking."""
    all_pivots: list[Pivot] = field(default_factory=list)
    # Alternating sequence (keeps only best of adjacent same-side)
    swing_sequence: list[Pivot] = field(default_factory=list)

    def add(self, pivot: Pivot) -> None:
        """Add pivot and update the alternating swing sequence."""
        self.all_pivots.append(pivot)

        if pivot.ambiguous:
            logger.debug("AMBIGUOUS_PIVOT at bar %d — skipping swing sequence", pivot.origin_bar)
            return

        seq = self.swing_sequence
        if not seq:
            seq.append(pivot)
            return

        last = seq[-1]
        if last.side == pivot.side:
            # Same side — keep the more extreme
            if pivot.side == "high" and pivot.price >= last.price:
                seq[-1] = pivot
            elif pivot.side == "low" and pivot.price <= last.price:
                seq[-1] = pivot
            # else keep existing (earlier equal extreme wins)
        else:
            seq.append(pivot)

    def last_high(self) -> Optional[Pivot]:
        """Return most recent confirmed high pivot."""
        for p in reversed(self.all_pivots):
            if p.side == "high" and not p.ambiguous:
                return p
        return None

    def last_low(self) -> Optional[Pivot]:
        for p in reversed(self.all_pivots):
            if p.side == "low" and not p.ambiguous:
                return p
        return None

    def swing_highs(self) -> list[Pivot]:
        return [p for p in self.swing_sequence if p.side == "high"]

    def swing_lows(self) -> list[Pivot]:
        return [p for p in self.swing_sequence if p.side == "low"]

    def confirmed_before(self, bar_idx: int) -> list[Pivot]:
        """All pivots confirmed at or before bar_idx."""
        return [p for p in self.all_pivots if p.confirmation_bar <= bar_idx]


class PivotDetector:
    """
    Detects high and low pivot bars using the L/R symmetric lookback rule.

    A high pivot at bar p requires:
      H[p] > H[p-1], ..., H[p-L]   (strictly greater than all L preceding)
      H[p] >= H[p+1], ..., H[p+R]  (greater-than-or-equal to all R following)
    This chooses the EARLIEST equal extreme.

    A low pivot is the inverse.

    A bar qualifying as BOTH high and low → AMBIGUOUS_PIVOT.
    """

    def __init__(self, L: int = 3, R: int = 3, is_external: bool = False):
        if L < 1 or R < 1:
            raise ValueError("L and R must be >= 1")
        self.L = L
        self.R = R
        self.is_external = is_external

    def detect(self, df: pd.DataFrame) -> PivotStore:
        """
        Run pivot detection over a validated OHLCV DataFrame.
        Pivot at bar p is confirmed at bar p+R (known_at close of p+R).

        Returns a PivotStore with all confirmed pivots.
        """
        store = PivotStore()
        highs = df["high"].values
        lows  = df["low"].values
        idx   = df.index
        n     = len(df)
        L, R  = self.L, self.R

        for p in range(L, n - R):
            h_p = highs[p]
            l_p = lows[p]

            # High pivot check
            is_high = (
                all(h_p > highs[p - j] for j in range(1, L + 1)) and
                all(h_p >= highs[p + j] for j in range(1, R + 1))
            )

            # Low pivot check
            is_low = (
                all(l_p < lows[p - j] for j in range(1, L + 1)) and
                all(l_p <= lows[p + j] for j in range(1, R + 1))
            )

            ambiguous = is_high and is_low
            conf_bar = p + R
            conf_time = idx[conf_bar]
            origin_time = idx[p]

            if is_high:
                pivot = Pivot(
                    id=_next_id(),
                    side="high",
                    price=h_p,
                    origin_bar=p,
                    confirmation_bar=conf_bar,
                    origin_time=origin_time,
                    confirmation_time=conf_time,
                    strength=L + R,
                    is_external=self.is_external,
                    ambiguous=ambiguous,
                )
                store.add(pivot)
                if ambiguous:
                    logger.debug(
                        "AMBIGUOUS_PIVOT at bar %d (high): H=%.5f, L=%.5f", p, h_p, l_p
                    )

            if is_low and not ambiguous:
                pivot = Pivot(
                    id=_next_id(),
                    side="low",
                    price=l_p,
                    origin_bar=p,
                    confirmation_bar=conf_bar,
                    origin_time=origin_time,
                    confirmation_time=conf_time,
                    strength=L + R,
                    is_external=self.is_external,
                    ambiguous=False,
                )
                store.add(pivot)

        logger.debug(
            "PivotDetector(L=%d,R=%d): found %d pivots (%d external)",
            L, R, len(store.all_pivots), sum(1 for p in store.all_pivots if p.is_external)
        )
        return store


def detect_equal_highs_lows(
    store: PivotStore,
    atr: pd.Series,
    min_bar_separation: int = 3,
    max_age_bars: int = 150,
    tolerance_atr_mult: float = 0.15,
    current_bar: int = 0,
) -> list[dict]:
    """
    Detect equal highs/lows clusters.
    At least two same-side confirmed pivots:
    - separated by >= min_bar_separation bars
    - within max(2 ticks, 0.15*A) price distance at the time the later pivot becomes known
    - expire after max_age_bars or decisive breakout

    Returns list of cluster dicts (tolerance frozen at creation).
    """
    clusters = []

    def _atr_at(bar: int) -> float:
        """Get ATR value at confirmation bar."""
        if bar < len(atr):
            v = atr.iloc[bar]
            return float(v) if not np.isnan(v) else 0.0
        return 0.0

    for side in ["high", "low"]:
        pivots = [p for p in store.all_pivots if p.side == side and not p.ambiguous]
        for i, p1 in enumerate(pivots):
            for p2 in pivots[i + 1:]:
                if p2.confirmation_bar - p1.confirmation_bar < min_bar_separation:
                    continue
                if current_bar - p2.confirmation_bar > max_age_bars:
                    continue

                atr_val = _atr_at(p2.confirmation_bar)
                tolerance = max(2 * 1e-5, tolerance_atr_mult * atr_val)  # 2 ticks approx

                if abs(p1.price - p2.price) <= tolerance:
                    clusters.append({
                        "side": side,
                        "pivot_ids": [p1.id, p2.id],
                        "price_level": (p1.price + p2.price) / 2,
                        "tolerance_frozen": tolerance,
                        "known_at_bar": p2.confirmation_bar,
                        "origin_bar": p1.origin_bar,
                        "expires_at_bar": p2.confirmation_bar + max_age_bars,
                    })

    return clusters
