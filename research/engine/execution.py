"""
ASTRA FUSION QUANT — Python Execution Simulator (Milestone 5)
Event-driven execution: next-bar market orders, cost modeling, gap handling.
Does not inspect future bars at signal time.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd
from research.engine.risk import RiskSnapshot

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    STOP_HIT      = "STOP_HIT"
    TP1_HIT       = "TP1_HIT"
    TP2_HIT       = "TP2_HIT"
    TP3_HIT       = "TP3_HIT"
    TIME_EXIT     = "TIME_EXIT"
    GAP_AT_FILL   = "GAP_INVALIDATED_AT_FILL"
    SESSION_END   = "SESSION_END"
    MANUAL        = "MANUAL"


@dataclass
class Fill:
    bar_idx: int
    timestamp: pd.Timestamp
    price: float
    quantity: float
    is_entry: bool
    cost: float = 0.0


@dataclass
class TradeRecord:
    signal_id: str
    side: str
    risk_snapshot: RiskSnapshot
    entry_fill: Optional[Fill] = None
    exit_fills: list[Fill] = field(default_factory=list)
    exit_reason: Optional[ExitReason] = None
    pnl_r: float = 0.0          # result in R units
    pnl_price: float = 0.0
    rejected_at_entry: bool = False
    rejection_reason: str = ""
    ambiguous_bar: bool = False  # stop and target both reachable in one bar


class ExecutionSimulator:
    """
    Event-driven execution simulator.
    - Entry: next-bar open (market order after signal close)
    - No same-close fills, no recalculation on fill
    - Costs: commission + slippage per venue config
    - Gap through stop or target → GAP_INVALIDATED_AT_FILL
    - Time exit after 40 bars
    - Conservative path: assume stop hit first when both reachable in one bar
    """
    TIME_EXIT_BARS = 40

    def __init__(
        self,
        commission_per_side_pct: float = 0.10,
        slippage_ticks: int = 1,
        tick_size: float = 1e-5,
        point_value: float = 1.0,
    ):
        self.commission_pct = commission_per_side_pct / 100.0
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.point_value = point_value
        self._open_trade: Optional[TradeRecord] = None
        self._completed: list[TradeRecord] = []

    def _entry_cost(self, price: float, qty: float) -> float:
        return price * qty * self.commission_pct + self.slippage_ticks * self.tick_size * qty

    def _exit_cost(self, price: float, qty: float) -> float:
        return price * qty * self.commission_pct + self.slippage_ticks * self.tick_size * qty

    def process_signal(
        self, signal_id: str, side: str, snap: RiskSnapshot, bar_idx: int, ts: pd.Timestamp
    ) -> Optional[TradeRecord]:
        """Register a signal for execution next bar."""
        if self._open_trade is not None:
            logger.debug("Signal %s: POSITION_ALREADY_OPEN — logged but not executed", signal_id)
            return None
        trade = TradeRecord(signal_id=signal_id, side=side, risk_snapshot=snap)
        self._open_trade = trade
        return trade

    def process_bar(
        self,
        bar_idx: int,
        ts: pd.Timestamp,
        open_: float,
        high: float,
        low: float,
        close: float,
    ) -> Optional[TradeRecord]:
        """
        Process one bar for the open trade. Returns completed trade if closed.
        """
        if self._open_trade is None:
            return None

        trade = self._open_trade
        snap = trade.risk_snapshot

        # ── Entry fill (next bar open) ────────────────────────────────────────
        if trade.entry_fill is None:
            slip = self.slippage_ticks * self.tick_size
            fill_price = open_ + slip if snap.side == "bullish" else open_ - slip

            # Gap check: if open gaps past stop, GAP_INVALIDATED_AT_FILL
            if snap.side == "bullish" and open_ <= snap.structural_stop:
                trade.rejected_at_entry = True
                trade.rejection_reason = ExitReason.GAP_AT_FILL.value
                trade.exit_reason = ExitReason.GAP_AT_FILL
                self._complete_trade(trade)
                return trade
            if snap.side == "bearish" and open_ >= snap.structural_stop:
                trade.rejected_at_entry = True
                trade.rejection_reason = ExitReason.GAP_AT_FILL.value
                trade.exit_reason = ExitReason.GAP_AT_FILL
                self._complete_trade(trade)
                return trade

            cost = self._entry_cost(fill_price, snap.quantity)
            trade.entry_fill = Fill(bar_idx, ts, fill_price, snap.quantity, True, cost)
            logger.debug("Entry fill %s: %.5f @ bar %d", trade.signal_id, fill_price, bar_idx)
            return None

        # ── Check exits ───────────────────────────────────────────────────────
        entry = trade.entry_fill.price
        stop  = snap.structural_stop
        tp1   = snap.tp1

        if snap.side == "bullish":
            stop_hit = low <= stop
            tp1_hit  = high >= tp1 if tp1 else False
        else:
            stop_hit = high >= stop
            tp1_hit  = low <= tp1 if tp1 else False

        # Both reachable in one bar — ambiguous; conservative = stop first
        if stop_hit and tp1_hit:
            trade.ambiguous_bar = True
            stop_hit = True
            tp1_hit  = False

        if stop_hit:
            fill_price = stop
            cost = self._exit_cost(fill_price, snap.quantity)
            trade.exit_fills.append(Fill(bar_idx, ts, fill_price, snap.quantity, False, cost))
            trade.exit_reason = ExitReason.STOP_HIT
            trade.pnl_price = (fill_price - entry) * (1 if snap.side == "bullish" else -1)
            trade.pnl_r = trade.pnl_price / snap.stop_distance if snap.stop_distance > 0 else 0.0
            self._complete_trade(trade)
            return trade

        if tp1_hit and tp1:
            fill_price = tp1
            cost = self._exit_cost(fill_price, snap.quantity)
            trade.exit_fills.append(Fill(bar_idx, ts, fill_price, snap.quantity, False, cost))
            trade.exit_reason = ExitReason.TP1_HIT
            trade.pnl_price = (fill_price - entry) * (1 if snap.side == "bullish" else -1)
            trade.pnl_r = trade.pnl_price / snap.stop_distance if snap.stop_distance > 0 else 0.0
            self._complete_trade(trade)
            return trade

        # Time exit after 40 bars
        entry_bar = trade.entry_fill.bar_idx
        if bar_idx - entry_bar >= self.TIME_EXIT_BARS:
            fill_price = close
            cost = self._exit_cost(fill_price, snap.quantity)
            trade.exit_fills.append(Fill(bar_idx, ts, fill_price, snap.quantity, False, cost))
            trade.exit_reason = ExitReason.TIME_EXIT
            trade.pnl_price = (fill_price - entry) * (1 if snap.side == "bullish" else -1)
            trade.pnl_r = trade.pnl_price / snap.stop_distance if snap.stop_distance > 0 else 0.0
            self._complete_trade(trade)
            return trade

        return None

    def _complete_trade(self, trade: TradeRecord) -> None:
        self._open_trade = None
        self._completed.append(trade)
        logger.debug("Trade completed: %s | reason=%s | R=%.2f",
                     trade.signal_id,
                     trade.exit_reason.value if trade.exit_reason else "N/A",
                     trade.pnl_r)

    @property
    def completed_trades(self) -> list[TradeRecord]:
        return list(self._completed)

    @property
    def has_open_position(self) -> bool:
        return self._open_trade is not None
