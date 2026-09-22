"""
ASTRA FUSION QUANT — Holdout Evaluation (Milestone 7)
Final holdout is evaluated ONCE after parameters are frozen.
Failure = failure. Do not auto-promote least-bad config.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from research.validation.metrics import compute_metrics, MetricReport

logger = logging.getLogger(__name__)

_HOLDOUT_UNSEALED = False  # flipped to True exactly once


@dataclass
class HoldoutResult:
    status: str   # "PASSED", "FAILED", "INCONCLUSIVE", "NOT_RUN"
    metrics: MetricReport
    promotion_gate_details: dict
    notes: str


# Promotion gate thresholds (project screening policies — not statistical guarantees)
GATE = {
    "min_trades_total":      200,
    "min_trades_per_scope":  50,
    "min_expectancy_r":      0.0,    # lower CI bound must exceed 0
    "max_drawdown_pct":      15.0,
    "min_positive_folds":    4,      # out of 5 walk-forward folds
    "cost_mult_robustness":  1.5,    # must be positive at 1.5x costs
}


def evaluate_holdout(
    run_fn: Callable[[pd.DataFrame], list[dict]],
    holdout_df: pd.DataFrame,
    fold_results: list[dict],
    high_cost_run_fn: Optional[Callable[[pd.DataFrame], list[dict]]] = None,
) -> HoldoutResult:
    """
    Evaluate sealed final holdout ONCE.
    Call this only after engine/features/costs/thresholds are frozen.

    Args:
        run_fn: function(df) → list of trade dicts for the frozen system
        holdout_df: the sealed holdout DataFrame
        fold_results: list of per-fold metric dicts (expectancy_r per fold)
        high_cost_run_fn: optional run at 1.5x costs for robustness gate
    """
    global _HOLDOUT_UNSEALED
    if _HOLDOUT_UNSEALED:
        raise RuntimeError(
            "HOLDOUT ALREADY UNSEALED. Cannot evaluate twice. "
            "Changed designs require future unseen data or a new genuinely untouched holdout."
        )
    _HOLDOUT_UNSEALED = True
    logger.warning("=== UNSEALING FINAL HOLDOUT — this can only happen once ===")

    trades = run_fn(holdout_df)
    m = compute_metrics(trades, len(holdout_df))

    gate_details = {}
    gate_passes = 0
    gate_total  = 0

    # Gate 1: Trade count
    gate_total += 1
    ok = m.trade_count >= GATE["min_trades_total"]
    gate_details["trades_>=200"] = {"value": m.trade_count, "passed": ok}
    if ok: gate_passes += 1

    # Gate 2: Positive expectancy (lower Wilson CI > 0 is proxy — use win_rate CI)
    gate_total += 1
    ok = m.expectancy_r > GATE["min_expectancy_r"]
    gate_details["positive_expectancy"] = {"value": m.expectancy_r, "passed": ok}
    if ok: gate_passes += 1

    # Gate 3: Max drawdown
    gate_total += 1
    ok = abs(m.max_drawdown_pct) <= GATE["max_drawdown_pct"]
    gate_details["drawdown_<=15pct"] = {"value": m.max_drawdown_pct, "passed": ok}
    if ok: gate_passes += 1

    # Gate 4: Walk-forward fold positivity
    gate_total += 1
    positive_folds = sum(1 for f in fold_results if f.get("expectancy_r", -1) > 0)
    ok = positive_folds >= GATE["min_positive_folds"]
    gate_details[f"wf_folds_positive_{positive_folds}_of_5"] = {"value": positive_folds, "passed": ok}
    if ok: gate_passes += 1

    # Gate 5: Cost robustness (optional)
    if high_cost_run_fn is not None:
        gate_total += 1
        hc_trades = high_cost_run_fn(holdout_df)
        hc_m = compute_metrics(hc_trades, len(holdout_df))
        ok = hc_m.expectancy_r > 0
        gate_details["positive_at_1.5x_costs"] = {"value": hc_m.expectancy_r, "passed": ok}
        if ok: gate_passes += 1

    # Determine status
    if m.trade_count < 50:
        status = "INCONCLUSIVE"
        notes = "Fewer than 50 trades — sample too small for reliable conclusions."
    elif gate_passes == gate_total:
        status = "PASSED"
        notes = f"All {gate_total} promotion gates passed."
    else:
        status = "FAILED"
        failed = [k for k, v in gate_details.items() if not v["passed"]]
        notes = f"Failed {gate_total - gate_passes}/{gate_total} gates: {failed}. " \
                "Do not auto-promote. Failure is a valid research outcome."

    logger.info("Holdout result: %s | %s", status, notes)
    return HoldoutResult(
        status=status,
        metrics=m,
        promotion_gate_details=gate_details,
        notes=notes,
    )
