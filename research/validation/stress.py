"""
ASTRA FUSION QUANT — Stress Testing (Milestone 6)
Parameter perturbation, family ablation, cost sensitivity, bootstrap.
All thresholds are UNVALIDATED hypotheses. Record every trial in the ledger.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import pandas as pd
from research.validation.metrics import compute_metrics, MetricReport

logger = logging.getLogger(__name__)


@dataclass
class StressResult:
    label: str
    metrics: MetricReport
    config_delta: dict


def perturb_parameter(
    base_value: float | int,
    delta_pct: float,
    is_integer: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> list[float]:
    """Return [base * (1-delta), base, base * (1+delta)] respecting constraints."""
    variants = [base_value * (1 - delta_pct), base_value, base_value * (1 + delta_pct)]
    if is_integer:
        variants = [max(1, round(v)) for v in variants]
    if min_val is not None:
        variants = [max(min_val, v) for v in variants]
    if max_val is not None:
        variants = [min(max_val, v) for v in variants]
    return list(dict.fromkeys(variants))   # deduplicate preserving order


def run_parameter_sensitivity(
    run_fn: Callable[[dict], list[dict]],  # fn(config) → list of trade dicts
    base_config: dict,
    param_keys: list[str],
    total_bars: int,
    deltas: list[float] = [0.10, 0.20],
) -> list[StressResult]:
    """
    For each param, test ±10% and ±20% variants.
    All other params held at base. Records each trial.
    """
    results = []
    for key in param_keys:
        base_val = base_config.get(key)
        if base_val is None or not isinstance(base_val, (int, float)):
            continue
        for delta in deltas:
            for variant in perturb_parameter(base_val, delta, is_integer=isinstance(base_val, int)):
                cfg = dict(base_config)
                cfg[key] = variant
                trades = run_fn(cfg)
                m = compute_metrics(trades, total_bars)
                label = f"{key}={variant:.4g} (base={base_val:.4g}, delta={delta*100:.0f}%)"
                results.append(StressResult(label=label, metrics=m, config_delta={key: variant}))
                logger.info("Sensitivity %s: E(R)=%.3f, N=%d", label, m.expectancy_r, m.trade_count)
    return results


def run_family_ablation(
    run_fn: Callable[[dict], list[dict]],
    base_config: dict,
    families: list[str],
    total_bars: int,
) -> list[StressResult]:
    """Remove each family separately and measure impact."""
    results = []
    for family in families:
        cfg = dict(base_config)
        cfg[f"disable_family_{family}"] = True
        trades = run_fn(cfg)
        m = compute_metrics(trades, total_bars)
        results.append(StressResult(
            label=f"ablate_family_{family}",
            metrics=m,
            config_delta={f"disable_family_{family}": True},
        ))
        logger.info("Ablation family=%s: E(R)=%.3f, N=%d", family, m.expectancy_r, m.trade_count)
    return results


def run_cost_sensitivity(
    run_fn: Callable[[dict], list[dict]],
    base_config: dict,
    total_bars: int,
    cost_multipliers: list[float] = [1.0, 1.5, 2.0],
) -> list[StressResult]:
    """Run at 1x, 1.5x, 2x base costs."""
    results = []
    base_commission = base_config.get("commission_per_side_pct", 0.10)
    for mult in cost_multipliers:
        cfg = dict(base_config)
        cfg["commission_per_side_pct"] = base_commission * mult
        trades = run_fn(cfg)
        m = compute_metrics(trades, total_bars)
        results.append(StressResult(
            label=f"cost_{mult}x",
            metrics=m,
            config_delta={"commission_per_side_pct": cfg["commission_per_side_pct"]},
        ))
        logger.info("Cost %sx: E(R)=%.3f, N=%d", mult, m.expectancy_r, m.trade_count)
    return results


def block_bootstrap(
    daily_returns: pd.Series,
    n_replicates: int = 1000,
    block_sizes: list[int] = [5, 10, 20],
    seed: int = 42,
) -> dict:
    """
    Moving block bootstrap of daily strategy returns.
    Returns distribution of terminal return and max drawdown per block size.

    NOTE: Reconstructs equity for drawdown. Simple reordering does not estimate
    terminal-return uncertainty — block bootstrap is used instead.
    """
    rng = np.random.default_rng(seed)
    results: dict[int, dict] = {}

    for bs in block_sizes:
        terminal_returns = []
        max_dds = []
        arr = daily_returns.values

        for _ in range(n_replicates):
            # Sample blocks with replacement
            n = len(arr)
            n_blocks = max(1, n // bs)
            idxs = rng.integers(0, n - bs + 1, size=n_blocks)
            sampled = np.concatenate([arr[i:i + bs] for i in idxs])[:n]
            equity = 1.0 + np.cumsum(sampled)
            terminal_returns.append(float(equity[-1] - 1.0))
            roll_max = np.maximum.accumulate(equity)
            dd = (equity - roll_max) / roll_max
            max_dds.append(float(dd.min()))

        results[bs] = {
            "terminal_return_mean":   float(np.mean(terminal_returns)),
            "terminal_return_p5":     float(np.percentile(terminal_returns, 5)),
            "terminal_return_p95":    float(np.percentile(terminal_returns, 95)),
            "max_dd_mean":            float(np.mean(max_dds)),
            "max_dd_p5":              float(np.percentile(max_dds, 5)),
            "max_dd_p95":             float(np.percentile(max_dds, 95)),
            "n_replicates":           n_replicates,
            "block_size":             bs,
        }
        logger.info(
            "Bootstrap block=%d: E[terminal]=%.3f, p5/p95=[%.3f, %.3f]",
            bs, results[bs]["terminal_return_mean"],
            results[bs]["terminal_return_p5"], results[bs]["terminal_return_p95"],
        )

    return results
