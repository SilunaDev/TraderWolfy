"""
ASTRA FUSION QUANT — Walk-Forward Validation (Milestone 6)
Chronological splits: 60% dev / 20% validation / 20% holdout (SEALED).
Expanding walk-forward with 5 sequential folds, purging, embargo.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    development: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame   # SEALED — do not access until final evaluation
    split_dates: dict


@dataclass
class WalkForwardFold:
    fold_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def chronological_split(
    df: pd.DataFrame,
    dev_frac: float = 0.60,
    val_frac: float = 0.20,
    holdout_frac: float = 0.20,
) -> DataSplit:
    """
    Split DataFrame chronologically.
    Holdout is reserved LAST — do not unseal until parameters are frozen.
    """
    assert abs(dev_frac + val_frac + holdout_frac - 1.0) < 1e-9, "Fractions must sum to 1"
    n = len(df)
    i_dev = int(n * dev_frac)
    i_val = int(n * (dev_frac + val_frac))

    dev       = df.iloc[:i_dev].copy()
    val       = df.iloc[i_dev:i_val].copy()
    holdout   = df.iloc[i_val:].copy()

    logger.info(
        "Split: dev=%d bars (%s–%s), val=%d bars, holdout=%d bars [SEALED]",
        len(dev), dev.index[0] if len(dev) else "N/A", dev.index[-1] if len(dev) else "N/A",
        len(val), len(holdout),
    )
    return DataSplit(
        development=dev,
        validation=val,
        holdout=holdout,
        split_dates={
            "dev_start":     str(dev.index[0])     if len(dev)     else None,
            "dev_end":       str(dev.index[-1])    if len(dev)     else None,
            "val_start":     str(val.index[0])     if len(val)     else None,
            "val_end":       str(val.index[-1])    if len(val)     else None,
            "holdout_start": str(holdout.index[0]) if len(holdout) else None,
            "holdout_end":   str(holdout.index[-1])if len(holdout) else None,
        }
    )


def expanding_walk_forward(
    dev_df: pd.DataFrame,
    n_folds: int = 5,
    min_train_frac: float = 0.50,
    embargo_bars: int = 0,
) -> list[WalkForwardFold]:
    """
    Expanding walk-forward over development data.
    - Initial train: first 50% of dev
    - Then 5 sequential test blocks covering the remaining 50%
    - Each fold adds the previous test block to training
    - Embargo: purge `embargo_bars` after each training end

    Returns list of WalkForwardFold (train/test do NOT overlap).
    """
    n = len(dev_df)
    initial_train_end = int(n * min_train_frac)
    remaining = n - initial_train_end
    fold_size = remaining // n_folds

    folds = []
    train_end_idx = initial_train_end

    for i in range(n_folds):
        test_start_idx = train_end_idx + embargo_bars
        test_end_idx   = test_start_idx + fold_size
        if i == n_folds - 1:
            test_end_idx = n   # last fold takes remainder

        if test_start_idx >= n or test_end_idx > n:
            logger.warning("Fold %d: insufficient data, skipping", i + 1)
            break

        train_df = dev_df.iloc[:train_end_idx].copy()
        test_df  = dev_df.iloc[test_start_idx:test_end_idx].copy()

        fold = WalkForwardFold(
            fold_id=i + 1,
            train=train_df,
            test=test_df,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=test_df.index[0],
            test_end=test_df.index[-1],
        )
        folds.append(fold)
        logger.info(
            "Fold %d: train %d bars (%s–%s), test %d bars (%s–%s)",
            i + 1, len(train_df), fold.train_start, fold.train_end,
            len(test_df), fold.test_start, fold.test_end,
        )
        train_end_idx = test_end_idx   # expand training window

    return folds
