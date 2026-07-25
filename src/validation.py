"""Time-aware validation for the forecasting task.

Random K-fold leaks future weeks into training, so every split here is by
``Date``: we always train on the past and validate on the immediately
following block of weeks, mirroring how Kaggle scores the future test period.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from .metrics import wmae


def time_holdout_split(df: pd.DataFrame, n_val_weeks: int = 12,
                       date_col: str = "Date"):
    """Single split: last ``n_val_weeks`` distinct weeks become validation."""
    dates = np.sort(df[date_col].unique())
    if n_val_weeks >= len(dates):
        raise ValueError("n_val_weeks must be smaller than the number of weeks")
    cutoff = dates[-n_val_weeks]
    train_mask = df[date_col] < cutoff
    return df[train_mask].copy(), df[~train_mask].copy()


def expanding_time_folds(df: pd.DataFrame, n_splits: int = 3,
                         n_val_weeks: int = 8, date_col: str = "Date"
                         ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV. Yields integer positional (train_idx, val_idx).

    Fold k validates on a block of ``n_val_weeks`` weeks; each later fold moves
    that block forward and grows the training window (walk-forward validation).
    """
    dates = np.sort(df[date_col].unique())
    total_val = n_splits * n_val_weeks
    if total_val >= len(dates):
        raise ValueError("Not enough weeks for the requested folds")
    pos = np.arange(len(df))
    for k in range(n_splits):
        val_start = len(dates) - (n_splits - k) * n_val_weeks
        val_dates = dates[val_start: val_start + n_val_weeks]
        val_mask = df[date_col].isin(val_dates).to_numpy()
        train_mask = (df[date_col] < val_dates[0]).to_numpy()
        yield pos[train_mask], pos[val_mask]


def cross_validate_wmae(pipeline, df: pd.DataFrame, target: str = "Weekly_Sales",
                        raw_cols=("Store", "Dept", "Date", "IsHoliday"),
                        n_splits: int = 3, n_val_weeks: int = 8) -> dict:
    """Walk-forward CV of a pipeline, scored with the competition WMAE."""
    from sklearn.base import clone

    raw_cols = list(raw_cols)
    scores = []
    for i, (tr, va) in enumerate(
            expanding_time_folds(df, n_splits=n_splits, n_val_weeks=n_val_weeks)):
        tr_df, va_df = df.iloc[tr], df.iloc[va]
        model = clone(pipeline)
        model.fit(tr_df[raw_cols], tr_df[target])
        pred = model.predict(va_df[raw_cols])
        score = wmae(va_df[target], pred, va_df["IsHoliday"])
        scores.append(score)
        print(f"  fold {i + 1}/{n_splits}: WMAE = {score:,.2f} "
              f"(train={len(tr_df):,}, val={len(va_df):,})")
    return {"fold_wmae": scores,
            "mean_wmae": float(np.mean(scores)),
            "std_wmae": float(np.std(scores))}
