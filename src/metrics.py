"""Competition metric: Weighted Mean Absolute Error (WMAE).

Kaggle weights every holiday week 5x and every other week 1x:

    WMAE = sum(w_i * |y_i - yhat_i|) / sum(w_i),   w_i = 5 if holiday else 1
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def holiday_weights(is_holiday) -> np.ndarray:
    """Map an IsHoliday boolean array to competition weights (5 / 1)."""
    is_holiday = np.asarray(is_holiday).astype(bool)
    return np.where(is_holiday, 5.0, 1.0)


def wmae(y_true, y_pred, is_holiday) -> float:
    """Weighted Mean Absolute Error used by the Kaggle leaderboard."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = holiday_weights(is_holiday)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


def wmae_from_frame(df: pd.DataFrame, y_true_col: str, y_pred_col: str,
                    holiday_col: str = "IsHoliday") -> float:
    """Convenience wrapper when true/pred/holiday all live in one DataFrame."""
    return wmae(df[y_true_col], df[y_pred_col], df[holiday_col])
