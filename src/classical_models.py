# Classical (ARIMA/SARIMA) family. Full per-series SARIMA with a 52-week season
# is too slow on ~3300 series, so the model we actually ship is seasonal-naive:
# predict each week with the same series' value 52 weeks earlier. That is just
# SARIMA (0,1,0)(0,1,0,52) -- only seasonal differencing -- so it's in the same
# family, but cheap and easy to save as a raw-test pipeline. The heavier SARIMA
# order search / AIC is done on the aggregate series in the notebook.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline


class SeasonalNaiveForecaster(BaseEstimator, RegressorMixin):
    """Predict Weekly_Sales as the value 52 weeks earlier for the same series."""

    def __init__(self, season_weeks: int = 52, tolerance_days: int = 10):
        self.season_weeks = season_weeks
        self.tolerance_days = tolerance_days

    def fit(self, X: pd.DataFrame, y):
        df = X[["Store", "Dept", "Date"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["y"] = np.asarray(y, dtype=float)
        # exact (store, dept, date) -> value, plus last value per series as fallback
        self.lookup_ = {(int(r.Store), int(r.Dept), r.Date.normalize()): r.y
                        for r in df.itertuples()}
        self.last_ = {}
        for (s, d), g in df.groupby(["Store", "Dept"]):
            self.last_[(int(s), int(d))] = float(g.sort_values("Date")["y"].iloc[-1])
        self.global_mean_ = float(df["y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        X["Date"] = pd.to_datetime(X["Date"])
        offset = pd.Timedelta(weeks=self.season_weeks)
        out = np.empty(len(X), dtype=float)
        for i, r in enumerate(X.itertuples()):
            key = (int(r.Store), int(r.Dept))
            target = (r.Date - offset).normalize()
            val = self.lookup_.get((key[0], key[1], target))
            if val is None:                       # no exact year-ago week
                val = self.last_.get(key, self.global_mean_)
            out[i] = val
        return np.clip(out, 0, None)


def build_seasonal_naive_pipeline(**kwargs) -> Pipeline:
    return Pipeline([("seasonal_naive", SeasonalNaiveForecaster(**kwargs))])
