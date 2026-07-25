# Prophet wrapper: one Prophet per Store x Dept series (Prophet has no global
# model). To save it as a Pipeline that runs on raw test rows, we keep only the
# training history inside the estimator and refit each series at predict time --
# much lighter than pickling ~3300 fitted Stan models.
from __future__ import annotations

import contextlib
import io
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline

from .pipeline import SUPER_BOWL, LABOR_DAY, THANKSGIVING, CHRISTMAS

for _lg in ("prophet", "cmdstanpy"):
    logging.getLogger(_lg).setLevel(logging.CRITICAL)


def competition_holidays() -> pd.DataFrame:
    """The 4 named holidays Kaggle weights 5x, as a Prophet holidays frame."""
    return pd.DataFrame({
        "holiday": (["SuperBowl"] * 4 + ["LaborDay"] * 4
                    + ["Thanksgiving"] * 4 + ["Christmas"] * 4),
        "ds": sorted(SUPER_BOWL) + sorted(LABOR_DAY)
              + sorted(THANKSGIVING) + sorted(CHRISTMAS),
    })


def _fit_one(hist_ds, hist_y, want_ds, hol_records, cfg):
    """Fit a single Prophet on one series and forecast the wanted dates."""
    import numpy as np
    import pandas as pd
    from prophet import Prophet

    y = np.asarray(hist_y, dtype=float)
    want_ds = pd.to_datetime(want_ds)
    # too little history -> just predict the series mean
    if len(y) < cfg["min_history"]:
        return np.full(len(want_ds), float(y.mean()) if len(y) else 0.0)

    dfp = pd.DataFrame({"ds": pd.to_datetime(hist_ds), "y": y})
    holidays = pd.DataFrame(hol_records) if hol_records else None
    try:
        m = Prophet(yearly_seasonality=cfg["yearly_seasonality"],
                    weekly_seasonality=cfg["weekly_seasonality"],
                    daily_seasonality=False,
                    seasonality_mode=cfg["seasonality_mode"],
                    changepoint_prior_scale=cfg["changepoint_prior_scale"],
                    holidays=holidays,
                    uncertainty_samples=0)
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            m.fit(dfp)
            fc = m.predict(pd.DataFrame({"ds": want_ds}))
        return fc["yhat"].to_numpy()
    except Exception:
        # Stan can fail on odd series; fall back to the mean instead of crashing
        return np.full(len(want_ds), float(y.mean()))


class ProphetForecaster(BaseEstimator, RegressorMixin):
    """One Prophet per Store x Dept. fit() stores history, predict() forecasts."""

    def __init__(self, yearly_seasonality: bool = True,
                 weekly_seasonality: bool = False,
                 seasonality_mode: str = "additive",
                 changepoint_prior_scale: float = 0.05,
                 use_holidays: bool = True, min_history: int = 8,
                 n_jobs: int = -1):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.use_holidays = use_holidays
        self.min_history = min_history
        self.n_jobs = n_jobs

    def _cfg(self) -> dict:
        return dict(yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    changepoint_prior_scale=self.changepoint_prior_scale,
                    min_history=self.min_history)

    def fit(self, X: pd.DataFrame, y):
        df = X[["Store", "Dept", "Date"]].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["y"] = np.asarray(y, dtype=float)
        self.history_ = {
            (int(s), int(d)): g.sort_values("Date")[["Date", "y"]]
            for (s, d), g in df.groupby(["Store", "Dept"])
        }
        self.global_mean_ = float(df["y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        X["Date"] = pd.to_datetime(X["Date"])
        X = X.reset_index(drop=True)
        hol = (competition_holidays().assign(ds=lambda d: d["ds"].astype(str))
               .to_dict("records")) if self.use_holidays else []
        cfg = self._cfg()

        tasks, slots, fallback_idx = [], [], []
        for (s, d), g in X.groupby(["Store", "Dept"]):
            hist = self.history_.get((int(s), int(d)))
            idx = g.index.to_numpy()
            if hist is None:                       # series never seen in train
                fallback_idx.append(idx)
                continue
            tasks.append((hist["Date"].astype(str).tolist(),
                          hist["y"].tolist(),
                          g["Date"].astype(str).tolist()))
            slots.append(idx)

        results = Parallel(n_jobs=self.n_jobs, backend="loky", batch_size=16)(
            delayed(_fit_one)(ds, y, wds, hol, cfg) for ds, y, wds in tasks
        ) if tasks else []

        out = np.zeros(len(X), dtype=float)
        for idx in fallback_idx:
            out[idx] = self.global_mean_
        for idx, yhat in zip(slots, results):
            out[idx] = yhat
        return np.clip(out, 0, None)


def build_prophet_pipeline(**kwargs) -> Pipeline:
    """Wrap the forecaster as a one-step Pipeline so it's saved like the others."""
    return Pipeline([("prophet", ProphetForecaster(**kwargs))])
