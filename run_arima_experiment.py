"""ARIMA / SARIMA family -> Weights & Biases (group ARIMA_Training).

Two parts, matching how much time the brief wants spent here:
  1. Seasonal-naive on every series (cheap, full 12-week holdout) -- the family's
     operational model, saved as walmart_arima:best.
  2. SARIMA parameter search on the aggregate weekly-total series (a few orders),
     logged with AIC to show the classical modelling / diagnostics.

    python3 run_arima_experiment.py
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import statsmodels.api as sm

from src.classical_models import build_seasonal_naive_pipeline
from src.data import load_raw
from src.metrics import wmae
from src.pipeline import RAW_COLS
from src.validation import time_holdout_split
from src.wandb_utils import init_run, log_pipeline

GROUP = "ARIMA_Training"

# SARIMA orders to compare on the aggregate series: (p,d,q)(P,D,Q,s), s=52.
SARIMA_ORDERS = [
    dict(name="SARIMA_agg_111_011", order=(1, 1, 1), seasonal_order=(0, 1, 1, 52)),
    dict(name="SARIMA_agg_011_011", order=(0, 1, 1), seasonal_order=(0, 1, 1, 52)),
    dict(name="SARIMA_agg_212_110", order=(2, 1, 2), seasonal_order=(1, 1, 0, 52)),
    dict(name="SARIMA_agg_110_011", order=(1, 1, 0), seasonal_order=(0, 1, 1, 52)),
]


def run_seasonal_naive(train):
    tr, val = time_holdout_split(train, n_val_weeks=12)
    val = val.reset_index(drop=True)
    run = init_run(group=GROUP, job_type="experiment", name="ARIMA_SeasonalNaive",
                   config={"model": "seasonal_naive", "season_weeks": 52})
    pipe = build_seasonal_naive_pipeline()
    pipe.fit(tr[RAW_COLS], tr["Weekly_Sales"])
    score = wmae(val["Weekly_Sales"], pipe.predict(val[RAW_COLS]), val["IsHoliday"])
    run.summary["holdout_wmae"] = score
    run.summary["wmae_val"] = score
    run.log({"holdout_wmae": score, "wmae_val": score})
    run.finish()
    print(f"seasonal-naive full holdout WMAE = {score:,.2f}")
    return score


def run_sarima_aggregate(train):
    """Fit a few SARIMA orders on the summed weekly series; log AIC + holdout."""
    agg = train.groupby("Date")["Weekly_Sales"].sum().sort_index()
    hol = train.groupby("Date")["IsHoliday"].max().sort_index()
    cutoff = agg.index[-12]
    a_tr, a_val = agg[agg.index < cutoff], agg[agg.index >= cutoff]
    h_val = hol[hol.index >= cutoff]

    best = None
    for spec in SARIMA_ORDERS:
        run = init_run(group=GROUP, job_type="experiment", name=spec["name"],
                       config={"order": str(spec["order"]),
                               "seasonal_order": str(spec["seasonal_order"]),
                               "level": "aggregate"})
        model = sm.tsa.statespace.SARIMAX(
            a_tr, order=spec["order"], seasonal_order=spec["seasonal_order"],
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=50)
        fc = np.asarray(model.forecast(len(a_val)))
        mape = float(np.mean(np.abs((a_val.values - fc) / a_val.values)) * 100)
        agg_wmae = wmae(a_val.values, fc, h_val.values)
        run.summary["aic"] = float(model.aic)
        run.summary["agg_mape_pct"] = mape
        run.summary["agg_wmae"] = agg_wmae
        run.log({"aic": float(model.aic), "agg_mape_pct": mape, "agg_wmae": agg_wmae})
        run.finish()
        print(f"{spec['name']:22s} AIC={model.aic:>8,.0f}  MAPE={mape:5.2f}%")
        if best is None or model.aic < best[1]:
            best = (spec["name"], float(model.aic), mape)
    print(f"best aggregate SARIMA by AIC: {best[0]} (AIC {best[1]:,.0f})")


def main():
    train = load_raw("data").train
    score = run_seasonal_naive(train)
    run_sarima_aggregate(train)

    # Final: fit seasonal-naive on ALL train, register walmart_arima:best.
    run = init_run(group=GROUP, job_type="final", name="ARIMA_Final",
                   config={"model": "seasonal_naive", "season_weeks": 52})
    final = build_seasonal_naive_pipeline()
    final.fit(train[RAW_COLS], train["Weekly_Sales"])
    run.summary["holdout_wmae"] = score
    run.summary["wmae_val"] = score
    log_pipeline(run, final, name="walmart_arima",
                 metadata={"holdout_wmae": score, "model": "seasonal_naive"},
                 aliases=["best"])
    run.finish()
    print("Final model registered: walmart_arima:best")


if __name__ == "__main__":
    main()
