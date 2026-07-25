"""Prophet experiments -> Weights & Biases (group Prophet_Training).

Runs several Prophet configs that differ in seasonality mode, trend
flexibility (changepoint_prior_scale) and whether holidays are used. Each is
scored on the same 12-week holdout with the competition WMAE. The best config
is refit on all of train and saved as the pipeline artifact walmart_prophet:best.

    python3 run_prophet_experiments.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_SILENT", "true")

from src.data import load_raw
from src.metrics import wmae
from src.pipeline import RAW_COLS
from src.prophet_model import build_prophet_pipeline
from src.validation import time_holdout_split
from src.wandb_utils import init_run, log_pipeline

GROUP = "Prophet_Training"

# 5 configs, each changing seasonality / trend flexibility / holidays.
EXPERIMENTS = [
    dict(name="Prophet_v1_additive_holidays",
         cfg=dict(seasonality_mode="additive", changepoint_prior_scale=0.05,
                  use_holidays=True)),
    dict(name="Prophet_v2_multiplicative",
         cfg=dict(seasonality_mode="multiplicative", changepoint_prior_scale=0.05,
                  use_holidays=True)),
    dict(name="Prophet_v3_flexible_trend",
         cfg=dict(seasonality_mode="additive", changepoint_prior_scale=0.5,
                  use_holidays=True)),
    dict(name="Prophet_v4_no_holidays",
         cfg=dict(seasonality_mode="additive", changepoint_prior_scale=0.05,
                  use_holidays=False)),
    dict(name="Prophet_v5_weekly_seasonality",
         cfg=dict(seasonality_mode="additive", changepoint_prior_scale=0.1,
                  use_holidays=True, weekly_seasonality=True)),
]


def main():
    raw = load_raw("data")
    train = raw.train
    tr, val = time_holdout_split(train, n_val_weeks=12)
    val = val.reset_index(drop=True)
    print(f"train {train.shape} | holdout 12 weeks | tr {tr.shape} val {val.shape}\n")

    results = []
    for exp in EXPERIMENTS:
        run = init_run(group=GROUP, job_type="experiment", name=exp["name"],
                       config=exp["cfg"])
        pipe = build_prophet_pipeline(**exp["cfg"])
        pipe.fit(tr[RAW_COLS], tr["Weekly_Sales"])
        t0 = time.time()
        pred = pipe.predict(val[RAW_COLS])
        dt = time.time() - t0
        score = wmae(val["Weekly_Sales"], pred, val["IsHoliday"])
        run.summary["holdout_wmae"] = score
        run.summary["wmae_val"] = score          # same key XGBoost/LightGBM use
        run.summary["fit_predict_min"] = dt / 60
        run.log({"holdout_wmae": score, "wmae_val": score})
        log_pipeline(run, pipe, name=f"walmart_prophet_{exp['name'].split('_', 1)[1]}",
                     metadata={"holdout_wmae": score, **exp["cfg"]})
        run.finish()
        results.append((exp, score))
        print(f"{exp['name']:34s} WMAE={score:>9,.2f}  ({dt/60:.1f} min)")

    best_exp, best_score = min(results, key=lambda r: r[1])
    print(f"\nBEST Prophet: {best_exp['name']}  (holdout WMAE {best_score:,.2f})")

    # Final: refit the best config on ALL train and register walmart_prophet:best.
    run = init_run(group=GROUP, job_type="final", name="Prophet_Final",
                   config={**best_exp["cfg"], "from_experiment": best_exp["name"]})
    final = build_prophet_pipeline(**best_exp["cfg"])
    final.fit(train[RAW_COLS], train["Weekly_Sales"])
    run.summary["holdout_wmae"] = best_score
    run.summary["wmae_val"] = best_score
    log_pipeline(run, final, name="walmart_prophet",
                 metadata={"holdout_wmae": best_score, **best_exp["cfg"]},
                 aliases=["best"])
    run.finish()
    print("Final model registered: walmart_prophet:best")


if __name__ == "__main__":
    main()
