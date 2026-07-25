"""XGBoost experiments -> Weights & Biases (group XGBoost_Training).

Each experiment changes BOTH the feature engineering (fe) and the
hyper-parameters (params), so the wandb runs table shows what actually moved the
holdout WMAE. The best config is refit on all of train, cross-validated, and
registered as walmart_xgboost:best.

    python3 run_xgboost_experiments.py
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_SILENT", "true")

from xgboost import XGBRegressor

from src.data import load_raw
from src.metrics import wmae, holiday_weights
from src.pipeline import build_pipeline, RAW_COLS
from src.validation import time_holdout_split, cross_validate_wmae
from src.wandb_utils import init_run, log_pipeline

ARCH = "XGBoost_Training"                      # wandb group
COMMON = dict(tree_method="hist", random_state=42, n_jobs=-1)

# fe flags: use_markdowns / use_cyclical / use_holiday_flags / add_interactions
EXPERIMENTS = [
    dict(name="XGBoost_v1_baseline",
         fe=dict(),
         params=dict(objective="reg:absoluteerror", n_estimators=400,
                     learning_rate=0.05, max_depth=8,
                     subsample=0.8, colsample_bytree=0.8)),
    dict(name="XGBoost_v2_no_markdowns_deep",
         fe=dict(use_markdowns=False),
         params=dict(objective="reg:absoluteerror", n_estimators=400,
                     learning_rate=0.03, max_depth=10, min_child_weight=5,
                     subsample=0.8, colsample_bytree=0.8)),
    dict(name="XGBoost_v3_interactions_no_cyclical",
         fe=dict(add_interactions=True, use_cyclical=False),
         params=dict(objective="reg:absoluteerror", n_estimators=400,
                     learning_rate=0.05, max_depth=8,
                     subsample=0.8, colsample_bytree=0.7)),
    dict(name="XGBoost_v4_rich_regularised",
         fe=dict(add_interactions=True),
         params=dict(objective="reg:absoluteerror", n_estimators=400,
                     learning_rate=0.03, max_depth=9, reg_lambda=2.0,
                     subsample=0.9, colsample_bytree=0.8)),
    dict(name="XGBoost_v5_no_markdowns_tuned",
         fe=dict(use_markdowns=False),
         params=dict(objective="reg:absoluteerror", n_estimators=700,
                     learning_rate=0.02, max_depth=10, min_child_weight=3,
                     reg_lambda=1.5, subsample=0.85, colsample_bytree=0.85)),
    dict(name="XGBoost_v6_shallow_many_trees",
         fe=dict(use_markdowns=False),
         params=dict(objective="reg:absoluteerror", n_estimators=1200,
                     learning_rate=0.02, max_depth=6, min_child_weight=2,
                     subsample=0.8, colsample_bytree=0.8)),
    dict(name="XGBoost_v7_squared_loss",   # ablation: MSE objective instead of MAE
         fe=dict(use_markdowns=False),
         params=dict(objective="reg:squarederror", n_estimators=700,
                     learning_rate=0.02, max_depth=10, min_child_weight=3,
                     subsample=0.85, colsample_bytree=0.85)),
    dict(name="XGBoost_v8_no_holiday_flags",   # ablation: drop named-holiday flags
         fe=dict(use_markdowns=False, use_holiday_flags=False),
         params=dict(objective="reg:absoluteerror", n_estimators=700,
                     learning_rate=0.02, max_depth=10, min_child_weight=3,
                     reg_lambda=1.5, subsample=0.85, colsample_bytree=0.85)),
    dict(name="XGBoost_v9_deep_strong_reg",
         fe=dict(use_markdowns=False),
         params=dict(objective="reg:absoluteerror", n_estimators=900,
                     learning_rate=0.02, max_depth=12, min_child_weight=4,
                     reg_lambda=3.0, reg_alpha=1.0,
                     subsample=0.8, colsample_bytree=0.8)),
    dict(name="XGBoost_v10_all_features_tuned",
         fe=dict(add_interactions=True),   # markdowns + cyclical + interactions
         params=dict(objective="reg:absoluteerror", n_estimators=800,
                     learning_rate=0.02, max_depth=10, min_child_weight=3,
                     reg_lambda=1.5, subsample=0.85, colsample_bytree=0.85)),
]


def main():
    raw = load_raw("data")
    train, features, stores = raw.train, raw.features, raw.stores
    tr_df, val_df = time_holdout_split(train, n_val_weeks=12)
    w_tr = holiday_weights(tr_df["IsHoliday"])
    print(f"train {train.shape} | holdout 12 weeks | tr {tr_df.shape} val {val_df.shape}\n")

    results = []
    for exp in EXPERIMENTS:
        cfg = {**exp["fe"], **exp["params"]}
        run = init_run(group=ARCH, job_type="experiment", name=exp["name"], config=cfg)
        pipe = build_pipeline(XGBRegressor(**COMMON, **exp["params"]),
                              features, stores, **exp["fe"])
        pipe.fit(tr_df[RAW_COLS], tr_df["Weekly_Sales"], model__sample_weight=w_tr)
        pred = pipe.predict(val_df[RAW_COLS])
        score = wmae(val_df["Weekly_Sales"], pred, val_df["IsHoliday"])
        n_features = len(pipe.named_steps["finalize"].columns_)
        run.summary["holdout_wmae"] = score
        run.summary["wmae_val"] = score          # same key the LightGBM runs use
        run.summary["n_features"] = n_features
        run.log({"holdout_wmae": score, "wmae_val": score, "n_features": n_features})
        log_pipeline(run, pipe, name=f"walmart_xgboost_{exp['name'].split('_', 1)[1]}",
                     metadata={"holdout_wmae": score, **cfg})
        run.finish()
        results.append((exp, score, n_features))
        print(f"{exp['name']:38s} WMAE={score:>9,.2f}  features={n_features}")

    best_exp, best_score, _ = min(results, key=lambda r: r[1])
    print(f"\nBEST config: {best_exp['name']}  (holdout WMAE {best_score:,.2f})")

    # Final: refit best config on ALL train, walk-forward CV, register best.
    cfg = {**best_exp["fe"], **best_exp["params"]}
    run = init_run(group=ARCH, job_type="final", name="XGBoost_Final",
                   config={**cfg, "from_experiment": best_exp["name"]})
    cv = cross_validate_wmae(
        build_pipeline(XGBRegressor(**COMMON, **best_exp["params"]), features, stores,
                       **best_exp["fe"]),
        train, n_splits=3, n_val_weeks=8)
    run.summary["cv_mean_wmae"] = cv["mean_wmae"]
    run.summary["cv_std_wmae"] = cv["std_wmae"]
    run.summary["wmae_cv_mean"] = cv["mean_wmae"]     # teammate-style keys
    run.summary["wmae_cv_std"] = cv["std_wmae"]
    for i, s in enumerate(cv["fold_wmae"], 1):
        run.summary[f"wmae_fold_{i}"] = s

    final = build_pipeline(XGBRegressor(**COMMON, **best_exp["params"]), features, stores,
                           **best_exp["fe"])
    final.fit(train[RAW_COLS], train["Weekly_Sales"],
              model__sample_weight=holiday_weights(train["IsHoliday"]))
    run.summary["holdout_wmae"] = best_score
    run.summary["wmae_val"] = best_score
    log_pipeline(run, final, name="walmart_xgboost",
                 metadata={"holdout_wmae": best_score, "cv_mean_wmae": cv["mean_wmae"], **cfg},
                 aliases=["best"])
    run.finish()
    print(f"\nFinal model registered: walmart_xgboost:best "
          f"| CV WMAE {cv['mean_wmae']:,.2f} +/- {cv['std_wmae']:,.2f}")


if __name__ == "__main__":
    main()
