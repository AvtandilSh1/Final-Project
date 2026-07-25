"""Log Cleaning / Feature_Engineering / CV staging runs to wandb (assignment layout)."""
from __future__ import annotations

import wandb

from .wandb_utils import init_run


def log_cleaning(group: str, train, test, features, stores):
    run = init_run(group=group, job_type="cleaning", name=f"{group.split('_')[0]}_Cleaning",
                   config={"stage": "cleaning"})
    null_train = int(train.isna().sum().sum())
    null_test = int(test.isna().sum().sum())
    null_feat = {c: int(features[c].isna().sum()) for c in features.columns if features[c].isna().any()}
    payload = {
        "train_rows": len(train), "test_rows": len(test),
        "n_stores": int(stores["Store"].nunique()),
        "n_depts": int(train["Dept"].nunique()),
        "null_train": null_train, "null_test": null_test,
        "feature_nulls": null_feat,
        "date_min": str(train["Date"].min().date()),
        "date_max": str(train["Date"].max().date()),
    }
    run.summary.update(payload)
    run.log(payload)
    run.finish()
    return payload


def log_feature_engineering(group: str, n_features: int, fe_config: dict):
    arch = group.split("_")[0]
    run = init_run(group=group, job_type="feature_engineering",
                   name=f"{arch}_Feature_Engineering", config=fe_config)
    payload = {"n_features": n_features, **{f"fe_{k}": v for k, v in fe_config.items()}}
    run.summary.update(payload)
    run.log(payload)
    run.finish()
    return payload


def log_cv(group: str, cv: dict, config: dict | None = None):
    arch = group.split("_")[0]
    run = init_run(group=group, job_type="cv", name=f"{arch}_CV", config=config or {})
    payload = {
        "wmae_cv_mean": cv["mean_wmae"],
        "wmae_cv_std": cv["std_wmae"],
        "cv_mean_wmae": cv["mean_wmae"],
        "cv_std_wmae": cv["std_wmae"],
    }
    for i, s in enumerate(cv["fold_wmae"], 1):
        payload[f"wmae_fold_{i}"] = s
    run.summary.update(payload)
    run.log(payload)
    run.finish()
    return payload
