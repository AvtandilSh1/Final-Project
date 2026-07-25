"""Pick the best registered pipeline from wandb, predict raw test, write submission.

Mirrors model_inference.ipynb (kept as a script because nbconvert is not
installed here). Scans every `final` run, takes the lowest holdout_wmae, loads
that model artifact from wandb and predicts on the RAW test set.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import pandas as pd
import wandb

from src.data import load_raw, make_submission_id
from src.pipeline import RAW_COLS
from src.wandb_utils import WANDB_PROJECT, WANDB_ENTITY, load_pipeline, project_path


def main():
    test = load_raw("data").test
    api = wandb.Api()
    path = project_path(api)

    rows, best_run, best_wmae = [], None, float("inf")
    for r in api.runs(path):
        if r.job_type != "final":
            continue
        w = r.summary.get("holdout_wmae")
        if not isinstance(w, (int, float)):
            continue
        rows.append((r.group, r.name, round(float(w), 1)))
        if w < best_wmae:
            best_wmae, best_run = float(w), r

    print(pd.DataFrame(rows, columns=["group", "run", "holdout_wmae"])
          .sort_values("holdout_wmae").to_string(index=False))
    print(f"\nBEST final: {best_run.group} / {best_run.name}  WMAE={best_wmae:,.1f}")

    model_art = next(a for a in best_run.logged_artifacts() if a.type == "model")
    collection = model_art.name.split(":")[0]
    if "best" not in model_art.aliases:
        model_art.aliases.append("best")
        model_art.save()

    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                     job_type="inference", name="inference")
    used = run.use_artifact(f"{collection}:best")
    model = load_pipeline(used.download())
    preds = np.clip(model.predict(test[RAW_COLS]), 0, None)

    os.makedirs("submissions", exist_ok=True)
    sub = pd.DataFrame({"Id": make_submission_id(test), "Weekly_Sales": preds})
    sub.to_csv("submissions/submission.csv", index=False)
    art = wandb.Artifact("submission", type="submission")
    art.add_file("submissions/submission.csv")
    run.log_artifact(art)
    run.summary["n_predictions"] = len(preds)
    run.summary["pred_mean"] = float(preds.mean())
    run.finish()
    print(f"\nwrote submissions/submission.csv  ({len(sub):,} rows, mean={preds.mean():,.1f})")


if __name__ == "__main__":
    main()
