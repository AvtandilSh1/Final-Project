"""PatchTST experiments -> Weights & Biases (group PatchTST_Training).

One global PatchTST trained on windows from every series. Variants change the
lookback length, patch size and Transformer capacity. Best config is refit on
all of train and saved as walmart_patchtst:best.

    python3 run_patchtst_experiments.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_SILENT", "true")

import torch

from src.data import load_raw
from src.metrics import wmae
from src.patchtst_model import build_patchtst_pipeline
from src.pipeline import RAW_COLS
from src.validation import time_holdout_split
from src.wandb_utils import init_run, log_pipeline

GROUP = "PatchTST_Training"

# window_stride=3 keeps every 3rd training window (+ the most recent) so a full
# sweep runs on CPU in minutes instead of ~20 min/variant.
_COMMON = dict(batch_size=2048, window_stride=3)
EXPERIMENTS = [
    dict(name="PatchTST_v1_baseline",
         cfg=dict(seq_len=52, patch_len=8, stride=4, d_model=64, n_heads=4,
                  depth=2, epochs=12, lr=1e-3, **_COMMON)),
    dict(name="PatchTST_v2_long_lookback",
         cfg=dict(seq_len=104, patch_len=16, stride=8, d_model=64, n_heads=4,
                  depth=2, epochs=12, lr=1e-3, **_COMMON)),
    dict(name="PatchTST_v3_deeper_wider",
         cfg=dict(seq_len=52, patch_len=8, stride=4, d_model=128, n_heads=8,
                  depth=2, epochs=10, lr=1e-3, **_COMMON)),
    dict(name="PatchTST_v4_small_patches",
         cfg=dict(seq_len=52, patch_len=4, stride=2, d_model=64, n_heads=4,
                  depth=2, epochs=12, lr=1e-3, **_COMMON)),
    dict(name="PatchTST_v5_long_train",
         cfg=dict(seq_len=52, patch_len=8, stride=4, d_model=64, n_heads=4,
                  depth=2, epochs=20, lr=5e-4, **_COMMON)),
]


def main():
    torch.manual_seed(42)
    train = load_raw("data").train
    tr, val = time_holdout_split(train, n_val_weeks=12)
    val = val.reset_index(drop=True)
    print(f"train {train.shape} | holdout 12 weeks | device cpu\n")

    results = []
    for exp in EXPERIMENTS:
        run = init_run(group=GROUP, job_type="experiment", name=exp["name"],
                       config=exp["cfg"])
        t0 = time.time()
        pipe = build_patchtst_pipeline(**exp["cfg"])
        pipe.fit(tr[RAW_COLS], tr["Weekly_Sales"])
        pred = pipe.predict(val[RAW_COLS])
        dt = time.time() - t0
        score = wmae(val["Weekly_Sales"], pred, val["IsHoliday"])
        run.summary["holdout_wmae"] = score
        run.summary["wmae_val"] = score
        run.summary["train_min"] = dt / 60
        run.log({"holdout_wmae": score, "wmae_val": score})
        log_pipeline(run, pipe, name=f"walmart_patchtst_{exp['name'].split('_', 1)[1]}",
                     metadata={"holdout_wmae": score, **exp["cfg"]})
        run.finish()
        results.append((exp, score))
        print(f"{exp['name']:28s} WMAE={score:>9,.2f}  ({dt/60:.1f} min)")

    best_exp, best_score = min(results, key=lambda r: r[1])
    print(f"\nBEST PatchTST: {best_exp['name']}  (holdout WMAE {best_score:,.2f})")

    run = init_run(group=GROUP, job_type="final", name="PatchTST_Final",
                   config={**best_exp["cfg"], "from_experiment": best_exp["name"]})
    final = build_patchtst_pipeline(**best_exp["cfg"])
    final.fit(train[RAW_COLS], train["Weekly_Sales"])
    run.summary["holdout_wmae"] = best_score
    run.summary["wmae_val"] = best_score
    log_pipeline(run, final, name="walmart_patchtst",
                 metadata={"holdout_wmae": best_score, **best_exp["cfg"]},
                 aliases=["best"])
    run.finish()
    print("Final model registered: walmart_patchtst:best")


if __name__ == "__main__":
    main()
