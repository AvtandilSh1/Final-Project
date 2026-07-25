"""TimesFM (Google foundation model) zero-shot -> Weights & Biases.

TimesFM is a pretrained time-series foundation model: no training on our data,
we just feed each series' history as context and ask for the next 12 weeks.
We sweep the context length. Runs go to wandb group TimesFM_Training.

Run with the isolated venv that has timesfm installed:
    scratchpad/tfm_venv/bin/python run_timesfm_experiment.py

Note: this is the optional/bonus model. We log the holdout scores but do NOT
push the 200M pretrained model into the shared registry -- it is reproducible
from `from_pretrained("google/timesfm-2.5-200m-pytorch")`.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import timesfm

from src.data import load_raw
from src.metrics import wmae
from src.validation import time_holdout_split
from src.wandb_utils import init_run

GROUP = "TimesFM_Training"
REPO = "google/timesfm-2.5-200m-pytorch"
CONTEXTS = [104, 256, 512, 1024]     # the swept hyper-parameter


def main():
    train = load_raw("data").train
    tr, val = time_holdout_split(train, n_val_weeks=12)
    val = val.reset_index(drop=True)
    holdout_dates = list(np.sort(val["Date"].unique()))
    pos_of = {d: i for i, d in enumerate(holdout_dates)}
    horizon = len(holdout_dates)

    trg = {k: g.sort_values("Date")["Weekly_Sales"].to_numpy(dtype=np.float32)
           for k, g in tr.groupby(["Store", "Dept"])}
    keys = [k for k in trg if len(trg[k]) >= 8]
    inputs = [trg[k] for k in keys]
    key_idx = {k: i for i, k in enumerate(keys)}
    global_mean = float(tr["Weekly_Sales"].mean())
    print(f"series to forecast: {len(keys)} | horizon {horizon}")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(REPO)

    for ctx in CONTEXTS:
        run = init_run(group=GROUP, job_type="experiment",
                       name=f"TimesFM_ctx{ctx}",
                       config={"model": REPO, "max_context": ctx, "horizon": horizon,
                               "zero_shot": True})
        model.compile(timesfm.ForecastConfig(
            max_context=ctx, max_horizon=horizon, normalize_inputs=True,
            use_continuous_quantile_head=True, infer_is_positive=True))
        t0 = time.time()
        point, _ = model.forecast(horizon=horizon, inputs=inputs)
        dt = time.time() - t0

        preds = np.empty(len(val), dtype=float)
        for i, r in enumerate(val.itertuples()):
            k = (int(r.Store), int(r.Dept))
            j = key_idx.get(k)
            preds[i] = global_mean if j is None else point[j][pos_of[r.Date]]
        preds = np.clip(preds, 0, None)
        score = wmae(val["Weekly_Sales"], preds, val["IsHoliday"])
        run.summary["holdout_wmae"] = score
        run.summary["wmae_val"] = score
        run.summary["forecast_sec"] = dt
        run.log({"holdout_wmae": score, "wmae_val": score})
        run.finish()
        print(f"TimesFM ctx={ctx:5d}  WMAE={score:>9,.2f}  ({dt:.0f}s)")


if __name__ == "__main__":
    main()
