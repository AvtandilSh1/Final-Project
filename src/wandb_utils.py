"""Weights & Biases tracking helpers for the Walmart project.

Structure mapping (classic MLflow layout -> wandb):
    experiment (per architecture)  ->  wandb *group*      (LightGBM / XGBoost)
    run (per stage)                ->  wandb *run*        (job_type = stage)
    Model Registry                 ->  versioned model *artifact* + `best` alias

Run ``wandb login`` once (or export ``WANDB_API_KEY``) before using the notebooks.
"""
from __future__ import annotations

import os
import tempfile

import joblib
import wandb

WANDB_PROJECT = "walmart-sales-forecasting-project"
# Shared team entity so both teammates (ikakh22 / AvtandilSh1) see the same runs.
# Set to None to fall back to your personal default entity.
WANDB_ENTITY: str | None = "ashos22-free-university-of-tbilisi-"


def init_run(group: str, job_type: str, name: str, config: dict | None = None):
    """Start one wandb run inside an architecture ``group``."""
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=group,
        job_type=job_type,
        name=name,
        config=config or {},
        reinit=True,
    )


def log_pipeline(run, pipeline, name: str, metadata: dict | None = None,
                 aliases: list[str] | None = None):
    """Pickle a fitted sklearn Pipeline and log it as a wandb model artifact.

    The Pipeline already has ``features``/``stores`` baked in, so the artifact is
    self-contained and can predict on the raw test set after download. Pass
    ``aliases=["best"]`` to promote it in the model registry.
    """
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "pipeline.joblib")
    joblib.dump(pipeline, path)
    art = wandb.Artifact(name, type="model", metadata=metadata or {})
    art.add_file(path, name="pipeline.joblib")
    run.log_artifact(art, aliases=aliases)
    return art


def load_pipeline(artifact_dir: str):
    """Load the Pipeline back from a downloaded artifact directory."""
    return joblib.load(os.path.join(artifact_dir, "pipeline.joblib"))


def project_path(api) -> str:
    """`entity/project` string for the wandb public API."""
    entity = WANDB_ENTITY or api.default_entity
    return f"{entity}/{WANDB_PROJECT}"
