"""Reusable code for the Walmart Store Sales Forecasting project.

Modules
-------
metrics     : WMAE (competition metric) and an sklearn-compatible scorer.
data        : loading the raw Kaggle CSVs and light cleaning helpers.
pipeline    : sklearn transformers + `build_pipeline` so a fitted model runs
              end-to-end on the RAW test set (Store, Dept, Date, IsHoliday).
validation  : time-based train/validation split and time-series CV.
"""

from . import metrics, data, pipeline, validation  # noqa: F401

__all__ = ["metrics", "data", "pipeline", "validation"]
