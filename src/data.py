"""Load the raw Kaggle CSVs and expose light cleaning / merge helpers.

Expected files in ``data_dir`` (download from the competition page and unzip):
    train.csv     Store, Dept, Date, Weekly_Sales, IsHoliday
    test.csv      Store, Dept, Date, IsHoliday
    features.csv  Store, Date, Temperature, Fuel_Price, MarkDown1..5, CPI,
                  Unemployment, IsHoliday
    stores.csv    Store, Type, Size
    sampleSubmission.csv  Id, Weekly_Sales
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "features": "features.csv",
    "stores": "stores.csv",
    "sample_submission": "sampleSubmission.csv",
}


@dataclass
class RawData:
    train: pd.DataFrame
    test: pd.DataFrame
    features: pd.DataFrame
    stores: pd.DataFrame
    sample_submission: pd.DataFrame | None = None


def load_raw(data_dir: str | Path = "data") -> RawData:
    """Read the raw competition CSVs. pandas reads .csv and .csv.zip alike."""
    data_dir = Path(data_dir)

    def _read(name: str, **kw) -> pd.DataFrame | None:
        for candidate in (data_dir / name, data_dir / f"{name}.zip"):
            if candidate.exists():
                return pd.read_csv(candidate, **kw)
        return None

    train = _read(RAW_FILES["train"], parse_dates=["Date"])
    test = _read(RAW_FILES["test"], parse_dates=["Date"])
    features = _read(RAW_FILES["features"], parse_dates=["Date"])
    stores = _read(RAW_FILES["stores"])
    sample = _read(RAW_FILES["sample_submission"])

    missing = [n for n, df in
               {"train": train, "test": test, "features": features, "stores": stores}.items()
               if df is None]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir.resolve()}. "
            "Download from the Kaggle competition page and unzip into this folder."
        )
    return RawData(train, test, features, stores, sample)


def make_submission_id(df: pd.DataFrame) -> pd.Series:
    """Kaggle submission Id is 'Store_Dept_Date' (Date as YYYY-MM-DD)."""
    d = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df["Store"].astype(str) + "_" + df["Dept"].astype(str) + "_" + d
