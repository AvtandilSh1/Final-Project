# sklearn Pipeline for the tree models. The point: it takes the raw test rows
# [Store, Dept, Date, IsHoliday] and does the merge + cleaning + feature building
# inside, so pipe.predict(test[RAW_COLS]) works with no preprocessing outside.
# features/stores live inside the first step, so the saved model is self-contained.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

RAW_COLS = ["Store", "Dept", "Date", "IsHoliday"]

# Holiday Fridays for this competition (used to build proximity flags).
SUPER_BOWL = {"2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"}
LABOR_DAY = {"2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"}
THANKSGIVING = {"2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"}
CHRISTMAS = {"2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"}


class ExternalMerge(BaseEstimator, TransformerMixin):
    """Merge ``stores.csv`` (on Store) and ``features.csv`` (on Store+Date)."""

    def __init__(self, features_df: pd.DataFrame, stores_df: pd.DataFrame):
        self.features_df = features_df
        self.stores_df = stores_df

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Date"] = pd.to_datetime(X["Date"])
        feats = self.features_df.copy()
        feats["Date"] = pd.to_datetime(feats["Date"])
        # features.csv also carries IsHoliday; keep the one from train/test.
        feats = feats.drop(columns=[c for c in ["IsHoliday"] if c in feats.columns])
        df = X.merge(self.stores_df, on="Store", how="left")
        df = df.merge(feats, on=["Store", "Date"], how="left")
        return df


class DateFeatures(BaseEstimator, TransformerMixin):
    """Calendar parts + optional cyclical (sin/cos) and named-holiday flags.

    The two flags let each experiment turn those feature groups on/off.
    """

    def __init__(self, use_cyclical: bool = True, use_holiday_flags: bool = True):
        self.use_cyclical = use_cyclical
        self.use_holiday_flags = use_holiday_flags

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        d = pd.to_datetime(X["Date"])
        X["Year"] = d.dt.year
        X["Month"] = d.dt.month
        X["Week"] = d.dt.isocalendar().week.astype(int)
        X["Day"] = d.dt.day
        X["DayOfYear"] = d.dt.dayofyear
        if self.use_cyclical:
            X["Week_sin"] = np.sin(2 * np.pi * X["Week"] / 52.0)
            X["Week_cos"] = np.cos(2 * np.pi * X["Week"] / 52.0)
            X["Month_sin"] = np.sin(2 * np.pi * X["Month"] / 12.0)
            X["Month_cos"] = np.cos(2 * np.pi * X["Month"] / 12.0)
        if self.use_holiday_flags:
            ds = d.dt.strftime("%Y-%m-%d")
            X["IsSuperBowl"] = ds.isin(SUPER_BOWL).astype(int)
            X["IsLaborDay"] = ds.isin(LABOR_DAY).astype(int)
            X["IsThanksgiving"] = ds.isin(THANKSGIVING).astype(int)
            X["IsChristmas"] = ds.isin(CHRISTMAS).astype(int)
        return X


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Optional pairwise interaction features (off by default)."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return X
        X = X.copy()
        if {"Size", "IsHoliday"}.issubset(X.columns):
            X["Size_x_Holiday"] = X["Size"] * X["IsHoliday"].astype(int)
        if {"CPI", "Unemployment"}.issubset(X.columns):
            X["CPI_x_Unemp"] = X["CPI"] * X["Unemployment"]
        if {"Temperature", "Month"}.issubset(X.columns):
            X["Temp_x_Month"] = X["Temperature"] * X["Month"]
        return X


class NumericImputer(BaseEstimator, TransformerMixin):
    """MarkDowns -> 0 (absent before Nov-2011); other numerics -> train median."""

    def __init__(self, markdown_fill: float = 0.0):
        self.markdown_fill = markdown_fill

    def fit(self, X, y=None):
        cols = ["CPI", "Unemployment", "Temperature", "Fuel_Price", "Size"]
        self.medians_ = {c: float(X[c].median()) for c in cols if c in X.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for i in range(1, 6):
            c = f"MarkDown{i}"
            if c in X.columns:
                X[c] = X[c].fillna(self.markdown_fill)
        for c, m in self.medians_.items():
            if c in X.columns:
                X[c] = X[c].fillna(m)
        return X


class FinalizeFeatures(BaseEstimator, TransformerMixin):
    """Encode Type, cast IsHoliday, drop non-features, fix the column order.

    Store/Dept/Type stay as integer codes so the output is all-numeric and any
    tree model takes it without special categorical handling.
    """

    TYPE_MAP = {"A": 0, "B": 1, "C": 2}
    DROP = {"Date", "Weekly_Sales"}

    def __init__(self, drop_cols=()):
        # extra feature columns to exclude (e.g. markdowns for an FE variant)
        self.drop_cols = tuple(drop_cols)

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "Type" in X.columns:
            X["Type"] = X["Type"].map(self.TYPE_MAP).fillna(-1).astype(int)
        if "IsHoliday" in X.columns:
            X["IsHoliday"] = X["IsHoliday"].astype(int)
        return X

    def fit(self, X, y=None):
        Xt = self._encode(X)
        drop = self.DROP.union(self.drop_cols)
        self.columns_ = [c for c in Xt.columns if c not in drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = self._encode(X)
        for c in self.columns_:            # guard against a column going missing
            if c not in Xt.columns:
                Xt[c] = 0
        return Xt[self.columns_]


MARKDOWN_COLS = tuple(f"MarkDown{i}" for i in range(1, 6))


def build_preprocessor(features_df: pd.DataFrame, stores_df: pd.DataFrame, *,
                       use_cyclical: bool = True, use_holiday_flags: bool = True,
                       use_markdowns: bool = True, markdown_fill: float = 0.0,
                       add_interactions: bool = False) -> Pipeline:
    """The full preprocessing chain (no estimator).

    The keyword flags select which feature-engineering groups are produced, so
    different experiment runs can compare feature-engineering choices.
    """
    drop_cols = () if use_markdowns else MARKDOWN_COLS
    return Pipeline([
        ("merge", ExternalMerge(features_df, stores_df)),
        ("dates", DateFeatures(use_cyclical=use_cyclical,
                               use_holiday_flags=use_holiday_flags)),
        ("impute", NumericImputer(markdown_fill=markdown_fill)),
        ("interactions", InteractionFeatures(enabled=add_interactions)),
        ("finalize", FinalizeFeatures(drop_cols=drop_cols)),
    ])


def build_pipeline(model, features_df: pd.DataFrame, stores_df: pd.DataFrame,
                   **fe_kwargs) -> Pipeline:
    """Preprocessing chain + a fitted-later estimator, runnable on raw test rows.

    Extra keyword args are forwarded to :func:`build_preprocessor` (FE toggles).
    """
    pre = build_preprocessor(features_df, stores_df, **fe_kwargs)
    return Pipeline(pre.steps + [("model", model)])
