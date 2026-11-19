"""Feature engineering and the preprocessing pipeline.

All transformations live inside sklearn estimators so they are fitted on training
folds only and serialised with the model. Nothing is computed on the full frame
before splitting, which is the leakage bug in most churn tutorials.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NEW_CUSTOMER_TENURE,
    NUMERIC_FEATURES,
    RAW_NUMERIC_FEATURES,
    SERVICE_COLUMNS,
)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Drops the identifier and adds four row-wise behavioural features.

    ``add_engineered=False`` keeps the raw fields only; the training script uses
    that mode to measure whether the engineered features actually earn their place.
    """

    def __init__(self, add_engineered: bool = True):
        self.add_engineered = add_engineered

    def fit(self, X: pd.DataFrame, y=None):
        self._check_columns(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        df = X.copy()

        if ID_COLUMN in df.columns:
            df = df.drop(columns=ID_COLUMN)

        tenure = df["tenure"].astype(float)
        total = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

        if self.add_engineered:
            df["has_internet"] = (df["InternetService"] != "No").astype(float)
            df["num_services"] = (
                (df[SERVICE_COLUMNS] == "Yes").sum(axis=1).astype(float) + df["has_internet"]
            )
            df["avg_charge_per_month"] = (total / tenure.where(tenure > 0)).fillna(
                df["MonthlyCharges"]
            )
            df["is_new_customer"] = (tenure <= NEW_CUSTOMER_TENURE).astype(float)

        df["tenure"] = tenure
        df["TotalCharges"] = total
        return df.reset_index(drop=True)

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureBuilder expects a pandas DataFrame.")
        required = set(SERVICE_COLUMNS) | {"tenure", "MonthlyCharges", "TotalCharges"}
        missing = required - set(X.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")


def build_preprocessor(engineered: bool = True) -> ColumnTransformer:
    numeric_columns = NUMERIC_FEATURES if engineered else RAW_NUMERIC_FEATURES
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, numeric_columns),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, engineered: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("features", FeatureBuilder(add_engineered=engineered)),
            ("preprocess", build_preprocessor(engineered)),
            ("model", estimator),
        ]
    )
