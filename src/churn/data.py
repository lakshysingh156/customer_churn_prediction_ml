"""Loading and cleaning the IBM Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn.config import POSITIVE_LABEL, RAW_DATA_PATH, TARGET

YES_NO = {"Yes", "No"}
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]
CATEGORY_VALUES = {
    "gender": {"Female", "Male"},
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{column: {"Yes", "No", "No internet service"} for column in INTERNET_ADDONS},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": YES_NO,
    "PaymentMethod": {
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
    },
}

REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET,
}


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"customerID": "string"}, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def _row_positions(mask: pd.Series) -> list[int]:
    return (np.flatnonzero(mask.to_numpy())[:5] + 1).tolist()


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate Telco inputs and convert numeric fields for training or scoring."""
    df = df.copy()
    for column, allowed in CATEGORY_VALUES.items():
        invalid = ~df[column].isin(allowed)
        if invalid.any():
            raise ValueError(
                f"Invalid {column} at input rows {_row_positions(invalid)}; "
                f"allowed values: {sorted(allowed)}."
            )

    for column in ["tenure", "MonthlyCharges", "SeniorCitizen"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any() or not np.isfinite(df[column]).all():
            raise ValueError(f"{column} must contain valid numbers.")
    if (df[["tenure", "MonthlyCharges"]] < 0).any().any():
        raise ValueError("Tenure and monthly charges cannot be negative.")
    if (df["tenure"] % 1 != 0).any():
        raise ValueError("tenure must be a whole number of months.")
    if not df["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError("SeniorCitizen must be 0 or 1.")

    raw_total = df["TotalCharges"]
    blanks = raw_total.isna() | raw_total.astype(str).str.strip().eq("")
    total = pd.to_numeric(raw_total, errors="coerce")
    if (total.isna() & ~blanks).any() or not np.isfinite(total.dropna()).all():
        raise ValueError("TotalCharges must contain valid numbers or be blank for zero-tenure customers.")
    if (blanks & df["tenure"].ne(0)).any():
        raise ValueError("Blank TotalCharges found for non-zero tenure; cannot impute safely.")
    if (total.dropna() < 0).any():
        raise ValueError("TotalCharges must be non-negative.")
    df["TotalCharges"] = total.fillna(0.0).astype(float)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(float)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)

    invalid_phone = df["PhoneService"].eq("No") != df["MultipleLines"].eq("No phone service")
    if invalid_phone.any():
        raise ValueError(
            f"MultipleLines must be 'No phone service' exactly when PhoneService is 'No' "
            f"(input rows {_row_positions(invalid_phone)})."
        )
    for column in INTERNET_ADDONS:
        invalid = df["InternetService"].eq("No") != df[column].eq("No internet service")
        if invalid.any():
            raise ValueError(
                f"{column} must be 'No internet service' exactly when InternetService is 'No' "
                f"(input rows {_row_positions(invalid)})."
            )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_features(df)
    customer_ids = df["customerID"].astype("string").str.strip()
    if customer_ids.isna().any() or customer_ids.eq("").any():
        raise ValueError("Training data must contain a customerID for every row.")
    if customer_ids.duplicated().any():
        raise ValueError("Training data contains duplicate customerID values.")
    unexpected = set(df[TARGET].unique()) - {"Yes", "No"}
    if unexpected:
        raise ValueError(f"Unexpected values in {TARGET}: {unexpected}")
    return df


def load_dataset(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def encode_target(y: pd.Series) -> np.ndarray:
    return (y == POSITIVE_LABEL).astype(int).to_numpy()


def churn_rate(df: pd.DataFrame) -> float:
    return float((df[TARGET] == POSITIVE_LABEL).mean())
