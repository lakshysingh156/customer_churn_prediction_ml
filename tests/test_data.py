import numpy as np
import pandas as pd
import pytest

from churn.config import TARGET
from churn.data import churn_rate, clean, encode_target, load_dataset, load_raw


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return load_raw()


def test_dataset_shape_and_columns(raw):
    assert raw.shape == (7043, 21)
    assert TARGET in raw.columns
    assert "customerID" in raw.columns


def test_total_charges_is_text_in_the_raw_file(raw):
    # The trap this project exists to handle correctly.
    assert raw["TotalCharges"].dtype == object
    assert raw["TotalCharges"].isna().sum() == 0


def test_clean_imputes_blank_total_charges_only_for_new_customers(raw):
    cleaned = clean(raw)
    assert cleaned["TotalCharges"].dtype == np.float64
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned["TotalCharges"].min() == 0.0
    zero_rows = cleaned[cleaned["TotalCharges"] == 0.0]
    assert (zero_rows["tenure"] == 0).all()


def test_clean_rejects_blank_total_charges_for_established_customers(raw):
    broken = raw.copy()
    broken.loc[broken["tenure"] > 0, "TotalCharges"] = " "
    with pytest.raises(ValueError, match="non-zero tenure"):
        clean(broken)


def test_clean_rejects_unknown_churn_labels(raw):
    broken = raw.copy()
    broken.loc[0, TARGET] = "Maybe"
    with pytest.raises(ValueError, match="Unexpected values"):
        clean(broken)


def test_load_raw_reports_missing_columns(tmp_path):
    partial = load_raw().drop(columns=["Contract"])
    path = tmp_path / "partial.csv"
    partial.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Contract"):
        load_raw(path)


@pytest.mark.parametrize("customer_ids", [["same", "same"], ["same", " same "], [" ", "other"], [None, "other"]])
def test_clean_rejects_missing_or_duplicate_training_ids(raw, customer_ids):
    broken = raw.head(2).copy()
    broken["customerID"] = customer_ids
    with pytest.raises(ValueError, match="customerID"):
        clean(broken)


def test_churn_rate_and_target_encoding():
    df = load_dataset()
    assert churn_rate(df) == pytest.approx(0.2654, abs=1e-3)
    encoded = encode_target(df[TARGET])
    assert set(np.unique(encoded)) == {0, 1}
    assert encoded.sum() == (df[TARGET] == "Yes").sum()


@pytest.mark.parametrize(
    "column,value,message",
    [
        ("SeniorCitizen", 0.8, "SeniorCitizen"),
        ("tenure", -1, "negative"),
        ("tenure", 2.5, "whole number"),
        ("MonthlyCharges", float("inf"), "MonthlyCharges"),
        ("MonthlyCharges", -1, "negative"),
        ("TotalCharges", -1, "TotalCharges"),
        ("TotalCharges", float("inf"), "TotalCharges"),
    ],
)
def test_clean_rejects_invalid_training_values(raw, column, value, message):
    broken = raw.head(2).copy()
    broken[column] = broken[column].astype(object)
    broken.loc[broken.index[0], column] = value
    with pytest.raises(ValueError, match=message):
        clean(broken)
