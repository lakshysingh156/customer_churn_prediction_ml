import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn.config import ENGINEERED_NUMERIC_FEATURES, ID_COLUMN, TARGET
from churn.data import encode_target, load_dataset
from churn.features import FeatureBuilder, build_pipeline


def sample_frame(rows: int = 300) -> pd.DataFrame:
    return load_dataset().sample(n=rows, random_state=0).reset_index(drop=True)


def test_feature_builder_adds_engineered_columns_and_drops_id():
    df = sample_frame()
    built = FeatureBuilder(add_engineered=True).fit_transform(df)
    for column in ENGINEERED_NUMERIC_FEATURES:
        assert column in built.columns
    assert ID_COLUMN not in built.columns
    assert len(built) == len(df)


def test_feature_builder_raw_mode_keeps_original_features_only():
    df = sample_frame()
    built = FeatureBuilder(add_engineered=False).fit_transform(df)
    for column in ENGINEERED_NUMERIC_FEATURES:
        assert column not in built.columns
    assert ID_COLUMN not in built.columns


def test_num_services_counts_only_positive_answers():
    row = sample_frame(1).copy()
    for column in ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                   "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        row[column] = "No"
    row["PhoneService"] = "Yes"
    row["MultipleLines"] = "No phone service"
    built = FeatureBuilder().fit_transform(row)
    assert built.loc[0, "num_services"] == 1.0
    assert built.loc[0, "has_internet"] == 0.0
    row["InternetService"] = "Fiber optic"
    built = FeatureBuilder().fit_transform(row)
    assert built.loc[0, "has_internet"] == 1.0
    assert built.loc[0, "num_services"] == 2.0


def test_average_charge_falls_back_to_monthly_for_new_customers():
    row = sample_frame(1).copy()
    row["tenure"] = 0
    row["TotalCharges"] = 0.0
    row["MonthlyCharges"] = 42.5
    built = FeatureBuilder().fit_transform(row)
    assert built.loc[0, "avg_charge_per_month"] == 42.5
    assert built.loc[0, "is_new_customer"] == 1.0


def test_unseen_category_is_tolerated_at_inference():
    df = sample_frame()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(df.drop(columns=[TARGET]), encode_target(df[TARGET]))
    novel = df.drop(columns=[TARGET]).iloc[[0]].copy()
    novel["Contract"] = "Lifetime"
    probabilities = pipeline.predict_proba(novel)
    assert probabilities.shape == (1, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_pipeline_output_is_probability_and_stable():
    df = sample_frame()
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    X = df.drop(columns=[TARGET])
    y = encode_target(df[TARGET])
    pipeline.fit(X, y)
    first = pipeline.predict_proba(X.head(20))[:, 1]
    second = pipeline.predict_proba(X.head(20))[:, 1]
    assert np.allclose(first, second)
    assert np.all((first > 0) & (first < 1))
