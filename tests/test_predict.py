import hashlib
import io
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from churn.config import ID_COLUMN, TARGET
from churn.data import encode_target, load_dataset
from churn.features import build_pipeline
from churn.predict import ChurnScorer, read_customer_csv, scored_to_csv, spreadsheet_safe_scores


@pytest.fixture(scope="module")
def frame():
    return load_dataset().sample(n=500, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="module")
def scorer(frame, tmp_path_factory):
    directory = tmp_path_factory.mktemp("artifact")
    X = frame.drop(columns=[TARGET])
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, encode_target(frame[TARGET]))
    joblib.dump(pipeline, directory / "model.joblib")
    (directory / "metadata.json").write_text(
        json.dumps({"decision_threshold": 0.5, "band_cutoffs": [0.2, 0.5, 0.6],
                    "model_sha256": hashlib.sha256((directory / "model.joblib").read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    return ChurnScorer(directory / "model.joblib", directory / "metadata.json")


def test_score_frame_returns_expected_columns(scorer, frame):
    X = frame.drop(columns=[TARGET])
    scored = scorer.score_frame(X)
    assert list(scored.columns) == [
        ID_COLUMN, "churn_probability", "churn_prediction", "risk_band", "recommended_action",
    ]
    assert len(scored) == len(frame)
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["churn_prediction"].unique()) <= {0, 1}
    assert set(scored["risk_band"].unique()) <= {"Low", "Medium", "High", "Critical"}
    assert scored["recommended_action"].notna().all()


def test_threshold_is_respected(scorer, frame):
    scored = scorer.score_frame(frame.drop(columns=[TARGET]))
    expected = (scored["churn_probability"] >= 0.5).astype(int)
    assert (scored["churn_prediction"] == expected).all()


def test_missing_columns_raise_a_helpful_error(scorer, frame):
    with pytest.raises(ValueError, match="missing required columns"):
        scorer.score_frame(frame.drop(columns=["Contract"]).drop(columns=[TARGET]).head(5))


def test_score_record_works_for_a_single_customer(scorer, frame):
    record = frame.drop(columns=[TARGET]).iloc[0].to_dict()
    result = scorer.score_record(record)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_band"] in {"Low", "Medium", "High", "Critical"}


def test_risk_bands_are_monotonic(scorer):
    bands = [scorer.risk_band(p) for p in (0.0, 0.3, 0.5, 0.9)]
    assert bands == ["Low", "Medium", "High", "Critical"]


def test_mismatched_model_is_rejected_before_loading(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"not a valid model")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"model_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Model and metadata do not match"):
        ChurnScorer(model, metadata)


@pytest.mark.parametrize(
    "identifier,exported",
    [("000123", "000123"), ("NA", "NA"), ("=2+2", "'=2+2"),
     ("  @formula", "'  @formula"), ("\ttext", "'\ttext"), ("-123", "'-123")],
)
def test_csv_identifiers_are_preserved_and_formula_safe(scorer, frame, identifier, exported):
    customer = frame.drop(columns=[TARGET]).head(1).copy()
    customer[ID_COLUMN] = identifier
    uploaded = read_customer_csv(io.StringIO(customer.to_csv(index=False)))
    assert uploaded.iloc[0][ID_COLUMN] == identifier
    scored = scorer.score_frame(uploaded)
    assert scored.iloc[0][ID_COLUMN] == identifier
    assert spreadsheet_safe_scores(scored).iloc[0][ID_COLUMN] == exported
    exported_frame = read_customer_csv(io.StringIO(scored_to_csv(scored)))
    assert exported_frame.iloc[0][ID_COLUMN] == exported


def test_missing_artifact_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/train.py"):
        ChurnScorer(tmp_path / "nope.joblib")


def test_missing_metadata_raises_clear_error(tmp_path):
    model = tmp_path / "model.joblib"
    model.touch()
    with pytest.raises(FileNotFoundError, match="metadata"):
        ChurnScorer(model, tmp_path / "missing.json")


def test_malformed_total_charges_are_rejected(scorer, frame):
    row = frame.drop(columns=[TARGET]).head(1).copy()
    row["TotalCharges"] = "not a number"
    with pytest.raises(ValueError, match="TotalCharges"):
        scorer.score_frame(row)


def test_blank_total_charges_for_new_customer_are_accepted(scorer, frame):
    row = frame.drop(columns=[TARGET]).head(1).copy()
    row["tenure"] = 0
    row["TotalCharges"] = " "
    assert len(scorer.score_frame(row)) == 1
    row["tenure"] = 5
    with pytest.raises(ValueError, match="TotalCharges"):
        scorer.score_frame(row)


def test_output_probability_matches_prediction_threshold(scorer, frame):
    scored = scorer.score_frame(frame.drop(columns=[TARGET]))
    assert (scored["churn_prediction"] == (scored["churn_probability"] >= scorer.threshold)).all()


def test_empty_customer_file_is_rejected(scorer, frame):
    with pytest.raises(ValueError, match="no customers"):
        scorer.score_frame(frame.drop(columns=[TARGET]).head(0))


@pytest.mark.parametrize("column,value", [("Contract", "Monthly"), ("PhoneService", "Maybe"), ("gender", None)])
def test_invalid_categories_are_rejected(scorer, frame, column, value):
    row = frame.drop(columns=[TARGET]).head(1).copy()
    row[column] = value
    with pytest.raises(ValueError, match=column):
        scorer.score_frame(row)


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"PhoneService": "No", "MultipleLines": "Yes"}, "MultipleLines"),
        ({"PhoneService": "Yes", "MultipleLines": "No phone service"}, "MultipleLines"),
        ({"InternetService": "No", "OnlineSecurity": "Yes"}, "OnlineSecurity"),
        ({"InternetService": "DSL", "OnlineSecurity": "No internet service"}, "OnlineSecurity"),
        ({"tenure": 1.5}, "tenure"),
    ],
)
def test_inconsistent_customer_profiles_are_rejected(scorer, frame, changes, field):
    row = frame.drop(columns=[TARGET]).head(1).copy()
    for column, value in changes.items():
        row[column] = value
    with pytest.raises(ValueError, match=field):
        scorer.score_frame(row)


def test_valid_but_unseen_training_category_is_rejected(scorer, frame, monkeypatch):
    from churn.config import CATEGORICAL_FEATURES

    encoder = scorer.pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    categories = list(encoder.categories_)
    categories[CATEGORICAL_FEATURES.index("Contract")] = np.array(["One year", "Two year"])
    monkeypatch.setattr(encoder, "categories_", categories)
    row = frame.drop(columns=[TARGET]).head(1).copy()
    row["Contract"] = "Month-to-month"
    with pytest.raises(ValueError, match="not seen during training"):
        scorer.score_frame(row)
