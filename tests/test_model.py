import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn.config import TARGET
from churn.data import encode_target, load_dataset
from churn.features import build_pipeline
from churn.model import (
    best_f1_threshold,
    candidate_models,
    compare_models,
    select_threshold,
    threshold_for_rate,
)


@pytest.fixture(scope="module")
def sample():
    df = load_dataset().sample(n=800, random_state=0).reset_index(drop=True)
    return df.drop(columns=[TARGET]), encode_target(df[TARGET])


def test_candidate_models_are_all_pipelines():
    models = candidate_models()
    assert "majority_baseline" in models
    assert len(models) >= 4
    for pipeline in models.values():
        assert list(pipeline.named_steps) == ["features", "preprocess", "model"]


def test_threshold_for_rate_flags_approximately_that_fraction():
    probabilities = np.linspace(0, 1, 1000)
    threshold = threshold_for_rate(probabilities, 0.30)
    assert (probabilities >= threshold).mean() == pytest.approx(0.30, abs=0.01)


def test_threshold_for_rate_rejects_invalid_rates():
    with pytest.raises(ValueError):
        threshold_for_rate(np.array([0.1, 0.9]), 0.0)


def test_select_threshold_returns_cost_sorted_curve():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.3, 0.6, 0.7])
    curve = select_threshold(y, probabilities)
    assert set(["threshold", "tp", "fp", "fn", "tn", "cost"]).issubset(curve.columns)
    assert curve["cost"].is_monotonic_increasing
    assert curve.iloc[0]["cost"] <= curve.iloc[-1]["cost"]


def test_f1_threshold_never_worse_than_any_grid_point():
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.45, 0.3, 0.55, 0.7, 0.15, 0.6])
    threshold = best_f1_threshold(y, probabilities)
    assert 0.0 < threshold < 1.0


def test_compare_models_ranks_by_pr_auc(sample):
    X, y = sample
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = compare_models(X, y, cv)
    assert len(results) == len(candidate_models())
    assert results["average_precision_mean"].is_monotonic_decreasing
    assert (results["average_precision_mean"] <= 1).all()
    assert results.iloc[-1]["model"] == "majority_baseline"


def test_no_leakage_with_shuffled_labels(sample):
    """A shuffled-label sanity check, not a proof that all leakage is absent."""
    X, y = sample
    rng = np.random.default_rng(42)
    shuffled = rng.permutation(y)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    scores = cross_val_score(
        pipeline, X, shuffled, cv=3, scoring="roc_auc", n_jobs=1
    )
    assert scores.mean() < 0.65
