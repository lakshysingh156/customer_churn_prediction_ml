"""Candidate models, cross-validated comparison and cost-aware threshold selection."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict

from churn.config import BusinessCost, N_SPLITS, RANDOM_STATE, TARGET_CONTACT_RATE
from churn.features import build_pipeline

SCORERS = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
}

SELECTION_METRIC = "average_precision"


def make_cv(n_splits: int = N_SPLITS) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def candidate_models() -> dict[str, object]:
    """No class_weight: probabilities stay roughly calibrated and imbalance is
    handled by the decision threshold instead. Weighting rebalances the scores,
    which silently changes what a "0.3 probability" means."""
    return {
        "majority_baseline": build_pipeline(DummyClassifier(strategy="prior")),
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
        "hist_gradient_boosting": build_pipeline(
            HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            )
        ),
    }


def compare_models(X, y, cv=None) -> pd.DataFrame:
    cv = cv or make_cv()
    rows = []
    for name, pipeline in candidate_models().items():
        with warnings.catch_warnings():
            # The majority baseline predicts no positives, so precision is undefined
            # on every fold. Expected, not a problem worth printing.
            warnings.filterwarnings("ignore", message=".*ill-defined.*")
            scores = cross_validate(
                pipeline, X, y, cv=cv, scoring=SCORERS, n_jobs=-1, error_score="raise"
            )
        row = {"model": name}
        for metric in SCORERS:
            row[f"{metric}_mean"] = float(np.mean(scores[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(scores[f"test_{metric}"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"{SELECTION_METRIC}_mean", ascending=False, ignore_index=True)


def oof_probabilities(pipeline, X, y, cv=None) -> np.ndarray:
    cv = cv or make_cv()
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]


def threshold_grid() -> np.ndarray:
    return np.round(np.arange(0.01, 1.00, 0.005), 3)


def threshold_for_rate(probabilities, rate: float = TARGET_CONTACT_RATE) -> float:
    """Choose an approximate target rate; ties can make the flagged share larger."""
    if not 0.0 < rate < 1.0:
        raise ValueError(f"rate must be in (0, 1), got {rate}")
    probabilities = np.asarray(probabilities)
    return float(np.quantile(probabilities, 1.0 - rate))


def select_threshold(y_true, probabilities, cost: BusinessCost = BusinessCost()) -> pd.DataFrame:
    """Sweep decision thresholds and return the full curve, cheapest row first.

    Thresholds must be chosen on out-of-fold probabilities, never on the test set.
    """
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    n_positive = int(y_true.sum())
    n_negative = int(len(y_true) - n_positive)

    records = []
    for threshold in threshold_grid():
        predicted = (probabilities >= threshold).astype(int)
        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        fn = n_positive - tp
        tn = n_negative - fp
        records.append(
            {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                "cost": cost.total(fp, fn),
            }
        )

    curve = pd.DataFrame(records)
    return curve.sort_values(["cost", "threshold"], ascending=[True, False], ignore_index=True)


def best_f1_threshold(y_true, probabilities) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = [(t, f1_score(y_true, (np.asarray(probabilities) >= t).astype(int))) for t in threshold_grid()]
    return float(max(scores, key=lambda pair: pair[1])[0])


def fit_final_model(pipeline, X, y):
    return clone(pipeline).fit(X, y)
