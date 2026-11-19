"""Train, evaluate and persist the churn model.

Protocol:
  1. Hold out 20% of rows before anything else touches them.
  2. Compare four pipelines with 5-fold CV on the training rows only.
  3. Pick the winner by PR-AUC, then set the decision threshold from out-of-fold
     scores so that it flags the riskiest TARGET_CONTACT_RATE of the base.
  4. Refit on all training rows, then score the holdout set exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.config import (  # noqa: E402
    ARTIFACT_DIR,
    DEFAULT_COST,
    METADATA_PATH,
    MODEL_PATH,
    N_SPLITS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORT_DIR,
    TARGET,
    TARGET_CONTACT_RATE,
    TEST_SIZE,
)
from churn.data import churn_rate, encode_target, load_dataset  # noqa: E402
from churn.evaluate import (  # noqa: E402
    compute_metrics,
    decile_table,
    plot_band_performance,
    plot_calibration,
    plot_confusion,
    plot_cost_curve,
    plot_cumulative_gains,
    plot_importance,
    plot_model_comparison,
    plot_roc_pr,
    plot_threshold_metrics,
    write_metrics,
)
from churn.features import build_pipeline  # noqa: E402
from churn.model import (  # noqa: E402
    SELECTION_METRIC,
    best_f1_threshold,
    candidate_models,
    compare_models,
    fit_final_model,
    oof_probabilities,
    select_threshold,
    threshold_for_rate,
)
from churn.predict import REQUIRED_INPUT_COLUMNS, band_labels  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Telco churn model.")
    parser.add_argument("--data", type=Path, default=None, help="Path to the raw CSV.")
    parser.add_argument("--quick", action="store_true", help="Subsample rows and folds; used by CI.")
    parser.add_argument("--no-ablation", action="store_true", help="Skip the engineered-feature ablation.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the model; custom directories also contain reports/.")
    args = parser.parse_args(argv)
    if args.out_dir is None:
        args.out_dir = ARTIFACT_DIR / "quick" if args.quick else ARTIFACT_DIR
    return args


def report_directory(out_dir: Path) -> Path:
    return REPORT_DIR if out_dir.resolve() == ARTIFACT_DIR.resolve() else out_dir / "reports"


def format_table(results: pd.DataFrame) -> str:
    columns = ["model", "average_precision_mean", "average_precision_std", "roc_auc_mean", "roc_auc_std"]
    view = results[columns].copy()
    for column in columns[1:]:
        view[column] = view[column].map(lambda v: f"{v:.3f}")
    view.columns = ["model", "PR-AUC", "± std", "ROC-AUC", "± std"]
    return view.to_string(index=False)


def cost_summary(metrics: dict, cost=DEFAULT_COST) -> dict:
    non_churners = metrics["n_samples"] - metrics["n_churners"]
    model_cost = cost.total(metrics["false_positives"], metrics["false_negatives"])
    no_action = cost.total(0, metrics["n_churners"])
    contact_all = cost.total(non_churners, 0)

    # Model beats blanket contact while fn * cost_fn + fp * cost_fp < non_churners * cost_fp.
    # Solving for the cost ratio gives the break-even point below.
    headroom = non_churners - metrics["false_positives"]
    breakeven = headroom / metrics["false_negatives"] if metrics["false_negatives"] else None

    return {
        "assumption_cost_false_negative": cost.false_negative,
        "assumption_cost_false_positive": cost.false_positive,
        "cost_ratio_fn_over_fp": cost.false_negative / cost.false_positive,
        "model_cost": model_cost,
        "no_action_cost": no_action,
        "contact_everyone_cost": contact_all,
        "saving_vs_best_naive": min(no_action, contact_all) - model_cost,
        "breakeven_ratio_vs_contact_all": breakeven,
    }


def json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = report_directory(args.out_dir)
    figure_dir = report_dir / "figures"

    data_path = args.data if args.data is not None else RAW_DATA_PATH
    dataset_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    df = load_dataset(data_path)
    if args.quick:
        df = df.sample(n=min(2000, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)

    X = df.drop(columns=[TARGET])
    y = encode_target(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    n_splits = 3 if args.quick else N_SPLITS
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    prevalence = churn_rate(df)
    print(f"Rows: {len(df):,} | churn rate: {prevalence:.1%}")
    print(f"Train: {len(X_train):,} | holdout: {len(X_test):,} (stratified, seed {RANDOM_STATE})")
    print(f"\nComparing models with {n_splits}-fold CV on the training rows...")

    results = compare_models(X_train, y_train, cv)
    print(format_table(results))
    if args.quick:
        print("\n[quick mode] Numbers are from a subsample; use them only to check the pipeline.")

    selected_name = str(results.iloc[0]["model"])
    selected_pipeline = candidate_models()[selected_name]
    print(f"\nSelected by {SELECTION_METRIC}: {selected_name}")

    ablation = []
    feature_set = "engineered"
    if not args.no_ablation:
        estimator = clone(selected_pipeline.named_steps["model"])
        variants = {
            "raw_features": build_pipeline(clone(estimator), engineered=False),
            "engineered_features": build_pipeline(clone(estimator), engineered=True),
        }
        for label, pipeline in variants.items():
            scores = cross_validate(
                pipeline, X_train, y_train, cv=cv,
                scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
                n_jobs=-1,
            )
            ablation.append(
                {
                    "feature_set": label,
                    "average_precision_mean": float(scores["test_average_precision"].mean()),
                    "roc_auc_mean": float(scores["test_roc_auc"].mean()),
                }
            )
        winner = max(ablation, key=lambda row: row["average_precision_mean"])
        feature_set = "engineered" if winner["feature_set"] == "engineered_features" else "raw"
        delta = abs(ablation[0]["average_precision_mean"] - ablation[1]["average_precision_mean"])
        print(f"Ablation: {winner['feature_set']} wins by {delta:.3f} PR-AUC")

    final_pipeline = build_pipeline(
        clone(selected_pipeline.named_steps["model"]), engineered=(feature_set == "engineered")
    )
    oof = oof_probabilities(final_pipeline, X_train, y_train, cv)
    threshold = threshold_for_rate(oof, TARGET_CONTACT_RATE)
    f1_threshold = best_f1_threshold(y_train, oof)
    oof_metrics = compute_metrics(y_train, oof, threshold)
    cost_curve = select_threshold(y_train, oof, DEFAULT_COST)
    cost_optimal = cost_curve.iloc[0]
    print(
        f"Operating point: target {TARGET_CONTACT_RATE:.0%} of training customers -> "
        f"cut-off {threshold:.3f} (out-of-fold precision {oof_metrics['precision']:.2f}, "
        f"recall {oof_metrics['recall']:.2f})"
    )
    print(
        f"  The illustrative penalty-optimal cut-off is {cost_optimal['threshold']:.3f}; "
        f"it flags {(oof >= cost_optimal['threshold']).mean():.0%} of training customers."
    )

    final = fit_final_model(final_pipeline, X_train, y_train)
    positive_index = int(np.flatnonzero(np.asarray(final.classes_) == 1)[0])
    test_probabilities = final.predict_proba(X_test)[:, positive_index]

    test_metrics = compute_metrics(y_test, test_probabilities, threshold)
    metrics_at_half = compute_metrics(y_test, test_probabilities, 0.5)
    metrics_at_f1 = compute_metrics(y_test, test_probabilities, f1_threshold)

    print("\nHoldout performance (scored once, at the operating point)")
    print(f"  ROC-AUC      {test_metrics['roc_auc']:.3f}")
    print(f"  PR-AUC       {test_metrics['average_precision']:.3f}   (prevalence {np.mean(y_test):.3f})")
    print(f"  Precision    {test_metrics['precision']:.3f}   -> {test_metrics['precision'] / np.mean(y_test):.1f}x lift over random")
    print(f"  Recall       {test_metrics['recall']:.3f}   -> caught {test_metrics['true_positives']} of {test_metrics['n_churners']} churners")
    print(f"  Contacts     {test_metrics['true_positives'] + test_metrics['false_positives']:,} "
          f"({(test_metrics['true_positives'] + test_metrics['false_positives']) / test_metrics['n_samples']:.0%} of customers)")
    print(f"  Accuracy     {test_metrics['accuracy']:.3f}  (reference only; imbalanced classes)")

    cost = cost_summary(test_metrics)
    print(f"\nIllustrative misclassification penalties (not campaign profit): "
          f"{DEFAULT_COST.false_negative:.0f} per false negative, "
          f"{DEFAULT_COST.false_positive:.0f} per false positive")
    print(f"  Model             {cost['model_cost']:,.0f}")
    print(f"  Predict no churn  {cost['no_action_cost']:,.0f}")
    print(f"  Predict all churn {cost['contact_everyone_cost']:,.0f}")

    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(results, SELECTION_METRIC, figure_dir / "model_comparison.png", n_splits)
    plot_confusion(y_test, test_probabilities, threshold, figure_dir / "confusion_matrix.png")
    plot_roc_pr(y_test, test_probabilities, figure_dir / "roc_pr_curves.png")
    plot_cost_curve(cost_curve, float(cost_optimal["threshold"]), figure_dir / "threshold_cost_curve.png")
    plot_calibration(y_test, test_probabilities, figure_dir / "calibration.png")

    importance = permutation_importance(
        final, X_test, y_test, scoring="average_precision", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance_frame = (
        pd.DataFrame({"feature": X_test.columns, "importance": importance.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    plot_importance(importance_frame["feature"].to_numpy(), importance_frame["importance"].to_numpy(),
                    figure_dir / "permutation_importance.png")
    print("\nTop features (permutation importance, PR-AUC drop)")
    print(importance_frame.head(8).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # Bands are population quantiles with the decision threshold spliced in, so
    # "High" and "Critical" are exactly the rows the team is asked to action.
    band_cutoffs = sorted(
        [
            float(np.quantile(oof, 0.50)),
            threshold,
            float(np.quantile(oof, 0.90)),
        ]
    )

    deciles = decile_table(y_test, test_probabilities)
    deciles.to_csv(report_dir / "decile_lift.csv", index=False, float_format="%.4f")
    plot_threshold_metrics(y_test, test_probabilities, threshold, figure_dir / "threshold_metrics.png")
    plot_cumulative_gains(y_test, test_probabilities, figure_dir / "cumulative_gains.png")
    plot_band_performance(y_test, test_probabilities, band_cutoffs, figure_dir / "band_performance.png")

    bands = band_labels(test_probabilities, band_cutoffs)
    band_performance = []
    for band in ["Low", "Medium", "High", "Critical"]:
        mask = bands == band
        band_performance.append(
            {
                "band": band,
                "customers": int(mask.sum()),
                "churners": int(y_test[mask].sum()),
                "observed_churn_rate": float(y_test[mask].mean()) if mask.any() else 0.0,
            }
        )
    print("\nRisk band performance (holdout)")
    for row in band_performance:
        print(f"  {row['band']:<9} n={row['customers']:>4}  churn {row['observed_churn_rate']:.1%}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / MODEL_PATH.name
    joblib.dump(final, model_path)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    metadata = {
        "model": selected_name,
        "model_sha256": model_hash,
        "dataset_sha256": dataset_hash,
        "feature_set": feature_set,
        "decision_threshold": threshold,
        "band_cutoffs": band_cutoffs,
        "target_contact_rate": TARGET_CONTACT_RATE,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": int(len(X_train)),
        "required_input_columns": REQUIRED_INPUT_COLUMNS,
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    dump(metadata, args.out_dir / METADATA_PATH.name)

    payload = {
        "generated_at": metadata["trained_at"],
        "model_sha256": model_hash,
        "quick_mode": bool(args.quick),
        "dataset": {"rows": int(len(df)), "churn_rate": prevalence, "raw_features": int(X.shape[1]),
                    "sha256": dataset_hash},
        "split": {
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_test)),
            "holdout_churn_rate": float(np.mean(y_test)),
            "strategy": "stratified 80/20",
            "random_state": RANDOM_STATE,
        },
        "cross_validation": {
            "folds": n_splits,
            "selection_metric": SELECTION_METRIC,
            "results": results.to_dict(orient="records"),
        },
        "selected_model": selected_name,
        "feature_set": feature_set,
        "operating_point": {
            "strategy": "flag the riskiest TARGET_CONTACT_RATE of the base",
            "target_contact_rate": TARGET_CONTACT_RATE,
            "decision_threshold": threshold,
            "band_cutoffs": band_cutoffs,
            "oof_metrics": oof_metrics,
            "test_metrics": test_metrics,
            "lift_over_random": test_metrics["precision"] / float(np.mean(y_test)),
        },
        "reference_operating_points": {
            "test_at_threshold_0_5": metrics_at_half,
            "test_at_f1_optimal": metrics_at_f1,
            "f1_optimal_threshold": f1_threshold,
        },
        "cost_analysis": {
            **cost,
            "cost_optimal_threshold": float(cost_optimal["threshold"]),
            "cost_optimal_flagged_fraction": float(
                (oof >= cost_optimal["threshold"]).mean()
            ),
            "curve": cost_curve.sort_values("threshold").iloc[::5].round(4).to_dict(orient="records"),
        },
        "feature_ablation": ablation,
        "decile_lift": deciles.round(4).to_dict(orient="records"),
        "band_performance": band_performance,
        "permutation_importance": importance_frame.round(6).to_dict(orient="records"),
        "environment": metadata["versions"],
    }
    metrics_path = write_metrics(payload, report_dir / "metrics.json")
    display_model_path = model_path.relative_to(ROOT) if model_path.is_relative_to(ROOT) else model_path
    display_metrics_path = metrics_path.relative_to(ROOT) if metrics_path.is_relative_to(ROOT) else metrics_path
    print(f"\nSaved model   -> {display_model_path}")
    print(f"Saved metrics -> {display_metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
