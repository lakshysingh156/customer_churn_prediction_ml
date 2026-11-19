"""Metrics and report figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from churn.config import REPORT_DIR

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#b44d3b"
COOL = "#3d6b7d"
GRID = "#e4e7eb"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "legend.frameon": False,
    }
)


def compute_metrics(y_true, probabilities, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    predicted = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "n_samples": int(len(y_true)),
        "n_churners": int(y_true.sum()),
    }


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model_comparison(results: pd.DataFrame, metric: str, path: Path, n_splits: int) -> Path:
    ordered = results.sort_values(f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(ordered) + 1.4))
    colors = [ACCENT if i == len(ordered) - 1 else COOL for i in range(len(ordered))]
    ax.barh(
        ordered["model"],
        ordered[f"{metric}_mean"],
        xerr=ordered[f"{metric}_std"],
        color=colors,
        error_kw={"ecolor": MUTED, "elinewidth": 1},
        height=0.55,
    )
    ax.set_xlabel(f"{metric.replace('_', ' ')} ({n_splits}-fold CV, mean ± std)")
    ax.set_xlim(0, 1)
    ax.set_title("Model comparison")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_confusion(y_true, probabilities, threshold: float, path: Path) -> Path:
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    matrix = confusion_matrix(np.asarray(y_true).astype(int), predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(matrix, cmap="BuGn")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{labels[i][j]}\n{matrix[i, j]:,}",
                ha="center",
                va="center",
                fontsize=12,
                color="white" if matrix[i, j] > matrix.max() / 2 else INK,
            )
    ax.set_xticks([0, 1], ["Stayed", "Churned"])
    ax.set_yticks([0, 1], ["Stayed", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix at threshold {threshold:.3f}")
    ax.grid(False)
    return _save(fig, path)


def plot_roc_pr(y_true, probabilities, path: Path) -> Path:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    baseline = y_true.mean()

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    left.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"AUC = {roc_auc_score(y_true, probabilities):.3f}")
    left.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    left.set_xlabel("False positive rate")
    left.set_ylabel("True positive rate")
    left.set_title("ROC curve")
    left.legend(loc="lower right")

    right.plot(recall, precision, color=COOL, linewidth=2,
               label=f"PR-AUC = {average_precision_score(y_true, probabilities):.3f}")
    right.axhline(baseline, color=MUTED, linestyle="--", linewidth=1, label=f"Prevalence = {baseline:.3f}")
    right.set_xlabel("Recall")
    right.set_ylabel("Precision")
    right.set_title("Precision-recall curve")
    right.legend(loc="upper right")
    return _save(fig, path)


def plot_cost_curve(curve: pd.DataFrame, chosen_threshold: float, path: Path) -> Path:
    ordered = curve.sort_values("threshold")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(ordered["threshold"], ordered["cost"], color=INK, linewidth=1.6)
    chosen = curve.iloc[0]
    ax.scatter([chosen_threshold], [chosen["cost"]], color=ACCENT, zorder=5,
               label=f"Chosen ({chosen_threshold:.3f})")
    default_row = ordered.iloc[(ordered["threshold"] - 0.5).abs().argmin()]
    ax.scatter([0.5], [default_row["cost"]], color=COOL, zorder=5, label="Default (0.500)")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost on out-of-fold data")
    ax.set_title("Cost-aware threshold selection")
    ax.legend(loc="upper center")
    return _save(fig, path)


def plot_calibration(y_true, probabilities, path: Path) -> Path:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
    ax.plot(predicted, observed, color=ACCENT, marker="o", linewidth=2, label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title(f"Calibration (Brier = {brier_score_loss(y_true, probabilities):.3f})")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_importance(names, values, path: Path, top_n: int = 15) -> Path:
    series = pd.Series(values, index=names).sort_values().tail(top_n)
    fig, ax = plt.subplots(figsize=(6.8, 0.36 * len(series) + 1.6))
    ax.barh(series.index, series.values, color=COOL, height=0.6)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.set_title("Permutation importance (holdout set)")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def decile_table(y_true, probabilities, n_bins: int = 10) -> pd.DataFrame:
    """Rank customers by score and report what each slice of the ranking delivers."""
    frame = pd.DataFrame(
        {"y": np.asarray(y_true).astype(int), "p": np.asarray(probabilities, dtype=float)}
    ).sort_values("p", ascending=False).reset_index(drop=True)

    frame["decile"] = np.floor(np.arange(len(frame)) * n_bins / len(frame)).astype(int) + 1
    grouped = frame.groupby("decile").agg(
        customers=("y", "size"),
        churners=("y", "sum"),
        mean_score=("p", "mean"),
    )
    overall = frame["y"].mean()
    grouped["churn_rate"] = grouped["churners"] / grouped["customers"]
    grouped["lift"] = grouped["churn_rate"] / overall
    grouped["captured"] = grouped["churners"].cumsum()
    grouped["capture_rate"] = grouped["captured"] / frame["y"].sum()
    grouped["population_share"] = grouped["customers"].cumsum() / len(frame)
    return grouped.reset_index()


def plot_threshold_metrics(y_true, probabilities, chosen_threshold: float, path: Path) -> Path:
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 3)

    precision, recall, f1 = [], [], []
    for threshold in grid:
        predicted = (probabilities >= threshold).astype(int)
        precision.append(precision_score(y_true, predicted, zero_division=0))
        recall.append(recall_score(y_true, predicted, zero_division=0))
        f1.append(f1_score(y_true, predicted, zero_division=0))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(grid, precision, color=COOL, linewidth=1.8, label="Precision")
    ax.plot(grid, recall, color=ACCENT, linewidth=1.8, label="Recall")
    ax.plot(grid, f1, color=INK, linewidth=1.2, linestyle="--", label="F1")
    ax.axvline(chosen_threshold, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate(f"chosen {chosen_threshold:.2f}", (chosen_threshold, 0.04),
                xytext=(6, 0), textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Precision, recall and F1 as the threshold moves")
    ax.legend(loc="center right")
    return _save(fig, path)


def plot_cumulative_gains(y_true, probabilities, path: Path) -> Path:
    table = decile_table(y_true, probabilities)
    x = np.concatenate([[0.0], table["population_share"].to_numpy()])
    y = np.concatenate([[0.0], table["capture_rate"].to_numpy()])

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Random targeting")
    ax.plot(x, y, color=ACCENT, marker="o", markersize=4, linewidth=2, label="Model")
    ax.fill_between(x, y, x, color=ACCENT, alpha=0.08)
    ax.set_xlabel("Share of customers contacted")
    ax.set_ylabel("Share of churners captured")
    ax.set_title("Cumulative gains")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_band_performance(y_true, probabilities, cutoffs, path: Path) -> Path:
    from churn.predict import band_labels

    y_true = np.asarray(y_true).astype(int)
    bands = band_labels(probabilities, cutoffs)
    order = ["Low", "Medium", "High", "Critical"]
    counts = [(bands == band).sum() for band in order]
    rates = [y_true[bands == band].mean() if (bands == band).any() else 0.0 for band in order]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bars = ax.bar(order, rates, color=[COOL, COOL, ACCENT, ACCENT], width=0.55)
    ax.axhline(y_true.mean(), color=MUTED, linestyle="--", linewidth=1)
    for bar, count, rate in zip(bars, counts, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{rate:.1%}\n(n={count})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Observed churn rate")
    ax.set_ylim(0, min(1.0, max(rates) * 1.35))
    ax.set_title("Churn actually observed in each risk band")
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def write_metrics(payload: dict, path: Path | None = None) -> Path:
    path = path or REPORT_DIR / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
