"""Exploratory plots, shared by the notebook and ``scripts/eda.py``."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churn.config import TARGET

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#b44d3b"
COOL = "#3d6b7d"
GRID = "#e4e7eb"


def apply_style() -> None:
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


def plot_target_balance(df: pd.DataFrame):
    counts = df[TARGET].value_counts()
    fig, ax = plt.subplots(figsize=(5.2, 1.8))
    ax.barh(counts.index, counts.values, color=[COOL, ACCENT], height=0.55)
    for position, value in enumerate(counts.values):
        ax.text(value + counts.max() * 0.02, position, f"{value:,} ({value / len(df):.1%})",
                va="center", fontsize=9)
    ax.set_xlim(0, counts.max() * 1.32)
    ax.set_title("Class balance — churn is the minority class")
    ax.grid(axis="y", visible=False)
    return fig


def plot_numeric_by_churn(df: pd.DataFrame):
    columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for ax, column in zip(axes, columns):
        for label, colour in [("No", COOL), ("Yes", ACCENT)]:
            values = df.loc[df[TARGET] == label, column]
            ax.hist(values, bins=35, density=True, alpha=0.55, color=colour, label=f"Churn = {label}")
        ax.set_title(column)
        ax.set_yticks([])
    axes[0].legend()
    fig.suptitle("Numeric features by outcome", y=1.04, fontweight="semibold")
    return fig


def plot_categorical_churn(df: pd.DataFrame):
    columns = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    overall = (df[TARGET] == "Yes").mean()
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3))
    for ax, column in zip(axes, columns):
        rates = df.groupby(column)[TARGET].apply(lambda s: (s == "Yes").mean()).sort_values()
        ax.barh(rates.index, rates.values, color=COOL, height=0.6)
        ax.axvline(overall, color=ACCENT, linestyle="--", linewidth=1)
        ax.set_xlim(0, 0.55)
        ax.set_title(column, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Churn rate by category (dashed line = overall {overall:.1%})",
                 y=1.04, fontweight="semibold")
    return fig


def plot_correlation(df: pd.DataFrame):
    numeric = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].copy()
    numeric["churn"] = (df[TARGET] == "Yes").astype(int)
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else INK)
    ax.set_title("Correlation between numeric features")
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def plot_tenure_churn(df: pd.DataFrame):
    """Churn rate by tenure bucket — the single strongest relationship in the data."""
    buckets = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 36, 48, 60, 72],
                     labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
    rates = df.groupby(buckets, observed=True)[TARGET].apply(lambda s: (s == "Yes").mean())
    sizes = df.groupby(buckets, observed=True)[TARGET].size()

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(rates.index.astype(str), rates.values, color=COOL, width=0.6)
    ax.axhline((df[TARGET] == "Yes").mean(), color=ACCENT, linestyle="--", linewidth=1,
               label="Overall rate")
    for bar, size, rate in zip(bars, sizes.values, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{rate:.0%}\nn={size}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(0, min(1.0, rates.max() * 1.35))
    ax.set_title("Churn falls sharply with tenure")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    return fig
