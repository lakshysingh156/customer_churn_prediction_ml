"""Regenerate the exploratory figures embedded in the README.

    python scripts/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.config import FIGURE_DIR  # noqa: E402
from churn.data import load_dataset  # noqa: E402
from churn.eda import (  # noqa: E402
    apply_style,
    plot_categorical_churn,
    plot_correlation,
    plot_numeric_by_churn,
    plot_target_balance,
    plot_tenure_churn,
)


def main() -> int:
    df = load_dataset()
    apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figures = {
        "eda_target_balance.png": plot_target_balance(df),
        "eda_numeric_by_churn.png": plot_numeric_by_churn(df),
        "eda_categorical_churn.png": plot_categorical_churn(df),
        "eda_correlation.png": plot_correlation(df),
        "eda_tenure_churn.png": plot_tenure_churn(df),
    }
    for name, figure in figures.items():
        figure.tight_layout()
        figure.savefig(FIGURE_DIR / name, bbox_inches="tight")
        plt.close(figure)
        print(f"wrote {FIGURE_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
