import numpy as np
import pandas as pd
import pytest

from churn.evaluate import decile_table, plot_model_comparison
from churn.predict import band_labels


def test_decile_table_preserves_all_rows_and_churners():
    y = np.array([0, 1] * 51)
    probabilities = np.linspace(0.01, 0.99, len(y))
    table = decile_table(y, probabilities)
    assert table["customers"].sum() == len(y)
    assert table["churners"].sum() == y.sum()
    assert table.iloc[-1]["capture_rate"] == pytest.approx(1.0)
    assert table.iloc[-1]["population_share"] == pytest.approx(1.0)


def test_decile_table_ranks_high_scores_first():
    y = np.array([0] * 90 + [1] * 10)
    probabilities = np.arange(100) / 100
    table = decile_table(y, probabilities)
    assert table.iloc[0]["churners"] == 10
    assert table.iloc[0]["lift"] == pytest.approx(10.0)
    assert table.iloc[0]["capture_rate"] == pytest.approx(1.0)


def test_model_comparison_labels_actual_cv_fold_count(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt

    def check_figure(fig, path):
        assert "3-fold CV" in fig.axes[0].get_xlabel()
        plt.close(fig)
        return path

    monkeypatch.setattr("churn.evaluate._save", check_figure)
    results = pd.DataFrame({"model": ["logistic"], "average_precision_mean": [0.5],
                            "average_precision_std": [0.1]})
    plot_model_comparison(results, "average_precision", tmp_path / "comparison.png", 3)


def test_band_labels_share_the_inference_boundary_rule():
    cutoffs = [0.2, 0.4, 0.6]
    bands = band_labels([0.1, 0.2, 0.399, 0.4, 0.599, 0.6], cutoffs)
    assert bands.tolist() == ["Low", "Medium", "Medium", "High", "High", "Critical"]
