from pathlib import Path

import pytest

from churn.config import MODEL_PATH

pytest.importorskip("streamlit")

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="train the model first: python scripts/train.py"
)


def run_app():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    return app


def test_app_boots_without_exception():
    app = run_app()
    assert not list(app.exception)
    assert [tab.label for tab in app.tabs] == ["Single customer", "Batch CSV", "Model card"]
    assert any("data/raw/telco_churn.csv" in caption.value for caption in app.caption)


def test_submitting_the_form_returns_a_score():
    app = run_app()
    app.button[0].click().run()
    assert not list(app.exception)

    labels = {metric.label: metric.value for metric in app.metric}
    assert "Churn probability" in labels
    assert labels["Churn probability"].endswith("%")


def test_longer_tenure_lowers_predicted_risk():
    app = run_app()
    app.slider[0].set_value(2)
    app.button[0].click().run()
    short = next(m.value for m in app.metric if m.label == "Churn probability")

    app.slider[0].set_value(70)
    app.button[0].click().run()
    long = next(m.value for m in app.metric if m.label == "Churn probability")

    assert float(long.rstrip("%")) < float(short.rstrip("%"))


def test_manual_total_is_available_without_scoring_first():
    app = run_app()
    app.checkbox[0].uncheck().run()
    total = next(widget for widget in app.number_input if widget.label == "Total charges")
    assert not total.disabled
    total.set_value(1500.0).run()
    app.button[0].click().run()
    assert not list(app.exception)
    assert any(metric.label == "Churn probability" for metric in app.metric)


def test_estimate_updates_with_tenure():
    app = run_app()
    assert any("840.00" in caption.value for caption in app.caption)
    app.slider[0].set_value(2).run()
    assert any("140.00" in caption.value for caption in app.caption)


def test_service_controls_follow_parent_selections():
    app = run_app()
    next(widget for widget in app.selectbox if widget.label == "Phone service").set_value("No").run()
    multiple = next(widget for widget in app.selectbox if widget.label == "Multiple lines")
    assert multiple.disabled and multiple.value == "No phone service"
    next(widget for widget in app.selectbox if widget.label == "Internet service").set_value("No").run()
    security = next(widget for widget in app.selectbox if widget.label == "Online security")
    assert security.disabled and security.value == "No internet service"
    app.button[0].click().run()
    assert not list(app.exception)
    assert any(metric.label == "Churn probability" for metric in app.metric)
