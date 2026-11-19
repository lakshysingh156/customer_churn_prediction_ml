from pathlib import Path

import pytest

from churn.config import MODEL_PATH

pytest.importorskip("streamlit")

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="train the model first: python scripts/train.py"
)


def run_app(page="Single customer"):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    if page != "Overview":
        app.radio(key="workspace").set_value(page).run()
    return app


def test_app_boots_without_exception():
    app = run_app("Overview")
    assert not list(app.exception)
    assert app.radio(key="workspace").options == ["Overview", "Single customer", "Batch CSV", "Model card"]
    assert any(metric.label == "ROC-AUC · holdout" for metric in app.metric)
    app.button[0].click().run()
    assert app.radio(key="workspace").value == "Single customer"
    assert not list(app.exception)


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


def test_demo_profiles_produce_different_predictions():
    app = run_app()
    next(widget for widget in app.radio if widget.label == "Choose a starting profile").set_value("New subscriber").run()
    app.button[0].click().run()
    new_risk = next(m.value for m in app.metric if m.label == "Churn probability")
    next(widget for widget in app.radio if widget.label == "Choose a starting profile").set_value("Long-term subscriber").run()
    app.button[0].click().run()
    loyal_risk = next(m.value for m in app.metric if m.label == "Churn probability")
    assert not list(app.exception)
    assert float(new_risk.rstrip("%")) > float(loyal_risk.rstrip("%"))


def test_changing_inputs_clears_the_old_prediction():
    app = run_app()
    app.button[0].click().run()
    assert any(m.label == "Churn probability" for m in app.metric)
    app.slider[0].set_value(24).run()
    assert not any(m.label == "Churn probability" for m in app.metric)


def test_local_batch_demo_and_clearing_it():
    app = run_app("Batch CSV")
    assert any("data/raw/telco_churn.csv" in caption.value for caption in app.caption)
    app.checkbox[0].check().run()
    assert not list(app.exception)
    assert any(m.label == "Customers scored" and m.value == "7,043" for m in app.metric)
    app.checkbox[0].uncheck().run()
    assert not list(app.metric)


def test_model_evaluation_views():
    app = run_app("Model card")
    for view in ["Confusion matrix", "Feature importance", "Calibration", "Cumulative gains", "ROC & precision-recall"]:
        app.selectbox[0].set_value(view).run()
        assert not list(app.exception)
        assert any("independent data" in item.value for item in app.markdown)


def test_navigation_without_local_dataset(monkeypatch, tmp_path):
    import churn.config

    monkeypatch.setattr(churn.config, "RAW_DATA_PATH", tmp_path / "missing.csv")
    app = run_app("Overview")
    assert not list(app.exception)
    app.radio(key="workspace").set_value("Batch CSV").run()
    assert not list(app.exception)
    assert app.checkbox[0].disabled
    assert any("download_data.py" in item.value for item in app.info)
