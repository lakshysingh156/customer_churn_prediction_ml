"""Streamlit demo for the churn model.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.config import METADATA_PATH, METRICS_PATH, MODEL_PATH  # noqa: E402
from churn.predict import (  # noqa: E402
    ChurnScorer, REQUIRED_INPUT_COLUMNS, read_customer_csv, scored_to_csv, spreadsheet_safe_scores,
)

st.set_page_config(page_title="Customer Churn Risk", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1220px; padding-top: 4.5rem; padding-bottom: 3rem;}
    h1 {letter-spacing: -0.045em; font-weight: 750;}
    h2, h3 {letter-spacing: -0.025em;}
    .eyebrow {color: #2563eb; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.14em;}
    .hero-note {color: #64748b; max-width: 760px; margin-bottom: 1.8rem; line-height: 1.7;}
    .st-key-customer_form {background: #ffffff; border-radius: 16px;}
    [data-testid="stMetric"] {background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 1.15rem; box-shadow: 0 3px 12px #0f172a04;}
    [data-testid="stMetricLabel"] {color: #64748b;}
    [data-testid="stMetricValue"] {letter-spacing: -0.035em;}
    [data-baseweb="tab-list"] {gap: 1.8rem; margin-bottom: 1.2rem;}
    [data-baseweb="tab"] {font-weight: 600;}
    [data-testid="stFileUploaderDropzone"] {border: 1px dashed #cbd5e1; border-radius: 14px;}
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button {border-radius: 9px;}
    @media (prefers-reduced-motion: no-preference) {
        button {transition: background-color 0.15s ease, box-shadow 0.15s ease;}
    }
    @media (max-width: 640px) {.block-container {padding-top: 3.5rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def load_metrics(metadata: dict) -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.warning("The metrics report could not be read. Retrain the model to regenerate it.")
        return None
    if (
        metrics.get("generated_at") != metadata.get("trained_at")
        or metrics.get("selected_model") != metadata.get("model")
        or metrics.get("feature_set") != metadata.get("feature_set")
        or metrics.get("model_sha256") != metadata.get("model_sha256")
        or metrics.get("operating_point", {}).get("decision_threshold") != metadata.get("decision_threshold")
    ):
        st.warning("The metrics report does not match the loaded model. Retrain to regenerate matching outputs.")
        return None
    return metrics


def band_style(band: str) -> None:
    message = {
        "Low": st.success,
        "Medium": st.info,
        "High": st.warning,
        "Critical": st.error,
    }[band]
    message(f"**{band} risk**")


def customer_profile() -> dict:
    with st.container(border=True, key="customer_form"):
        left, middle, right = st.columns(3)

        with left:
            st.markdown("#### Account")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0, step=0.05)
            estimate = st.checkbox("Estimate total charges from tenure × monthly", value=True)
            estimated_total = round(tenure * monthly, 2)
            if estimate:
                total = estimated_total
                st.caption(f"Estimated total charges: {total:,.2f}. Enter the actual amount when available.")
            else:
                total = st.number_input("Total charges", min_value=0.0, value=estimated_total, step=1.0)
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])

        with middle:
            st.markdown("#### Customer")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            st.markdown("#### Phone")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multi = st.selectbox(
                "Multiple lines", ["No phone service"] if phone == "No" else ["No", "Yes"],
                disabled=phone == "No",
            )

        with right:
            st.markdown("#### Internet & add-ons")
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            no_internet = internet == "No"
            addon_options = ["No internet service"] if no_internet else ["No", "Yes"]
            security = st.selectbox("Online security", addon_options, disabled=no_internet)
            backup = st.selectbox("Online backup", addon_options, disabled=no_internet)
            protection = st.selectbox("Device protection", addon_options, disabled=no_internet)
            support = st.selectbox("Tech support", addon_options, disabled=no_internet)
            tv = st.selectbox("Streaming TV", addon_options, disabled=no_internet)
            movies = st.selectbox("Streaming movies", addon_options, disabled=no_internet)

        submitted = st.button("Score customer", type="primary")

    record = {
        "gender": gender,
        "SeniorCitizen": int(senior == "Yes"),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    return record if submitted else {}


st.markdown('<div class="eyebrow">CUSTOMER INTELLIGENCE / TELCO PROJECT</div>', unsafe_allow_html=True)
st.title("Customer Churn Risk")
st.markdown(
    '<p class="hero-note">Understand who may leave. Make outreach more focused.<br>'
    'Explore customer risk and model performance on the fictional IBM Telco dataset.</p>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files are missing. Run `python scripts/train.py` to create the pipeline and metadata.")
    st.stop()

try:
    scorer = ChurnScorer()
except json.JSONDecodeError:
    st.error("Model metadata is invalid. Run `python scripts/train.py` to regenerate it.")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()
metrics = load_metrics(scorer.metadata)
single, batch, about = st.tabs(["Single customer", "Batch CSV", "Model card"])

with single:
    st.subheader("Customer profile")
    st.caption("Enter the available account and service details, then score the customer.")
    record = customer_profile()
    if record:
        try:
            result = scorer.score_record(record)
        except ValueError as error:
            st.error(f"Could not score this customer: {error}")
        else:
            probability = float(result["churn_probability"])
            band = str(result["risk_band"])
            st.subheader("Risk assessment")
            with st.container(border=True):
                left, right = st.columns([1, 1])
                with left:
                    st.metric("Churn probability", f"{probability:.1%}")
                    st.progress(min(max(probability, 0.0), 1.0))
                    band_style(band)
                with right:
                    st.markdown(f"**Suggested action**\n\n{result['recommended_action']}")
                    st.markdown(f"**Flag for outreach:** {'Yes' if result['churn_prediction'] else 'No'}")
                    st.caption(f"Decision threshold: {scorer.threshold:.3f}. Suggestions are illustrative, not validated interventions.")
    else:
        st.caption("The risk assessment appears after scoring.")

with batch:
    st.subheader("Score a customer list")
    st.caption("Upload a CSV with the original Telco feature columns to generate a risk score for each row.")
    st.caption("For a demo, upload data/raw/telco_churn.csv. This development dataset is not an independent evaluation.")
    with st.expander("Required columns and input rules"):
        st.code(", ".join(REQUIRED_INPUT_COLUMNS))
        st.markdown(
            "`customerID` is optional and `Churn` is ignored. Use the original Telco category values; "
            "service fields must agree with the phone and internet selections. "
            "See the README for the accepted values."
        )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            frame = read_customer_csv(uploaded)
            scored = scorer.score_frame(frame)
        except ValueError as error:
            st.error(f"Could not score that file: {error}")
        else:
            st.success(f"Scored {len(scored):,} customers.")
            flagged, critical, average = st.columns(3)
            flagged.metric("Flagged for outreach", f"{scored['churn_prediction'].sum():,}")
            critical.metric("Critical risk", f"{scored['risk_band'].eq('Critical').sum():,}")
            average.metric("Mean predicted risk", f"{scored['churn_probability'].mean():.1%}")
            st.caption(f"Preview: first {min(len(scored), 50):,} of {len(scored):,} rows. The download contains all rows. Formula-like IDs are escaped.")
            st.dataframe(spreadsheet_safe_scores(scored).head(50), hide_index=True)
            st.download_button(
                "Download scored CSV",
                scored_to_csv(scored).encode("utf-8"),
                file_name="churn_scores.csv",
                mime="text/csv",
            )

with about:
    st.subheader("Model performance")
    st.caption("Scores below describe the held-out portion of the fictional dataset, not a real-world deployment.")
    if metrics:
        point = metrics["operating_point"]
        test = point["test_metrics"]
        first, second, third, fourth = st.columns(4)
        first.metric("ROC-AUC (holdout)", f"{test['roc_auc']:.3f}")
        second.metric("PR-AUC (holdout)", f"{test['average_precision']:.3f}")
        third.metric("Precision", f"{test['precision']:.3f}")
        fourth.metric("Recall", f"{test['recall']:.3f}")
        st.markdown(
            f"""
**How the decision is made**

- Model: `{metrics['selected_model']}` on `{metrics['feature_set']}` features,
  selected by average precision across {metrics['cross_validation']['folds']}-fold CV.
- Operating point: a training-based cut-off targeting {point['target_contact_rate']:.0%}
  of customers; it caught {test['true_positives']} of {test['n_churners']} churners in the holdout set.
- Risk bands use fixed training-score cut-offs; proportions can change on new data.
- The secondary penalty analysis is illustrative and does not estimate real campaign profit.

**Known limitations**

- Trained on a public fictional dataset; behaviour may differ in a real customer base.
- The holdout has been inspected during development; use independent data for stronger evaluation.
- Accuracy is deliberately not reported as the headline metric — 73% of this
  dataset does not churn, so a do-nothing model scores 73.5%.
"""
        )
        roc_plot = ROOT / "reports" / "figures" / "roc_pr_curves.png"
        confusion_plot = ROOT / "reports" / "figures" / "confusion_matrix.png"
        if roc_plot.exists() and confusion_plot.exists():
            st.markdown("#### Evaluation diagrams")
            left, right = st.columns(2)
            left.image(str(roc_plot), caption="ROC and precision-recall curves")
            right.image(str(confusion_plot), caption="Confusion matrix at the selected threshold")
    else:
        st.info("Run the training script to populate matching model metrics.")
