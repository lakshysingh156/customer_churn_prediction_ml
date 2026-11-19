from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.config import METADATA_PATH, METRICS_PATH, MODEL_PATH, RAW_DATA_PATH
from churn.predict import (
    ChurnScorer, REQUIRED_INPUT_COLUMNS, read_customer_csv, scored_to_csv, spreadsheet_safe_scores,
)

st.set_page_config(page_title="Churn Studio | Customer Intelligence", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1420px; padding: 3.6rem 2.4rem 2rem;}
h1,h2,h3 {letter-spacing: -.035em;}
h1 {font-size: 2.15rem !important; padding-top: 0 !important;}
h3 {font-size: 1.2rem !important;}
[data-testid="stSidebar"] {background: #111e35;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {color: #c8d3e5;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {color: #c8d3e5; padding: .7rem .8rem; border-radius: 9px;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background: #263d65; color: white;}
.brand {display:flex; align-items:center; gap:12px; color:white; margin:0 0 32px;}
.brand-mark {display:grid; place-items:center; width:40px; height:40px; border-radius:12px; background:#4c78ff; font-size:24px; font-weight:800;}
.brand strong {font-size:21px; letter-spacing:-.7px;}
.brand small {display:block; color:#8ca4c7; font-size:10px; letter-spacing:1.4px; margin-top:3px;}
.side-label,.eyebrow {font-size:10px; font-weight:700; letter-spacing:1.7px; color:#617795; margin:16px 0 10px;}
.side-note {border:1px solid #30435e; border-radius:12px; padding:16px; margin-top:32px; color:#c9d7ec; font-size:13px; line-height:1.8;}
.side-note strong {color:#fff;}
.topline {display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:18px; color:#70809a; font-size:11px; letter-spacing:1.2px;}
.pill {display:inline-block; border-radius:20px; background:#e6f5ef; color:#18745d; padding:5px 10px; font-size:10px; letter-spacing:.4px; font-weight:700;}
.hero {background:linear-gradient(115deg,#152742,#244982); color:#fff; padding:27px 30px; border-radius:17px; display:flex; justify-content:space-between; align-items:center; gap:24px; margin:2px 0 20px;}
.hero h2 {color:white; font-size:28px; line-height:1.2; padding:8px 0 !important; margin:0;}
.hero p {color:#c2d2ec; font-size:13px; margin:5px 0 0; max-width:620px; line-height:1.7;}
.hero .eyebrow {color:#8eafff; margin:0;}
.hero-stat {text-align:center; flex-shrink:0; border-left:1px solid #52709e; padding-left:28px;}
.hero-stat strong {display:block; font-size:44px; letter-spacing:-2px;}
.hero-stat span {color:#c2d2ec; font-size:11px;}
[data-testid="stMetric"] {background:white; border:1px solid #e2e8f0; border-radius:13px; padding:15px 18px; border-top:3px solid #527cf2;}
[data-testid="stMetricLabel"] {color:#64748b; font-size:12px;}
[data-testid="stMetricValue"] {letter-spacing:-1px; font-size:30px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:14px;}
.st-key-customer_form,.st-key-outcome_panel,.st-key-comparison_panel,.st-key-risk_panel,.st-key-upload_panel {background:white; border-radius:14px;}
[data-baseweb="select"] > div,[data-testid="stNumberInput"] input {background:#f5f7fb;}
[data-testid="stFileUploaderDropzone"] {border:1px dashed #aabdda; border-radius:12px;}
[data-testid="stButton"] button,[data-testid="stDownloadButton"] button {border-radius:8px; font-weight:600;}
.comparison-row {margin:18px 0;}
.comparison-label {display:flex; justify-content:space-between; gap:10px; font-size:12px; color:#354763; margin-bottom:8px;}
.track {height:8px; background:#eef2f8; border-radius:8px; overflow:hidden;}
.track > div {height:100%; background:#527cf2; border-radius:8px;}
.outcome-grid {display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:14px 0;}
.outcome {border-radius:10px; padding:16px; background:#f0f5fd; color:#304967;}
.outcome.good {background:#edf8f3; color:#246b53;}
.outcome.warn {background:#fff4ec; color:#95613c;}
.outcome strong {display:block; font-size:27px; letter-spacing:-1px;}
.outcome span {font-size:11px;}
.workflow {display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:16px 0;}
.workflow > div {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:17px;}
.workflow b {font-size:11px; color:#527cf2; margin-right:8px;}
.workflow strong {font-size:13px; color:#253851;}
.workflow p {font-size:12px; color:#748298; line-height:1.6; margin:8px 0 0;}
.empty-state {text-align:center; padding:32px 16px; color:#708099;}
.empty-icon {display:inline-grid; place-items:center; width:56px; height:56px; background:#edf2ff; color:#527cf2; border-radius:16px; margin-bottom:14px; font-size:23px; font-weight:700;}
.empty-state h3 {color:#263952; margin:0;}
.empty-state p {font-size:13px; line-height:1.7;}
.risk-scale {position:relative; height:12px; border-radius:8px; background:linear-gradient(90deg,#66b6a0,#e6c964,#e99e62,#d86573); margin:22px 0 8px;}
.risk-marker {position:absolute; height:22px; width:4px; background:#182c49; top:-5px; border:1px solid white; border-radius:2px; transform:translateX(-50%);}
.scale-labels {display:flex; justify-content:space-between; font-size:11px; color:#7b8798; margin-bottom:22px;}
.section-note {font-size:12px; color:#77869a; line-height:1.7;}
@media (prefers-reduced-motion:no-preference) {button {transition:background-color .15s ease,box-shadow .15s ease;}}
@media (max-width:760px) {
    .block-container {padding:3.7rem 1rem 1.5rem;}
    .hero {padding:22px; flex-direction:column; align-items:flex-start;}
    .hero-stat {border-left:0; padding-left:0; text-align:left;}
    .hero-stat strong {font-size:30px;}
    .workflow {grid-template-columns:1fr;}
    .topline {flex-wrap:wrap;}
}
</style>
""", unsafe_allow_html=True)


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


def open_page(page: str) -> None:
    st.session_state.workspace = page


def performance_metrics(metrics: dict) -> None:
    test = metrics["operating_point"]["test_metrics"]
    for column, label, value, help_text in zip(st.columns(4),
        ["ROC-AUC · holdout", "Average precision · holdout", "Churners captured", "Precision lift"],
        [f"{test['roc_auc']:.3f}", f"{test['average_precision']:.3f}", f"{test['recall']:.1%}", f"{metrics['operating_point']['lift_over_random']:.2f}×"],
        ["Measures ranking across thresholds; chance is 0.5.", "Summary of the precision-recall curve.", "Recall at the training-selected threshold.", "Holdout precision divided by holdout churn prevalence, not campaign uplift."],
    ):
        column.metric(label, value, help=help_text)


def overview(metrics: dict | None) -> None:
    if not metrics:
        st.info("Matching evaluation metrics are unavailable. Customer scoring is still available in the sidebar.")
        return
    test = metrics["operating_point"]["test_metrics"]
    st.markdown(f"""<div class="hero"><div><div class="eyebrow">FROM PREDICTION TO PRIORITY</div>
<h2>Understand risk.<br>Focus the next conversation.</h2>
<p>A complete ML workflow: compare models, explore individual risk and prioritise a customer list. Built on the fictional IBM Telco dataset.</p></div>
<div class="hero-stat"><strong>{metrics['dataset']['rows']:,}</strong><span>customers in the development dataset</span></div></div>""", unsafe_allow_html=True)
    performance_metrics(metrics)
    st.write("")
    left, right = st.columns([1.15, 1], gap="large")
    with left, st.container(border=True, key="comparison_panel"):
        st.subheader("Model comparison")
        st.caption(f"{metrics['cross_validation']['folds']}-fold training CV · average precision, mean ± std")
        for item in metrics["cross_validation"]["results"]:
            name = html.escape(item["model"].replace("_", " ").title())
            score = item["average_precision_mean"]
            spread = item["average_precision_std"]
            st.markdown(f'<div class="comparison-row"><div class="comparison-label"><span>{name}</span><strong>{score:.3f} ± {spread:.3f}</strong></div><div class="track"><div style="width:{score * 100:.2f}%"></div></div></div>', unsafe_allow_html=True)
        st.caption("Candidates used engineered features; the selected model's ablation retained raw features. The AP difference was negligible.")
    with right, st.container(border=True, key="outcome_panel"):
        st.subheader("What happened on the holdout?")
        st.caption(f"{test['n_samples']:,} customers · threshold {scorer.threshold:.3f}")
        st.markdown(f"""<div class="outcome-grid">
<div class="outcome good"><strong>{test['true_positives']}</strong><span>Churners correctly flagged · TP</span></div>
<div class="outcome good"><strong>{test['true_negatives']}</strong><span>Stayers correctly unflagged · TN</span></div>
<div class="outcome warn"><strong>{test['false_positives']}</strong><span>Stayers incorrectly flagged · FP</span></div>
<div class="outcome warn"><strong>{test['false_negatives']}</strong><span>Churners missed · FN</span></div></div>""", unsafe_allow_html=True)
        st.caption("The holdout was inspected during development. These are benchmark results, not an independent production validation.")
    st.markdown("""<div class="workflow">
<div><b>01</b><strong>Explore a customer</strong><p>Start with a sample profile, adjust inputs and inspect the model's prediction.</p></div>
<div><b>02</b><strong>Score a customer list</strong><p>Upload a CSV, inspect risk bands and export the scored customer list.</p></div>
<div><b>03</b><strong>Explain the evidence</strong><p>Review curves, feature importance and the limits of the evaluation.</p></div>
</div>""", unsafe_allow_html=True)
    first, second, third = st.columns(3)
    first.button("Try a customer", on_click=open_page, args=("Single customer",), width="stretch", type="primary")
    second.button("Open batch scoring", on_click=open_page, args=("Batch CSV",), width="stretch")
    third.button("Explore model evidence", on_click=open_page, args=("Model card",), width="stretch")


def customer_profile(preset: str) -> dict:
    long_term = preset == "Long-term subscriber"
    new = preset == "New subscriber"
    def key(name: str) -> str:
        return f"{preset}:{name}"
    with st.container(border=True, key="customer_form"):
        st.subheader("Account details")
        left, right = st.columns(2)
        with left:
            tenure = st.slider("Tenure (months)", 0, 72, 60 if long_term else 2 if new else 12, key=key("tenure"))
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=2 if long_term else 0, key=key("contract"))
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=2 if long_term else 0, key=key("payment"))
        with right:
            monthly = st.number_input("Monthly charges", 0.0, 200.0, 65.0 if long_term else 95.0 if new else 70.0, step=0.05, key=key("monthly"))
            paperless = st.selectbox("Paperless billing", ["Yes", "No"], key=key("paperless"))
            estimate = st.checkbox("Estimate total charges from tenure × monthly", value=True, key=key("estimate"))
            estimated_total = round(tenure * monthly, 2)
            if estimate:
                total = estimated_total
                st.caption(f"Estimated total charges: {total:,.2f}. Enter the actual amount when available.")
            else:
                total = st.number_input("Total charges", min_value=0.0, value=estimated_total, step=1.0, key=key("total"))
        with st.expander("Customer, phone & internet details", expanded=False):
            left, right = st.columns(2)
            with left:
                gender = st.selectbox("Gender", ["Female", "Male"], key=key("gender"))
                senior = st.selectbox("Senior citizen", ["No", "Yes"], key=key("senior"))
                partner = st.selectbox("Partner", ["No", "Yes"], key=key("partner"))
                dependents = st.selectbox("Dependents", ["No", "Yes"], key=key("dependents"))
                phone = st.selectbox("Phone service", ["Yes", "No"], key=key("phone"))
                multi = st.selectbox("Multiple lines", ["No phone service"] if phone == "No" else ["No", "Yes"], disabled=phone == "No", key=key(f"multi:{phone}"))
            with right:
                internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], index=1 if new else 0, key=key("internet"))
                no_internet = internet == "No"
                addons = {}
                for label, column in [("Online security", "OnlineSecurity"), ("Online backup", "OnlineBackup"), ("Device protection", "DeviceProtection"), ("Tech support", "TechSupport"), ("Streaming TV", "StreamingTV"), ("Streaming movies", "StreamingMovies")]:
                    addons[column] = st.selectbox(label, ["No internet service"] if no_internet else ["No", "Yes"], index=1 if long_term and not no_internet else 0, disabled=no_internet, key=key(f"{column}:{no_internet}"))
        submitted = st.button("Score customer", type="primary", width="stretch")
    return {"gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, **addons, "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total} if submitted else {}


def single_customer() -> None:
    preset = st.radio("Choose a starting profile", ["Custom profile", "New subscriber", "Long-term subscriber"], horizontal=True)
    st.caption("Illustrative profiles, not real customers. You can edit every input before scoring.")
    form, assessment = st.columns([1.65, 1], gap="large")
    with form:
        record = customer_profile(preset)
    with assessment, st.container(border=True, key="risk_panel"):
        st.subheader("Risk assessment")
        if not record:
            st.markdown('<div class="empty-state"><div class="empty-icon">%</div><h3>A clearer view of risk</h3><p>Choose a sample profile or enter account details, then select <strong>Score customer</strong>.<br>No prediction has been made yet.</p></div>', unsafe_allow_html=True)
            st.caption(f"Outreach threshold · {scorer.threshold:.1%}")
            return
        try:
            result = scorer.score_record(record)
        except ValueError as error:
            st.error(f"Could not score this customer: {error}")
            return
        probability = float(result["churn_probability"])
        band = str(result["risk_band"])
        st.metric("Churn probability", f"{probability:.1%}")
        st.markdown(f'<div class="risk-scale"><div class="risk-marker" style="left:{probability * 100:.2f}%"></div></div><div class="scale-labels"><span>0% · lower risk</span><span>100% · higher risk</span></div>', unsafe_allow_html=True)
        {"Low": st.success, "Medium": st.info, "High": st.warning, "Critical": st.error}[band](f"**{band} risk**")
        st.markdown(f"**Suggested action**\n\n{result['recommended_action']}")
        st.markdown(f"**Flag for outreach:** {'Yes' if result['churn_prediction'] else 'No'}")
        st.caption(f"Decision threshold: {scorer.threshold:.3f}. Suggestions are illustrative, not validated interventions.")
        st.divider()
        st.caption(f"Profile scored: {record['tenure']} months · {record['Contract']} · {record['InternetService']}")


def batch_scoring() -> None:
    left, right = st.columns([1.65, 1], gap="large")
    with left, st.container(border=True, key="upload_panel"):
        st.subheader("Bring your customer list")
        st.caption("Upload the original Telco feature columns. Labels are not needed for scoring.")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        use_demo = st.checkbox("Use the local demo dataset", disabled=not RAW_DATA_PATH.exists())
        st.caption("For a demo, upload data/raw/telco_churn.csv. This development dataset is not an independent evaluation.")
        if not RAW_DATA_PATH.exists():
            st.info("To enable the local demo, run `python scripts/download_data.py`.")
    with right:
        st.markdown("""<div class="workflow" style="grid-template-columns:1fr;margin-top:0"><div><b>CSV → SCORES</b><strong>What you get</strong><p>Churn probability for every row<br>Risk band and outreach flag<br>Illustrative suggested action<br>A downloadable scored CSV</p></div></div>""", unsafe_allow_html=True)
        with st.expander("Required columns and input rules"):
            st.code(", ".join(REQUIRED_INPUT_COLUMNS))
            st.write("customerID is optional; Churn is ignored. Category values must match the original Telco schema, and phone/internet selections must be consistent.")
    source = uploaded if uploaded is not None else RAW_DATA_PATH if use_demo else None
    if source is None:
        st.markdown('<div class="empty-state"><div class="empty-icon">CSV</div><h3>Your customer intelligence starts here</h3><p>Upload a file above, or use the local demo dataset to explore the complete workflow.</p></div>', unsafe_allow_html=True)
        return
    if uploaded is not None and use_demo:
        st.caption("Using your uploaded file; it takes priority over the local demo.")
    try:
        scored = scorer.score_frame(read_customer_csv(source))
    except ValueError as error:
        st.error(f"Could not score that file: {error}")
        return
    st.success(f"Scored {len(scored):,} customers.")
    total, flagged, critical, average = st.columns(4)
    total.metric("Customers scored", f"{len(scored):,}")
    flagged.metric("Flagged for outreach", f"{scored['churn_prediction'].sum():,}")
    critical.metric("Critical risk", f"{scored['risk_band'].eq('Critical').sum():,}")
    average.metric("Mean predicted risk", f"{scored['churn_probability'].mean():.1%}")
    chart, table = st.columns([1, 2], gap="large")
    with chart:
        st.subheader("Risk distribution")
        counts = scored["risk_band"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
        st.bar_chart(counts.rename("Customers"), color="#527cf2", height=270, sort=False)
    with table:
        st.subheader("Customer scores")
        st.caption(f"Preview: first {min(len(scored), 50):,} of {len(scored):,} rows. The download contains all rows. Formula-like IDs are escaped.")
        st.dataframe(spreadsheet_safe_scores(scored).head(50), hide_index=True, height=270,
                     column_config={"churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="%.3f")})
        st.download_button("Download scored CSV", scored_to_csv(scored).encode("utf-8"), file_name="churn_scores.csv", mime="text/csv", type="primary")


def model_card(metrics: dict | None) -> None:
    if not metrics:
        st.info("Run the training script to populate matching model metrics.")
        return
    performance_metrics(metrics)
    point = metrics["operating_point"]
    st.caption(f"Saved model: {metrics['selected_model'].replace('_', ' ')} · {metrics['feature_set']} features · outreach threshold {scorer.threshold:.3f} · precision {point['test_metrics']['precision']:.1%}")
    chart = st.selectbox("Evaluation view", ["ROC & precision-recall", "Confusion matrix", "Feature importance", "Calibration", "Cumulative gains"])
    plots = {
        "ROC & precision-recall": ("roc_pr_curves.png", "ROC measures ranking across thresholds. Precision-recall focuses on the churn class; AP summarises this curve."),
        "Confusion matrix": ("confusion_matrix.png", "Rows are actual outcomes; columns are predictions at the training-selected threshold."),
        "Feature importance": ("permutation_importance.png", "Permutation importance measures model reliance on a feature, not a causal effect."),
        "Calibration": ("calibration.png", "Calibration compares predicted probabilities with observed frequencies on the holdout."),
        "Cumulative gains": ("cumulative_gains.png", "Gains show how many actual churners appear in the highest-scoring portion of the holdout."),
    }
    filename, explanation = plots[chart]
    path = ROOT / "reports" / "figures" / filename
    if path.exists():
        st.image(str(path), width=min(960, 650 if chart == "Confusion matrix" else 960))
        st.caption(explanation)
    with st.expander("Training workflow and decision policy"):
        st.markdown(f"""
- Stratified 80/20 split; preprocessing is fitted inside each training fold.
- Model comparison and feature ablation use {metrics['cross_validation']['folds']}-fold training CV.
- The training-based threshold targets approximately {point['target_contact_rate']:.0%} outreach; ties and new populations change the share.
- Captured {point['test_metrics']['true_positives']} of {point['test_metrics']['n_churners']} churners in the holdout.
- Fixed risk-band cut-offs and suggested actions are demonstration aids, not validated interventions.
""")
    with st.expander("Known limitations and responsible use", expanded=True):
        st.write("Fictional dataset; behaviour may differ in a real customer base. The holdout and full-data EDA were inspected during development, so use independent data for stronger evaluation. Model/feature selection affects training estimates. Demographic fields require fairness review before real use.")
        st.caption("A do-nothing model already reaches about 73.5% accuracy. Risk is not treatment effect; illustrative penalties are not measured campaign profits.")


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

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">C</div><div><strong>Churn Studio</strong><small>CUSTOMER INTELLIGENCE</small></div></div><div class="side-label">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio("Workspace", ["Overview", "Single customer", "Batch CSV", "Model card"], key="workspace", label_visibility="collapsed")
    st.markdown(f'<div class="side-note"><strong>Model workspace</strong><br>{html.escape(scorer.metadata["model"].replace("_", " ").title())}<br>Training rows · {scorer.metadata["training_rows"]:,}<br>Decision threshold · {scorer.threshold:.3f}</div><div class="side-note"><strong>Built to explain, not just predict.</strong><br>Fictional IBM Telco data.<br>Portfolio & classroom demonstration.</div>', unsafe_allow_html=True)

st.markdown('<div class="topline"><span>ANALYTICS / CUSTOMER RETENTION</span><span class="pill">SAVED MODEL · READY TO SCORE</span></div>', unsafe_allow_html=True)
titles = {"Overview": ("Retention overview", "The evidence behind the predictions, all in one workspace."),
          "Single customer": ("Customer simulator", "Explore how account and service details shape a prediction."),
          "Batch CSV": ("Batch intelligence", "Turn a customer list into a clear, scored view of risk."),
          "Model card": ("Model evidence", "Understand performance, trade-offs and the limits of the model.")}
st.title(titles[page][0])
st.caption(titles[page][1])
if page == "Overview":
    overview(metrics)
elif page == "Single customer":
    single_customer()
elif page == "Batch CSV":
    batch_scoring()
else:
    model_card(metrics)
st.caption("CHURN STUDIO / Research demonstration · Predictions support exploration, not automated customer decisions.")
