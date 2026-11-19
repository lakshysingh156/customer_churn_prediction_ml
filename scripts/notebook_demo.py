# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = None
RANDOM_STATE = 42
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

if CSV_PATH is None:
    roots = [Path.cwd(), Path.cwd().parent]
    if "__file__" in globals():
        roots.insert(0, Path(__file__).resolve().parents[1])
    candidates = [root / "data/raw/telco_churn.csv" for root in roots]
    candidates.append(Path.cwd() / "telco_churn.csv")
    CSV_PATH = next((path for path in candidates if path.is_file()), None)
if CSV_PATH is None:
    raise FileNotFoundError("Set CSV_PATH to the full path of data/raw/telco_churn.csv, then rerun.")

raw = pd.read_csv(CSV_PATH, dtype={"customerID": "string"}, keep_default_na=False)
print(f"Dataset: {len(raw):,} rows, {len(raw.columns)} columns")
display(raw.head())

# %%
df = raw.copy()
identifiers = df["customerID"].str.strip()
if identifiers.eq("").any() or identifiers.duplicated().any():
    raise ValueError("The training CSV must have non-empty, unique customer IDs.")
if not df["Churn"].isin(["Yes", "No"]).all():
    raise ValueError("Churn must contain only Yes and No.")
blank_total = df["TotalCharges"].astype(str).str.strip().eq("")
if (blank_total & df["tenure"].ne(0)).any():
    raise ValueError("Blank TotalCharges is allowed only for zero-tenure customers.")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].mask(blank_total, "0"), errors="raise")
print(f"Blank TotalCharges replaced with zero: {blank_total.sum()}")
print(f"Churn rate: {df['Churn'].eq('Yes').mean():.1%}")
display(df.describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
df["Churn"].value_counts().reindex(["No", "Yes"]).plot.bar(
    ax=axes[0], color=["#2563eb", "#ef6c52"], rot=0, title="Class balance")
axes[0].set_ylabel("Customers")
df.assign(churn=df["Churn"].eq("Yes")).groupby("Contract")["churn"].mean().plot.bar(
    ax=axes[1], color="#2563eb", rot=15, title="Churn rate by contract")
axes[1].set_ylabel("Churn proportion")
plt.tight_layout()
plt.show()

# %%
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
assert set(identifiers.loc[X_train.index]).isdisjoint(identifiers.loc[X_test.index])
numeric = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical = [column for column in X.columns if column not in numeric]
preprocessing = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
])
model = Pipeline([
    ("preprocessing", preprocessing),
    ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_validate(model, X_train, y_train, cv=cv,
                           scoring={"AP": "average_precision", "ROC_AUC": "roc_auc"})
display(pd.DataFrame({
    "Training CV metric": ["Average precision", "ROC-AUC"],
    "Mean": [cv_scores["test_AP"].mean(), cv_scores["test_ROC_AUC"].mean()],
    "Std": [cv_scores["test_AP"].std(), cv_scores["test_ROC_AUC"].std()],
}).round(4))
print(f"Train: {len(X_train):,}; holdout: {len(X_test):,}. Imputation, scaling and encoding fit within each fold.")

# %%
oof_probability = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
threshold = float(np.quantile(oof_probability, 0.70))
model.fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]
prediction = (probability >= threshold).astype(int)
baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
baseline_probability = baseline.predict_proba(X_test)[:, 1]
baseline_prediction = baseline.predict(X_test)
results = []
for name, scores, labels in [
    ("Majority baseline", baseline_probability, baseline_prediction),
    ("Logistic regression", probability, prediction),
]:
    results.append({"Model": name, "ROC-AUC": roc_auc_score(y_test, scores),
                    "Average precision": average_precision_score(y_test, scores),
                    "Precision": precision_score(y_test, labels, zero_division=0),
                    "Recall": recall_score(y_test, labels, zero_division=0),
                    "F1": f1_score(y_test, labels, zero_division=0),
                    "Accuracy": accuracy_score(y_test, labels)})
display(pd.DataFrame(results).set_index("Model").round(4))
print(f"Training-only threshold: {threshold:.4f}; holdout flagged: {prediction.mean():.1%}")
print("The 30% outreach target is approximate, not a hard capacity limit.")

# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=["Stayed", "Churned"],
                                        ax=axes[0, 0], cmap="Blues", colorbar=False)
axes[0, 0].set_title("Confusion matrix at training-selected threshold")
RocCurveDisplay.from_predictions(y_test, probability, ax=axes[0, 1], name="Logistic regression")
axes[0, 1].plot([0, 1], [0, 1], "--", color="gray", label="Chance")
axes[0, 1].legend()
PrecisionRecallDisplay.from_predictions(y_test, probability, ax=axes[1, 0], name="Logistic regression")
axes[1, 0].axhline(y_test.mean(), linestyle="--", color="gray", label="Churn prevalence")
axes[1, 0].legend()
CalibrationDisplay.from_predictions(y_test, probability, n_bins=8, strategy="quantile",
                                    ax=axes[1, 1], name="Logistic regression")
plt.tight_layout()
plt.show()

importance = permutation_importance(model, X_test, y_test, scoring="average_precision",
                                   n_repeats=3, random_state=RANDOM_STATE)
ranking = pd.Series(importance.importances_mean, index=X.columns).nlargest(8).sort_values()
ranking.plot.barh(figsize=(8, 4), color="#2563eb", title="Top model drivers — permutation importance")
plt.xlabel("Mean decrease in holdout average precision")
plt.tight_layout()
plt.show()

# %%
sample = X_test.iloc[:5].copy()
sample_scores = model.predict_proba(sample)[:, 1]
demo_predictions = pd.DataFrame({
    "customerID": df.loc[sample.index, "customerID"].to_numpy(),
    "churn_probability": sample_scores,
    "flag_for_outreach": sample_scores >= threshold,
    "actual_churn": y_test.loc[sample.index].to_numpy(),
})
display(demo_predictions.round(4))
print("Live prediction example:")
customer = X_test.iloc[[0]].copy()
display(customer)
print(f"Predicted churn risk: {model.predict_proba(customer)[0, 1]:.1%}")
print("Demo only: fictional data, previously inspected holdout, no causal or production claims.")
