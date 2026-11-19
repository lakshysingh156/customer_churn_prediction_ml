"""Single source of truth for paths, feature lists and evaluation settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
METRICS_PATH = REPORT_DIR / "metrics.json"

TARGET = "Churn"
POSITIVE_LABEL = "Yes"
ID_COLUMN = "customerID"

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2

RAW_NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

ENGINEERED_NUMERIC_FEATURES = [
    "num_services",
    "has_internet",
    "avg_charge_per_month",
    "is_new_customer",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NEW_CUSTOMER_TENURE = 6

# The retention team can work through this fraction of the customer base each
# cycle, so the operating point flags the riskiest TARGET_CONTACT_RATE of rows.
# Change this to match real team capacity rather than a made-up dollar figure.
TARGET_CONTACT_RATE = 0.30


@dataclass(frozen=True)
class BusinessCost:
    """Illustrative misclassification penalties, not measured campaign costs."""

    false_negative: float = 1000.0
    false_positive: float = 250.0

    def total(self, false_positives: int, false_negatives: int) -> float:
        return false_positives * self.false_positive + false_negatives * self.false_negative


DEFAULT_COST = BusinessCost()
