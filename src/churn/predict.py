"""Inference: turn a fitted pipeline into risk bands and recommended actions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn.config import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    METADATA_PATH,
    MODEL_PATH,
    SERVICE_COLUMNS,
)
from churn.data import clean_features

REQUIRED_INPUT_COLUMNS = sorted(
    set(CATEGORICAL_FEATURES)
    | set(SERVICE_COLUMNS)
    | {"tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"}
)

RECOMMENDED_ACTIONS = {
    "Low": "No action. Keep in standard lifecycle campaigns.",
    "Medium": "Monitor. Include in the next nurture campaign.",
    "High": "Proactive retention offer within 30 days.",
    "Critical": "Immediate outreach by the retention team.",
}

def read_customer_csv(source) -> pd.DataFrame:
    return pd.read_csv(source, dtype={ID_COLUMN: "string"}, keep_default_na=False)


def spreadsheet_safe_scores(scored: pd.DataFrame) -> pd.DataFrame:
    output = scored.copy()
    if ID_COLUMN in output.columns:
        identifiers = output[ID_COLUMN].astype("string")
        unsafe = identifiers.str.match(r"^[\t\r\n]|^\s*[=+\-@]", na=False)
        output.loc[unsafe, ID_COLUMN] = "'" + identifiers[unsafe]
    return output


def scored_to_csv(scored: pd.DataFrame) -> str:
    return spreadsheet_safe_scores(scored).to_csv(index=False)


def band_labels(probabilities, cutoffs) -> np.ndarray:
    """Vectorised version of the risk-band rule, shared by training and inference."""
    probabilities = np.asarray(probabilities, dtype=float)
    low, high, critical = cutoffs
    return np.select(
        [probabilities < low, probabilities < high, probabilities < critical],
        ["Low", "Medium", "High"],
        default="Critical",
    )


class ChurnScorer:
    """Wraps the persisted pipeline plus the business threshold chosen at training time."""

    def __init__(self, model_path: Path | str = MODEL_PATH, metadata_path: Path | str = METADATA_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model at {model_path}. Run `python scripts/train.py` first."
            )
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"No model metadata at {metadata_path}. Run `python scripts/train.py` first.")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if self.metadata.get("model_sha256") != model_hash:
            raise ValueError("Model and metadata do not match. Retrain or restore matching artifacts.")
        self.pipeline = joblib.load(model_path)

    @property
    def threshold(self) -> float:
        return float(self.metadata["decision_threshold"])

    @property
    def band_cutoffs(self) -> tuple[float, float, float]:
        return tuple(float(c) for c in self.metadata["band_cutoffs"])

    def risk_band(self, probability: float) -> str:
        return str(band_labels([probability], self.band_cutoffs)[0])

    @property
    def positive_class_index(self) -> int:
        return int(np.flatnonzero(np.asarray(self.pipeline.classes_) == 1)[0])

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        if df.empty:
            raise ValueError("Input CSV contains no customers.")
        df = clean_features(df)
        encoder = self.pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        for column, fitted_values in zip(CATEGORICAL_FEATURES, encoder.categories_):
            unseen = ~df[column].isin(fitted_values)
            if unseen.any():
                rows = (np.flatnonzero(unseen.to_numpy())[:5] + 1).tolist()
                raise ValueError(f"{column} contains categories not seen during training at input rows {rows}.")

        probabilities = self.pipeline.predict_proba(df)[:, self.positive_class_index]
        bands = band_labels(probabilities, self.band_cutoffs)
        result = pd.DataFrame(
            {
                "churn_probability": probabilities,
                "churn_prediction": (probabilities >= self.threshold).astype(int),
            },
            index=df.index,
        )
        result["risk_band"] = bands
        result["recommended_action"] = result["risk_band"].map(RECOMMENDED_ACTIONS)

        if ID_COLUMN in df.columns:
            result.insert(0, ID_COLUMN, df[ID_COLUMN].to_numpy())
        return result

    def score_record(self, record: dict) -> dict:
        scored = self.score_frame(pd.DataFrame([record])).iloc[0]
        return scored.to_dict()
