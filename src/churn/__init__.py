"""Customer churn prediction with a leakage-free sklearn pipeline."""

__version__ = "1.0.0"

from churn.config import MODEL_PATH, RAW_DATA_PATH  # noqa: F401
from churn.predict import ChurnScorer  # noqa: F401
