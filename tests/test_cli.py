from pathlib import Path
import importlib.util
from io import BytesIO
from unittest.mock import Mock
import subprocess
import sys

import pandas as pd
import pytest

from churn.config import MODEL_PATH, RAW_DATA_PATH

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run scripts/train.py first")


def test_batch_scoring_cli(tmp_path):
    sample = pd.read_csv(RAW_DATA_PATH).head(12).drop(columns=["Churn"])
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sample.to_csv(input_path, index=False)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "score_customers.py"), str(input_path), "--out", str(output_path)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    scored = pd.read_csv(output_path)
    assert "Scored 12 customers" in result.stdout
    assert len(scored) == 12
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["risk_band"]) <= {"Low", "Medium", "High", "Critical"}


def test_batch_scoring_preserves_ids_and_escapes_formulas(tmp_path):
    sample = pd.read_csv(RAW_DATA_PATH).head(2).drop(columns=["Churn"])
    sample["customerID"] = ["000123", "=1+1"]
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    sample.to_csv(input_path, index=False)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "score_customers.py"), str(input_path), "--out", str(output_path)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    scored = pd.read_csv(output_path, dtype={"customerID": str}, keep_default_na=False)
    assert scored["customerID"].tolist() == ["000123", "'=1+1"]


def test_batch_scoring_does_not_overwrite_input(tmp_path):
    path = tmp_path / "input.csv"
    contents = "example\n1\n"
    path.write_text(contents, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "score_customers.py"), str(path), "--out", str(path)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Output path must differ" in result.stderr
    assert path.read_text(encoding="utf-8") == contents


def test_custom_training_output_keeps_reports_separate(tmp_path):
    spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.report_directory(module.ARTIFACT_DIR) == module.REPORT_DIR
    assert module.report_directory(tmp_path) == tmp_path / "reports"
    assert module.parse_args([]).out_dir == module.ARTIFACT_DIR
    assert module.parse_args(["--quick"]).out_dir == module.ARTIFACT_DIR / "quick"
    assert module.parse_args(["--quick", "--out-dir", str(tmp_path)]).out_dir == tmp_path


@pytest.fixture
def download_script():
    spec = importlib.util.spec_from_file_location("download_script", ROOT / "scripts" / "download_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_download_verifies_bytes_and_reuses_existing(download_script, tmp_path, monkeypatch):
    contents = RAW_DATA_PATH.read_bytes()
    fetch = Mock(return_value=BytesIO(contents))
    monkeypatch.setattr(download_script, "urlopen", fetch)
    path = tmp_path / "data" / "telco.csv"
    assert download_script.download_dataset(path) == path
    assert path.read_bytes() == contents
    assert download_script.download_dataset(path) == path
    fetch.assert_called_once_with(download_script.SOURCE_URL, timeout=60)


def test_dataset_download_rejects_wrong_checksum(download_script, tmp_path, monkeypatch):
    monkeypatch.setattr(download_script, "urlopen", Mock(return_value=BytesIO(b"wrong dataset")))
    path = tmp_path / "telco.csv"
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_script.download_dataset(path)
    assert not path.exists()


def test_dataset_download_preserves_different_existing_file(download_script, tmp_path, monkeypatch):
    fetch = Mock()
    monkeypatch.setattr(download_script, "urlopen", fetch)
    path = tmp_path / "telco.csv"
    path.write_bytes(b"user data")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        download_script.download_dataset(path)
    assert path.read_bytes() == b"user data"
    fetch.assert_not_called()
