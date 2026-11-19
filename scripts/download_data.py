from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def download_dataset(path: Path) -> Path:
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != EXPECTED_SHA256:
            raise ValueError(f"Existing file differs from the reference dataset; refusing to overwrite: {path}")
        return path
    with urlopen(SOURCE_URL, timeout=60) as response:
        contents = response.read()
    if hashlib.sha256(contents).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Dataset checksum mismatch; no file was written.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(contents)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the reference IBM Telco CSV and verify its SHA-256.")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "telco_churn.csv")
    args = parser.parse_args(argv)
    path = download_dataset(args.out)
    print(f"Dataset verified: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
