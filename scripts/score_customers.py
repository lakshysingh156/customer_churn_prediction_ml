"""Score a CSV of Telco customers using the persisted model.

    python scripts/score_customers.py input.csv --out scored.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.predict import ChurnScorer, read_customer_csv, scored_to_csv  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Score customers for churn risk.")
    parser.add_argument("input", type=Path, help="CSV with the raw Telco columns.")
    parser.add_argument("--out", type=Path, default=Path("scored_customers.csv"), help="Output CSV path.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.input.resolve() == args.out.resolve():
        raise ValueError("Output path must differ from the input CSV.")
    frame = read_customer_csv(args.input)
    scored = ChurnScorer().score_frame(frame)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(scored_to_csv(scored), encoding="utf-8")
    print(f"Scored {len(scored):,} customers -> {args.out}")
    print(scored["risk_band"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
