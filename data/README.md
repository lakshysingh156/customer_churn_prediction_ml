# Dataset

`raw/telco_churn.csv` is the **IBM Telco Customer Churn** dataset: 7,043 rows, 21 columns, one row per customer, with a binary `Churn` label.

- Source: https://github.com/IBM/telco-customer-churn-on-icp4d (`data/Telco-Customer-Churn.csv`)
- A widely used public churn benchmark; this project uses the IBM source linked above.
- It describes a fictional telecom provider. No real customer data is involved.

The raw CSV is excluded from Git. Run `python scripts/download_data.py` from the project root to fetch it directly from IBM; an existing matching local copy is kept for offline use. `scripts/train.py` never modifies it; cleaning happens in memory (`churn.data.clean`). IBM describes its code pattern as Apache-2.0, but this project does not establish separate dataset redistribution rights. Review the source terms before reuse or redistribution. The project's MIT license covers project code, not the dataset.

## Schema notes

| Column | Type | Note |
|---|---|---|
| `customerID` | string | Identifier. Dropped before modelling — keeping it invites accidental leakage. |
| `TotalCharges` | string in the raw file | 11 blanks, all rows with `tenure == 0`. Coerced to float and set to `0.0`. |
| `SeniorCitizen` | int 0/1 | Already numeric. |
| `Churn` | `Yes`/`No` | Target, 26.5% positive. |
| 15 others | categorical | `Contract`, `InternetService`, `PaymentMethod`, the add-on services, etc. |

Category strings are inconsistent by design (`No`, `No internet service`, `No phone service` all mean "not subscribed"). The model treats them as distinct levels rather than collapsing them, because the distinction is informative.

## Provenance

The downloader retrieves the unmodified file at IBM commit `1fd6fd70906479be1712e64cfc0ea89f58466629` and verifies SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` before writing it. This pins the reference bytes, not their licensing status.

```bash
python scripts/download_data.py
```

Use `--out path/to/telco_churn.csv` for another destination. A different existing file is never overwritten. GitHub CI fetches and verifies the same reference before training and testing.
