# Model card

## Purpose

A portfolio demonstration that ranks telecom customers by predicted churn risk. It is not intended for decisions affecting real customers without validation and human oversight.

## Model and data

- **Dataset:** IBM Telco Customer Churn, 7,043 fictional records; about 26.5% are labelled as churned.
- **Target:** `Churn` (`Yes`/`No`).
- **Evaluation:** stratified 80/20 random split; model comparison and feature ablation use five-fold cross-validation on training data.
- **Selected pipeline:** logistic regression with raw numeric and one-hot encoded categorical fields. It excludes `customerID` from model features.
- **Operating threshold:** selected using out-of-fold training scores to target an illustrative 30% outreach capacity.
- **Artifacts:** `artifacts/churn_pipeline.joblib` and `artifacts/model_metadata.json`.

## Reported evaluation

The final full-run results, training configuration, confusion matrix counts, score bands and model comparison are recorded in [`metrics.json`](metrics.json). [`decile_lift.csv`](decile_lift.csv) contains holdout decile-level performance; diagrams are in [`figures/`](figures/).

The holdout is separate from model fitting and threshold selection, but full-dataset EDA and earlier project iterations have examined it. Use a genuinely independent or time-based test set for a stronger performance estimate.

## Limitations and responsible use

- A synthetic benchmark is not evidence that the model will generalise to a real provider or future customers.
- Churn risk is not treatment effect. The model does not tell us whether a customer would stay because of an offer.
- Fixed thresholds and score bands may flag different proportions when the population changes.
- Gender and senior-citizen indicators are included in the dataset and the model. Perform a fairness review before real-world use; the project does not establish fairness.
- Feature importance describes the fitted model, not a causal relationship.
- The cost curve uses arbitrary false-positive and false-negative penalties; it is not a measurement of campaign costs or savings.
- Joblib artifacts are pickle-based and should only be loaded from trusted sources.
