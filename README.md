# Alternative Credit Scoring Using Non-Traditional Data — Fast Prototype

**Important:** This project currently trains on generated synthetic data only. It is an engineering demonstration, not evidence that these features predict real borrower behaviour. Do not use the model for real lending decisions or enter real personal/financial data into the demo.

## What's included

- `train.py` — reproducibly generates 10,000 synthetic rows; trains a traditional-feature Logistic Regression baseline, a hybrid Logistic Regression model, and a hybrid HistGradientBoosting model; evaluates on a held-out stratified test set; saves models, metrics, plots and synthetic CSVs.
- `app.py` — optional Streamlit UI to demonstrate inference with explicitly simulated inputs.
- `data/synthetic_applicants.csv` — generated dataset.
- `reports/test_metrics.csv` — test-set metric comparison (synthetic data only).
- `reports/no_bureau_history_subgroup_metrics.csv` — subgroup analysis for the synthetic no-bureau-history group.
- `reports/permutation_importance.csv` and `.png` — permutation importance (association, not causation).
- `models/*.joblib` — serialized sklearn pipelines.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
```

If `streamlit` installation is unavailable, `python train.py` still runs training and produces the evaluation artifacts; the web UI is optional.

## Feature groups

**Traditional/application baseline:** age, monthly income, employment tenure, debt-to-income, number of existing loans, credit utilization, recent delinquencies, bureau-history availability and bureau score.

**Synthetic alternative examples:** utility-payment timeliness, rental-payment timeliness, digital-payment regularity, cash-flow stability and months of observed alternative history.

The names describe the synthetic scenario only. These columns are not sourced from real records. Their distributions and their relationships to the target are deliberately generated in `generate_synthetic_dataset` so the end-to-end comparison can be exercised.

## Evaluation design

- Fixed random seed (`2026`) for reproducibility.
- Stratified 80/20 train/test split.
- All imputation and scaling are fitted inside sklearn pipelines using training data only.
- Same held-out test rows for all model comparisons.
- ROC-AUC and average precision: higher is generally better for ranking/discrimination.
- Brier score and log loss: lower is generally better for probability quality.
- Permutation importance is descriptive and is not causal evidence.

## What is NOT claimed

- No real-borrower performance has been measured.
- No research novelty has been established by these synthetic experiments.
- The model is not calibrated/validated for production lending.
- The displayed risk-band cutoffs are illustrative, not approved decision thresholds.
- The synthetic no-bureau group is not a real-world thin-file population.

## Responsible next steps before any real-data experiment

1. Find a lawfully accessible real dataset with documented features and repayment outcomes.
2. Verify whether bureau-history status is measured and define the population supported by the records.
3. Confirm data provenance, permissions, target horizon and feature availability at the prediction time.
4. Freeze the train/validation/test design before tuning.
5. Refit and evaluate on real out-of-sample data; add calibration, error and subgroup fairness analysis where defensible.
6. Do not use sensitive behavioural features without a justified lawful basis and privacy review.

## Reproducibility

Run `python train.py` from this directory to regenerate the synthetic records, models and reports. Generated data and models should be versioned together with the script and seed. The saved models are only valid for the feature schema documented above.
