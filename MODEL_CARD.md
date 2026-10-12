# Model Card — Synthetic Credit-Risk Research Prototype

## Intended use
Software demonstration and student learning only: demonstrate data generation, preprocessing, baseline comparison, probability prediction, metrics and feature-importance tooling.

## Prohibited use
Do not use to approve, reject, price, rank or otherwise make decisions about real people or actual lending applications. Do not enter real financial or personal data.

## Data
10,000 simulated records generated locally by `train.py` using seed 2026. No real borrower records were used. Labels are sampled from a probability rule deliberately defined in the same generator. That rule explicitly uses both traditional and alternative feature examples, so a lift in hybrid performance is partly expected by construction.

## Features
Traditional/application: age, monthly income, employment tenure, debt-to-income ratio, existing loan count, credit utilization, past delinquencies, bureau-history availability, bureau score.

Synthetic alternatives: utility payment timeliness, rental payment timeliness, digital payment regularity, cash-flow stability index, and months of alternative history.

## Models
- Logistic Regression on traditional/application features.
- Logistic Regression on traditional plus alternative examples.
- HistGradientBoosting on the combined synthetic features.

## Validation and metrics
Fixed random seed; stratified 80/20 train/test split; preprocessing is fitted inside training pipelines. Metrics are ROC-AUC, average precision, Brier score and log loss. The random split does not constitute out-of-time validation. It supports an engineering demonstration only.

## Limitations
- Simulated population, features, missingness and target generation.
- Synthetic no-bureau-history group is not a validated representation of credit-invisible borrowers.
- No external validation, real-world calibration, drift testing or credible financial-cost model.
- Feature importance is predictive association, not causation.
- No fairness conclusions should be drawn from the current simulation.
- The illustrative risk bands in the optional UI are not validated thresholds.

## Revalidation required
Use a properly authorised real dataset; document target horizon and data provenance; define feature availability at decision time; evaluate chronologically when possible; check calibration and error costs; perform justified privacy and fairness audits; and independently review the deployment context before any real-world use.
