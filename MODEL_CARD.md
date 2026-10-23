# Model Card — Synthetic Credit-Risk Research Prototype

## Intended use
Software demonstration and student learning only: demonstrate data generation, preprocessing, baseline comparison, probability prediction, metrics, threshold sensitivity analysis, illustrative score transformation, and feature-importance/explainability tooling.

## Prohibited use
Do not use to approve, reject, price, rank or otherwise make decisions about real people or actual lending applications. Do not enter real financial or personal data.

## Data
10,000 simulated records generated locally by `train.py` using seed 2026. No real borrower records were used. Labels are sampled from a probability rule deliberately defined in the same generator. That rule explicitly uses both traditional and alternative feature examples, so a lift in hybrid performance is partly expected by construction.

## Features & Supported Domain
The supported bounds below reflect the generator distributions and input validator domain contracts for this specific synthetic prototype. They are **not** universal real-world lending bounds, borrower eligibility limits, or statutory regulatory standards.

- **Traditional/application (9 predictors):**
  - `age_years`: integer $\in [20, 65]$
  - `monthly_income_inr`: float $\in [8000.0, 250000.0]$
  - `employment_tenure_years`: float $\in [0.0, 35.0]$
  - `debt_to_income_ratio`: float $\in [0.0, 1.2]$
  - `existing_loan_count`: integer $\in [0, 8]$
  - `credit_utilization_ratio`: float $\in [0.0, 1.25]$
  - `prior_delinquencies_24m`: integer $\in [0, 5]$
  - `bureau_history_available`: binary $\in \{0, 1\}$
  - `bureau_score`: float $\in [300.0, 900.0]$ when bureau history is available; mandatory NaN when unavailable
- **Synthetic alternative features (5 predictors):**
  - `utility_payment_timeliness`: float $\in [0.05, 1.0]$, with permitted missing values (NaN)
  - `rental_payment_timeliness`: float $\in [0.05, 1.0]$, with permitted missing values (NaN)
  - `digital_payment_regularity`: float $\in [0.02, 1.0]$, with permitted missing values (NaN)
  - `cashflow_stability_index`: float $\in [0.02, 1.0]$, with permitted missing values (NaN)
  - `alternative_history_months`: integer $\in [0, 24]$, with inference missingness handled according to the existing validator contract (permitted missing at inference)

## Models
- Logistic Regression on traditional/application features.
- Logistic Regression on traditional plus alternative examples.
- HistGradientBoosting on the combined synthetic features.

## Validation, Metrics & Threshold Analysis
- Fixed random seed (2026); stratified 80/20 train/test split (8,000 train / 2,000 test).
- Preprocessing pipelines (median imputation, standard scaling) are fitted strictly on training data.
- Metrics evaluated: ROC-AUC, Gini coefficient, KS statistic, average precision (PR-AUC), Brier score, log loss, and Expected Calibration Error (ECE, 10 uniform bins).
- **Out-of-Fold Threshold Analysis:** Decision threshold evaluation ($\tau \in [0.05, 0.95]$) is performed strictly on 5-fold cross-validated out-of-fold predictions on the **training partition** (8,000 samples). The held-out test partition is reserved strictly for generalization evaluation and was **not** used for threshold selection, preventing data snooping leakage.

## Multi-Seed Robustness & Feature Ablation Analysis
To assess sampling variability and marginal feature importance across independent generations, a 20-seed ablation study is documented in `experiments/alternative_feature_ablation.py` and `reports/ablation_robustness/experiment_report.md`:
- **Seeds & Scope:** 20 independent simulation seeds (2026 to 2045), each generating 10,000 synthetic records split 80/20 train/test.
- **Scenarios Evaluated:** Full hybrid, traditional baseline, and 5 single-feature ablations (removing utility timeliness, rental timeliness, digital regularity, cash-flow stability, or alternative history length one at a time).
- **Leakage Prevention:** Preprocessing and model fitting are restricted strictly to each seed's training split; all 7 scenarios are evaluated on the identical held-out test split.
- **Findings & Ranking Variability:** Utility payment timeliness and cash-flow stability exhibit the largest paired performance drops (utility rank 1 in 15/20 seeds, mean rank 1.35; cash-flow rank 1 in 5/20 and rank 2 in 12/20 seeds, mean rank 1.90). While paired drops are consistently negative across seeds for key signals, relative feature importance ranks vary across simulation runs due to finite-sample sampling noise; the exact complete five-feature hierarchy holds in only 7 of 20 seeds (35.0%). Consistently negative deltas confirm directional importance under the synthetic process but do not imply deterministic rank ordering across datasets. All observed lifts are expected by construction due to the synthetic generative mechanism and do not demonstrate real-world credit underwriting efficacy.

## Bootstrap Uncertainty Quantification & Paired Model Comparison
To quantify test-metric variability conditional on the fixed fitted models, a non-parametric bootstrap study is documented in `uncertainty_analysis.py` and `reports/uncertainty/bootstrap_uncertainty_report.md`:
- **Protocol:** 1,000 bootstrap replicates (seed 2027) with 95% two-sided percentile confidence intervals $[CI_{\text{lower}}, CI_{\text{upper}}]$ evaluated across the overall test set (2,000 records) and the thin-file `no_bureau_history` subgroup (926 records with `bureau_history_available == 0`).
- **Paired Resampling:** Pairwise model differences ($\Delta = \text{Metric}(M_A) - \text{Metric}(M_B)$) evaluate predictions on the **exact same resampled observation indices** for every replicate, accounting for observation-level covariance rather than incorrectly treating models as independent samples.
- **Empirical Findings:** Both hybrid architectures (linear and boosted) maintain positive, non-zero paired discrimination advantages over the traditional baseline across 95% bootstrap intervals (overall test set ROC-AUC $\Delta = +0.1027$, 95% CI: $[+0.0848, +0.1206]$ for Hybrid LR; $\Delta = +0.0992$, 95% CI: $[+0.0806, +0.1179]$ for Hybrid HGB). The paired ROC-AUC interval between Hybrid HGB and Hybrid LR spans zero ($\Delta = -0.0035$, 95% CI: $[-0.0087, +0.0017]$), indicating comparable discrimination.
- **Conditioning Scope & Limitations:** These percentile intervals summarize sampling uncertainty from resampling held-out test observations for fixed, already-fitted model pipelines. They do **not** capture variation from retraining models on fresh training splits, modifying data generation rules, or real-world population distribution shifts. An interval crossing zero does not establish that two models are equivalent, and an interval excluding zero is not proof of commercial significance or deployment readiness. The models and default labels are entirely simulated; no fairness, credit-invisible, or regulatory compliance conclusions may be drawn.

## Illustrative Score Transformation & Risk Bands

To provide an intuitive interface familiar to credit professionals, default probabilities are mapped to a traditional 300–850 scoring scale using the industry-standard Points to Double the Odds (PDO) formulation:
$$\text{Score} = \text{Offset} - \frac{\text{PDO}}{\ln 2} \ln\left(\frac{p}{1-p}\right)$$
- **Base Score:** 600 at 50:1 good-to-bad odds ($p \approx 1.96\%$)
- **PDO:** 20 points
- **Scale Bounds:** $[300, 850]$ (clamped at extremes)
- **Illustrative Model Risk Bands:**
  - *Lower model-estimated risk:* $p < 0.10$
  - *Middle model-estimated risk:* $0.10 \le p < 0.20$
  - *Higher model-estimated risk:* $p \ge 0.20$
- **Score Transformation Note:** The illustrative credit score is a monotonic, bounded, and rounded transformation of modeled default probability (continuous raw scores at $p=0.10$ and $p=0.20$ evaluate to $\approx 550.52$ and $\approx 527.12$, respectively). Because display scores are subject to integer rounding and boundary clamping, scores should not be substituted for the probability-based band cutoffs.
- **Non-Regulatory Notice:** Scores and risk bands are mathematical transformations of model probabilities for educational demonstration only. They are not credit bureau scores and convey no underwriting authority.

## Local Explainability
For the hybrid Logistic Regression model, predictions provide exact additive local feature contributions decomposed in transformed-feature/log-odds space:
$$\text{decision\_function}(\mathbf{x}') = \beta_0 + \sum_{j=1}^M \beta_j x'_j$$
The sum of local feature contributions ($\beta_j x'_j$) and global intercept ($\beta_0$) reconstructs the model decision function to within machine precision ($< 10^{-14}$). Positive contributions increase the model's log-odds of synthetic default; negative contributions reduce them. These values represent the fitted model's mathematical behavior on normalized inputs, not causal effects. They are educational explanations and are not legally sufficient adverse-action reason codes or proof of regulatory compliance.

## Interactive Demonstration Interface
An interactive demonstration interface (`app.py`) provides graphical access to the hybrid Logistic Regression model:
- **Input Controls:** Widget ranges strictly match the validator domain bounds in `validation.py`.
- **Missingness Simulation:** Unbanked or cold-start scenarios are supported via toggles for unavailable bureau history (forcing `bureau_score = NaN`) and multiselect controls for unavailable alternative-data records (assigning actual `NaN` values to test imputation).
- **Mandatory Pre-Inference Validation:** Assembled rows are passed through `validate_feature_frame()` before inference. Model execution is rejected with user-facing alerts if any contract boundary is violated.
- **Shared Implementation:** All scoring, risk-band mappings, and additive local explanations invoke the core functions (`risk_scoring.py`, `explainability.py`, and `inference_helper.py`) without divergent UI logic.

## Limitations
- Simulated population, features, missingness and target generation.
- Synthetic no-bureau-history group is not a validated representation of credit-invisible borrowers.
- No external validation, real-world calibration, drift testing or credible financial-cost model.
- Feature importance and local feature contributions reflect predictive associations under the simulation rule, not causal impact.
- No fairness conclusions should be drawn from the current simulation.
- Thresholds and risk bands are not validated operational underwriting policies.

## Revalidation required
Use a properly authorised real dataset; document target horizon and data provenance; define feature availability at decision time; evaluate chronologically when possible; check calibration and error costs; perform justified privacy and fairness audits; and independently review the deployment context before any real-world use.
