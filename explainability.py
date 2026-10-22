"""Exact additive local feature explanations for linear credit-risk models.

This module decomposes predictions from the saved hybrid Logistic Regression model
into exact feature contributions in log-odds space:
    logit(p) = intercept + sum_j (coef_j * x'_j)
    contribution_j = coef_j * x'_j

IMPORTANT DISCLAIMER:
These contributions describe the mathematical mechanics of the fitted linear model
on normalized, imputed features. They do NOT represent causal explanations,
real-world borrower advice, or compliant adverse-action notice reason codes under
credit regulations (e.g., US FCRA/ECOA or Indian DPDP/RBI guidelines).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from validation import ALL_FEATURES, validate_feature_frame


# Mapping from transformed feature names to human-readable descriptors
# while keeping the underlying technical name explicit.
FEATURE_NAME_DESCRIPTIONS: dict[str, str] = {
    "trad__age_years": "Applicant age (age_years)",
    "trad__monthly_income_inr": "Monthly income in INR (monthly_income_inr)",
    "employment_tenure_years": "Employment tenure in years (employment_tenure_years)",
    "trad__employment_tenure_years": "Employment tenure in years (employment_tenure_years)",
    "trad__debt_to_income_ratio": "Debt-to-income ratio (debt_to_income_ratio)",
    "trad__existing_loan_count": "Active loan count (existing_loan_count)",
    "trad__credit_utilization_ratio": "Credit utilization ratio (credit_utilization_ratio)",
    "trad__prior_delinquencies_24m": "24-month past delinquencies (prior_delinquencies_24m)",
    "trad__bureau_history_available": "Bureau history availability flag (bureau_history_available)",
    "trad__bureau_score": "Credit bureau score (bureau_score)",
    "alt_ind__utility_payment_timeliness": "Utility payment timeliness ratio (utility_payment_timeliness)",
    "alt_ind__rental_payment_timeliness": "Rental payment timeliness ratio (rental_payment_timeliness)",
    "alt_ind__digital_payment_regularity": "Digital payment regularity index (digital_payment_regularity)",
    "alt_ind__cashflow_stability_index": "Cash-flow stability index (cashflow_stability_index)",
    "alt_ind__missingindicator_utility_payment_timeliness": (
        "Missing utility record indicator (missingindicator_utility_payment_timeliness)"
    ),
    "alt_ind__missingindicator_rental_payment_timeliness": (
        "Missing rental record indicator (missingindicator_rental_payment_timeliness)"
    ),
    "alt_ind__missingindicator_digital_payment_regularity": (
        "Missing digital regularity indicator (missingindicator_digital_payment_regularity)"
    ),
    "alt_ind__missingindicator_cashflow_stability_index": (
        "Missing cash-flow stability indicator (missingindicator_cashflow_stability_index)"
    ),
    "alt_no_ind__alternative_history_months": "Alternative history length in months (alternative_history_months)",
}


def _readable_name(technical_name: str) -> str:
    """Return a descriptive, human-readable name for a transformed feature."""
    if technical_name in FEATURE_NAME_DESCRIPTIONS:
        return FEATURE_NAME_DESCRIPTIONS[technical_name]
    # Clean fallback: strip transformer prefixes
    clean = technical_name.split("__")[-1]
    return f"{clean.replace('_', ' ').title()} ({technical_name})"


@dataclass
class ExplanationResult:
    """Container holding exact additive linear decomposition results."""
    intercept: float
    reconstructed_logit: float
    model_decision_function: float
    model_probability: float
    reconstruction_error: float
    contributions_table: pd.DataFrame
    top_risk_increasing: list[dict]
    top_risk_decreasing: list[dict]

    @property
    def all_contributions(self) -> list[dict]:
        return self.contributions_table.to_dict(orient="records")

    @property
    def feature_contributions(self) -> list[dict]:
        return self.all_contributions


def explain_logistic_prediction(
    pipeline: Pipeline,
    row: pd.DataFrame,
    top_k: int = 3,
    validate: bool = True,
    expected_features: Sequence[str] | None = None,
) -> ExplanationResult:
    """Decompose a Logistic Regression pipeline prediction into exact additive contributions.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted pipeline with a 'preprocess' ColumnTransformer and a 'classifier' LogisticRegression.
    row : pd.DataFrame
        Single-row applicant DataFrame containing input features.
    top_k : int, default=3
        Number of top risk-increasing and risk-decreasing contributions to extract.
    validate : bool, default=True
        Whether to validate the input row against the schema and domain contract.
    expected_features : Sequence[str] | None
        Expected feature names for validation (defaults to ALL_FEATURES).

    Returns
    -------
    ExplanationResult
        Structured explanation containing base intercept, reconstructed log-odds,
        exact error assertion, feature contributions table, and top factors.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected a scikit-learn Pipeline instance, got {type(pipeline).__name__}.")

    if not isinstance(row, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(row).__name__}.")
    if len(row) != 1:
        raise ValueError(f"Feature DataFrame must contain exactly 1 row; got {len(row)}.")

    if expected_features is None:
        expected_features = ALL_FEATURES

    if validate:
        validate_feature_frame(row, expected_features=expected_features)

    if "classifier" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain a 'classifier' step.")

    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, LogisticRegression):
        raise TypeError(
            f"Exact additive contribution explanation is only valid for LogisticRegression models; "
            f"got {type(classifier).__name__}. Tree-based models (such as HistGradientBoosting) "
            "do not have linear additive coefficients."
        )

    if "preprocess" not in pipeline.named_steps:
        raise ValueError("Pipeline must contain a 'preprocess' transformer step.")

    preprocessor = pipeline.named_steps["preprocess"]

    # 1. Transform single applicant row through fitted preprocessor
    x_transformed = preprocessor.transform(row)
    if hasattr(x_transformed, "toarray"):
        x_transformed = x_transformed.toarray()
    x_vec = np.asarray(x_transformed, dtype=float).ravel()

    # 2. Extract transformer feature names and classifier weights
    feature_names = list(preprocessor.get_feature_names_out())
    if len(feature_names) != len(x_vec):
        raise ValueError(
            f"Mismatch between transformed vector length ({len(x_vec)}) and "
            f"feature names count ({len(feature_names)})."
        )

    coefs = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])

    if len(coefs) != len(x_vec):
        raise ValueError(
            f"Mismatch between classifier coefficients ({len(coefs)}) and "
            f"transformed features ({len(x_vec)})."
        )

    # 3. Exact linear decomposition: contribution_j = coef_j * x'_j
    contributions = coefs * x_vec
    reconstructed_logit = float(intercept + np.sum(contributions))

    # 4. Numerical reconstruction verification
    model_decision = float(pipeline.decision_function(row)[0])
    model_prob = float(pipeline.predict_proba(row)[0, 1])
    reconstruction_error = abs(reconstructed_logit - model_decision)

    if reconstruction_error > 1e-5:
        raise RuntimeError(
            f"Reconstruction error ({reconstruction_error:.2e}) exceeded tolerance 1e-5. "
            f"Reconstructed: {reconstructed_logit:.6f}, Decision function: {model_decision:.6f}"
        )

    # 5. Build full contributions DataFrame
    table_data = []
    for i, name in enumerate(feature_names):
        val = x_vec[i]
        c = coefs[i]
        contrib = contributions[i]
        table_data.append({
            "feature": name,
            "readable_name": _readable_name(name),
            "transformed_value": float(val),
            "coefficient": float(c),
            "contribution": float(contrib),
        })

    df_contrib = pd.DataFrame(table_data)

    # 6. Extract top positive (risk-increasing log-odds) and top negative (risk-decreasing log-odds)
    increasing = df_contrib.sort_values("contribution", ascending=False)
    decreasing = df_contrib.sort_values("contribution", ascending=True)

    top_increasing = increasing.head(top_k).to_dict(orient="records")
    top_decreasing = decreasing.head(top_k).to_dict(orient="records")

    return ExplanationResult(
        intercept=intercept,
        reconstructed_logit=reconstructed_logit,
        model_decision_function=model_decision,
        model_probability=model_prob,
        reconstruction_error=reconstruction_error,
        contributions_table=df_contrib,
        top_risk_increasing=top_increasing,
        top_risk_decreasing=top_decreasing,
    )
