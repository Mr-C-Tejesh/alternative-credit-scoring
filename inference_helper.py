"""Shared inference and applicant row assembly helper.

Provides reusable utilities for applicant data construction, model loading with
scikit-learn backward-compatibility checks, and end-to-end inference/explanation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from explainability import ExplanationResult, explain_logistic_prediction
from risk_scoring import (
    DEFAULT_BASE_ODDS,
    DEFAULT_BASE_SCORE,
    DEFAULT_MAX_SCORE,
    DEFAULT_MIN_SCORE,
    DEFAULT_PDO,
    ScoreResult,
    map_probability_to_risk_band,
    probability_to_score,
)
from validation import (
    ALL_FEATURES,
    ALTERNATIVE_FEATURES,
    DOMAIN_BOUNDS,
    TRADITIONAL_FEATURES,
    ValidationError,
    validate_feature_frame,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "models" / "hybrid_logistic_regression.joblib"

# Supported alternative features with human-readable labels for UI controls
ALT_FEATURE_OPTIONS: list[tuple[str, str]] = [
    ("utility_payment_timeliness", "Utility payment timeliness"),
    ("rental_payment_timeliness", "Rental payment timeliness"),
    ("digital_payment_regularity", "Digital payment regularity"),
    ("cashflow_stability_index", "Cash-flow stability index"),
    ("alternative_history_months", "Alternative history length"),
]


@dataclass
class InferenceResult:
    """Structured container holding end-to-end prediction, score, and explanation."""
    probability: float
    score_result: ScoreResult
    risk_band: str
    explanation: ExplanationResult
    row: pd.DataFrame


def load_saved_model(model_path: Path | str | None = None) -> Pipeline:
    """Load the serialized hybrid pipeline and ensure scikit-learn attribute compatibility.

    scikit-learn 1.8 removed LogisticRegression.multi_class from the public
    constructor, while some serialized estimators/predict_proba paths still
    expect the attribute. Restores it if missing.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Saved model artifact not found at {path}. Run `python train.py` first.")

    model = joblib.load(path)
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        classifier = model.named_steps["classifier"]
        if classifier.__class__.__name__ == "LogisticRegression" and not hasattr(classifier, "multi_class"):
            classifier.multi_class = "auto"
    return model


def build_applicant_row(
    age_years: int,
    monthly_income_inr: float,
    employment_tenure_years: float,
    debt_to_income_ratio: float,
    existing_loan_count: int,
    credit_utilization_ratio: float,
    prior_delinquencies_24m: int,
    bureau_history_available: int | bool,
    bureau_score: float | None = 650.0,
    utility_payment_timeliness: float | None = 0.82,
    rental_payment_timeliness: float | None = 0.80,
    digital_payment_regularity: float | None = 0.75,
    cashflow_stability_index: float | None = 0.68,
    alternative_history_months: int | float | None = 15,
    unavailable_alternative_features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assemble a single-row applicant DataFrame with strict contract handling.

    Rules enforced:
    - If bureau_history_available is 0 or False: bureau_score is explicitly np.nan.
    - If bureau_history_available is 1 or True: bureau_score is float(bureau_score).
    - If an alternative feature is in `unavailable_alternative_features`, its value is set to np.nan.
    - Returns a DataFrame containing exactly ALL_FEATURES in expected order.
    """
    has_bureau = bool(bureau_history_available)
    bureau_hist_flag = 1 if has_bureau else 0

    if not has_bureau:
        final_bureau_score = np.nan
    else:
        final_bureau_score = (
            float(bureau_score)
            if bureau_score is not None and not (isinstance(bureau_score, float) and np.isnan(bureau_score))
            else np.nan
        )

    unavail = set(unavailable_alternative_features or [])

    def _resolve_alt(val: float | None, feature_name: str) -> float:
        if feature_name in unavail or val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
            return np.nan
        return float(val)

    def _resolve_alt_int(val: int | float | None, feature_name: str) -> float:
        if feature_name in unavail or val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
            return np.nan
        return float(val)

    row_data = {
        "age_years": int(age_years),
        "monthly_income_inr": float(monthly_income_inr),
        "employment_tenure_years": float(employment_tenure_years),
        "debt_to_income_ratio": float(debt_to_income_ratio),
        "existing_loan_count": int(existing_loan_count),
        "credit_utilization_ratio": float(credit_utilization_ratio),
        "prior_delinquencies_24m": int(prior_delinquencies_24m),
        "bureau_history_available": bureau_hist_flag,
        "bureau_score": final_bureau_score,
        "utility_payment_timeliness": _resolve_alt(utility_payment_timeliness, "utility_payment_timeliness"),
        "rental_payment_timeliness": _resolve_alt(rental_payment_timeliness, "rental_payment_timeliness"),
        "digital_payment_regularity": _resolve_alt(digital_payment_regularity, "digital_payment_regularity"),
        "cashflow_stability_index": _resolve_alt(cashflow_stability_index, "cashflow_stability_index"),
        "alternative_history_months": _resolve_alt_int(alternative_history_months, "alternative_history_months"),
    }

    df = pd.DataFrame([row_data], columns=ALL_FEATURES)
    return df


def run_inference_pipeline(
    model: Pipeline,
    row: pd.DataFrame,
    validate: bool = True,
    base_score: float = DEFAULT_BASE_SCORE,
    base_odds: float = DEFAULT_BASE_ODDS,
    pdo: float = DEFAULT_PDO,
    min_score: int = DEFAULT_MIN_SCORE,
    max_score: int = DEFAULT_MAX_SCORE,
    top_k_explanations: int = 3,
) -> InferenceResult:
    """Run validation, prediction, illustrative scoring, risk band mapping, and local explanation.

    Parameters
    ----------
    model : Pipeline
        The fitted hybrid Logistic Regression pipeline.
    row : pd.DataFrame
        Single applicant row DataFrame.
    validate : bool, default=True
        Whether to run validate_feature_frame prior to inference.
    base_score, base_odds, pdo, min_score, max_score : scoring parameters
    top_k_explanations : int, default=3

    Returns
    -------
    InferenceResult
        Container holding probability, score result, risk band, explanation, and validated row.
    """
    if validate:
        validate_feature_frame(row, expected_features=ALL_FEATURES)

    probability = float(model.predict_proba(row)[0, 1])

    score_result = probability_to_score(
        p=probability,
        base_score=base_score,
        base_odds=base_odds,
        pdo=pdo,
        min_score=min_score,
        max_score=max_score,
    )

    risk_band = map_probability_to_risk_band(probability)

    explanation = explain_logistic_prediction(
        model,
        row,
        top_k=top_k_explanations,
        validate=False,  # Already validated above
    )

    return InferenceResult(
        probability=probability,
        score_result=score_result,
        risk_band=risk_band,
        explanation=explanation,
        row=row,
    )
