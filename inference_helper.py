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


def _validate_and_coerce_int(val: object, field_name: str) -> int:
    """Validate that val is a finite integer-compatible number (rejects booleans and fractions)."""
    if isinstance(val, bool):
        raise ValidationError(f"Field '{field_name}' must be an integer; got boolean '{val}'.")
    if val is None or not isinstance(val, (int, float, np.integer, np.floating)):
        raise ValidationError(
            f"Field '{field_name}' must be numeric; got {type(val).__name__} ({val!r})."
        )
    float_val = float(val)
    if not np.isfinite(float_val):
        raise ValidationError(f"Field '{field_name}' must be finite; got {val}.")
    if not float_val.is_integer():
        raise ValidationError(
            f"Field '{field_name}' must be an integer; got fractional value {val}."
        )
    return int(float_val)


def _validate_and_coerce_float(val: object, field_name: str) -> float:
    """Validate that val is a finite numeric float (rejects booleans and non-finite values)."""
    if isinstance(val, bool):
        raise ValidationError(f"Field '{field_name}' must be a float; got boolean '{val}'.")
    if val is None or not isinstance(val, (int, float, np.integer, np.floating)):
        raise ValidationError(
            f"Field '{field_name}' must be numeric; got {type(val).__name__} ({val!r})."
        )
    float_val = float(val)
    if not np.isfinite(float_val):
        raise ValidationError(f"Field '{field_name}' must be finite; got {val}.")
    return float_val


def _validate_bureau_flag(val: object) -> int:
    """Validate bureau history indicator flag strictly in {0, 1}."""
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, np.integer)):
        if val in (0, 1):
            return int(val)
        raise ValidationError(f"Field 'bureau_history_available' must be 0 or 1; got {val}.")
    if isinstance(val, (float, np.floating)):
        float_val = float(val)
        if np.isfinite(float_val) and float_val.is_integer() and int(float_val) in (0, 1):
            return int(float_val)
        raise ValidationError(f"Field 'bureau_history_available' must be 0 or 1; got {val}.")
    raise ValidationError(
        f"Field 'bureau_history_available' must be binary {{0, 1}} or bool; got {type(val).__name__}."
    )


def build_applicant_row(
    age_years: int | float,
    monthly_income_inr: float,
    employment_tenure_years: float,
    debt_to_income_ratio: float,
    existing_loan_count: int | float,
    credit_utilization_ratio: float,
    prior_delinquencies_24m: int | float,
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
    - Integer-valued fields (age, loans, delinquencies, alternative months) reject
      fractional or non-finite inputs instead of silently truncating.
    - Required float fields reject non-finite inputs.
    - If bureau_history_available is 0 or False: bureau_score is explicitly np.nan.
    - If bureau_history_available is 1 or True: bureau_score must be present and finite float.
    - If an alternative feature is in `unavailable_alternative_features`, its value is set to np.nan.
    - `unavailable_alternative_features` rejects any feature name not in ALTERNATIVE_FEATURES.
    - Returns a DataFrame containing exactly ALL_FEATURES in expected order.
    """
    # 1. Validate unavailable alternative features list
    if unavailable_alternative_features is not None:
        unavail_list = list(unavailable_alternative_features)
        invalid_features = set(unavail_list) - set(ALTERNATIVE_FEATURES)
        if invalid_features:
            raise ValidationError(
                f"Invalid unavailable alternative feature(s): {sorted(invalid_features)}. "
                f"Supported alternative features are: {sorted(ALTERNATIVE_FEATURES)}"
            )
        unavail = set(unavail_list)
    else:
        unavail = set()

    # 2. Bureau history flag and bureau score
    bureau_hist_flag = _validate_bureau_flag(bureau_history_available)
    if bureau_hist_flag == 0:
        final_bureau_score = np.nan
    else:
        if bureau_score is None or (isinstance(bureau_score, (float, np.floating)) and np.isnan(bureau_score)):
            raise ValidationError("Field 'bureau_score' cannot be missing when 'bureau_history_available' is 1.")
        final_bureau_score = _validate_and_coerce_float(bureau_score, "bureau_score")

    # 3. Optional alternative feature resolvers
    def _resolve_alt_float(val: object, feature_name: str) -> float:
        if feature_name in unavail or val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
            return np.nan
        return _validate_and_coerce_float(val, feature_name)

    def _resolve_alt_int(val: object, feature_name: str) -> float:
        if feature_name in unavail or val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
            return np.nan
        int_val = _validate_and_coerce_int(val, feature_name)
        return float(int_val)

    # 4. Assemble row data with strict coercion and validation
    row_data = {
        "age_years": _validate_and_coerce_int(age_years, "age_years"),
        "monthly_income_inr": _validate_and_coerce_float(monthly_income_inr, "monthly_income_inr"),
        "employment_tenure_years": _validate_and_coerce_float(employment_tenure_years, "employment_tenure_years"),
        "debt_to_income_ratio": _validate_and_coerce_float(debt_to_income_ratio, "debt_to_income_ratio"),
        "existing_loan_count": _validate_and_coerce_int(existing_loan_count, "existing_loan_count"),
        "credit_utilization_ratio": _validate_and_coerce_float(credit_utilization_ratio, "credit_utilization_ratio"),
        "prior_delinquencies_24m": _validate_and_coerce_int(prior_delinquencies_24m, "prior_delinquencies_24m"),
        "bureau_history_available": bureau_hist_flag,
        "bureau_score": final_bureau_score,
        "utility_payment_timeliness": _resolve_alt_float(utility_payment_timeliness, "utility_payment_timeliness"),
        "rental_payment_timeliness": _resolve_alt_float(rental_payment_timeliness, "rental_payment_timeliness"),
        "digital_payment_regularity": _resolve_alt_float(digital_payment_regularity, "digital_payment_regularity"),
        "cashflow_stability_index": _resolve_alt_float(cashflow_stability_index, "cashflow_stability_index"),
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
    if not isinstance(row, pd.DataFrame):
        raise ValidationError(f"Expected applicant row to be a pandas DataFrame, got {type(row).__name__}.")

    if len(row) != 1:
        raise ValidationError(f"Inference pipeline expects exactly 1 applicant row; got {len(row)}.")

    if validate:
        validate_feature_frame(row, expected_features=ALL_FEATURES)

    proba_out = model.predict_proba(row)
    proba_arr = np.asarray(proba_out)

    if proba_arr.shape != (1, 2):
        raise ValueError(
            f"Model predict_proba returned unexpected output shape {proba_arr.shape}; "
            "expected exactly (1, 2) for single-applicant binary classification."
        )

    prob_val = proba_arr[0, 1]
    if isinstance(prob_val, bool) or not isinstance(prob_val, (int, float, np.floating)):
        raise TypeError(f"Model default probability must be numeric; got {type(prob_val).__name__}.")

    float_prob = float(prob_val)
    if not np.isfinite(float_prob) or not (0.0 <= float_prob <= 1.0):
        raise ValueError(
            f"Model returned invalid default probability {float_prob}; must be a finite float in [0.0, 1.0]."
        )

    probability = float_prob

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
