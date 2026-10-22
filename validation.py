"""Data contract and schema validation for synthetic credit scoring.

Phase-02 implementation:
- Formal dataset schema validation (validate_dataset_schema).
- Feature frame validation for training and inference (validate_feature_frame).
- Explicit domain bounds and constraints documented with justification.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

TARGET = "default_within_12_months"

TRADITIONAL_FEATURES: list[str] = [
    "age_years",
    "monthly_income_inr",
    "employment_tenure_years",
    "debt_to_income_ratio",
    "existing_loan_count",
    "credit_utilization_ratio",
    "prior_delinquencies_24m",
    "bureau_history_available",
    "bureau_score",
]

ALTERNATIVE_FEATURES: list[str] = [
    "utility_payment_timeliness",
    "rental_payment_timeliness",
    "digital_payment_regularity",
    "cashflow_stability_index",
    "alternative_history_months",
]

ALL_FEATURES: list[str] = TRADITIONAL_FEATURES + ALTERNATIVE_FEATURES

# Features permitted to be missing (NaN) during inference.
# For the full generated dataset, bureau_score is NaN iff bureau_history_available == 0,
# and alternative timeliness/regularity/stability features have controlled missingness.
# At inference, all alternative fields may be missing (e.g. cold-start borrower).
_PERMITTED_MISSING_INFERENCE: set[str] = {
    "bureau_score",
    "utility_payment_timeliness",
    "rental_payment_timeliness",
    "digital_payment_regularity",
    "cashflow_stability_index",
    "alternative_history_months",
}


class ValidationError(ValueError):
    """Raised when data fails schema, dtype, missingness, or domain bounds validation."""
    pass


# ---------------------------------------------------------------------------
# Domain bounds & constraints
#
# NOTE ON SOURCE AND JUSTIFICATION:
# These constraints reflect the documented simulation parameters and physical boundaries
# of the prototype lending demonstration. They are NOT universal regulatory limits.
# Accepting values outside these ranges in a real application requires a deliberate
# extrapolation policy rather than relying on unvalidated linear model projections.
# ---------------------------------------------------------------------------
DOMAIN_BOUNDS: dict[str, dict] = {
    "age_years": {
        "min": 18,
        "max": 100,
        "integer_only": True,
        "justification": "Legal credit age (18+) up to 100; generator simulates 20-65.",
    },
    "monthly_income_inr": {
        "min": 0.0,
        "max": 10_000_000.0,
        "integer_only": False,
        "justification": "Monthly income must be non-negative; capped at 10M INR for demo.",
    },
    "employment_tenure_years": {
        "min": 0.0,
        "max": 60.0,
        "integer_only": False,
        "justification": "Tenure must be non-negative; reasonable career span bound.",
    },
    "debt_to_income_ratio": {
        "min": 0.0,
        "max": 1.2,
        "integer_only": False,
        "justification": "CLI and generator support range is [0.0, 1.2].",
    },
    "existing_loan_count": {
        "min": 0,
        "max": 50,
        "integer_only": True,
        "justification": "Active loan count must be a non-negative integer.",
    },
    "credit_utilization_ratio": {
        "min": 0.0,
        "max": 1.5,
        "integer_only": False,
        "justification": "Credit card / line utilization ratio (generator support up to 1.25).",
    },
    "prior_delinquencies_24m": {
        "min": 0,
        "max": 50,
        "integer_only": True,
        "justification": "Count of past 24-month delinquencies must be a non-negative integer.",
    },
    "bureau_history_available": {
        "min": 0,
        "max": 1,
        "integer_only": True,
        "allowed_values": {0, 1},
        "justification": "Binary indicator: 0 = thin-file / no history, 1 = bureau record exists.",
    },
    "bureau_score": {
        "min": 300.0,
        "max": 900.0,
        "integer_only": False,
        "justification": "Standard Indian bureau score range (e.g. CIBIL 300-900).",
    },
    "utility_payment_timeliness": {
        "min": 0.0,
        "max": 1.0,
        "integer_only": False,
        "justification": "Payment timeliness ratio bounded in [0.0, 1.0] as documented.",
    },
    "rental_payment_timeliness": {
        "min": 0.0,
        "max": 1.0,
        "integer_only": False,
        "justification": "Payment timeliness ratio bounded in [0.0, 1.0] as documented.",
    },
    "digital_payment_regularity": {
        "min": 0.0,
        "max": 1.0,
        "integer_only": False,
        "justification": "Digital transaction regularity score bounded in [0.0, 1.0].",
    },
    "cashflow_stability_index": {
        "min": 0.0,
        "max": 1.0,
        "integer_only": False,
        "justification": "Cash-flow stability index bounded in [0.0, 1.0].",
    },
    "alternative_history_months": {
        "min": 0,
        "max": 120,
        "integer_only": True,
        "justification": "Months of recorded alternative history; non-negative integer.",
    },
}


def _check_numeric_and_finite(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Ensure specified columns have numeric dtypes and contain no infinities."""
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValidationError(
                f"Column '{col}' must be numeric, but has dtype '{df[col].dtype}'. "
                "String or object values are not permitted and will not be silently coerced."
            )
        col_vals = df[col].dropna().to_numpy()
        if len(col_vals) > 0 and np.isinf(col_vals).any():
            raise ValidationError(
                f"Column '{col}' contains infinite values (inf or -inf), which are not allowed."
            )


def _check_domain_bounds(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate that non-null values comply with documented domain bounds."""
    for col in columns:
        if col not in DOMAIN_BOUNDS:
            continue
        spec = DOMAIN_BOUNDS[col]
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        if spec.get("allowed_values") is not None:
            allowed = spec["allowed_values"]
            invalid = set(non_null.unique()) - allowed
            if invalid:
                raise ValidationError(
                    f"Column '{col}' contains invalid value(s) {invalid}. Allowed values are {allowed}."
                )

        min_val = spec["min"]
        max_val = spec["max"]
        if (non_null < min_val).any() or (non_null > max_val).any():
            violating = non_null[(non_null < min_val) | (non_null > max_val)].iloc[0]
            raise ValidationError(
                f"Column '{col}' value {violating} is outside the supported domain [{min_val}, {max_val}]. "
                f"Justification: {spec['justification']}"
            )

        if spec.get("integer_only", False):
            vals = non_null.to_numpy()
            if not np.all(np.isclose(vals, np.round(vals))):
                raise ValidationError(
                    f"Column '{col}' must contain integer values; found non-integer values."
                )


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """Validate the complete synthetic dataset schema before splitting or training.

    Enforces:
    - Expected DataFrame type and non-empty rows.
    - No duplicate columns.
    - Presence of all predictor columns and the binary target column.
    - Target is non-null, integer binary {0, 1}, with both classes present.
    - Synthetic applicant IDs, when present, are unique.
    - All feature columns are numeric with no infinities.
    - Strict missingness contract:
        * Traditional features (except bureau_score) must have no missing values.
        * bureau_score is NaN if and only if bureau_history_available == 0.
        * alternative_history_months must have no missing values.
    - All values adhere to DOMAIN_BOUNDS.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if len(df) == 0:
        raise ValidationError("Dataset DataFrame is empty (0 rows).")

    # Duplicate column names check
    if len(df.columns) != len(set(df.columns)):
        counts = pd.Series(df.columns).value_counts()
        duplicates = counts[counts > 1].index.tolist()
        raise ValidationError(f"Duplicate column names found in dataset: {duplicates}")

    # Check required columns
    expected_cols = set(ALL_FEATURES) | {TARGET}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValidationError(f"Missing required dataset column(s): {sorted(missing_cols)}")

    # Target validation
    if df[TARGET].isna().any():
        raise ValidationError(f"Target column '{TARGET}' must not contain missing values.")
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        raise ValidationError(f"Target column '{TARGET}' must be numeric binary, got {df[TARGET].dtype}.")
    target_vals = df[TARGET].to_numpy()
    if not np.all(np.isclose(target_vals, np.round(target_vals))):
        raise ValidationError(f"Target column '{TARGET}' must contain integer binary values.")
    unique_targets = set(df[TARGET].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValidationError(f"Target column '{TARGET}' must have binary values in {{0, 1}}; got {unique_targets}.")
    if len(unique_targets) < 2:
        raise ValidationError(f"Target column '{TARGET}' must contain both classes (0 and 1); got {unique_targets}.")

    # Applicant ID uniqueness
    if "synthetic_applicant_id" in df.columns:
        if df["synthetic_applicant_id"].duplicated().any():
            dup_id = df["synthetic_applicant_id"][df["synthetic_applicant_id"].duplicated()].iloc[0]
            raise ValidationError(f"Duplicate synthetic applicant identifier detected: '{dup_id}'.")

    # Numeric and finite checks
    _check_numeric_and_finite(df, ALL_FEATURES)

    # Missingness contract for dataset:
    # 1. Non-bureau traditional features must have no NaNs
    for col in TRADITIONAL_FEATURES:
        if col == "bureau_score":
            continue
        if df[col].isna().any():
            raise ValidationError(
                f"Traditional predictor '{col}' has {df[col].isna().sum()} missing values, "
                "which is prohibited by the dataset contract."
            )

    # 2. Bureau history relationship: bureau_score is NaN iff bureau_history_available == 0
    no_bureau = df["bureau_history_available"] == 0
    has_bureau = df["bureau_history_available"] == 1

    bureau_score_is_nan = df["bureau_score"].isna()
    if not (bureau_score_is_nan == no_bureau).all():
        inconsistent_count = (bureau_score_is_nan != no_bureau).sum()
        raise ValidationError(
            f"Dataset violates bureau contract in {inconsistent_count} rows: "
            "bureau_score must be NaN exactly when bureau_history_available == 0, and non-NaN when 1."
        )

    # 3. alternative_history_months must have no NaNs in synthetic dataset
    if df["alternative_history_months"].isna().any():
        raise ValidationError(
            "Column 'alternative_history_months' contains missing values in dataset generation."
        )

    # Domain bounds check
    _check_domain_bounds(df, ALL_FEATURES)


def validate_feature_frame(
    X: pd.DataFrame,
    expected_features: Sequence[str] | None = None,
) -> None:
    """Validate a predictor feature frame prior to model fitting or inference.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix or single-row prediction frame.
    expected_features : Sequence[str] | None
        The sequence of features expected (e.g. TRADITIONAL_FEATURES or ALL_FEATURES).
        Defaults to ALL_FEATURES if None.

    Enforces:
    - Input is a non-empty pandas DataFrame.
    - No duplicate columns.
    - Target column is NOT inadvertently passed as a predictor.
    - All expected features are present.
    - No unknown or unexpected columns are present.
    - All expected features are numeric with no infinities.
    - bureau_history_available is binary {0, 1}.
    - Bureau score relationship:
        * If bureau_history_available == 0, bureau_score must be NaN.
        * If bureau_history_available == 1, bureau_score must be non-null and in [300, 900].
    - Traditional features (except bureau_score) must not be missing.
    - Alternative features may be missing (NaN) to support cold-start / unbanked inference.
    - All non-null values comply with DOMAIN_BOUNDS.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValidationError(f"Expected a pandas DataFrame for feature inputs, got {type(X).__name__}.")
    if len(X) == 0:
        raise ValidationError("Feature DataFrame contains 0 rows.")

    if expected_features is None:
        expected_features = ALL_FEATURES

    expected_set = set(expected_features)

    # Duplicate column names check
    if len(X.columns) != len(set(X.columns)):
        counts = pd.Series(X.columns).value_counts()
        duplicates = counts[counts > 1].index.tolist()
        raise ValidationError(f"Duplicate column names found in input features: {duplicates}")

    # Accidental inclusion of target check
    if TARGET in X.columns:
        raise ValidationError(
            f"Target column '{TARGET}' cannot inadvertently be passed as an input feature predictor."
        )

    # Missing predictor columns check
    missing = expected_set - set(X.columns)
    if missing:
        raise ValidationError(f"Missing required predictor column(s): {sorted(missing)}")

    # Unexpected columns check
    unexpected = set(X.columns) - expected_set
    if unexpected:
        raise ValidationError(
            f"Unexpected column(s) not in expected feature set: {sorted(unexpected)}. "
            f"Expected: {sorted(expected_features)}"
        )

    # Numeric and finite checks
    _check_numeric_and_finite(X, expected_features)

    # Bureau relationship check
    if "bureau_history_available" in expected_set and "bureau_score" in expected_set:
        no_bureau = X["bureau_history_available"] == 0
        has_bureau = X["bureau_history_available"] == 1

        # Where bureau_history_available == 0, bureau_score must be NaN
        violating_no_bureau = no_bureau & X["bureau_score"].notna()
        if violating_no_bureau.any():
            raise ValidationError(
                "Invalid bureau history combination: bureau_score must be NaN / missing when "
                "bureau_history_available is 0."
            )

        # Where bureau_history_available == 1, bureau_score must NOT be NaN
        violating_has_bureau = has_bureau & X["bureau_score"].isna()
        if violating_has_bureau.any():
            raise ValidationError(
                "Invalid bureau history combination: bureau_score cannot be missing when "
                "bureau_history_available is 1."
            )

    # Traditional features missingness check
    for col in expected_features:
        if col in TRADITIONAL_FEATURES and col != "bureau_score":
            if X[col].isna().any():
                nan_count = X[col].isna().sum()
                raise ValidationError(
                    f"Traditional predictor '{col}' does not permit missing values (found {nan_count} NaNs)."
                )

    # Domain bounds check
    _check_domain_bounds(X, expected_features)
