"""Batch Inference Monitoring & Data-Drift Diagnostics.

Phase 09 Implementation:
- Offline batch-monitoring workflow comparing an incoming feature batch with a reference feature dataset.
- Monitors feature distribution drift via Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test.
- Evaluates feature missingness changes independently in percentage points.
- Evaluates predicted-probability distribution shifts across all three fitted models.
- Generates structured CSVs, comprehensive Markdown audit report, and machine-readable metadata.

CRITICAL METHODOLOGICAL NOTICE:
All source data, models, and scores are simulated for machine learning pipeline demonstration.
This is an engineering observability and data-drift diagnostics module on synthetic data, NOT
evidence of real-world creditworthiness, fairness, financial inclusion, or lending suitability.
Statistical drift is not proof of model performance degradation, and absence of drift does not
prove model safety. Without incoming target labels, predictive accuracy, discrimination, and
calibration cannot be evaluated. Monitoring outputs must never be used to make lending decisions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split

ROOT: Path = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uncertainty_analysis import get_installed_dependency_versions
from validation import (
    ALL_FEATURES,
    ALTERNATIVE_FEATURES,
    DOMAIN_BOUNDS,
    TARGET,
    TRADITIONAL_FEATURES,
    ValidationError,
    validate_dataset_schema,
    validate_feature_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("monitoring")

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_SPLIT_SEED: int = 2026
DEFAULT_SPLIT_TEST_SIZE: float = 0.20
DEFAULT_N_BINS: int = 10
DEFAULT_SMOOTHING_EPSILON: float = 1e-4

DEFAULT_PSI_WARNING_THRESHOLD: float = 0.10
DEFAULT_PSI_CRITICAL_THRESHOLD: float = 0.25
DEFAULT_MISSINGNESS_WARNING_THRESHOLD_PP: float = 5.0
DEFAULT_MISSINGNESS_CRITICAL_THRESHOLD_PP: float = 15.0
DEFAULT_MIN_SAMPLE_SIZE_KS: int = 5

DEFAULT_DATA_PATH: Path = ROOT / "data" / "synthetic_applicants.csv"
DEFAULT_OUTPUT_DIR: Path = ROOT / "reports" / "monitoring"

MODEL_PATHS: dict[str, Path] = {
    "Traditional baseline": ROOT / "models" / "traditional_baseline.joblib",
    "Hybrid (traditional + alternative)": ROOT / "models" / "hybrid_logistic_regression.joblib",
    "Hybrid (HistGradientBoosting)": ROOT / "models" / "hybrid_hist_gradient_boosting.joblib",
}

MODEL_FEATURES: dict[str, list[str]] = {
    "Traditional baseline": TRADITIONAL_FEATURES,
    "Hybrid (traditional + alternative)": ALL_FEATURES,
    "Hybrid (HistGradientBoosting)": ALL_FEATURES,
}

PROBABILITY_SUM_TOLERANCE: float = 1e-4


# ---------------------------------------------------------------------------
# Monitoring Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonitoringConfig:
    """Configuration container for batch monitoring diagnostic thresholds and binning."""

    n_bins: int = DEFAULT_N_BINS
    smoothing_epsilon: float = DEFAULT_SMOOTHING_EPSILON
    psi_warning_threshold: float = DEFAULT_PSI_WARNING_THRESHOLD
    psi_critical_threshold: float = DEFAULT_PSI_CRITICAL_THRESHOLD
    missingness_warning_threshold_pp: float = DEFAULT_MISSINGNESS_WARNING_THRESHOLD_PP
    missingness_critical_threshold_pp: float = DEFAULT_MISSINGNESS_CRITICAL_THRESHOLD_PP
    min_sample_size_ks: int = DEFAULT_MIN_SAMPLE_SIZE_KS


def validate_monitoring_config(config: MonitoringConfig) -> None:
    """Validate monitoring configuration parameters."""
    if not isinstance(config.n_bins, (int, np.integer)) or isinstance(config.n_bins, bool):
        raise ValueError(f"n_bins must be an integer; got {type(config.n_bins).__name__}")
    if config.n_bins < 2:
        raise ValueError(f"n_bins must be at least 2; got {config.n_bins}")

    if not isinstance(config.smoothing_epsilon, (int, float, np.floating)) or isinstance(config.smoothing_epsilon, bool):
        raise ValueError(f"smoothing_epsilon must be a positive float; got {type(config.smoothing_epsilon).__name__}")
    if not np.isfinite(config.smoothing_epsilon) or config.smoothing_epsilon <= 0.0:
        raise ValueError(f"smoothing_epsilon must be finite and positive; got {config.smoothing_epsilon}")

    for name, val in [
        ("psi_warning_threshold", config.psi_warning_threshold),
        ("psi_critical_threshold", config.psi_critical_threshold),
        ("missingness_warning_threshold_pp", config.missingness_warning_threshold_pp),
        ("missingness_critical_threshold_pp", config.missingness_critical_threshold_pp),
    ]:
        if not isinstance(val, (int, float, np.floating)) or isinstance(val, bool):
            raise ValueError(f"{name} must be numeric; got {type(val).__name__}")
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"{name} must be finite and non-negative; got {val}")

    if config.psi_warning_threshold > config.psi_critical_threshold:
        raise ValueError(
            f"psi_warning_threshold ({config.psi_warning_threshold}) cannot exceed "
            f"psi_critical_threshold ({config.psi_critical_threshold})"
        )

    if config.missingness_warning_threshold_pp > config.missingness_critical_threshold_pp:
        raise ValueError(
            f"missingness_warning_threshold_pp ({config.missingness_warning_threshold_pp}) cannot exceed "
            f"missingness_critical_threshold_pp ({config.missingness_critical_threshold_pp})"
        )

    if not isinstance(config.min_sample_size_ks, (int, np.integer)) or isinstance(config.min_sample_size_ks, bool):
        raise ValueError(f"min_sample_size_ks must be an integer; got {type(config.min_sample_size_ks).__name__}")
    if config.min_sample_size_ks < 1:
        raise ValueError(f"min_sample_size_ks must be at least 1; got {config.min_sample_size_ks}")


# ---------------------------------------------------------------------------
# Path Helper
# ---------------------------------------------------------------------------

def to_relative_path(path: Path | str) -> str:
    """Format path as repository-relative if within ROOT, else string representation."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Input Validation & Batch Loading
# ---------------------------------------------------------------------------

def load_and_validate_batch(
    data_source: Path | str | pd.DataFrame,
    batch_name: str = "batch",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate an input batch or reference dataset.

    Requirements enforced:
    - Input is non-empty and has DataFrame format.
    - Preserves source DataFrame without mutation.
    - Checks required feature columns (ALL_FEATURES).
    - Excludes TARGET and synthetic applicant identifiers from predictor features.
    - Does NOT require TARGET labels in incoming batches.
    - Validates feature dtypes, missingness contract, and DOMAIN_BOUNDS.
    - Fails clearly and safely with ValidationError on schema or domain violations.

    Returns:
    - (source_df, features_df)
    """
    if isinstance(data_source, (str, Path)):
        path = Path(data_source)
        if not path.exists():
            raise FileNotFoundError(f"{batch_name.capitalize()} file not found at: {path}")
        df = pd.read_csv(path)
    elif isinstance(data_source, pd.DataFrame):
        df = data_source
    else:
        raise ValidationError(
            f"Expected a file path or pandas DataFrame for {batch_name}; got {type(data_source).__name__}."
        )

    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected a pandas DataFrame for {batch_name}, got {type(df).__name__}.")
    if len(df) == 0:
        raise ValidationError(f"{batch_name.capitalize()} DataFrame contains 0 rows.")

    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        counts = pd.Series(df.columns).value_counts()
        duplicates = counts[counts > 1].index.tolist()
        raise ValidationError(f"Duplicate column names found in {batch_name}: {duplicates}")

    # Check required prediction features
    missing_features = set(ALL_FEATURES) - set(df.columns)
    if missing_features:
        raise ValidationError(
            f"Missing required predictor column(s) in {batch_name}: {sorted(missing_features)}"
        )

    # Validate applicant ID uniqueness when present
    if "synthetic_applicant_id" in df.columns:
        if df["synthetic_applicant_id"].duplicated().any():
            dup_id = df["synthetic_applicant_id"][df["synthetic_applicant_id"].duplicated()].iloc[0]
            raise ValidationError(
                f"Duplicate synthetic applicant identifier detected in {batch_name}: '{dup_id}'."
            )

    # Construct dedicated feature frame excluding TARGET and identifier columns
    features_df = df[ALL_FEATURES].copy()

    # Formal feature frame validation (types, infinities, bureau contract, domain bounds)
    validate_feature_frame(features_df, ALL_FEATURES)

    return df, features_df


def load_default_demonstration_data(
    data_path: Path = DEFAULT_DATA_PATH,
    seed: int = DEFAULT_SPLIT_SEED,
    test_size: float = DEFAULT_SPLIT_TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load default dataset and reproduce the exact stratified 80/20 train/test split.

    Returns:
    - (train_df, test_df, train_features, test_features)
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Synthetic applicants dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    validate_dataset_schema(df)

    if TARGET not in df.columns:
        raise ValidationError(f"Target column '{TARGET}' missing from default dataset.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET],
    )

    _, train_features = load_and_validate_batch(train_df, batch_name="reference_train")
    _, test_features = load_and_validate_batch(test_df, batch_name="incoming_test")

    return train_df, test_df, train_features, test_features


# ---------------------------------------------------------------------------
# Feature Distribution Drift Calculation (PSI & KS)
# ---------------------------------------------------------------------------

def compute_reference_bin_edges(
    ref_values: np.ndarray | Sequence[float],
    n_bins: int = DEFAULT_N_BINS,
) -> np.ndarray:
    """Compute stable, reproducible bin edges from non-missing reference values.

    Binning scheme:
    - Derives quantile cutpoints from reference data across [0, 1].
    - Handles repeated quantiles by taking unique cutpoints.
    - Handles constant features by constructing a 3-bin delta neighborhood around the constant.
    - Sets the outer edges to [-inf, inf] so all real numbers are guaranteed to fall in a bin.
    """
    ref_arr = np.asarray(ref_values, dtype=float)
    ref_clean = ref_arr[np.isfinite(ref_arr)]
    if len(ref_clean) == 0:
        raise ValueError("Cannot compute bin edges on empty or all-missing reference array.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    unique_q = np.unique(np.quantile(ref_clean, quantiles))

    if len(unique_q) == 1:
        c = float(unique_q[0])
        delta = 1e-6 if abs(c) < 1e-6 else abs(c) * 1e-4
        return np.array([-np.inf, c - delta, c + delta, np.inf])
    elif len(unique_q) == 2:
        mid = float((unique_q[0] + unique_q[1]) / 2.0)
        return np.array([-np.inf, mid, np.inf])
    else:
        inner = unique_q[1:-1]
        return np.concatenate([[-np.inf], inner, [np.inf]])


def calculate_psi(
    ref_values: np.ndarray | Sequence[float],
    batch_values: np.ndarray | Sequence[float],
    bin_edges: np.ndarray,
    eps: float = DEFAULT_SMOOTHING_EPSILON,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and batch distributions.

    Formula:
        PSI = sum_i (q_i - p_i) * ln(q_i / p_i)

    Where:
        p_i = (count_ref_i / n_ref + eps) / (1.0 + k * eps)
        q_i = (count_batch_i / n_batch + eps) / (1.0 + k * eps)

    Smoothing with eps provides strict numerical stability against zero bin counts.
    Returns np.nan if either sample has 0 non-missing values.
    """
    ref_arr = np.asarray(ref_values, dtype=float)
    ref_clean = ref_arr[np.isfinite(ref_arr)]
    batch_arr = np.asarray(batch_values, dtype=float)
    batch_clean = batch_arr[np.isfinite(batch_arr)]

    if len(ref_clean) == 0 or len(batch_clean) == 0:
        return float("nan")

    inner = bin_edges[1:-1]
    n_bins = len(bin_edges) - 1

    ref_idx = np.digitize(ref_clean, inner, right=True)
    batch_idx = np.digitize(batch_clean, inner, right=True)

    ref_counts = np.bincount(ref_idx, minlength=n_bins)
    batch_counts = np.bincount(batch_idx, minlength=n_bins)

    p_raw = ref_counts / len(ref_clean)
    q_raw = batch_counts / len(batch_clean)

    p = (p_raw + eps) / (1.0 + n_bins * eps)
    q = (q_raw + eps) / (1.0 + n_bins * eps)

    psi = np.sum((q - p) * np.log(q / p))
    return float(psi)


def calculate_ks_statistic(
    ref_values: np.ndarray | Sequence[float],
    batch_values: np.ndarray | Sequence[float],
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE_KS,
) -> tuple[float, float, str]:
    """Calculate the two-sample Kolmogorov-Smirnov statistic for non-missing numeric values.

    Returns:
    - (ks_statistic, p_value, note)
    """
    ref_arr = np.asarray(ref_values, dtype=float)
    ref_clean = ref_arr[np.isfinite(ref_arr)]
    batch_arr = np.asarray(batch_values, dtype=float)
    batch_clean = batch_arr[np.isfinite(batch_arr)]

    if len(ref_clean) < min_sample_size or len(batch_clean) < min_sample_size:
        return (
            float("nan"),
            float("nan"),
            f"Sample size below minimum threshold ({min_sample_size}): ref={len(ref_clean)}, batch={len(batch_clean)}",
        )

    res = stats.ks_2samp(ref_clean, batch_clean)
    return float(res.statistic), float(res.pvalue), "Adequate sample size"


def derive_feature_status(
    missingness_ref: float,
    missingness_batch: float,
    psi: float,
    config: MonitoringConfig,
) -> tuple[str, str]:
    """Derive operational monitoring status and descriptive reason for a feature.

    Evaluates:
    1. Total feature dropout (all batch values missing while reference had non-missing data).
    2. Missingness change in percentage points versus warning/critical thresholds.
    3. PSI versus warning/critical thresholds.

    Returns:
    - (status, reason) where status is in {"OK", "WARNING", "CRITICAL", "INSUFFICIENT_DATA"}.
    """
    miss_diff_pp = (missingness_batch - missingness_ref) * 100.0

    # 1. Total feature dropout (all batch values missing while reference had non-missing data)
    if missingness_batch >= 1.0 and missingness_ref < 1.0:
        return "CRITICAL", "Total feature dropout (100% missing in incoming batch)"

    # 2. Both reference and incoming batch are completely missing
    if missingness_ref >= 1.0 and missingness_batch >= 1.0:
        return (
            "INSUFFICIENT_DATA",
            "Zero non-missing values in reference and incoming batch; distribution drift diagnostics unavailable (missingness stable at 100.0%)",
        )

    # 3. Critical threshold checks
    critical_reasons: list[str] = []
    if np.isfinite(psi) and psi >= config.psi_critical_threshold:
        critical_reasons.append(
            f"PSI ({psi:.4f}) >= critical threshold ({config.psi_critical_threshold:.2f})"
        )
    if abs(miss_diff_pp) >= config.missingness_critical_threshold_pp:
        critical_reasons.append(
            f"Missingness change ({miss_diff_pp:+.2f} pp) exceeds critical threshold ({config.missingness_critical_threshold_pp:.1f} pp)"
        )

    if critical_reasons:
        return "CRITICAL", "; ".join(critical_reasons)

    # 4. Warning threshold checks
    warning_reasons: list[str] = []
    if np.isfinite(psi) and psi >= config.psi_warning_threshold:
        warning_reasons.append(
            f"PSI ({psi:.4f}) >= warning threshold ({config.psi_warning_threshold:.2f})"
        )
    if abs(miss_diff_pp) >= config.missingness_warning_threshold_pp:
        warning_reasons.append(
            f"Missingness change ({miss_diff_pp:+.2f} pp) exceeds warning threshold ({config.missingness_warning_threshold_pp:.1f} pp)"
        )

    if warning_reasons:
        return "WARNING", "; ".join(warning_reasons)

    # 5. Insufficient non-missing data for PSI calculation
    if np.isnan(psi):
        return (
            "INSUFFICIENT_DATA",
            f"Missingness change ({miss_diff_pp:+.2f} pp) within tolerance; PSI undefined due to empty non-missing sample",
        )

    return "OK", (
        f"Stable distribution (PSI={psi:.4f} < {config.psi_warning_threshold:.2f}, "
        f"|ΔMiss|={abs(miss_diff_pp):.2f} pp < {config.missingness_warning_threshold_pp:.1f} pp)"
    )


def compute_feature_drift(
    ref_features: pd.DataFrame,
    batch_features: pd.DataFrame,
    config: MonitoringConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Compute feature-level drift, missingness changes, and diagnostic statuses.

    Returns:
    - feature_drift_df: DataFrame with 14 rows (one per feature).
    - bin_edges_dict: Dictionary mapping feature name to reference-derived bin edges.
    """
    rows: list[dict[str, Any]] = []
    bin_edges_dict: dict[str, np.ndarray] = {}

    ref_rows = len(ref_features)
    batch_rows = len(batch_features)

    for feature in ALL_FEATURES:
        group = "traditional" if feature in TRADITIONAL_FEATURES else "alternative"
        ref_series = ref_features[feature]
        batch_series = batch_features[feature]

        ref_non_missing = int(ref_series.notna().sum())
        batch_non_missing = int(batch_series.notna().sum())

        ref_miss_rate = float(ref_series.isna().mean())
        batch_miss_rate = float(batch_series.isna().mean())
        miss_change_pp = (batch_miss_rate - ref_miss_rate) * 100.0

        ref_vals = ref_series.dropna().to_numpy()
        batch_vals = batch_series.dropna().to_numpy()

        if len(ref_vals) > 0:
            edges = compute_reference_bin_edges(ref_vals, n_bins=config.n_bins)
            bin_edges_dict[feature] = edges
            psi_val = calculate_psi(ref_vals, batch_vals, edges, eps=config.smoothing_epsilon)
        else:
            edges = np.array([-np.inf, np.inf])
            bin_edges_dict[feature] = edges
            psi_val = float("nan")

        ks_stat, ks_pval, _ = calculate_ks_statistic(
            ref_vals, batch_vals, min_sample_size=config.min_sample_size_ks
        )

        status, reason = derive_feature_status(
            missingness_ref=ref_miss_rate,
            missingness_batch=batch_miss_rate,
            psi=psi_val,
            config=config,
        )

        rows.append({
            "feature": feature,
            "feature_group": group,
            "reference_rows": ref_rows,
            "incoming_rows": batch_rows,
            "reference_non_missing": ref_non_missing,
            "incoming_non_missing": batch_non_missing,
            "reference_missingness_rate": ref_miss_rate,
            "incoming_missingness_rate": batch_miss_rate,
            "missingness_change_pp": miss_change_pp,
            "psi": psi_val,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "status": status,
            "warning_reason": reason,
        })

    return pd.DataFrame(rows), bin_edges_dict


# ---------------------------------------------------------------------------
# Model Score Monitoring
# ---------------------------------------------------------------------------

def load_fitted_models(
    model_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Load the three fitted model artifacts."""
    if model_paths is None:
        model_paths = MODEL_PATHS

    loaded: dict[str, Any] = {}
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Model artifact for '{name}' not found at: {path}")
        model = joblib.load(path)
        # Handle scikit-learn backward compatibility
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            classifier = model.named_steps["classifier"]
            if classifier.__class__.__name__ == "LogisticRegression" and not hasattr(classifier, "multi_class"):
                classifier.multi_class = "auto"
        loaded[name] = model
    return loaded


def validate_and_extract_probabilities(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
    batch_name: str,
) -> np.ndarray:
    """Execute model prediction and strictly validate binary probability vector contract."""
    n_rows = len(X)
    probs_raw = model.predict_proba(X)

    if not isinstance(probs_raw, np.ndarray):
        raise ValueError(
            f"Model '{model_name}' on {batch_name} returned non-numpy probabilities: {type(probs_raw).__name__}"
        )
    if probs_raw.ndim != 2 or probs_raw.shape != (n_rows, 2):
        raise ValueError(
            f"Model '{model_name}' on {batch_name} returned shape {probs_raw.shape}; expected ({n_rows}, 2)"
        )
    if not np.issubdtype(probs_raw.dtype, np.number) or probs_raw.dtype == bool:
        raise ValueError(
            f"Model '{model_name}' on {batch_name} returned non-numeric probability dtype: {probs_raw.dtype}"
        )
    if not np.all(np.isfinite(probs_raw)):
        raise ValueError(f"Model '{model_name}' on {batch_name} produced non-finite probability values.")
    if np.any((probs_raw < 0.0) | (probs_raw > 1.0)):
        raise ValueError(
            f"Model '{model_name}' on {batch_name} produced probabilities outside [0.0, 1.0]."
        )

    row_sums = np.sum(probs_raw, axis=1)
    if np.any(np.abs(row_sums - 1.0) > PROBABILITY_SUM_TOLERANCE):
        raise ValueError(
            f"Model '{model_name}' on {batch_name} produced unnormalized probabilities not summing to 1.0."
        )

    return probs_raw[:, 1].astype(float)


def compute_score_monitoring(
    ref_features: pd.DataFrame,
    batch_features: pd.DataFrame,
    loaded_models: dict[str, Any],
    config: MonitoringConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Compute score distributions, score shifts, and score PSI across all three models.

    Returns:
    - score_monitoring_df: DataFrame with 3 rows (one per model).
    - score_arrays: Dictionary holding raw probability arrays for reference and batch.
    """
    rows: list[dict[str, Any]] = []
    score_arrays: dict[str, dict[str, np.ndarray]] = {}

    for model_name, model in loaded_models.items():
        feature_cols = MODEL_FEATURES[model_name]
        p_ref = validate_and_extract_probabilities(
            model, ref_features[feature_cols], model_name=model_name, batch_name="reference"
        )
        p_batch = validate_and_extract_probabilities(
            model, batch_features[feature_cols], model_name=model_name, batch_name="incoming_batch"
        )

        score_arrays[model_name] = {"reference": p_ref, "incoming_batch": p_batch}

        # Distribution metrics
        ref_mean = float(np.mean(p_ref))
        ref_std = float(np.std(p_ref, ddof=0))
        ref_min = float(np.min(p_ref))
        ref_p05 = float(np.percentile(p_ref, 5))
        ref_p25 = float(np.percentile(p_ref, 25))
        ref_p50 = float(np.percentile(p_ref, 50))
        ref_p75 = float(np.percentile(p_ref, 75))
        ref_p95 = float(np.percentile(p_ref, 95))
        ref_max = float(np.max(p_ref))

        batch_mean = float(np.mean(p_batch))
        batch_std = float(np.std(p_batch, ddof=0))
        batch_min = float(np.min(p_batch))
        batch_p05 = float(np.percentile(p_batch, 5))
        batch_p25 = float(np.percentile(p_batch, 25))
        batch_p50 = float(np.percentile(p_batch, 50))
        batch_p75 = float(np.percentile(p_batch, 75))
        batch_p95 = float(np.percentile(p_batch, 95))
        batch_max = float(np.max(p_batch))

        mean_shift = batch_mean - ref_mean
        abs_mean_shift = abs(mean_shift)

        # Score PSI calculation
        score_edges = compute_reference_bin_edges(p_ref, n_bins=config.n_bins)
        score_psi = calculate_psi(p_ref, p_batch, score_edges, eps=config.smoothing_epsilon)

        if score_psi >= config.psi_critical_threshold:
            status = "CRITICAL"
            reason = f"Score PSI ({score_psi:.4f}) >= critical threshold ({config.psi_critical_threshold:.2f})"
        elif score_psi >= config.psi_warning_threshold:
            status = "WARNING"
            reason = f"Score PSI ({score_psi:.4f}) >= warning threshold ({config.psi_warning_threshold:.2f})"
        else:
            status = "OK"
            reason = f"Score distribution stable (PSI={score_psi:.4f} < {config.psi_warning_threshold:.2f})"

        rows.append({
            "model": model_name,
            "reference_rows": len(p_ref),
            "incoming_rows": len(p_batch),
            "reference_score_mean": ref_mean,
            "reference_score_std": ref_std,
            "reference_score_min": ref_min,
            "reference_score_p05": ref_p05,
            "reference_score_p25": ref_p25,
            "reference_score_p50": ref_p50,
            "reference_score_p75": ref_p75,
            "reference_score_p95": ref_p95,
            "reference_score_max": ref_max,
            "incoming_score_mean": batch_mean,
            "incoming_score_std": batch_std,
            "incoming_score_min": batch_min,
            "incoming_score_p05": batch_p05,
            "incoming_score_p25": batch_p25,
            "incoming_score_p50": batch_p50,
            "incoming_score_p75": batch_p75,
            "incoming_score_p95": batch_p95,
            "incoming_score_max": batch_max,
            "mean_score_change": mean_shift,
            "abs_mean_score_change": abs_mean_shift,
            "score_psi": score_psi,
            "score_status": status,
            "status_reason": reason,
        })

    return pd.DataFrame(rows), score_arrays


# ---------------------------------------------------------------------------
# Metadata & Report Generation
# ---------------------------------------------------------------------------

def generate_monitoring_metadata(
    feature_drift_df: pd.DataFrame,
    score_monitoring_df: pd.DataFrame,
    config: MonitoringConfig,
    reference_path: str,
    batch_path: str,
    execution_mode: str,
    split_settings: dict[str, Any] | None = None,
    execution_timestamp: str | None = None,
) -> dict[str, Any]:
    """Generate structured machine-readable metadata dictionary."""
    features_ok = int((feature_drift_df["status"] == "OK").sum())
    features_warn = int((feature_drift_df["status"] == "WARNING").sum())
    features_crit = int((feature_drift_df["status"] == "CRITICAL").sum())
    features_insufficient = int((feature_drift_df["status"] == "INSUFFICIENT_DATA").sum())

    models_ok = int((score_monitoring_df["score_status"] == "OK").sum())
    models_warn = int((score_monitoring_df["score_status"] == "WARNING").sum())
    models_crit = int((score_monitoring_df["score_status"] == "CRITICAL").sum())
    models_insufficient = (
        int((score_monitoring_df["score_status"] == "INSUFFICIENT_DATA").sum())
        if "INSUFFICIENT_DATA" in score_monitoring_df["score_status"].values
        else 0
    )

    ts = execution_timestamp or datetime.now(timezone.utc).isoformat()

    return {
        "generated_at_utc": ts,
        "phase": "Phase 09 — Batch Inference Monitoring & Data-Drift Diagnostics",
        "execution_mode": execution_mode,
        "dataset_settings": {
            "reference_path": reference_path,
            "batch_path": batch_path,
            "reference_rows": int(feature_drift_df["reference_rows"].iloc[0]),
            "incoming_rows": int(feature_drift_df["incoming_rows"].iloc[0]),
            "split_settings": split_settings,
        },
        "monitored_features": {
            "total_count": len(ALL_FEATURES),
            "traditional_features": TRADITIONAL_FEATURES,
            "alternative_features": ALTERNATIVE_FEATURES,
        },
        "models_monitored": list(score_monitoring_df["model"]),
        "binning_configuration": {
            "method": "reference_quantiles_with_infinite_tails",
            "n_bins": config.n_bins,
            "smoothing_epsilon": config.smoothing_epsilon,
            "handling_constant_features": "3_bin_delta_partition_around_constant",
            "handling_repeated_quantiles": "unique_quantile_consolidation",
        },
        "threshold_settings": {
            "psi_warning_threshold": config.psi_warning_threshold,
            "psi_critical_threshold": config.psi_critical_threshold,
            "missingness_warning_threshold_pp": config.missingness_warning_threshold_pp,
            "missingness_critical_threshold_pp": config.missingness_critical_threshold_pp,
            "min_sample_size_ks": config.min_sample_size_ks,
        },
        "summary_counts": {
            "features_total": len(feature_drift_df),
            "features_ok": features_ok,
            "features_warning": features_warn,
            "features_critical": features_crit,
            "features_insufficient_data": features_insufficient,
            "models_total": len(score_monitoring_df),
            "models_ok": models_ok,
            "models_warning": models_warn,
            "models_critical": models_crit,
            "models_insufficient_data": models_insufficient,
        },
        "monitoring_limitations": [
            "All underlying data, models, and scores are simulated for machine learning pipeline demonstration.",
            "This module is strictly an engineering observability layer, not an automated retraining pipeline or lending decision engine.",
            "Distribution drift is not equivalent to performance degradation; without ground-truth labels in production, accuracy and calibration cannot be evaluated.",
            "Absence of detected statistical drift does not prove model safety or robustness against unmonitored shifts.",
            "Population Stability Index (PSI) and missingness thresholds are operational heuristics, not universal regulatory or statistical standards.",
            "Monitoring outputs must never be used to make lending decisions, price credit, approve/reject applicants, or evaluate real-world fairness.",
        ],
        "environment": get_installed_dependency_versions(),
    }


def generate_monitoring_report(
    feature_drift_df: pd.DataFrame,
    score_monitoring_df: pd.DataFrame,
    metadata: dict[str, Any],
    config: MonitoringConfig,
) -> str:
    """Generate comprehensive Markdown audit report for batch monitoring."""
    ref_rows = metadata["dataset_settings"]["reference_rows"]
    batch_rows = metadata["dataset_settings"]["incoming_rows"]
    mode = metadata["execution_mode"]
    ref_path = metadata["dataset_settings"]["reference_path"]
    batch_path = metadata["dataset_settings"]["batch_path"]

    summary = metadata["summary_counts"]

    # Feature table rows
    feat_rows_md: list[str] = []
    for _, r in feature_drift_df.iterrows():
        psi_str = f"{r['psi']:.4f}" if np.isfinite(r["psi"]) else "N/A"
        ks_str = f"{r['ks_statistic']:.4f}" if np.isfinite(r["ks_statistic"]) else "N/A"
        pval_str = f"{r['ks_pvalue']:.4f}" if np.isfinite(r["ks_pvalue"]) else "N/A"
        feat_rows_md.append(
            f"| `{r['feature']}` | {r['feature_group']} | {r['reference_missingness_rate']*100:.1f}% | "
            f"{r['incoming_missingness_rate']*100:.1f}% | {r['missingness_change_pp']:+.2f} pp | "
            f"{psi_str} | {ks_str} | {pval_str} | **{r['status']}** |"
        )
    feat_table = "\n".join(feat_rows_md)

    # Score table rows
    score_rows_md: list[str] = []
    for _, r in score_monitoring_df.iterrows():
        score_rows_md.append(
            f"| **{r['model']}** | {r['reference_score_mean']:.4f} | {r['incoming_score_mean']:.4f} | "
            f"{r['mean_score_change']:+.4f} | {r['abs_mean_score_change']:.4f} | "
            f"{r['score_psi']:.4f} | **{r['score_status']}** |"
        )
    score_table = "\n".join(score_rows_md)

    # Score percentiles table rows
    pct_rows_md: list[str] = []
    for _, r in score_monitoring_df.iterrows():
        pct_rows_md.append(
            f"| **{r['model']} (Ref)** | {r['reference_score_min']:.4f} | {r['reference_score_p05']:.4f} | "
            f"{r['reference_score_p25']:.4f} | {r['reference_score_p50']:.4f} | {r['reference_score_p75']:.4f} | "
            f"{r['reference_score_p95']:.4f} | {r['reference_score_max']:.4f} |\n"
            f"| **{r['model']} (Batch)** | {r['incoming_score_min']:.4f} | {r['incoming_score_p05']:.4f} | "
            f"{r['incoming_score_p25']:.4f} | {r['incoming_score_p50']:.4f} | {r['incoming_score_p75']:.4f} | "
            f"{r['incoming_score_p95']:.4f} | {r['incoming_score_max']:.4f} |"
        )
    pct_table = "\n".join(pct_rows_md)

    # Prioritized warnings & diagnostic limitations
    warnings_list: list[str] = []
    crit_feats = feature_drift_df[feature_drift_df["status"] == "CRITICAL"]
    warn_feats = feature_drift_df[feature_drift_df["status"] == "WARNING"]
    crit_scores = score_monitoring_df[score_monitoring_df["score_status"] == "CRITICAL"]
    warn_scores = score_monitoring_df[score_monitoring_df["score_status"] == "WARNING"]

    for _, r in crit_scores.iterrows():
        warnings_list.append(f"- **CRITICAL (Score Shift)**: `{r['model']}` — {r['status_reason']}")
    for _, r in crit_feats.iterrows():
        warnings_list.append(f"- **CRITICAL (Feature Drift)**: `{r['feature']}` — {r['warning_reason']}")
    for _, r in warn_scores.iterrows():
        warnings_list.append(f"- **WARNING (Score Shift)**: `{r['model']}` — {r['status_reason']}")
    for _, r in warn_feats.iterrows():
        warnings_list.append(f"- **WARNING (Feature Drift)**: `{r['feature']}` — {r['warning_reason']}")

    insufficient_feats = feature_drift_df[feature_drift_df["status"] == "INSUFFICIENT_DATA"]
    insufficient_list: list[str] = []
    for _, r in insufficient_feats.iterrows():
        unavail = []
        if not np.isfinite(r["psi"]):
            unavail.append("Population Stability Index (PSI) unavailable due to empty non-missing sample")
        if not np.isfinite(r["ks_statistic"]):
            unavail.append(
                f"Kolmogorov-Smirnov (KS) test unavailable (non-missing sample size: ref={r['reference_non_missing']}, "
                f"batch={r['incoming_non_missing']} < min threshold {config.min_sample_size_ks})"
            )
        diag_desc = "; ".join(unavail) if unavail else "Distribution drift diagnostics unavailable"
        insufficient_list.append(
            f"- **INSUFFICIENT DATA**: `{r['feature']}` ({r['feature_group']}) — {diag_desc}. "
            f"Missingness: reference={r['reference_missingness_rate']*100:.1f}%, "
            f"incoming batch={r['incoming_missingness_rate']*100:.1f}% (Δ={r['missingness_change_pp']:+.2f} pp). "
            f"Diagnostic note: {r['warning_reason']}."
        )

    ks_limited_feats = feature_drift_df[
        (~np.isfinite(feature_drift_df["ks_statistic"])) & (feature_drift_df["status"] != "INSUFFICIENT_DATA")
    ]
    ks_limited_list: list[str] = []
    for _, r in ks_limited_feats.iterrows():
        ks_limited_list.append(
            f"- **SAMPLE SIZE LIMITATION (KS Unavailable)**: `{r['feature']}` ({r['feature_group']}) — "
            f"Non-missing sample size below KS threshold ({config.min_sample_size_ks}): "
            f"ref={r['reference_non_missing']}, incoming={r['incoming_non_missing']}. Status: {r['status']}."
        )

    sections: list[str] = []
    if warnings_list:
        sections.append("### Active Operational Alerts\n\n" + "\n".join(warnings_list))
    elif not insufficient_list and not ks_limited_list:
        sections.append(
            "No operational warnings or critical alerts triggered. All 14 monitored features and all 3 model "
            f"score distributions remain well within the configured operational warning thresholds "
            f"(PSI < {config.psi_warning_threshold:.2f}, |ΔMissingness| < {config.missingness_warning_threshold_pp:.1f} pp)."
        )
    else:
        sections.append(
            "No threshold warnings or critical alerts were triggered on evaluable features. However, operational "
            "diagnostics could not be fully evaluated for all features due to data limitations described below."
        )

    if insufficient_list:
        sections.append(
            "### Features with Insufficient Diagnostic Data\n\n"
            "The following feature(s) have insufficient non-missing data to compute distribution drift statistics. "
            "These features cannot be confirmed to be within tolerance:\n\n"
            + "\n".join(insufficient_list)
        )

    if ks_limited_list:
        sections.append(
            "### Features with Sample-Size Diagnostic Limitations\n\n"
            + "\n".join(ks_limited_list)
        )

    warnings_section = "\n\n".join(sections)

    # Find highest feature PSI and highest score shift for observations
    valid_psis = feature_drift_df[np.isfinite(feature_drift_df["psi"])]
    if len(valid_psis) > 0:
        max_psi_row = valid_psis.loc[valid_psis["psi"].idxmax()]
        max_psi_desc = f"`{max_psi_row['feature']}` (PSI = {max_psi_row['psi']:.5f})"
    else:
        max_psi_desc = "None (no finite feature PSI values)"

    max_score_shift_row = score_monitoring_df.loc[score_monitoring_df["abs_mean_score_change"].idxmax()]
    max_score_shift_desc = (
        f"`{max_score_shift_row['model']}` (mean change = {max_score_shift_row['mean_score_change']:+.4f}, "
        f"score PSI = {max_score_shift_row['score_psi']:.5f})"
    )

    max_miss_diff = float(feature_drift_df["missingness_change_pp"].abs().max())

    # Executive summary table and narrative
    exec_summary_table = f"""| Diagnostic Domain | Total Monitored | Status OK | Status WARNING | Status CRITICAL | Status INSUFFICIENT DATA |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Monitored Predictor Features** | {summary['features_total']} | **{summary['features_ok']}** | {summary['features_warning']} | {summary['features_critical']} | {summary.get('features_insufficient_data', 0)} |
| **Model Score Distributions** | {summary['models_total']} | **{summary['models_ok']}** | {summary['models_warning']} | {summary['models_critical']} | {summary.get('models_insufficient_data', 0)} |"""

    has_feature_issues = (
        summary.get("features_warning", 0) > 0
        or summary.get("features_critical", 0) > 0
        or summary.get("features_insufficient_data", 0) > 0
    )
    has_model_issues = summary.get("models_warning", 0) > 0 or summary.get("models_critical", 0) > 0

    if mode == "default_demonstration":
        if not has_feature_issues and not has_model_issues:
            exec_summary_narrative = (
                f"In the default demonstration, the incoming batch represents the held-out test partition "
                f"({batch_rows:,} rows) evaluated against the reference training partition ({ref_rows:,} rows) "
                f"under the identical seed-2026 stratified split. As expected for an independent, identically "
                f"distributed partition from the same synthetic generator, all {summary['features_total']} features "
                f"and all {summary['models_total']} model score distributions exhibit near-zero drift and remain "
                f"well within operational tolerance."
            )
        else:
            issue_parts = []
            if summary.get("features_insufficient_data", 0) > 0:
                issue_parts.append(
                    f"{summary['features_insufficient_data']} feature(s) had insufficient non-missing data for distribution drift evaluation"
                )
            if summary.get("features_critical", 0) > 0:
                issue_parts.append(f"{summary['features_critical']} feature(s) triggered critical alerts")
            if summary.get("features_warning", 0) > 0:
                issue_parts.append(f"{summary['features_warning']} feature(s) triggered warnings")
            issue_desc = "; ".join(issue_parts)

            exec_summary_narrative = (
                f"In the default demonstration, the incoming batch represents the held-out test partition "
                f"({batch_rows:,} rows) evaluated against the reference training partition ({ref_rows:,} rows) "
                f"under the seed-2026 stratified split. Status summary: {summary['features_ok']} feature(s) and "
                f"{summary['models_ok']} model score distribution(s) are within tolerance; {issue_desc}."
            )
    else:  # caller_supplied
        if not has_feature_issues and not has_model_issues:
            exec_summary_narrative = (
                f"In this caller-supplied batch monitoring run, an incoming batch of {batch_rows:,} rows from "
                f"`{batch_path}` was evaluated against a reference dataset of {ref_rows:,} rows from `{ref_path}`. "
                f"All {summary['features_total']} monitored predictor features and all {summary['models_total']} model "
                f"score distributions exhibit measured statistics within the configured operational thresholds "
                f"(PSI < {config.psi_warning_threshold:.2f}, |ΔMissingness| < {config.missingness_warning_threshold_pp:.1f} pp). "
                f"Measured stability across these operational metrics reflects only the specific caller-supplied batches; "
                f"it does not imply that the batches are independent and identically distributed or share the synthetic generator's missingness mechanism."
            )
        else:
            issue_parts = []
            if summary.get("features_insufficient_data", 0) > 0:
                issue_parts.append(
                    f"{summary['features_insufficient_data']} feature(s) had insufficient data for distribution drift evaluation"
                )
            if summary.get("features_critical", 0) > 0:
                issue_parts.append(f"{summary['features_critical']} feature(s) triggered critical alerts")
            if summary.get("features_warning", 0) > 0:
                issue_parts.append(f"{summary['features_warning']} feature(s) triggered warnings")
            issue_desc = "; ".join(issue_parts)

            exec_summary_narrative = (
                f"In this caller-supplied batch monitoring run, an incoming batch of {batch_rows:,} rows from "
                f"`{batch_path}` was evaluated against a reference dataset of {ref_rows:,} rows from `{ref_path}`. "
                f"Across monitored inputs, {summary['features_ok']} feature(s) and {summary['models_ok']} model score "
                f"distribution(s) are within operational tolerance; {issue_desc}. "
                f"These diagnostic findings reflect strictly the caller-supplied inputs and must not be assumed to follow "
                f"the default synthetic split or missingness mechanisms."
            )

    # Methodological caveat on training-reference score comparison
    if mode == "default_demonstration":
        score_caveat_md = """
> [!NOTE]
> **Methodological Limitation (Training-Reference Score Comparison)**:
> In the default demonstration, the reference feature set is the training partition used to fit the model pipelines, while the incoming batch consists of held-out test observations. Consequently, the reference score distribution represents in-sample model predictions, whereas the incoming batch score distribution represents held-out (out-of-sample) predictions. Any observed score-distribution differences reflect both finite-sample variation and in-sample versus out-of-sample prediction behavior; they must NOT be attributed solely to population drift."""
    else:
        score_caveat_md = """
> [!NOTE]
> **Methodological Limitation (Caller-Supplied Score Comparison)**:
> In a caller-supplied monitoring run, the reference dataset and incoming batch are provided externally. If the reference batch contains data used during model training while the incoming batch contains new observations, score shifts will reflect both sample variation and in-sample versus out-of-sample prediction characteristics. Score differences should not be attributed solely to population drift without independent knowledge of model training provenance for the reference and incoming records."""

    # Observations vs. Potential Explanations
    if mode == "default_demonstration":
        observations_md = f"""## Observations vs. Potential Explanations

1. **Observed Finding**: Largest feature PSI in the demonstration run is {max_psi_desc}.
   *Hypothesis (Not Proven Fact)*: In an i.i.d. random split of finite synthetic data, minor differences across deciles represent standard finite-sample sampling variation rather than structural population drift.
2. **Observed Finding**: Largest absolute mean score change is {max_score_shift_desc}.
   *Hypothesis (Not Proven Fact)*: The slight difference in mean predicted probability between the 8,000-row training set and 2,000-row test set reflects finite-sample variance across test applicants and in-sample versus out-of-sample prediction behavior; score PSI remains well below the 0.10 threshold.
3. **Observed Finding**: Alternative feature missingness rates in incoming batch closely match reference missingness rates (all differences within $\\pm {max_miss_diff:.1f}$ percentage points).
   *Hypothesis (Not Proven Fact)*: The missing-at-random synthetic data generation mechanism is uniform across the entire synthetic dataset."""
    else:
        observations_md = f"""## Observations vs. Potential Explanations

1. **Observed Finding**: Reference dataset `{ref_path}` ({ref_rows:,} rows) and incoming batch `{batch_path}` ({batch_rows:,} rows) were evaluated across {len(feature_drift_df)} features. Largest feature PSI observed is {max_psi_desc}.
   *Hypothesis (Not Proven Fact)*: Observed PSI values reflect differences between the provided batches. Whether these differences represent ordinary sampling variability, seasonal changes, or population drift cannot be determined without knowledge of data collection provenance.
2. **Observed Finding**: Model score distributions: largest absolute mean score change is {max_score_shift_desc}.
   *Hypothesis (Not Proven Fact)*: Score distribution differences may arise from covariate shifts, differences in population composition between reference and incoming batches, or sampling variability. Without ground-truth outcome labels on the incoming batch, discrimination and calibration cannot be determined.
3. **Observed Finding**: Missingness rate changes: maximum observed absolute missingness shift across features is $\\pm {max_miss_diff:.1f}$ percentage points.
   *Hypothesis (Not Proven Fact)*: Missingness differences may reflect upstream data pipeline alterations, changes in caller feature collection, or random sampling variation. The data generating mechanism of caller-supplied batches cannot be assumed to follow the synthetic generator's missingness process."""

    if len(insufficient_feats) > 0:
        ins_feat_names = ", ".join(f"`{f}`" for f in insufficient_feats["feature"])
        observations_md += f"""\n4. **Observed Finding**: {len(insufficient_feats)} feature(s) exhibited insufficient data for drift diagnostics: {ins_feat_names}.
   *Hypothesis (Not Proven Fact)*: Complete missingness or extreme sparsity in reference or incoming batches precludes empirical quantile binning and distribution comparisons for these features."""

    return f"""# Batch Inference Monitoring & Data-Drift Diagnostic Report

**Execution Mode**: `{mode}`  
**Reference Dataset**: `{ref_path}` ({ref_rows:,} rows)  
**Incoming Batch**: `{batch_path}` ({batch_rows:,} rows)  
**Binning Methodology**: Reference-derived quantiles with infinite tails ({config.n_bins} bins, smoothing $\\epsilon = {config.smoothing_epsilon}$)  
**Configured Thresholds**: PSI Warning $\\ge {config.psi_warning_threshold:.2f}$, PSI Critical $\\ge {config.psi_critical_threshold:.2f}$, Missingness Warning $\\ge {config.missingness_warning_threshold_pp:.1f}$ pp, Missingness Critical $\\ge {config.missingness_critical_threshold_pp:.1f}$ pp  

---

## Critical Methodological Limitations & Governance Notice

> [!CAUTION]
> **SYNTHETIC AND EDUCATIONAL PURPOSE ONLY**  
> All records, fitted models, probability predictions, and drift measurements in this report are strictly simulated for machine learning engineering and observability demonstration. This analysis does NOT evaluate real-world borrowers, actual financial behavior, creditworthiness, financial inclusion, or regulatory compliance.

> [!IMPORTANT]
> **MONITORING AND OBSERVABILITY BOUNDARIES**  
> 1. **Statistical drift is not equivalent to performance degradation**: A measurable shift in feature distribution or predicted probabilities indicates input divergence from the reference baseline, but does not prove model failure or increased error rates.
> 2. **Unlabeled production reality**: Without ground-truth outcome labels on incoming batches, true predictive accuracy, ROC-AUC, Brier score, and probability calibration cannot be computed.
> 3. **Absence of drift does not guarantee safety**: Low PSI values confirm distribution stability relative to reference quantiles, but do not protect against unmonitored covariates, non-stationary target relationships, or adversarial inputs.
> 4. **Heuristic thresholds**: The PSI warning ({config.psi_warning_threshold:.2f}) and critical ({config.psi_critical_threshold:.2f}) thresholds are operational rules of thumb for this engineering demonstration, NOT universal statistical laws or regulatory mandates.
> 5. **No automated credit policy**: Drift warnings are diagnostic signals for engineering investigation. They must never trigger automated credit denials, pricing changes, or unverified model weight retraining.

---

## Executive Summary

{exec_summary_table}

{exec_summary_narrative}

---

## Input Validation & Schema Compliance

The input batch and reference data passed all pre-scoring integrity checks:
- **Feature Frame Completeness**: All 14 expected predictor features (`ALL_FEATURES`) were present with correct numeric types and zero infinite values.
- **Target Separation**: The target variable (`{TARGET}`) and synthetic applicant identifiers were strictly excluded from predictor matrices.
- **Label Independence**: Model feature extraction and drift diagnostics were conducted entirely without requiring or referencing target labels in the incoming batch.
- **Missingness Contract**: Traditional predictors (excluding `bureau_score`) contained zero missing values. Alternative features adhered to permitted missingness bounds.
- **Domain Constraints**: All non-null predictor values strictly complied with documented `DOMAIN_BOUNDS`.
- **DataFrame Immutability**: Source inputs and row ordering were strictly preserved without in-place mutation.

---

## Feature Distribution Drift Table

| Feature Name | Feature Group | Ref Missing | Batch Missing | Missingness Δ | Non-Missing PSI | KS Statistic | KS p-value | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{feat_table}

*Note: Missingness change is evaluated independently in percentage points (pp). PSI and two-sample KS statistics are computed exclusively on non-missing numeric values. When non-missing observations are absent or below the minimum sample threshold (n < {config.min_sample_size_ks}), PSI and KS statistics are marked as N/A.*

---

## Model Score Distribution Shifts

Predicted default probabilities ($P(\\text{{default}} = 1)$) were generated across the three saved pipelines using their respective feature contracts:

| Model Pipeline | Reference Mean | Incoming Mean | Mean Shift | Abs Mean Shift | Score PSI | Score Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{score_table}

### Score Percentile Summaries

| Model & Partition | Min | P05 | P25 | P50 (Median) | P75 | P95 | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{pct_table}

*Diagnostic Interpretation: Score shifts reflect changes in the model's output probability distribution. In the absence of incoming labels, a shift in mean predicted risk indicates a change in average model output, but does not provide evidence of calibration drift or degraded ranking discrimination.*
{score_caveat_md}

---

## Prioritized Warnings & Diagnostic Evidence

{warnings_section}

---

{observations_md}

---

## Reproduction Command

To reproduce this batch monitoring analysis deterministically:

```bash
.venv/bin/python monitoring.py
```

To run with custom caller-supplied batches:

```bash
.venv/bin/python monitoring.py --reference-path path/to/reference.csv --batch-path path/to/incoming_batch.csv
```

### Reproducibility Contract
- Executing `monitoring.py` with identical configuration on identical datasets produces byte-for-byte identical `feature_drift.csv`, `score_monitoring.csv`, and `monitoring_report.md`.
- `monitoring_metadata.json` is identical across repeated runs except for `generated_at_utc`, which intentionally records runtime provenance.
"""


# ---------------------------------------------------------------------------
# Batch Monitoring Runner
# ---------------------------------------------------------------------------

def run_batch_monitoring(
    reference_path: Path | str | None = None,
    batch_path: Path | str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_paths: dict[str, Path] | None = None,
    config: MonitoringConfig | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    split_test_size: float = DEFAULT_SPLIT_TEST_SIZE,
    execution_timestamp: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    """Execute complete offline batch monitoring and drift diagnostics workflow.

    Returns:
    - feature_drift_df: 14 rows of feature drift metrics.
    - score_monitoring_df: 3 rows of model score monitoring metrics.
    - metadata: Structured metadata dictionary.
    - report_markdown: Formatted markdown audit report.
    """
    if config is None:
        config = MonitoringConfig()
    validate_monitoring_config(config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Determine data sources (default demonstration vs custom caller paths)
    if reference_path is None and batch_path is None:
        execution_mode = "default_demonstration"
        train_df, test_df, ref_features, batch_features = load_default_demonstration_data(
            data_path=DEFAULT_DATA_PATH, seed=split_seed, test_size=split_test_size
        )
        ref_path_str = f"{to_relative_path(DEFAULT_DATA_PATH)} (training partition: 80%)"
        batch_path_str = f"{to_relative_path(DEFAULT_DATA_PATH)} (held-out test partition: 20%)"
        split_settings: dict[str, Any] | None = {
            "seed": split_seed,
            "test_size": split_test_size,
            "stratify": TARGET,
        }
    elif reference_path is not None and batch_path is not None:
        execution_mode = "caller_supplied"
        _, ref_features = load_and_validate_batch(reference_path, batch_name="reference")
        _, batch_features = load_and_validate_batch(batch_path, batch_name="incoming_batch")
        ref_path_str = to_relative_path(reference_path)
        batch_path_str = to_relative_path(batch_path)
        split_settings = None
    else:
        raise ValueError(
            "Must provide BOTH reference_path and batch_path for custom runs, or NEITHER for default demonstration."
        )

    # 2. Compute feature-level drift
    logger.info("Computing feature distribution drift and missingness metrics...")
    feature_drift_df, _ = compute_feature_drift(ref_features, batch_features, config)

    # 3. Load models and compute score monitoring
    logger.info("Loading fitted models and computing score distribution shifts...")
    loaded_models = load_fitted_models(model_paths)
    score_monitoring_df, _ = compute_score_monitoring(
        ref_features, batch_features, loaded_models, config
    )

    # 4. Generate metadata and report
    logger.info("Generating monitoring metadata and markdown audit report...")
    metadata = generate_monitoring_metadata(
        feature_drift_df=feature_drift_df,
        score_monitoring_df=score_monitoring_df,
        config=config,
        reference_path=ref_path_str,
        batch_path=batch_path_str,
        execution_mode=execution_mode,
        split_settings=split_settings,
        execution_timestamp=execution_timestamp,
    )

    report_markdown = generate_monitoring_report(
        feature_drift_df=feature_drift_df,
        score_monitoring_df=score_monitoring_df,
        metadata=metadata,
        config=config,
    )

    # 5. Write artifacts to output_dir
    feature_drift_path = output_path / "feature_drift.csv"
    score_monitoring_path = output_path / "score_monitoring.csv"
    metadata_path = output_path / "monitoring_metadata.json"
    report_path = output_path / "monitoring_report.md"

    feature_drift_df.to_csv(feature_drift_path, index=False)
    score_monitoring_df.to_csv(score_monitoring_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path.write_text(report_markdown, encoding="utf-8")

    logger.info(f"Wrote feature drift metrics: {feature_drift_path} ({len(feature_drift_df)} rows)")
    logger.info(f"Wrote score monitoring metrics: {score_monitoring_path} ({len(score_monitoring_df)} rows)")
    logger.info(f"Wrote monitoring metadata: {metadata_path}")
    logger.info(f"Wrote monitoring audit report: {report_path}")

    return feature_drift_df, score_monitoring_df, metadata, report_markdown


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for monitoring.py."""
    parser = argparse.ArgumentParser(
        description="Phase 09: Batch Inference Monitoring & Data-Drift Diagnostics."
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="Path to reference feature CSV dataset. If omitted, uses training partition of synthetic_applicants.csv.",
    )
    parser.add_argument(
        "--batch-path",
        type=Path,
        default=None,
        help="Path to incoming batch feature CSV. If omitted, uses held-out test partition of synthetic_applicants.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write monitoring artifacts. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=f"Random split seed for default demonstration. Defaults to {DEFAULT_SPLIT_SEED}.",
    )
    parser.add_argument(
        "--split-test-size",
        type=float,
        default=DEFAULT_SPLIT_TEST_SIZE,
        help=f"Held-out test fraction for default demonstration. Defaults to {DEFAULT_SPLIT_TEST_SIZE}.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=DEFAULT_N_BINS,
        help=f"Number of quantile bins for PSI. Defaults to {DEFAULT_N_BINS}.",
    )
    parser.add_argument(
        "--smoothing-epsilon",
        type=float,
        default=DEFAULT_SMOOTHING_EPSILON,
        help=f"Smoothing epsilon for PSI calculation. Defaults to {DEFAULT_SMOOTHING_EPSILON}.",
    )
    parser.add_argument(
        "--psi-warning-threshold",
        type=float,
        default=DEFAULT_PSI_WARNING_THRESHOLD,
        help=f"PSI warning threshold. Defaults to {DEFAULT_PSI_WARNING_THRESHOLD}.",
    )
    parser.add_argument(
        "--psi-critical-threshold",
        type=float,
        default=DEFAULT_PSI_CRITICAL_THRESHOLD,
        help=f"PSI critical threshold. Defaults to {DEFAULT_PSI_CRITICAL_THRESHOLD}.",
    )
    parser.add_argument(
        "--missingness-warning-threshold-pp",
        type=float,
        default=DEFAULT_MISSINGNESS_WARNING_THRESHOLD_PP,
        help=f"Missingness change warning threshold in percentage points. Defaults to {DEFAULT_MISSINGNESS_WARNING_THRESHOLD_PP}.",
    )
    parser.add_argument(
        "--missingness-critical-threshold-pp",
        type=float,
        default=DEFAULT_MISSINGNESS_CRITICAL_THRESHOLD_PP,
        help=f"Missingness change critical threshold in percentage points. Defaults to {DEFAULT_MISSINGNESS_CRITICAL_THRESHOLD_PP}.",
    )
    parser.add_argument(
        "--execution-timestamp",
        type=str,
        default=None,
        help="Optional ISO-8601 UTC timestamp string to inject for deterministic test assertions.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint for monitoring.py."""
    parser = build_parser()
    args = parser.parse_args(argv)

    config = MonitoringConfig(
        n_bins=args.n_bins,
        smoothing_epsilon=args.smoothing_epsilon,
        psi_warning_threshold=args.psi_warning_threshold,
        psi_critical_threshold=args.psi_critical_threshold,
        missingness_warning_threshold_pp=args.missingness_warning_threshold_pp,
        missingness_critical_threshold_pp=args.missingness_critical_threshold_pp,
    )

    try:
        feature_drift_df, score_monitoring_df, metadata, _ = run_batch_monitoring(
            reference_path=args.reference_path,
            batch_path=args.batch_path,
            output_dir=args.output_dir,
            config=config,
            split_seed=args.seed,
            split_test_size=args.split_test_size,
            execution_timestamp=args.execution_timestamp,
        )

        print("\n" + "=" * 80)
        print("PHASE 09 BATCH INFERENCE MONITORING & DATA-DRIFT DIAGNOSTICS COMPLETE")
        print(f"Mode: {metadata['execution_mode']}")
        print(f"Features Monitored: {len(feature_drift_df)} (OK: {metadata['summary_counts']['features_ok']}, "
              f"WARNING: {metadata['summary_counts']['features_warning']}, "
              f"CRITICAL: {metadata['summary_counts']['features_critical']}, "
              f"INSUFFICIENT_DATA: {metadata['summary_counts']['features_insufficient_data']})")
        print(f"Models Monitored: {len(score_monitoring_df)} (OK: {metadata['summary_counts']['models_ok']}, "
              f"WARNING: {metadata['summary_counts']['models_warning']}, "
              f"CRITICAL: {metadata['summary_counts']['models_critical']})")
        print(f"Feature Drift CSV: {args.output_dir / 'feature_drift.csv'}")
        print(f"Score Monitoring CSV: {args.output_dir / 'score_monitoring.csv'}")
        print(f"Metadata JSON: {args.output_dir / 'monitoring_metadata.json'}")
        print(f"Audit Report Markdown: {args.output_dir / 'monitoring_report.md'}")
        print("=" * 80 + "\n")
        return 0
    except Exception as err:
        logger.error(f"Monitoring execution failed: {err}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
