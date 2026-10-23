"""Bootstrap uncertainty quantification and paired model comparison for alternative credit scoring.

Phase 07 implementation:
- Quantifies test-metric variability using non-parametric bootstrap resampling on fixed fitted models.
- Performs paired model comparisons evaluating metric differences on identical resampled observation indices.
- Evaluates two cohorts: overall held-out test set and no-bureau-history subgroup.
- Generates reproducible CSV intervals, run metadata JSON, and a comprehensive audit report.

EDUCATIONAL AND SYNTHETIC USE ONLY:
All data, models, and scores are simulated for machine learning pipeline demonstration.
They do not represent real-world borrowers or validated underwriting policies.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import train
from train import (
    calculate_gini,
    expected_calibration_error,
    kolmogorov_smirnov_statistic,
)
from validation import (
    ALL_FEATURES,
    TARGET,
    TRADITIONAL_FEATURES,
    validate_dataset_schema,
)

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_BOOTSTRAP: int = 1000
DEFAULT_SEED: int = 2027
DEFAULT_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_TEST_SIZE: float = 0.20
DEFAULT_SPLIT_SEED: int = 2026
DEFAULT_N_BINS: int = 10

ROOT: Path = Path(__file__).resolve().parent
DATA_PATH: Path = ROOT / "data" / "synthetic_applicants.csv"
OUTPUT_DIR: Path = ROOT / "reports" / "uncertainty"

MODEL_PATHS: dict[str, Path] = {
    "Traditional baseline": ROOT / "models" / "traditional_baseline.joblib",
    "Hybrid (traditional + alternative)": ROOT / "models" / "hybrid_logistic_regression.joblib",
    "Hybrid (HistGradientBoosting)": ROOT / "models" / "hybrid_hist_gradient_boosting.joblib",
}

PAIRED_COMPARISONS: list[tuple[str, str]] = [
    ("Hybrid (traditional + alternative)", "Traditional baseline"),
    ("Hybrid (HistGradientBoosting)", "Traditional baseline"),
    ("Hybrid (HistGradientBoosting)", "Hybrid (traditional + alternative)"),
]

METRIC_NAMES: list[str] = [
    "roc_auc",
    "gini_coefficient",
    "ks_statistic",
    "average_precision",
    "brier_score",
    "log_loss",
    "ece_10_bins",
]

# Higher is better for discrimination metrics; lower is better for loss/calibration metrics
METRIC_DIRECTIONS: dict[str, str] = {
    "roc_auc": "higher",
    "gini_coefficient": "higher",
    "ks_statistic": "higher",
    "average_precision": "higher",
    "brier_score": "lower",
    "log_loss": "lower",
    "ece_10_bins": "lower",
}


# ---------------------------------------------------------------------------
# Argument Validation & Environment Metadata
# ---------------------------------------------------------------------------

def validate_arguments(n_bootstrap: int, seed: int, confidence_level: float) -> None:
    """Validate bootstrap command-line and configuration parameters."""
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)):
        raise ValueError(
            f"n_bootstrap must be an integer, got {type(n_bootstrap).__name__}: {n_bootstrap!r}"
        )
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be a positive integer, got {n_bootstrap}")

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(
            f"seed must be an integer, got {type(seed).__name__}: {seed!r}"
        )
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    if isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float, np.floating)):
        raise ValueError(
            f"confidence_level must be numeric, got {type(confidence_level).__name__}: {confidence_level!r}"
        )
    if not np.isfinite(confidence_level):
        raise ValueError(f"confidence_level must be finite, got {confidence_level}")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError(
            f"confidence_level must be in open interval (0.0, 1.0), got {confidence_level}"
        )


def get_installed_dependency_versions() -> dict[str, str]:
    """Retrieve runtime versions of key modeling and analysis dependencies dynamically."""
    packages = {
        "numpy_version": "numpy",
        "pandas_version": "pandas",
        "scikit_learn_version": "scikit-learn",
        "matplotlib_version": "matplotlib",
    }
    versions: dict[str, str] = {
        "python_version": sys.version,
    }
    for key, pkg_name in packages.items():
        try:
            versions[key] = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "not_installed"
        except Exception as exc:  # pragma: no cover
            versions[key] = f"error: {exc}"
    return versions


# ---------------------------------------------------------------------------
# Core Metric Computation & Percentile Interval Helpers
# ---------------------------------------------------------------------------

def compute_model_metrics(
    y_true: np.ndarray | Sequence[int],
    probabilities: np.ndarray | Sequence[float],
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, float]:
    """Compute complete discrimination and calibration evaluation metrics reusing train.py definitions."""
    y = np.asarray(y_true)
    probs = np.asarray(probabilities, dtype=float)

    if y.ndim != 1 or probs.ndim != 1:
        raise ValueError("y_true and probabilities must be 1-dimensional.")
    if len(y) != len(probs):
        raise ValueError(f"Length mismatch: y_true ({len(y)}) vs probabilities ({len(probs)}).")
    if len(y) < 2:
        raise ValueError("At least 2 samples are required.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities must be finite numbers (no NaN or Inf).")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Probabilities must be bounded in [0.0, 1.0].")

    unique_labels = np.unique(y)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels must be binary in {{0, 1}}; got {unique_labels}.")
    if len(unique_labels) < 2:
        raise ValueError("Both classes (0 and 1) must be present to evaluate classification metrics.")

    auc = float(roc_auc_score(y, probs))
    gini = calculate_gini(auc)
    ks = kolmogorov_smirnov_statistic(y, probs)
    ap = float(average_precision_score(y, probs))
    brier = float(brier_score_loss(y, probs))
    ll = float(log_loss(y, np.clip(probs, 1e-8, 1.0 - 1e-8)))
    ece = expected_calibration_error(y, probs, n_bins=n_bins)

    return {
        "roc_auc": auc,
        "gini_coefficient": gini,
        "ks_statistic": ks,
        "average_precision": ap,
        "brier_score": brier,
        "log_loss": ll,
        f"ece_{n_bins}_bins": ece,
    }


def compute_percentile_interval(
    values: np.ndarray | Sequence[float],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Compute two-sided percentile interval for an array of bootstrap replicate values."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("values must be a non-empty 1-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Bootstrap replicate values must be finite numbers.")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float, np.floating)):
        raise ValueError(f"confidence_level must be numeric, got {type(confidence_level).__name__}")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError(f"confidence_level must be in open interval (0, 1), got {confidence_level}")

    alpha = (1.0 - float(confidence_level)) / 2.0
    lower_pct = 100.0 * alpha
    upper_pct = 100.0 * (1.0 - alpha)

    ci_lower = float(np.percentile(arr, lower_pct))
    ci_upper = float(np.percentile(arr, upper_pct))
    if ci_lower > ci_upper:
        ci_lower, ci_upper = ci_upper, ci_lower

    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Cohort Bootstrap Engine
# ---------------------------------------------------------------------------

def bootstrap_cohort_analysis(
    cohort_name: str,
    y_true: np.ndarray | Sequence[int],
    model_predictions: dict[str, np.ndarray | Sequence[float]],
    paired_comparisons: list[tuple[str, str]] = PAIRED_COMPARISONS,
    metric_names: list[str] = METRIC_NAMES,
    metric_directions: dict[str, str] = METRIC_DIRECTIONS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_bins: int = DEFAULT_N_BINS,
    min_valid_replicates: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Execute bootstrap uncertainty quantification and paired comparisons for a single cohort.

    Guarantees:
    - Resamples the exact same observation indices across all models in each replicate.
    - Computes point estimates from original, unresampled cohort.
    - Explicitly catches single-class or non-evaluable replicates.
    - Formulates two-sided percentile intervals.
    """
    validate_arguments(n_bootstrap, seed, confidence_level)

    y = np.asarray(y_true, dtype=int)
    n_samples = len(y)
    if n_samples < 2:
        raise ValueError(f"Cohort '{cohort_name}' has insufficient samples: {n_samples}")

    unique_classes = np.unique(y)
    if not set(unique_classes).issubset({0, 1}):
        raise ValueError(f"y_true must contain binary labels in {{0, 1}}, got {unique_classes}")
    if len(unique_classes) < 2:
        raise ValueError(f"Cohort '{cohort_name}' does not contain both classes: {unique_classes}")

    # Validate prediction arrays
    clean_preds: dict[str, np.ndarray] = {}
    for model_name, probs in model_predictions.items():
        p = np.asarray(probs, dtype=float)
        if p.ndim != 1 or len(p) != n_samples:
            raise ValueError(
                f"Length mismatch for model '{model_name}': {len(p)} vs {n_samples} samples."
            )
        if not np.all(np.isfinite(p)):
            raise ValueError(f"Model '{model_name}' has non-finite probabilities.")
        if np.any(p < 0.0) or np.any(p > 1.0):
            raise ValueError(f"Model '{model_name}' has probabilities outside [0.0, 1.0].")
        clean_preds[model_name] = p

    n_positive = int(np.sum(y == 1))
    n_negative = int(np.sum(y == 0))

    # 1. Point estimates on original, unresampled cohort
    original_metrics: dict[str, dict[str, float]] = {}
    for model_name, p in clean_preds.items():
        original_metrics[model_name] = compute_model_metrics(y, p, n_bins=n_bins)

    # 2. Resampling loop
    rng = np.random.default_rng(seed)

    replicate_metrics: dict[str, dict[str, list[float]]] = {
        m: {metric: [] for metric in metric_names} for m in clean_preds
    }
    replicate_deltas: dict[tuple[str, str], dict[str, list[float]]] = {
        pair: {metric: [] for metric in metric_names} for pair in paired_comparisons
    }

    n_valid = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b = y[idx]

        # Check that resampled replicate contains both classes
        if len(np.unique(y_b)) < 2:
            continue

        staged_model_metrics: dict[str, dict[str, float]] = {}
        staged_paired_deltas: dict[tuple[str, str], dict[str, float]] = {}

        try:
            # 1. Compute resampled labels and all model metrics into staged dictionary
            for model_name, p in clean_preds.items():
                p_b = p[idx]
                staged_model_metrics[model_name] = compute_model_metrics(y_b, p_b, n_bins=n_bins)

            # 2. Compute all requested pairwise metric deltas using the same indices
            for m_a, m_b in paired_comparisons:
                staged_paired_deltas[(m_a, m_b)] = {}
                for metric in metric_names:
                    delta = staged_model_metrics[m_a][metric] - staged_model_metrics[m_b][metric]
                    staged_paired_deltas[(m_a, m_b)][metric] = delta

            # 3. Validate that staged results contain all expected finite metric values
            for model_name in clean_preds:
                for metric in metric_names:
                    val = staged_model_metrics[model_name][metric]
                    if not np.isfinite(val):
                        raise ValueError(f"Non-finite metric '{metric}' for model '{model_name}': {val}")

            for pair in paired_comparisons:
                for metric in metric_names:
                    delta_val = staged_paired_deltas[pair][metric]
                    if not np.isfinite(delta_val):
                        raise ValueError(f"Non-finite delta for metric '{metric}', pair {pair}: {delta_val}")

        except (ValueError, FloatingPointError, ZeroDivisionError):
            # Expected non-evaluable replicate failure: skip atomically without appending partial metrics
            continue

        # 4. Atomic commit: append all staged metrics and deltas only after replicate is fully valid
        for model_name in clean_preds:
            for metric in metric_names:
                replicate_metrics[model_name][metric].append(staged_model_metrics[model_name][metric])

        for pair in paired_comparisons:
            for metric in metric_names:
                replicate_deltas[pair][metric].append(staged_paired_deltas[pair][metric])

        n_valid += 1

    # Verify transactional alignment across all metrics
    for model_name in clean_preds:
        for metric in metric_names:
            if len(replicate_metrics[model_name][metric]) != n_valid:
                raise RuntimeError(
                    f"Replicate count mismatch for model '{model_name}', metric '{metric}': "
                    f"{len(replicate_metrics[model_name][metric])} vs {n_valid} valid"
                )

    for pair in paired_comparisons:
        for metric in metric_names:
            if len(replicate_deltas[pair][metric]) != n_valid:
                raise RuntimeError(
                    f"Replicate count mismatch for pair {pair}, metric '{metric}': "
                    f"{len(replicate_deltas[pair][metric])} vs {n_valid} valid"
                )

    if n_valid < min_valid_replicates:
        raise ValueError(
            f"Insufficient valid bootstrap replicates for cohort '{cohort_name}': "
            f"{n_valid} valid out of {n_bootstrap} requested (minimum required: {min_valid_replicates})"
        )

    # 3. Compile absolute model metrics DataFrame
    model_rows: list[dict] = []
    for model_name in clean_preds:
        for metric in metric_names:
            pe = original_metrics[model_name][metric]
            ci_lower, ci_upper = compute_percentile_interval(
                replicate_metrics[model_name][metric], confidence_level=confidence_level
            )
            model_rows.append({
                "cohort": cohort_name,
                "n_samples": n_samples,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "model": model_name,
                "metric": metric,
                "point_estimate": pe,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "confidence_level": confidence_level,
                "method": "percentile",
                "n_bootstrap_requested": n_bootstrap,
                "n_bootstrap_valid": n_valid,
                "bootstrap_seed": seed,
            })

    # 4. Compile paired comparisons DataFrame
    paired_rows: list[dict] = []
    for m_a, m_b in paired_comparisons:
        for metric in metric_names:
            point_delta = original_metrics[m_a][metric] - original_metrics[m_b][metric]
            ci_lower, ci_upper = compute_percentile_interval(
                replicate_deltas[(m_a, m_b)][metric], confidence_level=confidence_level
            )
            excludes_zero = bool(ci_lower > 0.0 or ci_upper < 0.0)
            paired_rows.append({
                "cohort": cohort_name,
                "n_samples": n_samples,
                "model_a": m_a,
                "model_b": m_b,
                "metric": metric,
                "metric_direction": metric_directions.get(metric, "higher"),
                "point_delta": point_delta,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "confidence_level": confidence_level,
                "method": "percentile",
                "n_bootstrap_requested": n_bootstrap,
                "n_bootstrap_valid": n_valid,
                "bootstrap_seed": seed,
                "interval_excludes_zero": excludes_zero,
            })

    return pd.DataFrame(model_rows), pd.DataFrame(paired_rows), n_bootstrap, n_valid


# ---------------------------------------------------------------------------
# Report Formatting Helper
# ---------------------------------------------------------------------------

def generate_markdown_report(
    model_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    metadata: dict,
) -> str:
    """Generate comprehensive audit report summarizing bootstrap uncertainty and paired comparisons."""
    conf_level = float(metadata["bootstrap_configuration"]["confidence_level"])
    conf_pct_val = conf_level * 100.0
    conf_pct = f"{conf_pct_val:g}"

    lines: list[str] = [
        "# Phase 07: Bootstrap Uncertainty Quantification & Paired Model Comparison Report",
        "",
        "> **Notice: Synthetic Educational Prototype**",
        "> This report analyzes uncertainty across fixed model artifacts evaluated on synthetic applicant records.",
        "> All data, feature values, bureau scores, and target default labels are mathematically generated.",
        "> These results do NOT represent real-world credit risk, real lending policy, or credit-invisible fairness.",
        "",
        "## 1. Executive Summary & Objective",
        "",
        "In credit risk model evaluation, point estimates computed on a single held-out test split can obscure sampling variability.",
        "To quantify this variability rigorously without retraining or altering production model artifacts, we perform non-parametric",
        f"paired bootstrap resampling ({metadata['bootstrap_configuration']['n_bootstrap_requested']:,} replicates, {conf_pct}% two-sided percentile intervals,",
        f"random seed {metadata['bootstrap_configuration']['seed']}).",
        "",
        "### Key Principles",
        "1. **Fixed Model Conditioning**: The bootstrap procedure resamples held-out test observations conditioning on existing fitted models.",
        "   It measures sampling variance of evaluation metrics on test sets of size $N$, NOT retraining variance across new training sets.",
        "2. **Strict Paired Resampling**: Every pairwise comparison between models evaluates predictions on the **exact same resampled observation indices**",
        "   for every bootstrap replicate. This accounts for observation-level covariance between model predictions rather than incorrectly assuming independent samples.",
        "3. **Zero Data Leakage / Artifact Preservation**: No model was refitted, re-tuned, or modified. Existing models and datasets are purely evaluated.",
        "",
        "## 2. Experimental Setup & Cohort Definitions",
        "",
        f"- **Dataset Source**: `{metadata['dataset_source_path']}` (generation seed {metadata['dataset_generation_seed']})",
        f"- **Held-out Test Partition**: {metadata['test_split_configuration']['test_size']*100:.0f}% stratified test split ({metadata['test_split_configuration']['n_test_observations']:,} rows, split random state {metadata['test_split_configuration']['random_state']})",
        f"- **Target Column**: `{metadata['test_split_configuration']['stratify_column']}`",
        "",
        "### Cohorts Evaluated",
        "| Cohort | Definition | Sample Size ($N$) | Positives (Defaults) | Negatives | Synthetic Default Rate |",
        "|---|---|---|---|---|---|",
    ]

    for c_name, c_info in metadata["cohorts"].items():
        lines.append(
            f"| `{c_name}` | {c_info['definition']} | {c_info['n_samples']:,} | "
            f"{c_info['n_positive']:,} | {c_info['n_negative']:,} | {c_info['synthetic_default_rate']:.4f} |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> The `no_bureau_history` cohort is a synthetic simulation-defined subgroup (`bureau_history_available == 0`).",
        "> It does not represent real-world credit-invisible populations, and this analysis must not be construed as a fairness or compliance assessment.",
        "",
        "## 3. Absolute Model Metric Confidence Intervals",
        "",
        f"Point estimates and {conf_pct}% two-sided percentile confidence intervals $[CI_{{lower}}, CI_{{upper}}]$ across all models and cohorts:",
        "",
    ])

    for cohort in model_df["cohort"].unique():
        lines.append(f"### Cohort: `{cohort}`")
        lines.append("")
        lines.append(f"| Model | Metric | Point Estimate | {conf_pct}% CI Lower | {conf_pct}% CI Upper | Replicates Valid |")
        lines.append("|---|---|---|---|---|---|")
        sub = model_df[model_df["cohort"] == cohort]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['model']} | `{r['metric']}` | {r['point_estimate']:.4f} | "
                f"{r['ci_lower']:.4f} | {r['ci_upper']:.4f} | {r['n_bootstrap_valid']}/{r['n_bootstrap_requested']} |"
            )
        lines.append("")

    lines.extend([
        "## 4. Paired Model Comparison & Performance Differences",
        "",
        f"For every bootstrap replicate $b$, the paired difference is evaluated on identical resampled indices:",
        "$$\\Delta_b = \\text{metric}(M_A) - \\text{metric}(M_B)$$",
        "",
        "### Metric Direction Interpretation",
        "- **Higher is better** (`roc_auc`, `gini_coefficient`, `ks_statistic`, `average_precision`): Positive $\\Delta$ favors Model A.",
        "- **Lower is better** (`brier_score`, `log_loss`, `ece_10_bins`): Negative $\\Delta$ favors Model A.",
        f"- **Interval Excludes Zero (`interval_excludes_zero`)**: Descriptive indicator that the {conf_pct}% percentile interval does not contain $0.0$.",
        "  *(Note: This is a property of the empirical interval, not a p-value or proof of commercial/practical significance).* ",
        "",
    ])

    for cohort in paired_df["cohort"].unique():
        lines.append(f"### Cohort: `{cohort}`")
        lines.append("")
        lines.append(f"| Model A | Model B | Metric | Direction | Point $\\Delta$ | {conf_pct}% CI Lower | {conf_pct}% CI Upper | Excludes Zero? |")
        lines.append("|---|---|---|---|---|---|---|---|")
        sub = paired_df[paired_df["cohort"] == cohort]

        for _, r in sub.iterrows():
            excl = "Yes" if r["interval_excludes_zero"] else "No"
            lines.append(
                f"| {r['model_a']} | {r['model_b']} | `{r['metric']}` | {r['metric_direction']} | "
                f"{r['point_delta']:+.4f} | {r['ci_lower']:+.4f} | {r['ci_upper']:+.4f} | {excl} |"
            )
        lines.append("")

    lines.extend([
        "## 5. Statistical Interpretation Guardrails & Limitations",
        "",
        "1. **Conditioning Scope**: These intervals reflect test-sample variability conditional on the **fixed fitted models**.",
        "   They do NOT capture variation from retraining models on fresh training splits or variation in synthetic data generation parameters.",
        "2. **Paired vs Independent Evaluation**: Independent confidence intervals for two models may overlap even when paired differences",
        "   consistently favor one model, because paired differences cancel out sample-wide difficulty variations across resampled test sets.",
        "3. **Zero Inclusion / Exclusion**: An interval crossing zero does not establish that two models are equivalent. An interval excluding zero",
        "   does not establish business significance or deployment readiness.",
        "4. **No Fairness or Legal Inferences**: The models and labels are entirely synthetic. No assertions regarding algorithmic bias,",
        "   disparate impact, or regulatory compliance (e.g. ECOA, FCRA, RBI guidelines) can be derived from this analysis.",
        "",
        "## 6. Execution Environment & Dependencies",
        "",
        f"- **Python Version**: `{metadata['environment_versions']['python_version']}`",
        f"- **NumPy**: `{metadata['environment_versions']['numpy_version']}`",
        f"- **pandas**: `{metadata['environment_versions']['pandas_version']}`",
        f"- **scikit-learn**: `{metadata['environment_versions']['scikit_learn_version']}`",
        f"- **matplotlib**: `{metadata['environment_versions']['matplotlib_version']}`",
        "",
        "---",
        "*Report generated automatically by `uncertainty_analysis.py`.*",
    ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Workflow Runner
# ---------------------------------------------------------------------------

def run_uncertainty_analysis(
    data_path: Path = DATA_PATH,
    model_paths: dict[str, Path] = MODEL_PATHS,
    output_dir: Path = OUTPUT_DIR,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    """Execute complete Phase 07 uncertainty analysis workflow."""
    validate_arguments(n_bootstrap, seed, confidence_level)

    # 1. Verify existence of dataset and model artifacts
    if not data_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at: {data_path}")

    for model_name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Model artifact for '{model_name}' not found at: {path}")

    # 2. Load dataset and reproduce exact test partition
    df = pd.read_csv(data_path)
    validate_dataset_schema(df)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing from dataset.")

    train_df, test_df = train_test_split(
        df,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_SPLIT_SEED,
        stratify=df[TARGET],
    )
    y_test = test_df[TARGET].to_numpy()

    # 3. Load models and generate predicted probabilities
    loaded_models: dict[str, object] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for model_name, path in model_paths.items():
        model = joblib.load(path)
        loaded_models[model_name] = model

        feature_cols = TRADITIONAL_FEATURES if "Traditional" in model_name else ALL_FEATURES
        probs_raw = model.predict_proba(test_df[feature_cols])

        if probs_raw.ndim != 2 or probs_raw.shape != (len(test_df), 2):
            raise ValueError(
                f"Model '{model_name}' returned probability shape {probs_raw.shape}, expected ({len(test_df)}, 2)."
            )

        p = probs_raw[:, 1]
        if not np.all(np.isfinite(p)):
            raise ValueError(f"Model '{model_name}' produced non-finite probabilities.")
        if np.any(p < 0.0) or np.any(p > 1.0):
            raise ValueError(f"Model '{model_name}' produced probabilities outside [0, 1].")

        test_predictions[model_name] = p

    # 4. Cohort masks
    cohort_masks: dict[str, tuple[np.ndarray, str]] = {
        "overall_test_set": (
            np.ones(len(test_df), dtype=bool),
            "All held-out synthetic test records",
        ),
        "no_bureau_history": (
            test_df["bureau_history_available"].eq(0).to_numpy(),
            "Held-out synthetic test records with bureau_history_available == 0",
        ),
    }

    # 5. Run bootstrap analysis per cohort
    all_model_dfs: list[pd.DataFrame] = []
    all_paired_dfs: list[pd.DataFrame] = []
    cohort_meta: dict[str, dict] = {}
    valid_replicates_by_cohort: dict[str, int] = {}

    for cohort_name, (mask, definition) in cohort_masks.items():
        y_cohort = y_test[mask]
        preds_cohort = {m: p[mask] for m, p in test_predictions.items()}

        m_df, p_df, req, val = bootstrap_cohort_analysis(
            cohort_name=cohort_name,
            y_true=y_cohort,
            model_predictions=preds_cohort,
            paired_comparisons=PAIRED_COMPARISONS,
            metric_names=METRIC_NAMES,
            metric_directions=METRIC_DIRECTIONS,
            n_bootstrap=n_bootstrap,
            seed=seed,
            confidence_level=confidence_level,
            n_bins=n_bins,
        )

        all_model_dfs.append(m_df)
        all_paired_dfs.append(p_df)
        valid_replicates_by_cohort[cohort_name] = val

        cohort_meta[cohort_name] = {
            "definition": definition,
            "n_samples": int(len(y_cohort)),
            "n_positive": int(np.sum(y_cohort == 1)),
            "n_negative": int(np.sum(y_cohort == 0)),
            "synthetic_default_rate": float(np.mean(y_cohort)),
        }

    combined_model_df = pd.concat(all_model_dfs, ignore_index=True)
    combined_paired_df = pd.concat(all_paired_dfs, ignore_index=True)

    # 6. Save output artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    model_csv_path = output_dir / "model_metric_confidence_intervals.csv"
    paired_csv_path = output_dir / "paired_model_comparison_confidence_intervals.csv"
    meta_json_path = output_dir / "run_metadata.json"
    report_md_path = output_dir / "bootstrap_uncertainty_report.md"

    combined_model_df.to_csv(model_csv_path, index=False)
    combined_paired_df.to_csv(paired_csv_path, index=False)

    env_versions = get_installed_dependency_versions()
    metadata: dict = {
        "experiment": "bootstrap_uncertainty_quantification_and_paired_model_comparison",
        "dataset_type": "SIMULATED_DATA_ONLY",
        "dataset_source_path": str(data_path.relative_to(ROOT)),
        "dataset_generation_seed": DEFAULT_SPLIT_SEED,
        "test_split_configuration": {
            "test_size": DEFAULT_TEST_SIZE,
            "random_state": DEFAULT_SPLIT_SEED,
            "stratify_column": TARGET,
            "n_test_observations": len(test_df),
        },
        "model_artifact_paths": {
            k: str(v.relative_to(ROOT)) for k, v in model_paths.items()
        },
        "cohorts": cohort_meta,
        "model_pair_definitions": [
            {
                "model_a": a,
                "model_b": b,
                "subtraction_direction": f"{a} minus {b}",
            }
            for a, b in PAIRED_COMPARISONS
        ],
        "metric_names": METRIC_NAMES,
        "metric_directions": METRIC_DIRECTIONS,
        "bootstrap_configuration": {
            "n_bootstrap_requested": n_bootstrap,
            "n_bootstrap_valid": valid_replicates_by_cohort,
            "confidence_level": confidence_level,
            "method": "two-sided percentile interval",
            "seed": seed,
        },
        "environment_versions": env_versions,
        "conditioning_scope_and_limitations": (
            "These percentile intervals summarize uncertainty arising strictly from resampling held-out observations "
            "for fixed, already-fitted model artifacts. They do NOT include uncertainty from retraining on a new training set, "
            "changing the synthetic data-generation process, temporal distribution shifts, or real-world population changes. "
            "Paired intervals use shared observation indices, preserving observation-level covariance across model predictions. "
            "An interval crossing zero does not establish that two models are equivalent, and an interval excluding zero "
            "is not by itself proof of business significance or deployment readiness."
        ),
        "synthetic_data_disclaimer": (
            "This project is an educational credit-risk prototype trained and evaluated exclusively on synthetic data. "
            "All applicant profiles, bureau scores, alternative features, and default outcomes are mathematically simulated. "
            "No real-world financial, underwriting, credit-invisible fairness, or legal compliance inferences may be drawn."
        ),
    }

    meta_json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report_text = generate_markdown_report(combined_model_df, combined_paired_df, metadata)
    report_md_path.write_text(report_text, encoding="utf-8")

    return combined_model_df, combined_paired_df, model_csv_path, paired_csv_path, report_md_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="Bootstrap uncertainty quantification and paired model comparisons."
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Number of bootstrap replicates (default: {DEFAULT_N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for bootstrap resampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help=f"Confidence level in (0, 1) for two-sided percentile interval (default: {DEFAULT_CONFIDENCE_LEVEL}).",
    )
    parsed = parser.parse_args(args)
    validate_arguments(parsed.n_bootstrap, parsed.seed, parsed.confidence_level)
    return parsed


def main() -> None:
    parsed = parse_args(sys.argv[1:])
    print("=" * 70)
    print("PHASE 07 — BOOTSTRAP UNCERTAINTY QUANTIFICATION & PAIRED MODEL COMPARISON")
    print("=" * 70)
    print(f"Replicates requested : {parsed.n_bootstrap:,}")
    print(f"Bootstrap seed       : {parsed.seed}")
    print(f"Confidence level     : {parsed.confidence_level*100:.1f}%")
    print(f"Output directory     : {OUTPUT_DIR}")
    print()

    m_df, p_df, m_path, p_path, r_path = run_uncertainty_analysis(
        n_bootstrap=parsed.n_bootstrap,
        seed=parsed.seed,
        confidence_level=parsed.confidence_level,
    )

    print(f"Analysis complete.")
    print(f"Model metric intervals saved to    : {m_path}")
    print(f"Paired comparison intervals saved to: {p_path}")
    print(f"Comprehensive report saved to      : {r_path}")
    print()
    print("Sample paired comparison output (Overall test set, ROC-AUC):")
    auc_pair = p_df[
        (p_df["cohort"] == "overall_test_set") & (p_df["metric"] == "roc_auc")
    ][["model_a", "model_b", "point_delta", "ci_lower", "ci_upper", "interval_excludes_zero"]]
    print(auc_pair.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()
