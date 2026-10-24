"""Train an illustrative alternative-credit-risk model on SIMULATED data.

This script is an engineering demonstration only. The generated records and targets
are synthetic and must not be represented as real borrower evidence.

Phase-01 changes (pipeline integrity):
- Replaced the dead make_pipeline() stub and the three independent inline pipeline
  constructions with three reusable, clearly named builder functions:
    build_traditional_lr_pipeline()
    build_hybrid_lr_pipeline()
    build_hybrid_hgb_pipeline()
- Corrected a redundant bureau-score missingness indicator (Objective B). See the
  _make_preprocessor() docstring for the full explanation.

Phase-02 changes (input validation & credit risk metrics):
- Integrated formal dataset schema validation and feature frame checks via validation.py.
- Added credit-risk domain evaluation metrics:
    * Gini coefficient (Somers' D = 2 * ROC_AUC - 1).
    * Kolmogorov-Smirnov (KS) statistic (max separation between default and non-default distributions).
    * Expected Calibration Error (ECE) with uniform 10-bin partitioning.
- Automated reliability diagram generation: reports/calibration_curves.png.

Phase-03 changes (illustrative scoring, threshold analysis & local explainability):
- Out-of-fold leakage-safe threshold sensitivity analysis on the training partition:
    * 5-fold StratifiedKFold on training data only (test set strictly preserved).
    * Precision, recall, specificity, F1, balanced accuracy, and counts across 0.05-0.95 grid.
    * Artifacts: reports/threshold_analysis.csv and reports/threshold_analysis.png.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
from inference_helper import generate_model_manifest

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 2026
N_SAMPLES = 10_000

# Alternative features whose missingness is independently informative and should
# therefore produce a binary missingness indicator during imputation.
# NOTE: bureau_score is deliberately excluded here — see _make_preprocessor().
# NOTE: alternative_history_months has no missingness by design, so it also needs
# no indicator.
_ALT_FEATURES_WITH_MISSINGNESS: list[str] = [
    "utility_payment_timeliness",
    "rental_payment_timeliness",
    "digital_payment_regularity",
    "cashflow_stability_index",
]
_ALT_FEATURES_NO_MISSINGNESS: list[str] = ["alternative_history_months"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_synthetic_dataset(n: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    """Generate fully synthetic records with a known, invented outcome mechanism."""
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 66, size=n)
    log_income = rng.normal(np.log(42_000), 0.55, size=n)
    income = np.clip(np.exp(log_income), 8_000, 250_000)
    employ = np.clip(rng.gamma(shape=2.1, scale=2.6, size=n), 0, 35)
    dti = np.clip(rng.beta(2.0, 4.2, size=n) * 1.25 + 0.08 * (income < 20_000), 0, 1.2)
    existing_loans = np.clip(rng.poisson(1.2, size=n), 0, 8)
    utilization = np.clip(rng.beta(2.0, 3.3, size=n) + 0.05 * existing_loans, 0, 1.25)
    prior_delinquencies = np.clip(rng.poisson(0.28, size=n), 0, 5)

    # Availability is simulated, not inferred from real population statistics.
    bureau_available = rng.binomial(1, sigmoid(-0.15 + 0.035 * (age - 35) + 0.000006 * (income - 42_000)), size=n)

    # Latent financial stability creates modest correlations across generated features.
    stability = (
        0.27 * ((log_income - np.log(42_000)) / 0.55)
        + 0.22 * ((employ - 5.0) / 4.0)
        - 0.30 * ((dti - 0.35) / 0.22)
        - 0.12 * prior_delinquencies
        + rng.normal(0, 0.85, size=n)
    )
    utility = np.clip(0.82 + 0.10 * stability + rng.normal(0, 0.12, size=n), 0.05, 1.0)
    rental = np.clip(0.80 + 0.09 * stability + rng.normal(0, 0.15, size=n), 0.05, 1.0)
    digital_regularity = np.clip(0.75 + 0.11 * stability + rng.normal(0, 0.16, size=n), 0.02, 1.0)
    cashflow_stability = np.clip(0.68 + 0.14 * stability + rng.normal(0, 0.17, size=n), 0.02, 1.0)
    alt_months = np.clip(np.round(rng.normal(15 + 2.0 * stability, 6, size=n)), 0, 24).astype(float)

    # Simulated bureau scores are noisy measurements of latent stability. Scores do not
    # exist for applicants flagged as having no bureau history.
    bureau_score_full = np.clip(650 + 42 * stability + rng.normal(0, 42, size=n), 300, 900)
    bureau_score = np.where(bureau_available == 1, bureau_score_full, np.nan)

    # Synthetic outcome mechanism. Alternative indicators are deliberately informative
    # here so that feature ablation can be demonstrated. This is NOT empirical evidence.
    z = (
        -2.35
        + 0.58 * ((dti - 0.35) / 0.22)
        + 0.48 * ((utilization - 0.40) / 0.25)
        + 0.52 * prior_delinquencies
        - 0.30 * ((employ - 5.0) / 4.0)
        - 0.28 * ((log_income - np.log(42_000)) / 0.55)
        - 0.25 * ((age - 35) / 15)
        - 0.38 * stability
        + 0.95 * (0.82 - utility) / 0.18
        + 0.72 * (0.80 - rental) / 0.22
        + 0.78 * (0.75 - digital_regularity) / 0.25
        + 0.85 * (0.68 - cashflow_stability) / 0.25
        - 0.20 * ((alt_months - 15) / 6)
    )
    true_prob = sigmoid(z)
    default = rng.binomial(1, true_prob, size=n)

    df = pd.DataFrame({
        "synthetic_applicant_id": [f"SYN{i:06d}" for i in range(1, n + 1)],
        "age_years": age,
        "monthly_income_inr": np.round(income, 2),
        "employment_tenure_years": np.round(employ, 2),
        "debt_to_income_ratio": np.round(dti, 4),
        "existing_loan_count": existing_loans,
        "credit_utilization_ratio": np.round(utilization, 4),
        "prior_delinquencies_24m": prior_delinquencies,
        "bureau_history_available": bureau_available,
        "bureau_score": np.round(bureau_score, 1),
        "utility_payment_timeliness": np.round(utility, 4),
        "rental_payment_timeliness": np.round(rental, 4),
        "digital_payment_regularity": np.round(digital_regularity, 4),
        "cashflow_stability_index": np.round(cashflow_stability, 4),
        "alternative_history_months": alt_months.astype(int),
        TARGET: default,
    })
    # Missing alternative records demonstrate normal pipeline imputation; the missing
    # values are synthetic and are generated independently of the target.
    for col, rate in [("utility_payment_timeliness", 0.07), ("rental_payment_timeliness", 0.12),
                      ("digital_payment_regularity", 0.05), ("cashflow_stability_index", 0.08)]:
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _make_preprocessor(
    feature_cols: Sequence[str],
    add_scaler: bool = True,
) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes correctly without a redundant indicator.

    WHY THE SPLIT IS NEEDED (Objective B)
    ======================================
    ``bureau_score`` is NaN *if and only if* ``bureau_history_available == 0``.
    That relationship holds by construction (see ``generate_synthetic_dataset``):
    bureau_score is set to NaN wherever bureau_available is 0, and is always
    non-NaN wherever bureau_available is 1.

    A naïve ``SimpleImputer(add_indicator=True)`` applied uniformly to all features
    therefore creates a ``missingindicator_bureau_score`` column that is algebraically
    identical to ``1 - bureau_history_available``. Passing both to a linear model:
      • inflates parameter variance via exact collinearity;
      • splits the credit given to bureau availability between two perfectly correlated
        columns, making the learned coefficients and permutation importances
        uninterpretable;
      • produced the observed equal-and-opposite coefficient artefact
        (±0.026 in the traditional model, ±0.021 in the hybrid model).

    THE FIX
    ========
    The preprocessor uses two separate transformer groups:
      1. ``trad`` — all traditional features (including bureau_score), imputed
         with ``add_indicator=False``. Missingness for bureau_score is fully
         captured by the explicit ``bureau_history_available`` column.
      2. ``alt_ind`` — the four alternative features that can be genuinely missing
         (not already covered by any other feature), imputed with
         ``add_indicator=True`` so that their independent MCAR missingness pattern
         is preserved as predictive signal.
      3. ``alt_no_ind`` — ``alternative_history_months``, which has no missingness
         in the synthetic data and requires no indicator.

    This structure preserves all legitimate missingness information while eliminating
    the redundant bureau-score indicator.
    """
    trad_cols = [c for c in TRADITIONAL_FEATURES if c in feature_cols]
    alt_ind_cols = [c for c in _ALT_FEATURES_WITH_MISSINGNESS if c in feature_cols]
    alt_no_ind_cols = [c for c in _ALT_FEATURES_NO_MISSINGNESS if c in feature_cols]

    transformers = []

    if trad_cols:
        if add_scaler:
            trad_pipe: Pipeline | SimpleImputer = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
                ("scaler", StandardScaler()),
            ])
        else:
            trad_pipe = SimpleImputer(strategy="median", add_indicator=False)
        transformers.append(("trad", trad_pipe, trad_cols))

    if alt_ind_cols:
        if add_scaler:
            alt_ind_pipe: Pipeline | SimpleImputer = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ])
        else:
            alt_ind_pipe = SimpleImputer(strategy="median", add_indicator=True)
        transformers.append(("alt_ind", alt_ind_pipe, alt_ind_cols))

    if alt_no_ind_cols:
        if add_scaler:
            alt_no_ind_pipe: Pipeline | SimpleImputer = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
                ("scaler", StandardScaler()),
            ])
        else:
            alt_no_ind_pipe = SimpleImputer(strategy="median", add_indicator=False)
        transformers.append(("alt_no_ind", alt_no_ind_pipe, alt_no_ind_cols))

    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_traditional_lr_pipeline() -> Pipeline:
    """Logistic Regression on traditional/application features only.

    Preprocessing: median imputation (no bureau-score indicator; see
    _make_preprocessor) + StandardScaler, fitted on training data.
    """
    return Pipeline([
        ("preprocess", _make_preprocessor(TRADITIONAL_FEATURES, add_scaler=True)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def build_hybrid_lr_pipeline() -> Pipeline:
    """Logistic Regression on traditional + alternative features.

    Preprocessing: corrected ColumnTransformer (no bureau-score indicator, but
    retains indicators for the four genuinely-missing alternative features) +
    StandardScaler, fitted on training data.
    """
    return Pipeline([
        ("preprocess", _make_preprocessor(ALL_FEATURES, add_scaler=True)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def build_hybrid_hgb_pipeline() -> Pipeline:
    """HistGradientBoosting on traditional + alternative features.

    HistGradientBoostingClassifier handles NaN natively; the explicit
    SimpleImputer is retained so the pipeline is self-contained and the corrected
    preprocessor keeps the feature-name contract consistent with the linear models.
    No StandardScaler is used (not needed for tree-based models).
    """
    return Pipeline([
        ("preprocess", _make_preprocessor(ALL_FEATURES, add_scaler=False)),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=160, learning_rate=0.08, max_leaf_nodes=15, random_state=SEED
        )),
    ])


# ---------------------------------------------------------------------------
# Credit-risk domain evaluation metrics (Objective B)
# ---------------------------------------------------------------------------

def calculate_gini(roc_auc: float) -> float:
    """Calculate Gini coefficient (Somers' D) from ROC-AUC for binary default prediction.

    Formula:
        Gini = 2 * ROC_AUC - 1

    Ranking discrimination interpretation:
        Gini measures the degree of separation between default and non-default risk scores.
        Gini = 1.0 indicates perfect ranking separation; Gini = 0.0 indicates no discrimination
        beyond random ordering.
        It is a rank discrimination metric, NOT a measure of probability calibration.
    """
    if not np.isfinite(roc_auc):
        raise ValueError("ROC-AUC must be a finite number.")
    if not (0.0 <= roc_auc <= 1.0):
        raise ValueError(f"ROC-AUC must be in [0.0, 1.0]; got {roc_auc}.")
    return float(2.0 * roc_auc - 1.0)


def kolmogorov_smirnov_statistic(
    y_true: Sequence | np.ndarray,
    probabilities: Sequence | np.ndarray,
) -> float:
    """Calculate the two-sample Kolmogorov-Smirnov (KS) statistic for credit risk scoring.

    In credit risk evaluation, the KS statistic measures the maximum vertical separation
    between the cumulative default distribution (True Positive Rate) and cumulative
    non-default distribution (False Positive Rate) across all decision thresholds:
        KS = max_t |TPR(t) - FPR(t)|

    Score orientation:
        Probabilities represent the modeled probability of borrower default (positive class: 1).
        Higher predicted default probability indicates higher modeled default risk.

    Parameters
    ----------
    y_true : 1D array-like of binary targets in {0, 1}. Both classes must be present.
    probabilities : 1D array-like of predicted default probabilities in [0.0, 1.0].

    Returns
    -------
    ks : float in [0.0, 1.0], representing maximum separation between risk distributions.
    """
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
        raise ValueError("Both classes (0 and 1) must be present to evaluate KS statistic.")

    fpr, tpr, _ = roc_curve(y, probs)
    ks = float(np.max(np.abs(tpr - fpr)))
    return ks


def expected_calibration_error(
    y_true: Sequence | np.ndarray,
    probabilities: Sequence | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate the Expected Calibration Error (ECE) using uniform probability bins.

    Binning policy:
        Uniformly partitions the probability interval [0.0, 1.0] into `n_bins` equal-width intervals.
        Bin edges: np.linspace(0.0, 1.0, n_bins + 1).
        Samples are assigned using numpy.searchsorted on internal edges bins[1:-1].
        Observations with probability 1.0 fall into the final bin [1 - 1/n_bins, 1.0].

    Weighting:
        Each nonempty bin is weighted by its sample count fraction (|B_m| / N).
        Empty bins contribute 0 to the weighted sum and avoid NaN results.
        ECE = sum_{m: |B_m| > 0} (|B_m| / N) * |accuracy(B_m) - confidence(B_m)|
        where accuracy(B_m) is the fraction of positive labels in bin m, and
        confidence(B_m) is the mean predicted probability in bin m.

    Calibration vs discrimination:
        ECE measures probability reliability (calibration) and depends on the chosen bin count.
        It is complementary to Brier score and log loss, which evaluate probability accuracy
        without requiring binning discretization.

    Parameters
    ----------
    y_true : 1D array-like of binary labels {0, 1}. Both classes must be present.
    probabilities : 1D array-like of predicted default probabilities in [0.0, 1.0].
    n_bins : int, number of uniform bins (default: 10).

    Returns
    -------
    ece : float non-negative calibration error in [0.0, 1.0].
    """
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
        raise ValueError("Both classes (0 and 1) must be present to evaluate ECE.")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer >= 1; got {n_bins}.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], probs)
    bin_total = np.bincount(binids, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y.astype(float), minlength=n_bins)
    bin_conf = np.bincount(binids, weights=probs, minlength=n_bins)

    nonzero = bin_total > 0
    if not np.any(nonzero):
        return 0.0

    acc = bin_true[nonzero] / bin_total[nonzero]
    conf = bin_conf[nonzero] / bin_total[nonzero]
    weights = bin_total[nonzero] / len(y)

    ece = float(np.sum(weights * np.abs(acc - conf)))
    return ece


def metric_row(
    name: str,
    y_true: pd.Series | np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute complete discrimination and calibration evaluation metrics for a model."""
    auc = float(roc_auc_score(y_true, probs))
    gini = calculate_gini(auc)
    ks = kolmogorov_smirnov_statistic(y_true, probs)
    ap = float(average_precision_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))
    ll = float(log_loss(y_true, np.clip(probs, 1e-8, 1 - 1e-8)))
    ece = expected_calibration_error(y_true, probs, n_bins=n_bins)
    return {
        "model": name,
        "roc_auc": auc,
        "gini_coefficient": gini,
        "ks_statistic": ks,
        "average_precision": ap,
        "brier_score": brier,
        "log_loss": ll,
        f"ece_{n_bins}_bins": ece,
    }


def plot_comparison(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(metrics))
    width = 0.25
    for i, metric in enumerate(["roc_auc", "average_precision"]):
        ax.bar(x + (i - 0.5) * width, metrics[metric], width, label=metric.replace("_", " ").title())
    ax.set_xticks(x, metrics["model"], rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Synthetic test-set discrimination metrics")
    ax.set_ylabel("Score (higher is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "metric_comparison.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Threshold sensitivity analysis (Phase 03 — Objective C)
# ---------------------------------------------------------------------------

def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float] | None = None,
    model_name: str = "Model",
    data_partition: str = "training_out_of_fold",
    method: str = "5-fold stratified cross-validation",
    threshold: float | None = None,
) -> Any:
    """Calculate confusion-matrix and classification metrics across a threshold grid.

    Decision policy:
        Positive class is default (1).
        Applicant is predicted as default if predicted probability >= threshold.
        Zero denominators are safely handled to prevent NaN metrics.
    """
    single_threshold_mode = False
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating)):
            raise TypeError(f"threshold must be numeric; got {type(threshold).__name__}.")
        if not np.isfinite(threshold):
            raise ValueError(f"threshold must be a finite number; got {threshold}.")
        if not (0.0 <= float(threshold) <= 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0]; got {threshold}.")
        threshold_list = [float(threshold)]
        single_threshold_mode = True
    elif thresholds is None:
        threshold_list = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    else:
        try:
            threshold_list = [float(t) for t in thresholds]
        except (TypeError, ValueError) as err:
            raise TypeError(f"thresholds must be an iterable of numeric values: {err}") from err
        if len(threshold_list) == 0:
            raise ValueError("thresholds sequence must be non-empty.")
        for idx, t in enumerate(threshold_list):
            if not np.isfinite(t):
                raise ValueError(f"Threshold at index {idx} must be a finite number; got {t}.")
            if not (0.0 <= t <= 1.0):
                raise ValueError(f"Threshold at index {idx} must be in [0.0, 1.0]; got {t}.")

    # Validate 1D array shapes and non-empty lengths
    y_arr = np.asarray(y_true)
    probs_arr = np.asarray(probabilities)
    if y_arr.ndim != 1 or probs_arr.ndim != 1:
        raise ValueError("y_true and probabilities must be 1D arrays.")
    if len(y_arr) == 0:
        raise ValueError("y_true and probabilities arrays must be non-empty.")
    if len(y_arr) != len(probs_arr):
        raise ValueError(f"Length mismatch: y_true ({len(y_arr)}) vs probabilities ({len(probs_arr)}).")

    # Validate genuine binary labels without silent coercion
    if not np.issubdtype(y_arr.dtype, np.number):
        raise TypeError(f"y_true must contain numeric binary labels; got dtype {y_arr.dtype}.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y_true contains NaN or infinite values.")
    unique_labels = set(np.unique(y_arr))
    if not unique_labels.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"y_true must contain only binary labels in {{0, 1}}; got unique values {unique_labels}.")
    y = y_arr.astype(int)

    # Validate finite probabilities within [0.0, 1.0]
    if not np.issubdtype(probs_arr.dtype, np.number):
        raise TypeError(f"probabilities must be numeric; got dtype {probs_arr.dtype}.")
    if not np.all(np.isfinite(probs_arr)):
        raise ValueError("probabilities array contains NaN or infinite values.")
    if np.any((probs_arr < 0.0) | (probs_arr > 1.0)):
        raise ValueError(
            f"All probabilities must be within [0.0, 1.0]; "
            f"min={float(np.min(probs_arr))}, max={float(np.max(probs_arr))}."
        )
    probs = probs_arr.astype(float)

    rows = []
    for t in threshold_list:
        pred = (probs >= t).astype(int)
        tp = int(np.sum((y == 1) & (pred == 1)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        tn = int(np.sum((y == 0) & (pred == 0)))
        fn = int(np.sum((y == 1) & (pred == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )
        balanced_acc = float(0.5 * (recall + specificity))

        rows.append({
            "model": model_name,
            "threshold": float(t),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "balanced_accuracy": balanced_acc,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "data_partition": data_partition,
            "method": method,
        })
    return rows[0] if single_threshold_mode else rows


def compute_oof_predictions(
    train_df: pd.DataFrame,
    seed: int = SEED,
    n_splits: int = 5,
) -> dict[str, np.ndarray]:
    """Obtain leakage-safe out-of-fold predicted probabilities on the training partition.

    Uses StratifiedKFold on the training partition so that preprocessing and model
    fitting are strictly isolated to each fold's training slice. Final test labels
    are never touched during threshold analysis.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_train = train_df[TARGET].to_numpy()

    oof_trad = np.zeros(len(train_df), dtype=float)
    oof_hybrid_lr = np.zeros(len(train_df), dtype=float)
    oof_hybrid_hgb = np.zeros(len(train_df), dtype=float)

    for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        fold_train = train_df.iloc[trn_idx]
        fold_val = train_df.iloc[val_idx]
        fold_y_train = y_train[trn_idx]

        # 1. Traditional baseline pipeline
        pipe_trad = build_traditional_lr_pipeline()
        pipe_trad.fit(fold_train[TRADITIONAL_FEATURES], fold_y_train)
        oof_trad[val_idx] = pipe_trad.predict_proba(fold_val[TRADITIONAL_FEATURES])[:, 1]

        # 2. Hybrid LR pipeline
        pipe_hlr = build_hybrid_lr_pipeline()
        pipe_hlr.fit(fold_train[ALL_FEATURES], fold_y_train)
        oof_hybrid_lr[val_idx] = pipe_hlr.predict_proba(fold_val[ALL_FEATURES])[:, 1]

        # 3. Hybrid HGB pipeline
        pipe_hhgb = build_hybrid_hgb_pipeline()
        pipe_hhgb.fit(fold_train[ALL_FEATURES], fold_y_train)
        oof_hybrid_hgb[val_idx] = pipe_hhgb.predict_proba(fold_val[ALL_FEATURES])[:, 1]

    return {
        "Traditional baseline": oof_trad,
        "Hybrid (traditional + alternative)": oof_hybrid_lr,
        "Hybrid (HistGradientBoosting)": oof_hybrid_hgb,
    }


def plot_threshold_analysis(threshold_df: pd.DataFrame) -> None:
    """Plot precision, recall, and F1 across thresholds for the models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    models = [
        "Traditional baseline",
        "Hybrid (traditional + alternative)",
        "Hybrid (HistGradientBoosting)",
    ]
    for ax, model_name in zip(axes, models):
        sub = threshold_df[threshold_df["model"] == model_name]
        ax.plot(sub["threshold"], sub["precision"], label="Precision", color="#1f77b4", linewidth=1.8)
        ax.plot(sub["threshold"], sub["recall"], label="Recall (Sensitivity)", color="#d62728", linewidth=1.8)
        ax.plot(sub["threshold"], sub["f1_score"], label="F1 Score", color="#2ca02c", linewidth=1.8, linestyle="--")
        ax.axvline(0.10, color="gray", linestyle=":", alpha=0.7, label="Demo risk cutoff (0.10)" if model_name == models[0] else "")
        ax.axvline(0.20, color="gray", linestyle="-.", alpha=0.7, label="Demo risk cutoff (0.20)" if model_name == models[0] else "")
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("Decision threshold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Metric score")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Threshold sensitivity analysis — 5-fold cross-validation out-of-fold training data\n"
        "(Synthetic demonstration only; trade-off analysis, not a lending decision policy)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "threshold_analysis.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training workflow
# ---------------------------------------------------------------------------

def main() -> None:
    for folder in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_dataset()

    # Formal dataset schema validation (Objective A)
    validate_dataset_schema(df)

    df.to_csv(DATA_DIR / "synthetic_applicants.csv", index=False)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df[TARGET])
    y_train, y_test = train_df[TARGET], test_df[TARGET]

    # Validate feature frames before training
    validate_feature_frame(train_df[TRADITIONAL_FEATURES], TRADITIONAL_FEATURES)
    validate_feature_frame(train_df[ALL_FEATURES], ALL_FEATURES)
    validate_feature_frame(test_df[TRADITIONAL_FEATURES], TRADITIONAL_FEATURES)
    validate_feature_frame(test_df[ALL_FEATURES], ALL_FEATURES)

    # Phase 03: Out-of-fold threshold sensitivity analysis on training partition (leakage-safe)
    print("Computing 5-fold cross-validation out-of-fold predictions on training partition...")
    oof_scores = compute_oof_predictions(train_df, seed=SEED, n_splits=5)
    threshold_rows = []
    for model_name, oof_probs in oof_scores.items():
        threshold_rows.extend(calculate_threshold_metrics(
            y_true=y_train.to_numpy(),
            probabilities=oof_probs,
            model_name=model_name,
        ))
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(REPORT_DIR / "threshold_analysis.csv", index=False)
    plot_threshold_analysis(threshold_df)

    # --- Traditional Logistic Regression baseline ---
    trad_pipe = build_traditional_lr_pipeline()
    trad_pipe.fit(train_df[TRADITIONAL_FEATURES], y_train)
    trad_probs = trad_pipe.predict_proba(test_df[TRADITIONAL_FEATURES])[:, 1]
    joblib.dump(trad_pipe, MODEL_DIR / "traditional_baseline.joblib")

    # --- Hybrid Logistic Regression ---
    hybrid_lr_pipe = build_hybrid_lr_pipeline()
    hybrid_lr_pipe.fit(train_df[ALL_FEATURES], y_train)
    hybrid_lr_probs = hybrid_lr_pipe.predict_proba(test_df[ALL_FEATURES])[:, 1]
    hybrid_lr_path = MODEL_DIR / "hybrid_logistic_regression.joblib"
    joblib.dump(hybrid_lr_pipe, hybrid_lr_path)

    # Regenerate model artifact manifest to keep manifest and saved artifact mutually consistent
    manifest = generate_model_manifest(
        model_path=hybrid_lr_path,
        output_path=MODEL_DIR / "model_manifest.json",
        training_seed=SEED,
        train_size=len(train_df),
    )

    # --- Hybrid HistGradientBoosting ---
    hybrid_hgb_pipe = build_hybrid_hgb_pipeline()
    hybrid_hgb_pipe.fit(train_df[ALL_FEATURES], y_train)
    hybrid_hgb_probs = hybrid_hgb_pipe.predict_proba(test_df[ALL_FEATURES])[:, 1]
    joblib.dump(hybrid_hgb_pipe, MODEL_DIR / "hybrid_hist_gradient_boosting.joblib")

    pipelines = {
        "Traditional baseline": trad_pipe,
        "Hybrid (traditional + alternative)": hybrid_lr_pipe,
        "Hybrid (HistGradientBoosting)": hybrid_hgb_pipe,
    }
    scores = {
        "Traditional baseline": trad_probs,
        "Hybrid (traditional + alternative)": hybrid_lr_probs,
        "Hybrid (HistGradientBoosting)": hybrid_hgb_probs,
    }

    rows = [
        metric_row("Traditional baseline", y_test, trad_probs),
        metric_row("Hybrid (traditional + alternative)", y_test, hybrid_lr_probs),
        metric_row("Hybrid (HistGradientBoosting)", y_test, hybrid_hgb_probs),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORT_DIR / "test_metrics.csv", index=False)

    prevalence = float(y_test.mean())
    (REPORT_DIR / "run_summary.json").write_text(json.dumps({
        "data_type": "SIMULATED ONLY",
        "seed": SEED,
        "n_rows": len(df),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "synthetic_default_rate_all": float(df[TARGET].mean()),
        "synthetic_test_default_rate": prevalence,
        "features_traditional": TRADITIONAL_FEATURES,
        "features_alternative": ALTERNATIVE_FEATURES,
        "metrics_evaluated": [
            "roc_auc",
            "gini_coefficient",
            "ks_statistic",
            "average_precision",
            "brier_score",
            "log_loss",
            "ece_10_bins",
        ],
        "calibration_bins": 10,
        "calibration_strategy": "uniform",
        "threshold_analysis": {
            "data_partition": "training_out_of_fold",
            "method": "5-fold stratified cross-validation",
            "threshold_grid_min": 0.05,
            "threshold_grid_max": 0.95,
            "threshold_grid_step": 0.05,
            "n_thresholds": 19,
            "note": (
                "Threshold metrics evaluated strictly on out-of-fold training data to prevent leakage; "
                "final held-out test data was NOT used for threshold selection."
            ),
        },
        "note": (
            "Metrics demonstrate an engineering pipeline on generated data only. "
            "They are not estimates of real-world borrower risk or proof of "
            "alternative-data value."
        ),
    }, indent=2))

    # ROC and precision-recall curves
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, probs in scores.items():
        RocCurveDisplay.from_predictions(y_test, probs, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC curves — synthetic test data only")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curves.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, probs in scores.items():
        PrecisionRecallDisplay.from_predictions(y_test, probs, name=name, ax=ax)
    ax.set_title("Precision-recall curves — synthetic test data only")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "precision_recall_curves.png", dpi=160)
    plt.close(fig)

    plot_comparison(metrics)

    # Calibration curves (reliability diagram) — Objective C
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, probs in scores.items():
        CalibrationDisplay.from_predictions(
            y_test, probs, n_bins=10, strategy="uniform", name=name, ax=ax
        )
    ax.set_title("Calibration curves (reliability diagram) — synthetic test data only")
    ax.set_xlabel("Mean predicted default probability")
    ax.set_ylabel("Observed default frequency")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "calibration_curves.png", dpi=160)
    plt.close(fig)

    # Test-set permutation importance for the hybrid logistic pipeline.
    # Evaluate importance on the test set for demonstration, not for tuning the model.
    hybrid_lr = pipelines["Hybrid (traditional + alternative)"]
    perm = permutation_importance(
        hybrid_lr, test_df[ALL_FEATURES], y_test,
        scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": ALL_FEATURES,
        "auc_drop_mean": perm.importances_mean,
        "auc_drop_std": perm.importances_std,
    })
    imp.sort_values("auc_drop_mean", ascending=False).to_csv(
        REPORT_DIR / "permutation_importance.csv", index=False
    )
    top = imp.sort_values("auc_drop_mean", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["auc_drop_mean"], xerr=top["auc_drop_std"])
    ax.set_title(
        "Permutation importance — hybrid logistic model\n"
        "Synthetic test data only; association, not causation"
    )
    ax.set_xlabel("Mean decrease in ROC-AUC after permutation")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "permutation_importance.png", dpi=160)
    plt.close(fig)

    # Thin-file-like synthetic subgroup: by construction, applicants with no bureau history.
    subgroup_mask = test_df["bureau_history_available"].eq(0)
    subgroup_rows = []
    if subgroup_mask.sum() > 0:
        subgroup_y = y_test.loc[subgroup_mask]
        for name, probs in scores.items():
            p = probs[subgroup_mask.to_numpy()]
            if subgroup_y.nunique() == 2:
                subgroup_rows.append(metric_row(name, subgroup_y, p))
        pd.DataFrame(subgroup_rows).to_csv(
            REPORT_DIR / "no_bureau_history_subgroup_metrics.csv", index=False
        )

    # A small labelled example set for the UI. This file contains synthetic test samples.
    display_cols = ["synthetic_applicant_id"] + ALL_FEATURES + [TARGET]
    test_df[display_cols].head(200).to_csv(DATA_DIR / "synthetic_demo_cases.csv", index=False)

    print("TRAINING COMPLETE — SYNTHETIC DATA ONLY")
    print(f"Rows: {len(df):,} | Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Synthetic default rate: {df[TARGET].mean():.3f}")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Artifacts saved to data/, models/, and reports/.")
    print(f"Artifact manifest updated: {MODEL_DIR / 'model_manifest.json'} (SHA-256: {manifest['sha256'][:8]}...)")
    print("IMPORTANT: These results do not validate real-world credit scoring.")


if __name__ == "__main__":
    main()
