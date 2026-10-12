"""Train an illustrative alternative-credit-risk model on SIMULATED data.

This script is an engineering demonstration only. The generated records and targets
are synthetic and must not be represented as real borrower evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    ConfusionMatrixDisplay,
    log_loss,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SEED = 2026
N_SAMPLES = 10_000

TRADITIONAL_FEATURES = [
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
ALTERNATIVE_FEATURES = [
    "utility_payment_timeliness",
    "rental_payment_timeliness",
    "digital_payment_regularity",
    "cashflow_stability_index",
    "alternative_history_months",
]
TARGET = "default_within_12_months"
ALL_FEATURES = TRADITIONAL_FEATURES + ALTERNATIVE_FEATURES


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


def make_pipeline() -> Pipeline:
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    prep = ColumnTransformer([("numeric", numeric, ALL_FEATURES)], remainder="drop")
    return Pipeline([
        ("preprocess", prep),
        ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def metric_row(name: str, y_true: pd.Series, probs: np.ndarray) -> dict:
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision": float(average_precision_score(y_true, probs)),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "log_loss": float(log_loss(y_true, np.clip(probs, 1e-8, 1 - 1e-8))),
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


def main() -> None:
    for folder in (DATA_DIR, MODEL_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_dataset()
    df.to_csv(DATA_DIR / "synthetic_applicants.csv", index=False)
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df[TARGET])
    y_train, y_test = train_df[TARGET], test_df[TARGET]

    feature_sets = {
        "Traditional baseline": TRADITIONAL_FEATURES,
        "Hybrid (traditional + alternative)": ALL_FEATURES,
    }
    pipelines = {}
    scores = {}
    rows = []
    for name, cols in feature_sets.items():
        # Separate columns for each fit; all preprocessing is fit only on training data.
        numeric = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
        prep = ColumnTransformer([("numeric", numeric, cols)], remainder="drop")
        pipe = Pipeline([
            ("preprocess", prep),
            ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
        ])
        pipe.fit(train_df[cols], y_train)
        # Compatibility with scikit-learn 1.8 inference behaviour.
        classifier = pipe.named_steps["classifier"]
        if classifier.__class__.__name__ == "LogisticRegression" and not hasattr(classifier, "multi_class"):
            classifier.multi_class = "auto"
        probs = pipe.predict_proba(test_df[cols])[:, 1]
        pipelines[name] = pipe
        scores[name] = probs
        rows.append(metric_row(name, y_test, probs))
        joblib.dump(pipe, MODEL_DIR / ("traditional_baseline.joblib" if name == "Traditional baseline" else "hybrid_logistic_regression.joblib"))

    # An additional, modest non-linear model on the same synthetic feature set.
    # Keep median imputation inside a pipeline and fit exclusively on the training split.
    tree_numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    tree_prep = ColumnTransformer([("numeric", tree_numeric, ALL_FEATURES)], remainder="drop")
    tree_pipe = Pipeline([
        ("preprocess", tree_prep),
        ("classifier", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.08, max_leaf_nodes=15, random_state=SEED)),
    ])
    tree_pipe.fit(train_df[ALL_FEATURES], y_train)
    tree_probs = tree_pipe.predict_proba(test_df[ALL_FEATURES])[:, 1]
    pipelines["Hybrid (HistGradientBoosting)"] = tree_pipe
    scores["Hybrid (HistGradientBoosting)"] = tree_probs
    rows.append(metric_row("Hybrid (HistGradientBoosting)", y_test, tree_probs))
    joblib.dump(tree_pipe, MODEL_DIR / "hybrid_hist_gradient_boosting.joblib")

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
        "note": "Metrics demonstrate an engineering pipeline on generated data only. They are not estimates of real-world borrower risk or proof of alternative-data value."
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

    # Test-set permutation importance for the hybrid logistic pipeline.
    # Evaluate importance on the test set for demonstration, not for tuning the model.
    hybrid = pipelines["Hybrid (traditional + alternative)"]
    perm = permutation_importance(
        hybrid, test_df[ALL_FEATURES], y_test,
        scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    imp = pd.DataFrame({"feature": ALL_FEATURES, "auc_drop_mean": perm.importances_mean, "auc_drop_std": perm.importances_std})
    imp.sort_values("auc_drop_mean", ascending=False).to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    top = imp.sort_values("auc_drop_mean", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["auc_drop_mean"], xerr=top["auc_drop_std"])
    ax.set_title("Permutation importance — hybrid logistic model\nSynthetic test data only; association, not causation")
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
        pd.DataFrame(subgroup_rows).to_csv(REPORT_DIR / "no_bureau_history_subgroup_metrics.csv", index=False)

    # A small labelled example set for the UI. This file contains synthetic test samples.
    display_cols = ["synthetic_applicant_id"] + ALL_FEATURES + [TARGET]
    test_df[display_cols].head(200).to_csv(DATA_DIR / "synthetic_demo_cases.csv", index=False)

    print("TRAINING COMPLETE — SYNTHETIC DATA ONLY")
    print(f"Rows: {len(df):,} | Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Synthetic default rate: {df[TARGET].mean():.3f}")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Artifacts saved to data/, models/, and reports/.")
    print("IMPORTANT: These results do not validate real-world credit scoring.")


if __name__ == "__main__":
    main()
