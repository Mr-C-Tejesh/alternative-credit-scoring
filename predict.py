"""Run a sample prediction with the saved HYBRID model (synthetic demonstration only)."""
from pathlib import Path
import argparse
import sys
import joblib
import numpy as np
import pandas as pd

from explainability import explain_logistic_prediction
from risk_scoring import (
    DEFAULT_BASE_ODDS,
    DEFAULT_BASE_SCORE,
    DEFAULT_MAX_SCORE,
    DEFAULT_MIN_SCORE,
    DEFAULT_PDO,
    map_probability_to_risk_band,
    probability_to_score,
)
from validation import ALL_FEATURES, ValidationError, validate_feature_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hybrid_logistic_regression.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic credit-risk model demo; not for real lending decisions.")
    parser.add_argument("--income", type=float, default=40000, help="monthly income in INR")
    parser.add_argument("--age", type=int, default=30)
    parser.add_argument("--employment", type=float, default=4.0, help="employment tenure years")
    parser.add_argument("--dti", type=float, default=0.35, help="debt-to-income ratio, 0 to 1.2")
    parser.add_argument("--loans", type=int, default=1)
    parser.add_argument("--utilization", type=float, default=0.40)
    parser.add_argument("--delinquencies", type=int, default=0)
    parser.add_argument("--no-bureau", action="store_true", help="simulate no bureau history")
    parser.add_argument("--bureau-score", type=float, default=650)
    parser.add_argument("--utility", type=float, default=0.82, help="payment timeliness, 0.05 to 1")
    parser.add_argument("--rental", type=float, default=0.80, help="payment timeliness, 0.05 to 1")
    parser.add_argument("--digital-regularity", type=float, default=0.75)
    parser.add_argument("--cashflow-stability", type=float, default=0.68)
    parser.add_argument("--alternative-months", type=int, default=15)
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        sys.exit("Saved model not found. Run `python train.py` first.")

    row = pd.DataFrame([{
        "age_years": args.age,
        "monthly_income_inr": args.income,
        "employment_tenure_years": args.employment,
        "debt_to_income_ratio": args.dti,
        "existing_loan_count": args.loans,
        "credit_utilization_ratio": args.utilization,
        "prior_delinquencies_24m": args.delinquencies,
        "bureau_history_available": 0 if args.no_bureau else 1,
        "bureau_score": np.nan if args.no_bureau else args.bureau_score,
        "utility_payment_timeliness": args.utility,
        "rental_payment_timeliness": args.rental,
        "digital_payment_regularity": args.digital_regularity,
        "cashflow_stability_index": args.cashflow_stability,
        "alternative_history_months": args.alternative_months,
    }])

    # Pre-inference validation
    try:
        validate_feature_frame(row, expected_features=ALL_FEATURES)
    except ValidationError as err:
        sys.exit(f"Input validation error: {err}")

    model = joblib.load(MODEL_PATH)
    p = float(model.predict_proba(row)[0, 1])

    # Phase 03: Illustrative credit score & risk band
    score_res = probability_to_score(p)
    risk_band = map_probability_to_risk_band(p)

    # Phase 03: Exact additive local feature explanation
    explanation = explain_logistic_prediction(model, row, top_k=3, validate=False)

    print("================================================================================")
    print("SYNTHETIC MODEL DEMO ONLY — PREDICTION & EXPLANATION")
    print("================================================================================")
    print("Model: Hybrid Logistic Regression (synthetic demonstration prototype)")
    print()
    print(f"Estimated probability of default (12m): {p:.1%}")
    print(f"Illustrative Credit Score:             {score_res.display_score} / {DEFAULT_MAX_SCORE} (range: {DEFAULT_MIN_SCORE}–{DEFAULT_MAX_SCORE})")
    print(f"Illustrative Risk Band:                {risk_band}")
    print()
    print("Scoring Model Assumptions:")
    print(f"  • Base score: {int(DEFAULT_BASE_SCORE)} at {int(DEFAULT_BASE_ODDS)}:1 good-to-bad odds (p ≈ {1.0 / (DEFAULT_BASE_ODDS + 1.0):.2%})")
    print(f"  • Points to Double the Odds (PDO): {int(DEFAULT_PDO)}")
    print(f"  • Raw score: {score_res.raw_score:.1f} (clamped to [{DEFAULT_MIN_SCORE}, {DEFAULT_MAX_SCORE}])")
    print("  • Note: Educational transformation only; NOT an official credit bureau score.")
    print()
    print("Local Feature Contributions (Additive log-odds decomposition):")
    print(f"  Baseline Log-Odds Intercept: {explanation.intercept:+.4f}")
    print(f"  Model Decision Log-Odds:     {explanation.model_decision_function:+.4f} (reconstructed: {explanation.reconstructed_logit:+.4f})")
    print()
    print("  Top factors increasing modeled default risk (+ log-odds):")
    for item in explanation.top_risk_increasing:
        print(f"    + {item['readable_name']}: {item['contribution']:+.4f}")
    print()
    print("  Top factors decreasing modeled default risk (- log-odds):")
    for item in explanation.top_risk_decreasing:
        print(f"    - {item['readable_name']}: {item['contribution']:+.4f}")
    print()
    print("IMPORTANT DISCLAIMER:")
    print("This output is generated from synthetic simulation data. It is an educational")
    print("demonstration only. It has NOT been validated for real borrowers, does NOT")
    print("convey credit approval or rejection, and MUST NOT be used for real lending decisions.")
    print("================================================================================")


if __name__ == "__main__":
    main()
