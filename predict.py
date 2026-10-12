"""Run a sample prediction with the saved HYBRID model (synthetic demonstration only)."""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd

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
    parser.add_argument("--utility", type=float, default=0.82, help="payment timeliness, 0 to 1")
    parser.add_argument("--rental", type=float, default=0.80, help="payment timeliness, 0 to 1")
    parser.add_argument("--digital-regularity", type=float, default=0.75)
    parser.add_argument("--cashflow-stability", type=float, default=0.68)
    parser.add_argument("--alternative-months", type=int, default=15)
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit("Saved model not found. Run `python train.py` first.")

    model = joblib.load(MODEL_PATH)
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
    p = float(model.predict_proba(row)[0, 1])
    print("SYNTHETIC MODEL DEMO ONLY — NOT A REAL CREDIT SCORE")
    print(f"Estimated probability under the simulated model: {p:.1%}")
    print("This output is not validated for real borrowers and must not be used for lending decisions.")


if __name__ == "__main__":
    main()
