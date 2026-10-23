"""Streamlit demonstration prototype for synthetic alternative credit scoring.

Connects the validated Phase 01–03 ML core components (data validation, hybrid Logistic
Regression inference, PDO risk scoring, risk bands, and exact additive local explanations)
to an interactive interface.

IMPORTANT DISCLAIMER:
This is an educational research demonstration trained strictly on simulated data.
It is NOT validated for real people, credit underwriting, or lending decisions.
"""
from pathlib import Path
import numpy as np
import streamlit as st

from inference_helper import (
    ALT_FEATURE_OPTIONS,
    build_applicant_row,
    load_saved_model,
    run_inference_pipeline,
)
from validation import ALL_FEATURES, ValidationError, validate_feature_frame

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "hybrid_logistic_regression.joblib"

st.set_page_config(
    page_title="Alternative Credit Risk — Research Demo",
    page_icon="📊",
    layout="wide",
)

st.title("Alternative Credit Risk — Research Prototype")

# Mandatory synthetic demonstration warning
st.error(
    "SIMULATED DATA DEMO ONLY. This tool is an educational research prototype trained on simulated data. "
    "It is NOT validated for real people, credit underwriting, or lending decisions. "
    "Do not enter real personal or financial data."
)
st.markdown(
    "Estimate synthetic default risk, illustrative credit score, and exact local feature contributions "
    "for a simulated applicant using the validated hybrid machine learning pipeline."
)

if not MODEL_PATH.exists():
    st.warning("Saved model artifact not found. Run `python train.py` first.")
    st.stop()

try:
    model = load_saved_model(MODEL_PATH)
except Exception as load_err:
    st.error(f"Failed to load model pipeline: {load_err}")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Traditional / Application Features")
    st.caption("Features aligned with synthetic training domain bounds.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years)", min_value=20, max_value=65, value=30, step=1, help="Synthetic generator support: [20, 65]")
        income = st.number_input(
            "Monthly income (₹)",
            min_value=8000,
            max_value=250000,
            value=40000,
            step=1000,
            help="Synthetic generator support: ₹8,000–₹250,000",
        )
        employment = st.slider(
            "Employment tenure (years)",
            min_value=0.0,
            max_value=35.0,
            value=4.0,
            step=0.5,
            help="Synthetic generator support: [0.0, 35.0]",
        )
    with col2:
        dti = st.slider(
            "Debt-to-income ratio",
            min_value=0.0,
            max_value=1.20,
            value=0.35,
            step=0.01,
            help="Synthetic generator support: [0.0, 1.20]",
        )
        loans = st.slider("Existing loan count", min_value=0, max_value=8, value=1, step=1, help="Synthetic generator support: [0, 8]")
        utilization = st.slider(
            "Credit utilization ratio",
            min_value=0.0,
            max_value=1.25,
            value=0.40,
            step=0.01,
            help="Synthetic generator support: [0.0, 1.25]",
        )
    with col3:
        delinquencies = st.slider(
            "Prior delinquencies (last 24 months)",
            min_value=0,
            max_value=5,
            value=0,
            step=1,
            help="Synthetic generator support: [0, 5]",
        )
        has_bureau = st.checkbox(
            "Bureau history available",
            value=True,
            help="Simulates whether formal bureau records exist. If unchecked, bureau score is passed as NaN.",
        )
        bureau_score_input = st.slider(
            "Bureau score (ignored if no bureau history)",
            min_value=300,
            max_value=900,
            value=650,
            step=5,
            help="Synthetic generator support: [300, 900]. Automatically ignored if bureau history is unchecked.",
        )

    st.subheader("Alternative Feature Examples & Record Availability")
    st.caption(
        "These are simulated behavioural summaries. In real research, their lawful availability, "
        "validity, and fairness would require separate evidence and review."
    )

    unavailable_alt = st.multiselect(
        "Mark alternative records as unavailable (simulates missing / unobserved data):",
        options=[k for k, _ in ALT_FEATURE_OPTIONS],
        format_func=lambda k: dict(ALT_FEATURE_OPTIONS).get(k, k),
        default=[],
        help="Selected records will be assigned NaN in the model input, testing the model's missingness indicators.",
    )

    alt1, alt2, alt3 = st.columns(3)
    with alt1:
        utility = st.slider(
            "Utility payment timeliness",
            min_value=0.05,
            max_value=1.0,
            value=0.82,
            step=0.01,
            help="Synthetic generator support: [0.05, 1.0]. Ignored if marked unavailable above.",
        )
        rental = st.slider(
            "Rental payment timeliness",
            min_value=0.05,
            max_value=1.0,
            value=0.80,
            step=0.01,
            help="Synthetic generator support: [0.05, 1.0]. Ignored if marked unavailable above.",
        )
    with alt2:
        digital = st.slider(
            "Digital payment regularity",
            min_value=0.02,
            max_value=1.0,
            value=0.75,
            step=0.01,
            help="Synthetic generator support: [0.02, 1.0]. Ignored if marked unavailable above.",
        )
        cashflow = st.slider(
            "Cash-flow stability index",
            min_value=0.02,
            max_value=1.0,
            value=0.68,
            step=0.01,
            help="Synthetic generator support: [0.02, 1.0]. Ignored if marked unavailable above.",
        )
    with alt3:
        history_months = st.slider(
            "Alternative history observed (months)",
            min_value=0,
            max_value=24,
            value=15,
            step=1,
            help="Synthetic generator support: [0, 24]. Ignored if marked unavailable above.",
        )

    submitted = st.form_submit_button("Estimate synthetic model risk", type="primary")

if submitted:
    row = build_applicant_row(
        age_years=age,
        monthly_income_inr=float(income),
        employment_tenure_years=float(employment),
        debt_to_income_ratio=float(dti),
        existing_loan_count=int(loans),
        credit_utilization_ratio=float(utilization),
        prior_delinquencies_24m=int(delinquencies),
        bureau_history_available=has_bureau,
        bureau_score=float(bureau_score_input),
        utility_payment_timeliness=float(utility),
        rental_payment_timeliness=float(rental),
        digital_payment_regularity=float(digital),
        cashflow_stability_index=float(cashflow),
        alternative_history_months=int(history_months),
        unavailable_alternative_features=unavailable_alt,
    )

    # Pre-inference validation
    try:
        validate_feature_frame(row, expected_features=ALL_FEATURES)
    except ValidationError as val_err:
        st.error(f"Input validation error: {val_err}")
        st.stop()

    try:
        result = run_inference_pipeline(model, row, validate=False)
    except Exception as inf_err:
        st.error(f"Inference execution error: {inf_err}")
        st.stop()

    st.subheader("Model Output (Simulated Research Prototype)")
    st.caption(
        "All metrics are generated by the fitted hybrid Logistic Regression model on simulated data. "
        "Outputs are illustrative and must not be used for credit underwriting."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Estimated Default Probability (12m)",
        f"{result.probability:.1%}",
        help="Synthetic hybrid Logistic Regression model predicted default probability.",
    )
    c2.metric(
        "Illustrative Credit Score",
        f"{result.score_result.display_score} / {result.score_result.max_score}",
        help=f"Raw PDO score: {result.score_result.raw_score:.1f}. Clamped to [{result.score_result.min_score}, {result.score_result.max_score}].",
    )
    c3.metric(
        "Illustrative Risk Band",
        result.risk_band,
        help="Cutoffs: Lower (< 10%), Middle (10%–20%), Higher (>= 20%).",
    )

    st.progress(float(np.clip(result.probability, 0.0, 1.0)))

    if result.score_result.is_clamped:
        st.info(
            f"Score Clamping Notice: The unclamped mathematical raw score was "
            f"{result.score_result.raw_score:.1f}, which was clamped to the display bounds "
            f"[{result.score_result.min_score}, {result.score_result.max_score}]."
        )

    with st.expander("Illustrative Score Assumptions & PDO Parameters", expanded=True):
        st.markdown(
            f"- **Base Score:** {int(result.score_result.base_score)} points at "
            f"{int(result.score_result.base_odds)}:1 good-to-bad odds "
            f"(implied default probability $\\approx {1.0 / (result.score_result.base_odds + 1.0):.2%}$)\n"
            f"- **Points to Double the Odds (PDO):** {int(result.score_result.pdo)} points\n"
            f"- **Display Range:** [{result.score_result.min_score}, {result.score_result.max_score}]\n"
            f"- **Raw Mathematical Score:** `{result.score_result.raw_score:.2f}` "
            f"(Implied good-to-bad odds: `{result.score_result.good_to_bad_odds:.2f}:1`)\n"
            f"- **Educational Transformation:** This score is an illustrative mathematical transformation of default probability. "
            "It is **not** an official credit bureau score (such as CIBIL or FICO), a regulator-approved credit rating, or a commercial lending cutoff."
        )

    st.subheader("Local Feature Explanation (Exact Additive Decomposition)")
    st.markdown(
        "Contributions decompose the fitted hybrid Logistic Regression model's decision function "
        "in log-odds space: $\\text{logit}(p) = \\text{intercept} + \\sum_j (\\text{coef}_j \\cdot x'_j)$. "
        "**Positive contributions (+)** increase the model's estimated log-odds of synthetic default; "
        "**negative contributions (-)** reduce modeled default log-odds. "
        "These values reflect the model's internal mathematical weighting on standardized synthetic features, "
        "**not** real-world causal factors or statutory adverse-action reason codes."
    )
    st.caption(
        f"Baseline Intercept: {result.explanation.intercept:+.4f} | "
        f"Decision Function Log-Odds: {result.explanation.model_decision_function:+.4f} "
        f"(reconstructed: {result.explanation.reconstructed_logit:+.4f}, "
        f"reconstruction error: {result.explanation.reconstruction_error:.2e})"
    )

    col_inc, col_dec = st.columns(2)
    with col_inc:
        st.markdown("##### 🔺 Top Factors Increasing Modeled Default Risk (+ log-odds)")
        for item in result.explanation.top_risk_increasing:
            st.markdown(f"- **{item['readable_name']}**: `{item['contribution']:+.4f}`")

    with col_dec:
        st.markdown("##### 🔻 Top Factors Decreasing Modeled Default Risk (- log-odds)")
        for item in result.explanation.top_risk_decreasing:
            st.markdown(f"- **{item['readable_name']}**: `{item['contribution']:+.4f}`")

    with st.expander("View Full Transformed Feature Contributions Table"):
        st.dataframe(
            result.explanation.contributions_table[[
                "feature", "readable_name", "transformed_value", "coefficient", "contribution"
            ]],
            hide_index=True,
        )

    st.divider()
    st.subheader("Methodological & Regulatory Limitations")
    st.markdown(
        "- **Synthetic Data Demonstration:** This prototype was trained and evaluated strictly on synthetic benchmark records. "
        "Feature distributions, missingness patterns, and outcome mechanics are simulated and do not represent any real lender's customer portfolio.\n"
        "- **No Underwriting or Policy Authority:** Risk bands and PDO scores do not represent credit approval, rejection, pricing, or credit line assignment.\n"
        "- **Not Adverse Action Reason Codes:** The local additive explanations describe mathematical model behavior in transformed log-odds space. "
        "They are not legally compliant adverse-action reason codes under credit regulations (e.g., US ECOA/FCRA or Indian DPDP/RBI guidelines).\n"
        "- **Alternative Data Governance:** In real-world credit systems, non-traditional behavioral data requires strict statutory basis, verifiable data lineage, consumer consent, and thorough fairness/disparate-impact auditing."
    )
