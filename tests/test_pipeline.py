"""Unit tests for the synthetic credit-scoring pipeline.

Phase-01 additions:
- Preserved all three original tests.
- Added coverage for:
    * Dataset invariants (target exclusion from predictors, correct schema, binary
      values, expected missingness patterns).
    * Pipeline construction: each builder returns the correct estimator type and
      step names.
    * Pipeline fit/predict: shape, finite values, probability bounds.
    * Redundant-indicator correction: trained pipelines must NOT expose a
      missingindicator_bureau_score column.
    * Missing alternative data: hybrid inference works when all five alternative
      fields are NaN in a prediction row.
    * Missing bureau history: bureau_score NaN + bureau_history_available==0 is
      handled without error.
    * Serialized artifacts: all three saved models load and return valid probs.

Phase-02 additions:
- Comprehensive schema and input contract validation tests (TestSchemaValidation):
    * Valid dataset and feature frames pass without error.
    * Missing target, missing predictors, duplicate columns, duplicate applicant IDs.
    * Non-binary and null targets rejected early.
    * Malformed string values and infinite numbers rejected.
    * Invalid binary flags and inconsistent bureau history / score combinations.
    * Prohibited missing traditional values vs. permitted missing alternative values.
    * Accidental inclusion of target as a predictor rejected.
- Credit-risk domain evaluation metric tests (TestCreditRiskMetrics):
    * Gini identity with ROC-AUC (Gini = 2 * ROC_AUC - 1).
    * Kolmogorov-Smirnov (KS) statistic on perfect, random, and toy examples.
    * Expected Calibration Error (ECE) on perfectly calibrated toy distributions.
    * ECE handling of empty bins, probability bounds 0.0 and 1.0.
    * Input validation and rejection of malformed or invalid metric inputs.
    * Metric row dictionary keys and finite values.
- Calibration and reporting integration tests (TestCalibrationAndReporting):
    * Generated metrics CSV files contain all expected new metric columns.
    * Calibration curves image artifact exists and is non-empty.
    * Run summary JSON records evaluated metrics and binning configuration.
    * Serialized models remain loadable and output valid predictions.

Phase-02.1 additions:
- Focused generator-aligned boundary tests (TestBoundaryConditionsAndGeneratorSupport):
    * Exact minimum boundary support acceptance across all features.
    * Exact maximum boundary support acceptance across all features.
    * Out-of-support lower boundary rejection for all features.
    * Out-of-support upper boundary rejection for all features.
    * Rejection of fractional values on integer-only features.
    * Structural validity of no-bureau boundary case.
    * Support for all-alternative-missing cold-start inference.
    * CLI subprocess smoke tests for valid default, valid no-bureau, and out-of-support inputs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train  # noqa: E402
import validation  # noqa: E402
import joblib  # noqa: E402
import explainability  # noqa: E402
import risk_scoring  # noqa: E402
import inference_helper  # noqa: E402
import experiments.alternative_feature_ablation as ablation_exp  # noqa: E402
import uncertainty_analysis  # noqa: E402
import stress_testing  # noqa: E402



# ---------------------------------------------------------------------------
# Shared test fixtures (small, deterministic)
# ---------------------------------------------------------------------------

def _small_dataset(n: int = 300, seed: int = 7) -> pd.DataFrame:
    return train.generate_synthetic_dataset(n=n, seed=seed)


def _train_test_split(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=0.25, random_state=42, stratify=df[train.TARGET])


# ---------------------------------------------------------------------------
# 1. Dataset behavior
# ---------------------------------------------------------------------------

class TestDatasetBehavior(unittest.TestCase):

    def setUp(self):
        self.df = _small_dataset()

    def test_generator_reproducibility(self):
        """Identical seeds must produce identical DataFrames."""
        a = train.generate_synthetic_dataset(n=100, seed=7)
        b = train.generate_synthetic_dataset(n=100, seed=7)
        self.assertTrue(a.equals(b))

    def test_different_seeds_differ(self):
        """Different seeds must NOT produce identical DataFrames."""
        a = train.generate_synthetic_dataset(n=100, seed=7)
        b = train.generate_synthetic_dataset(n=100, seed=99)
        self.assertFalse(a.equals(b))

    def test_schema_and_binary_target(self):
        """Required predictor columns and a binary target must be present."""
        df = self.df
        self.assertTrue(set(train.ALL_FEATURES).issubset(df.columns))
        self.assertEqual(set(df[train.TARGET].unique()), {0, 1})

    def test_target_not_in_predictor_lists(self):
        """Target column must not appear in any feature list."""
        for feature_list in (
            train.TRADITIONAL_FEATURES,
            train.ALTERNATIVE_FEATURES,
            train.ALL_FEATURES,
        ):
            self.assertNotIn(train.TARGET, feature_list)

    def test_expected_missingness_bureau_score(self):
        """bureau_score must be NaN exactly when bureau_history_available == 0."""
        df = self.df
        self.assertTrue(df["bureau_score"].isna().any())
        no_bureau = df["bureau_history_available"] == 0
        self.assertTrue(df.loc[no_bureau, "bureau_score"].isna().all())
        self.assertTrue(df.loc[~no_bureau, "bureau_score"].notna().all())

    def test_expected_missingness_alt_features(self):
        """Several alternative features must contain some NaN values."""
        df = self.df
        for col in ["utility_payment_timeliness", "rental_payment_timeliness",
                    "digital_payment_regularity", "cashflow_stability_index"]:
            self.assertTrue(df[col].isna().any(), f"{col} has no NaN values")

    def test_alternative_history_months_no_missingness(self):
        """alternative_history_months must have no NaN values."""
        df = self.df
        self.assertFalse(df["alternative_history_months"].isna().any())


# ---------------------------------------------------------------------------
# 2. Pipeline construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction(unittest.TestCase):
    """Builder functions must return properly structured Pipeline objects."""

    def test_build_traditional_lr_pipeline_structure(self):
        pipe = train.build_traditional_lr_pipeline()
        self.assertIn("preprocess", pipe.named_steps)
        self.assertIn("classifier", pipe.named_steps)
        from sklearn.linear_model import LogisticRegression
        self.assertIsInstance(pipe.named_steps["classifier"], LogisticRegression)

    def test_build_hybrid_lr_pipeline_structure(self):
        pipe = train.build_hybrid_lr_pipeline()
        self.assertIn("preprocess", pipe.named_steps)
        self.assertIn("classifier", pipe.named_steps)
        from sklearn.linear_model import LogisticRegression
        self.assertIsInstance(pipe.named_steps["classifier"], LogisticRegression)

    def test_build_hybrid_hgb_pipeline_structure(self):
        pipe = train.build_hybrid_hgb_pipeline()
        self.assertIn("preprocess", pipe.named_steps)
        self.assertIn("classifier", pipe.named_steps)
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.assertIsInstance(pipe.named_steps["classifier"], HistGradientBoostingClassifier)

    def test_pipeline_builders_return_new_instances(self):
        """Each call to a builder must return a fresh, unfitted instance."""
        p1 = train.build_traditional_lr_pipeline()
        p2 = train.build_traditional_lr_pipeline()
        self.assertIsNot(p1, p2)
        self.assertIsNot(p1.named_steps["classifier"], p2.named_steps["classifier"])


# ---------------------------------------------------------------------------
# 3. Pipeline fit and predict
# ---------------------------------------------------------------------------

class TestPipelineFitPredict(unittest.TestCase):
    """Pipelines must fit without error and produce valid probabilities."""

    @classmethod
    def setUpClass(cls):
        df = _small_dataset(n=400, seed=42)
        cls.train_df, cls.test_df = _train_test_split(df)
        cls.y_train = cls.train_df[train.TARGET]
        cls.y_test = cls.test_df[train.TARGET]

    def test_traditional_lr_fit_predict_shape(self):
        pipe = train.build_traditional_lr_pipeline()
        pipe.fit(self.train_df[train.TRADITIONAL_FEATURES], self.y_train)
        probs = pipe.predict_proba(self.test_df[train.TRADITIONAL_FEATURES])
        self.assertEqual(probs.shape, (len(self.test_df), 2))

    def test_traditional_lr_probabilities_valid(self):
        pipe = train.build_traditional_lr_pipeline()
        pipe.fit(self.train_df[train.TRADITIONAL_FEATURES], self.y_train)
        probs = pipe.predict_proba(self.test_df[train.TRADITIONAL_FEATURES])[:, 1]
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_hybrid_lr_fit_predict_shape(self):
        pipe = train.build_hybrid_lr_pipeline()
        pipe.fit(self.train_df[train.ALL_FEATURES], self.y_train)
        probs = pipe.predict_proba(self.test_df[train.ALL_FEATURES])
        self.assertEqual(probs.shape, (len(self.test_df), 2))

    def test_hybrid_lr_probabilities_valid(self):
        pipe = train.build_hybrid_lr_pipeline()
        pipe.fit(self.train_df[train.ALL_FEATURES], self.y_train)
        probs = pipe.predict_proba(self.test_df[train.ALL_FEATURES])[:, 1]
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_hybrid_hgb_fit_predict_shape(self):
        pipe = train.build_hybrid_hgb_pipeline()
        pipe.fit(self.train_df[train.ALL_FEATURES], self.y_train)
        probs = pipe.predict_proba(self.test_df[train.ALL_FEATURES])
        self.assertEqual(probs.shape, (len(self.test_df), 2))

    def test_hybrid_hgb_probabilities_valid(self):
        pipe = train.build_hybrid_hgb_pipeline()
        pipe.fit(self.train_df[train.ALL_FEATURES], self.y_train)
        probs = pipe.predict_proba(self.test_df[train.ALL_FEATURES])[:, 1]
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))


# ---------------------------------------------------------------------------
# 4. Redundant indicator correction
# ---------------------------------------------------------------------------

class TestRedundantIndicatorCorrection(unittest.TestCase):
    """Trained pipelines must NOT expose missingindicator_bureau_score."""

    @classmethod
    def setUpClass(cls):
        df = _small_dataset(n=400, seed=99)
        train_df, cls.test_df = _train_test_split(df)
        y_train = train_df[train.TARGET]

        cls.trad_pipe = train.build_traditional_lr_pipeline()
        cls.trad_pipe.fit(train_df[train.TRADITIONAL_FEATURES], y_train)

        cls.hybrid_lr_pipe = train.build_hybrid_lr_pipeline()
        cls.hybrid_lr_pipe.fit(train_df[train.ALL_FEATURES], y_train)

        cls.hybrid_hgb_pipe = train.build_hybrid_hgb_pipeline()
        cls.hybrid_hgb_pipe.fit(train_df[train.ALL_FEATURES], y_train)

    def _feature_names(self, pipeline):
        return list(pipeline.named_steps["preprocess"].get_feature_names_out())

    def test_traditional_pipeline_no_bureau_score_indicator(self):
        names = self._feature_names(self.trad_pipe)
        bureau_indicators = [n for n in names if "missingindicator_bureau_score" in n]
        self.assertEqual(bureau_indicators, [])

    def test_traditional_pipeline_retains_bureau_history_feature(self):
        names = self._feature_names(self.trad_pipe)
        self.assertTrue(any("bureau_history_available" in n for n in names))

    def test_hybrid_lr_pipeline_no_bureau_score_indicator(self):
        names = self._feature_names(self.hybrid_lr_pipe)
        bureau_indicators = [n for n in names if "missingindicator_bureau_score" in n]
        self.assertEqual(bureau_indicators, [])

    def test_hybrid_lr_pipeline_retains_alt_indicators(self):
        names = self._feature_names(self.hybrid_lr_pipe)
        alt_indicators = [n for n in names if "missingindicator" in n]
        self.assertEqual(len(alt_indicators), 4)

    def test_hybrid_hgb_pipeline_no_bureau_score_indicator(self):
        names = self._feature_names(self.hybrid_hgb_pipe)
        bureau_indicators = [n for n in names if "missingindicator_bureau_score" in n]
        self.assertEqual(bureau_indicators, [])

    def test_hybrid_hgb_pipeline_retains_alt_indicators(self):
        names = self._feature_names(self.hybrid_hgb_pipe)
        alt_indicators = [n for n in names if "missingindicator" in n]
        self.assertEqual(len(alt_indicators), 4)


# ---------------------------------------------------------------------------
# 5. Missing alternative data inference handling
# ---------------------------------------------------------------------------

class TestMissingAlternativeData(unittest.TestCase):
    """Hybrid models must handle entirely missing alternative features at inference."""

    @classmethod
    def setUpClass(cls):
        df = _small_dataset(n=400, seed=11)
        train_df, cls.test_df = _train_test_split(df)
        y_train = train_df[train.TARGET]

        cls.hybrid_lr_pipe = train.build_hybrid_lr_pipeline()
        cls.hybrid_lr_pipe.fit(train_df[train.ALL_FEATURES], y_train)

        cls.hybrid_hgb_pipe = train.build_hybrid_hgb_pipeline()
        cls.hybrid_hgb_pipe.fit(train_df[train.ALL_FEATURES], y_train)

        base_row = train_df[train.ALL_FEATURES].iloc[[0]].copy()
        for col in train.ALTERNATIVE_FEATURES:
            base_row[col] = float("nan")
        cls.all_alt_missing_row = base_row

        no_bureau_row = train_df[train.ALL_FEATURES].iloc[[0]].copy()
        no_bureau_row["bureau_history_available"] = 0
        no_bureau_row["bureau_score"] = float("nan")
        cls.no_bureau_row = no_bureau_row

    def test_hybrid_lr_all_alt_features_missing(self):
        probs = self.hybrid_lr_pipe.predict_proba(self.all_alt_missing_row)[:, 1]
        self.assertEqual(len(probs), 1)
        self.assertTrue(np.isfinite(probs[0]))
        self.assertTrue(0.0 <= probs[0] <= 1.0)

    def test_hybrid_hgb_all_alt_features_missing(self):
        probs = self.hybrid_hgb_pipe.predict_proba(self.all_alt_missing_row)[:, 1]
        self.assertEqual(len(probs), 1)
        self.assertTrue(np.isfinite(probs[0]))
        self.assertTrue(0.0 <= probs[0] <= 1.0)

    def test_hybrid_lr_no_bureau_history_row(self):
        probs = self.hybrid_lr_pipe.predict_proba(self.no_bureau_row)[:, 1]
        self.assertEqual(len(probs), 1)
        self.assertTrue(np.isfinite(probs[0]))
        self.assertTrue(0.0 <= probs[0] <= 1.0)

    def test_hybrid_hgb_no_bureau_history_row(self):
        probs = self.hybrid_hgb_pipe.predict_proba(self.no_bureau_row)[:, 1]
        self.assertEqual(len(probs), 1)
        self.assertTrue(np.isfinite(probs[0]))
        self.assertTrue(0.0 <= probs[0] <= 1.0)


# ---------------------------------------------------------------------------
# 6. Serialized artifact tests
# ---------------------------------------------------------------------------

class TestSerializedArtifacts(unittest.TestCase):

    TRAD_PATH = ROOT / "models" / "traditional_baseline.joblib"
    HYBRID_LR_PATH = ROOT / "models" / "hybrid_logistic_regression.joblib"
    HYBRID_HGB_PATH = ROOT / "models" / "hybrid_hist_gradient_boosting.joblib"

    def setUp(self):
        self.df = _small_dataset(n=100)

    def test_traditional_baseline_loads(self):
        import joblib
        if not self.TRAD_PATH.exists():
            self.skipTest("Model file not found; run train.py first")
        model = joblib.load(self.TRAD_PATH)
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_traditional_baseline_inference(self):
        import joblib
        if not self.TRAD_PATH.exists():
            self.skipTest("Model file not found; run train.py first")
        model = joblib.load(self.TRAD_PATH)
        probs = model.predict_proba(self.df[train.TRADITIONAL_FEATURES])[:, 1]
        self.assertEqual(len(probs), len(self.df))
        self.assertTrue(np.all(np.isfinite(probs)))

    def test_hybrid_lr_loads(self):
        import joblib
        if not self.HYBRID_LR_PATH.exists():
            self.skipTest("Model file not found; run train.py first")
        model = joblib.load(self.HYBRID_LR_PATH)
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_hybrid_lr_inference(self):
        import joblib
        if not self.HYBRID_LR_PATH.exists():
            self.skipTest("Model file not found; run train.py first")
        model = joblib.load(self.HYBRID_LR_PATH)
        probs = model.predict_proba(self.df[train.ALL_FEATURES])[:, 1]
        self.assertEqual(len(probs), len(self.df))
        self.assertTrue(np.all(np.isfinite(probs)))

    def test_hybrid_lr_probability_in_range(self):
        import joblib
        if not self.HYBRID_LR_PATH.exists():
            self.skipTest("Model file not found; run train.py first")
        model = joblib.load(self.HYBRID_LR_PATH)
        probs = model.predict_proba(self.df[train.ALL_FEATURES])[:, 1]
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_hybrid_hgb_loads(self):
        import joblib
        if not self.HYBRID_HGB_PATH.exists():
            self.skipTest("Model file not found; run train.py first")
        model = joblib.load(self.HYBRID_HGB_PATH)
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_hybrid_hgb_inference(self):
        import joblib
        if not self.HYBRID_HGB_PATH.exists():
            self.skipTest("Model file not found; run train.py first")
        model = joblib.load(self.HYBRID_HGB_PATH)
        probs = model.predict_proba(self.df[train.ALL_FEATURES])[:, 1]
        self.assertEqual(len(probs), len(self.df))
        self.assertTrue(np.all(np.isfinite(probs)))


# ---------------------------------------------------------------------------
# 7. Backward-compatibility check for serialized models
# ---------------------------------------------------------------------------

class TestSerializedArtifactCompatibility(unittest.TestCase):
    HYBRID_FILES = [
        ("hybrid_logistic_regression.joblib",    train.ALL_FEATURES),
        ("hybrid_hist_gradient_boosting.joblib", train.ALL_FEATURES),
    ]
    TRAD_FILES = [
        ("traditional_baseline.joblib", train.TRADITIONAL_FEATURES),
    ]

    @classmethod
    def setUpClass(cls):
        import joblib
        cls.models = {}
        for filename, _ in cls.HYBRID_FILES + cls.TRAD_FILES:
            path = ROOT / "models" / filename
            if path.exists():
                try:
                    cls.models[filename] = joblib.load(path)
                except Exception:
                    pass

    def _feature_names(self, model):
        return list(model.named_steps["preprocess"].get_feature_names_out())

    def _check_no_redundant_indicator(self, filename):
        if filename not in self.models:
            self.skipTest(f"{filename} not found; run train.py first")
        names = self._feature_names(self.models[filename])
        redundant = [n for n in names if "missingindicator_bureau_score" in n]
        self.assertEqual(
            redundant, [],
            msg=(
                f"{filename} still contains the redundant bureau-score indicator "
                f"{redundant}. Run `python train.py` to regenerate the artifacts "
                f"with the corrected preprocessor."
            ),
        )

    def test_traditional_baseline_no_redundant_indicator(self):
        self._check_no_redundant_indicator("traditional_baseline.joblib")

    def test_hybrid_lr_no_redundant_indicator(self):
        self._check_no_redundant_indicator("hybrid_logistic_regression.joblib")

    def test_hybrid_hgb_no_redundant_indicator(self):
        self._check_no_redundant_indicator("hybrid_hist_gradient_boosting.joblib")


# ---------------------------------------------------------------------------
# 8. Schema validation tests (Phase 02 — Objective A & E)
# ---------------------------------------------------------------------------

class TestSchemaValidation(unittest.TestCase):
    """Test suite covering dataset schema and feature frame data contracts."""

    def setUp(self):
        self.df = _small_dataset(n=200, seed=123)

    def test_valid_dataset_schema_passes(self):
        """The standard generated dataset must validate with zero errors."""
        validation.validate_dataset_schema(self.df)

    def test_dataset_missing_target_raises(self):
        """Omitting the binary target column from the dataset must fail validation."""
        bad_df = self.df.drop(columns=[train.TARGET])
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("Missing required dataset column", str(cm.exception))

    def test_dataset_missing_predictor_raises(self):
        """Omitting a required feature column from the dataset must fail validation."""
        bad_df = self.df.drop(columns=["monthly_income_inr"])
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("monthly_income_inr", str(cm.exception))

    def test_dataset_duplicate_column_raises(self):
        """Duplicate column names in dataset must be rejected."""
        bad_df = pd.concat([self.df, self.df[["age_years"]]], axis=1)
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("Duplicate column names", str(cm.exception))

    def test_dataset_duplicate_applicant_id_raises(self):
        """Duplicate synthetic applicant IDs must be rejected."""
        bad_df = self.df.copy()
        bad_df.loc[1, "synthetic_applicant_id"] = bad_df.loc[0, "synthetic_applicant_id"]
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("Duplicate synthetic applicant identifier", str(cm.exception))

    def test_dataset_non_binary_target_raises(self):
        """Non-binary target values (e.g. 2) must be rejected."""
        bad_df = self.df.copy()
        bad_df.loc[0, train.TARGET] = 2
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("binary values in {0, 1}", str(cm.exception))

    def test_dataset_null_target_raises(self):
        """Missing values in the target column must be rejected."""
        bad_df = self.df.copy()
        bad_df.loc[0, train.TARGET] = np.nan
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("must not contain missing values", str(cm.exception))

    def test_dataset_malformed_string_feature_raises(self):
        """String or object columns masquerading as features must fail without silent coercion."""
        bad_df = self.df.copy()
        bad_df["monthly_income_inr"] = bad_df["monthly_income_inr"].astype(str)
        bad_df.loc[0, "monthly_income_inr"] = "forty_thousand"
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("must be numeric", str(cm.exception))

    def test_dataset_infinite_feature_raises(self):
        """Infinite numeric values (+inf or -inf) must be rejected."""
        bad_df = self.df.copy()
        bad_df.loc[0, "debt_to_income_ratio"] = np.inf
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("infinite values", str(cm.exception))

    def test_dataset_invalid_binary_flag_raises(self):
        """bureau_history_available values outside {0, 1} must be rejected."""
        bad_df = self.df.copy()
        bad_df.loc[0, "bureau_history_available"] = 5
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("Allowed values are {0, 1}", str(cm.exception))

    def test_dataset_inconsistent_bureau_score_no_history_raises(self):
        """A row with bureau_history_available == 0 but non-NaN bureau_score violates contract."""
        bad_df = self.df.copy()
        no_bureau_idx = bad_df[bad_df["bureau_history_available"] == 0].index[0]
        bad_df.loc[no_bureau_idx, "bureau_score"] = 650.0
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("bureau contract", str(cm.exception))

    def test_dataset_inconsistent_bureau_score_has_history_raises(self):
        """A row with bureau_history_available == 1 but NaN bureau_score violates contract."""
        bad_df = self.df.copy()
        has_bureau_idx = bad_df[bad_df["bureau_history_available"] == 1].index[0]
        bad_df.loc[has_bureau_idx, "bureau_score"] = np.nan
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("bureau contract", str(cm.exception))

    def test_dataset_prohibited_missing_traditional_raises(self):
        """Missing values in traditional features other than bureau_score are prohibited."""
        bad_df = self.df.copy()
        bad_df.loc[0, "age_years"] = np.nan
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("prohibited by the dataset contract", str(cm.exception))

    def test_dataset_negative_value_where_prohibited_raises(self):
        """Negative values where prohibited by domain bounds must be rejected."""
        bad_df = self.df.copy()
        bad_df.loc[0, "monthly_income_inr"] = -500.0
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("outside the supported domain", str(cm.exception))

    def test_dataset_out_of_range_value_raises(self):
        """Values exceeding maximum domain bounds must be rejected."""
        bad_df = self.df.copy()
        bad_df.loc[0, "debt_to_income_ratio"] = 2.5  # bound is [0.0, 1.2]
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_dataset_schema(bad_df)
        self.assertIn("outside the supported domain", str(cm.exception))

    def test_feature_frame_valid_traditional_passes(self):
        """Valid traditional feature frame passes validation."""
        X_trad = self.df[train.TRADITIONAL_FEATURES]
        validation.validate_feature_frame(X_trad, train.TRADITIONAL_FEATURES)

    def test_feature_frame_valid_hybrid_passes(self):
        """Valid hybrid feature frame passes validation."""
        X_all = self.df[train.ALL_FEATURES]
        validation.validate_feature_frame(X_all, train.ALL_FEATURES)

    def test_feature_frame_rejects_target(self):
        """Passing target column inside feature frame must be rejected."""
        bad_X = self.df[train.ALL_FEATURES + [train.TARGET]]
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_feature_frame(bad_X, train.ALL_FEATURES)
        self.assertIn("cannot inadvertently be passed as an input feature", str(cm.exception))

    def test_feature_frame_rejects_unexpected_column(self):
        """Extra columns not in expected_features must be rejected."""
        bad_X = self.df[train.TRADITIONAL_FEATURES].copy()
        bad_X["unregistered_col"] = 42
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_feature_frame(bad_X, train.TRADITIONAL_FEATURES)
        self.assertIn("Unexpected column(s)", str(cm.exception))

    def test_feature_frame_rejects_missing_column(self):
        """Missing expected columns must be rejected."""
        bad_X = self.df[train.TRADITIONAL_FEATURES].drop(columns=["age_years"])
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_feature_frame(bad_X, train.TRADITIONAL_FEATURES)
        self.assertIn("Missing required predictor column", str(cm.exception))

    def test_feature_frame_rejects_non_dataframe(self):
        """Passing non-DataFrame object must be rejected."""
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_feature_frame([1, 2, 3])  # type: ignore
        self.assertIn("Expected a pandas DataFrame", str(cm.exception))

    def test_feature_frame_rejects_empty_dataframe(self):
        """Passing an empty DataFrame must be rejected."""
        empty_df = pd.DataFrame(columns=train.ALL_FEATURES)
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_feature_frame(empty_df, train.ALL_FEATURES)
        self.assertIn("contains 0 rows", str(cm.exception))

    def test_feature_frame_permits_all_alternative_missing(self):
        """Inference with all 5 alternative fields missing (NaN) must be accepted."""
        row = self.df[train.ALL_FEATURES].iloc[[0]].copy()
        for col in train.ALTERNATIVE_FEATURES:
            row[col] = np.nan
        validation.validate_feature_frame(row, train.ALL_FEATURES)

    def test_feature_frame_permits_valid_missing_bureau(self):
        """Inference row with bureau_history_available==0 and bureau_score NaN must be accepted."""
        row = self.df[train.ALL_FEATURES].iloc[[0]].copy()
        row["bureau_history_available"] = 0
        row["bureau_score"] = np.nan
        validation.validate_feature_frame(row, train.ALL_FEATURES)

    def test_feature_frame_rejects_invalid_bureau_combination(self):
        """Inference row with bureau_history_available==0 but bureau_score present must fail."""
        row = self.df[train.ALL_FEATURES].iloc[[0]].copy()
        row["bureau_history_available"] = 0
        row["bureau_score"] = 650.0
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_feature_frame(row, train.ALL_FEATURES)
        self.assertIn("bureau_score must be NaN", str(cm.exception))


# ---------------------------------------------------------------------------
# 9. Credit-risk domain evaluation metric tests (Phase 02 — Objective B & E)
# ---------------------------------------------------------------------------

class TestCreditRiskMetrics(unittest.TestCase):
    """Test suite covering Gini, KS statistic, ECE, and evaluation helpers."""

    def test_gini_identity_with_roc_auc(self):
        """Gini must exactly satisfy Gini = 2 * ROC_AUC - 1."""
        test_cases = [
            (0.5, 0.0),
            (1.0, 1.0),
            (0.0, -1.0),
            (0.75, 0.5),
            (0.85, 0.7),
        ]
        for auc, expected_gini in test_cases:
            self.assertAlmostEqual(train.calculate_gini(auc), expected_gini, places=7)

    def test_gini_input_validation(self):
        """Non-finite or out-of-range AUC values must raise ValueError."""
        with self.assertRaises(ValueError):
            train.calculate_gini(float("nan"))
        with self.assertRaises(ValueError):
            train.calculate_gini(1.2)
        with self.assertRaises(ValueError):
            train.calculate_gini(-0.1)

    def test_ks_statistic_perfect_separation(self):
        """Perfect separation between defaults and non-defaults must yield KS == 1.0."""
        y_true = np.array([0, 0, 1, 1])
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        ks = train.kolmogorov_smirnov_statistic(y_true, probs)
        self.assertAlmostEqual(ks, 1.0, places=7)

    def test_ks_statistic_no_separation(self):
        """Identical scores for all samples must yield KS == 0.0."""
        y_true = np.array([0, 0, 1, 1])
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        ks = train.kolmogorov_smirnov_statistic(y_true, probs)
        self.assertAlmostEqual(ks, 0.0, places=7)

    def test_ks_statistic_range(self):
        """KS statistic must always fall within [0.0, 1.0]."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        probs = np.array([0.2, 0.4, 0.3, 0.7, 0.6, 0.8])
        ks = train.kolmogorov_smirnov_statistic(y_true, probs)
        self.assertTrue(0.0 <= ks <= 1.0)
        self.assertTrue(np.isfinite(ks))

    def test_ks_statistic_input_validation(self):
        """Malformed arrays, single-class targets, or invalid probabilities must raise ValueError."""
        # Single class
        with self.assertRaises(ValueError):
            train.kolmogorov_smirnov_statistic([0, 0, 0], [0.1, 0.2, 0.3])
        # Non-binary labels
        with self.assertRaises(ValueError):
            train.kolmogorov_smirnov_statistic([0, 1, 2], [0.1, 0.2, 0.3])
        # Length mismatch
        with self.assertRaises(ValueError):
            train.kolmogorov_smirnov_statistic([0, 1], [0.1])
        # Non-finite probability
        with self.assertRaises(ValueError):
            train.kolmogorov_smirnov_statistic([0, 1], [0.1, np.nan])
        # Out-of-bounds probability
        with self.assertRaises(ValueError):
            train.kolmogorov_smirnov_statistic([0, 1], [-0.1, 0.5])

    def test_ece_perfectly_calibrated_toy(self):
        """A toy distribution where bin accuracy equals confidence must have near-zero ECE."""
        # 100 samples with prob 0.2 (20 defaults), 100 with prob 0.8 (80 defaults)
        y_true = np.array([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
        probs = np.array([0.2] * 100 + [0.8] * 100)
        ece = train.expected_calibration_error(y_true, probs, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=5)

    def test_ece_boundary_zero_and_one(self):
        """Probabilities of exactly 0.0 and 1.0 must be handled correctly without error."""
        y_true = np.array([0, 1])
        probs = np.array([0.0, 1.0])
        ece = train.expected_calibration_error(y_true, probs, n_bins=10)
        self.assertAlmostEqual(ece, 0.0, places=7)

    def test_ece_empty_bins_handled(self):
        """Sparse distributions with empty bins must produce a valid finite number without NaN."""
        y_true = np.array([0] * 9 + [1])
        probs = np.array([0.05] * 9 + [0.95])
        ece = train.expected_calibration_error(y_true, probs, n_bins=10)
        self.assertTrue(np.isfinite(ece))
        self.assertTrue(0.0 <= ece <= 1.0)

    def test_ece_input_validation(self):
        """Invalid inputs to ECE must raise ValueError."""
        # Single class
        with self.assertRaises(ValueError):
            train.expected_calibration_error([0, 0, 0], [0.1, 0.2, 0.3])
        # Length mismatch
        with self.assertRaises(ValueError):
            train.expected_calibration_error([0, 1], [0.1])
        # Non-finite prob
        with self.assertRaises(ValueError):
            train.expected_calibration_error([0, 1], [np.nan, 0.5])
        # Out of bounds prob
        with self.assertRaises(ValueError):
            train.expected_calibration_error([0, 1], [1.1, 0.5])
        # Invalid n_bins
        with self.assertRaises(ValueError):
            train.expected_calibration_error([0, 1], [0.2, 0.8], n_bins=0)

    def test_metric_row_returns_all_columns(self):
        """metric_row helper must return all expected discrimination and calibration fields."""
        y_true = pd.Series([0, 1, 0, 1, 1, 0, 1, 0])
        probs = np.array([0.1, 0.85, 0.2, 0.75, 0.9, 0.15, 0.8, 0.3])
        row = train.metric_row("TestModel", y_true, probs, n_bins=10)

        expected_keys = {
            "model",
            "roc_auc",
            "gini_coefficient",
            "ks_statistic",
            "average_precision",
            "brier_score",
            "log_loss",
            "ece_10_bins",
        }
        self.assertEqual(set(row.keys()), expected_keys)
        for k in expected_keys - {"model"}:
            self.assertTrue(np.isfinite(row[k]), f"{k} is not finite")


# ---------------------------------------------------------------------------
# 10. Calibration & reporting integration tests (Phase 02 — Objective C & D)
# ---------------------------------------------------------------------------

class TestCalibrationAndReporting(unittest.TestCase):
    """Verify generated report artifacts and metrics files contain the required content."""

    def test_test_metrics_csv_contains_new_fields(self):
        csv_path = ROOT / "reports" / "test_metrics.csv"
        if not csv_path.exists():
            self.skipTest("test_metrics.csv not found; run train.py first")
        df = pd.read_csv(csv_path)
        expected_cols = {
            "model",
            "roc_auc",
            "gini_coefficient",
            "ks_statistic",
            "average_precision",
            "brier_score",
            "log_loss",
            "ece_10_bins",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        self.assertEqual(len(df), 3)

    def test_subgroup_metrics_csv_contains_new_fields(self):
        csv_path = ROOT / "reports" / "no_bureau_history_subgroup_metrics.csv"
        if not csv_path.exists():
            self.skipTest("no_bureau_history_subgroup_metrics.csv not found; run train.py first")
        df = pd.read_csv(csv_path)
        expected_cols = {
            "model",
            "roc_auc",
            "gini_coefficient",
            "ks_statistic",
            "average_precision",
            "brier_score",
            "log_loss",
            "ece_10_bins",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_calibration_curves_image_exists(self):
        img_path = ROOT / "reports" / "calibration_curves.png"
        self.assertTrue(img_path.exists(), "reports/calibration_curves.png was not generated")
        self.assertGreater(img_path.stat().st_size, 10_000, "calibration_curves.png is unexpectedly small")

    def test_run_summary_json_contains_calibration_info(self):
        summary_path = ROOT / "reports" / "run_summary.json"
        if not summary_path.exists():
            self.skipTest("run_summary.json not found; run train.py first")
        data = json.loads(summary_path.read_text())
        self.assertIn("metrics_evaluated", data)
        self.assertIn("gini_coefficient", data["metrics_evaluated"])
        self.assertIn("ks_statistic", data["metrics_evaluated"])
        self.assertIn("ece_10_bins", data["metrics_evaluated"])
        self.assertEqual(data.get("calibration_bins"), 10)
        self.assertEqual(data.get("calibration_strategy"), "uniform")


# ---------------------------------------------------------------------------
# 11. Generator support boundary tests (Phase 02.1)
# ---------------------------------------------------------------------------

class TestBoundaryConditionsAndGeneratorSupport(unittest.TestCase):
    """Test boundary conditions aligned strictly with synthetic generator support."""

    @classmethod
    def setUpClass(cls):
        # A valid baseline applicant within generator support
        cls.valid_base = {
            "age_years": 35,
            "monthly_income_inr": 45000.0,
            "employment_tenure_years": 5.0,
            "debt_to_income_ratio": 0.35,
            "existing_loan_count": 2,
            "credit_utilization_ratio": 0.40,
            "prior_delinquencies_24m": 1,
            "bureau_history_available": 1,
            "bureau_score": 680.0,
            "utility_payment_timeliness": 0.85,
            "rental_payment_timeliness": 0.82,
            "digital_payment_regularity": 0.78,
            "cashflow_stability_index": 0.70,
            "alternative_history_months": 15,
        }

    def _make_row(self, overrides: dict | None = None) -> pd.DataFrame:
        row = self.valid_base.copy()
        if overrides:
            row.update(overrides)
        return pd.DataFrame([row])

    def test_exact_minimum_boundaries_accepted(self):
        """Values exactly at generator-supported minimums must pass validation."""
        min_row = pd.DataFrame([{
            "age_years": 20,
            "monthly_income_inr": 8000.0,
            "employment_tenure_years": 0.0,
            "debt_to_income_ratio": 0.0,
            "existing_loan_count": 0,
            "credit_utilization_ratio": 0.0,
            "prior_delinquencies_24m": 0,
            "bureau_history_available": 1,
            "bureau_score": 300.0,
            "utility_payment_timeliness": 0.05,
            "rental_payment_timeliness": 0.05,
            "digital_payment_regularity": 0.02,
            "cashflow_stability_index": 0.02,
            "alternative_history_months": 0,
        }])
        validation.validate_feature_frame(min_row, train.ALL_FEATURES)

    def test_exact_maximum_boundaries_accepted(self):
        """Values exactly at generator-supported maximums must pass validation."""
        max_row = pd.DataFrame([{
            "age_years": 65,
            "monthly_income_inr": 250000.0,
            "employment_tenure_years": 35.0,
            "debt_to_income_ratio": 1.2,
            "existing_loan_count": 8,
            "credit_utilization_ratio": 1.25,
            "prior_delinquencies_24m": 5,
            "bureau_history_available": 1,
            "bureau_score": 900.0,
            "utility_payment_timeliness": 1.0,
            "rental_payment_timeliness": 1.0,
            "digital_payment_regularity": 1.0,
            "cashflow_stability_index": 1.0,
            "alternative_history_months": 24,
        }])
        validation.validate_feature_frame(max_row, train.ALL_FEATURES)

    def test_values_below_minimum_boundaries_rejected(self):
        """Values just below supported minimum boundary must raise ValidationError."""
        below_min_cases = [
            ("age_years", 19),
            ("monthly_income_inr", 7999.0),
            ("employment_tenure_years", -0.1),
            ("debt_to_income_ratio", -0.01),
            ("existing_loan_count", -1),
            ("credit_utilization_ratio", -0.01),
            ("prior_delinquencies_24m", -1),
            ("bureau_score", 299.9),
            ("utility_payment_timeliness", 0.049),
            ("rental_payment_timeliness", 0.049),
            ("digital_payment_regularity", 0.019),
            ("cashflow_stability_index", 0.019),
            ("alternative_history_months", -1),
        ]
        for feature, val in below_min_cases:
            with self.subTest(feature=feature, val=val):
                bad_row = self._make_row({feature: val})
                with self.assertRaises(validation.ValidationError) as cm:
                    validation.validate_feature_frame(bad_row, train.ALL_FEATURES)
                self.assertIn(f"Column '{feature}'", str(cm.exception))

    def test_values_above_maximum_boundaries_rejected(self):
        """Values just above supported maximum boundary must raise ValidationError."""
        above_max_cases = [
            ("age_years", 66),
            ("monthly_income_inr", 250001.0),
            ("employment_tenure_years", 35.1),
            ("debt_to_income_ratio", 1.21),
            ("existing_loan_count", 9),
            ("credit_utilization_ratio", 1.26),
            ("prior_delinquencies_24m", 6),
            ("bureau_score", 900.1),
            ("utility_payment_timeliness", 1.01),
            ("rental_payment_timeliness", 1.01),
            ("digital_payment_regularity", 1.01),
            ("cashflow_stability_index", 1.01),
            ("alternative_history_months", 25),
        ]
        for feature, val in above_max_cases:
            with self.subTest(feature=feature, val=val):
                bad_row = self._make_row({feature: val})
                with self.assertRaises(validation.ValidationError) as cm:
                    validation.validate_feature_frame(bad_row, train.ALL_FEATURES)
                self.assertIn(f"Column '{feature}'", str(cm.exception))

    def test_fractional_values_rejected_for_integer_features(self):
        """Integer-only features must reject non-integer fractional values."""
        fractional_cases = [
            ("age_years", 30.5),
            ("existing_loan_count", 2.5),
            ("prior_delinquencies_24m", 1.5),
            ("alternative_history_months", 10.5),
            ("bureau_history_available", 0.5),
        ]
        for feature, val in fractional_cases:
            with self.subTest(feature=feature, val=val):
                bad_row = self._make_row({feature: val})
                with self.assertRaises(validation.ValidationError) as cm:
                    validation.validate_feature_frame(bad_row, train.ALL_FEATURES)
                self.assertTrue(
                    "must contain integer values" in str(cm.exception)
                    or "invalid value" in str(cm.exception),
                    f"Unexpected exception message: {cm.exception}",
                )

    def test_valid_no_bureau_boundary_case(self):
        """Applicants without bureau record are valid when bureau_score is NaN and flag is 0."""
        no_bureau_row = self._make_row({
            "bureau_history_available": 0,
            "bureau_score": np.nan,
        })
        validation.validate_feature_frame(no_bureau_row, train.ALL_FEATURES)

    def test_all_alternative_features_missing_boundary_case(self):
        """Inference with all five alternative features missing must be accepted."""
        all_alt_missing = self._make_row({
            col: np.nan for col in train.ALTERNATIVE_FEATURES
        })
        validation.validate_feature_frame(all_alt_missing, train.ALL_FEATURES)

    def test_cli_smoke_valid_default(self):
        """Default CLI predict arguments must succeed with returncode 0."""
        res = subprocess.run(
            [sys.executable, str(ROOT / "predict.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("SYNTHETIC MODEL DEMO ONLY", res.stdout)
        self.assertIn("Estimated probability", res.stdout)

    def test_cli_smoke_valid_no_bureau(self):
        """Valid no-bureau CLI invocation must succeed with returncode 0."""
        res = subprocess.run(
            [
                sys.executable,
                str(ROOT / "predict.py"),
                "--no-bureau",
                "--dti", "0.8",
                "--delinquencies", "3",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("SYNTHETIC MODEL DEMO ONLY", res.stdout)

    def test_cli_smoke_out_of_support_rejected(self):
        """Out-of-support CLI arguments must exit cleanly with code 1 and descriptive error."""
        cases = [
            (["--age", "70"], "Column 'age_years'"),
            (["--income", "5000"], "Column 'monthly_income_inr'"),
            (["--loans", "10"], "Column 'existing_loan_count'"),
            (["--alternative-months", "30"], "Column 'alternative_history_months'"),
            (["--utilization", "1.4"], "Column 'credit_utilization_ratio'"),
            (["--delinquencies", "8"], "Column 'prior_delinquencies_24m'"),
        ]
        for cli_args, expected_msg in cases:
            with self.subTest(args=cli_args):
                res = subprocess.run(
                    [sys.executable, str(ROOT / "predict.py")] + cli_args,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(res.returncode, 1)
                self.assertIn("Input validation error", res.stderr)
                self.assertIn(expected_msg, res.stderr)


# ---------------------------------------------------------------------------
# Phase 03 Tests: Probability-to-Score Mapping & PDO Math
# ---------------------------------------------------------------------------

class TestProbabilityToScoreMapping(unittest.TestCase):
    """Verify points-to-double-the-odds (PDO) probability-to-score transformation."""

    def test_base_score_at_base_odds(self):
        """p = 1 / (base_odds + 1) must evaluate exactly to base_score."""
        p_base = 1.0 / (risk_scoring.DEFAULT_BASE_ODDS + 1.0)  # 1 / 51
        res = risk_scoring.probability_to_score(p_base)
        self.assertAlmostEqual(res.raw_score, 600.0, places=5)
        self.assertEqual(res.display_score, 600)
        self.assertFalse(res.is_clamped)

    def test_odds_doubling_pdo_step(self):
        """Doubling the good-to-bad odds (100:1) increases score by exactly PDO (+20 points)."""
        p_double = 1.0 / (2.0 * risk_scoring.DEFAULT_BASE_ODDS + 1.0)  # 1 / 101
        res = risk_scoring.probability_to_score(p_double)
        self.assertAlmostEqual(res.raw_score, 620.0, places=5)
        self.assertEqual(res.display_score, 620)

    def test_odds_halving_pdo_step(self):
        """Halving the good-to-bad odds (25:1) decreases score by exactly PDO (-20 points)."""
        p_halve = 1.0 / (0.5 * risk_scoring.DEFAULT_BASE_ODDS + 1.0)  # 1 / 26
        res = risk_scoring.probability_to_score(p_halve)
        self.assertAlmostEqual(res.raw_score, 580.0, places=5)
        self.assertEqual(res.display_score, 580)

    def test_monotonicity(self):
        """Higher default probability must strictly produce lower or equal score."""
        probs = np.linspace(0.001, 0.999, 100)
        scores = [risk_scoring.probability_to_score(float(p)).raw_score for p in probs]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1], f"Monotonicity violated at p={probs[i]}")

    def test_extremes_and_clamping(self):
        """Extreme probabilities map to min and max bounds with is_clamped=True."""
        res_zero = risk_scoring.probability_to_score(0.0)
        self.assertEqual(res_zero.display_score, 850)
        self.assertGreaterEqual(res_zero.raw_score, 850.0)
        self.assertTrue(res_zero.is_clamped)

        res_one = risk_scoring.probability_to_score(1.0)
        self.assertEqual(res_one.display_score, 300)
        self.assertLessEqual(res_one.raw_score, 300.0)
        self.assertTrue(res_one.is_clamped)

    def test_scores_strictly_bounded(self):
        """All scores for p in [0, 1] must stay within [min_score, max_score]."""
        for p in np.linspace(0.0, 1.0, 50):
            res = risk_scoring.probability_to_score(float(p))
            self.assertGreaterEqual(res.display_score, risk_scoring.DEFAULT_MIN_SCORE)
            self.assertLessEqual(res.display_score, risk_scoring.DEFAULT_MAX_SCORE)

    def test_invalid_probability_raises(self):
        """Probabilities outside [0, 1] raise ValueError; non-numbers raise TypeError."""
        for invalid_p in [-0.01, 1.01, -10.0, 2.5, float("nan"), float("inf")]:
            with self.subTest(invalid_p=invalid_p):
                with self.assertRaises(ValueError):
                    risk_scoring.probability_to_score(invalid_p)

        for invalid_type in ["0.5", None, [0.5]]:
            with self.subTest(invalid_type=invalid_type):
                with self.assertRaises(TypeError):
                    risk_scoring.probability_to_score(invalid_type)  # type: ignore

    def test_custom_score_parameters(self):
        """Custom base score, base odds, PDO, min/max score work as expected."""
        res = risk_scoring.probability_to_score(
            probability=0.1,
            base_score=700.0,
            base_odds=20.0,
            pdo=40.0,
            min_score=400,
            max_score=900,
        )
        self.assertIsInstance(res.display_score, int)
        self.assertGreaterEqual(res.display_score, 400)
        self.assertLessEqual(res.display_score, 900)

    def test_score_result_namedtuple_fields(self):
        """ScoreResult namedtuple must expose all expected fields."""
        res = risk_scoring.probability_to_score(0.05)
        self.assertTrue(hasattr(res, "display_score"))
        self.assertTrue(hasattr(res, "raw_score"))
        self.assertTrue(hasattr(res, "is_clamped"))
        self.assertTrue(hasattr(res, "default_probability"))
        self.assertTrue(hasattr(res, "odds_good_to_bad"))

    def test_formula_at_risk_band_boundaries_p10_and_p20(self):
        """Verify PDO score formula evaluates to ~550.52 at p=0.10 and ~527.12 at p=0.20."""
        res_10 = risk_scoring.probability_to_score(0.10)
        self.assertAlmostEqual(res_10.raw_score, 550.5218, places=3)
        self.assertEqual(res_10.display_score, 551)
        self.assertFalse(res_10.is_clamped)

        res_20 = risk_scoring.probability_to_score(0.20)
        self.assertAlmostEqual(res_20.raw_score, 527.1229, places=3)
        self.assertEqual(res_20.display_score, 527)
        self.assertFalse(res_20.is_clamped)

    def test_invalid_epsilon_raises(self):
        """Epsilon must be a finite float strictly between 0.0 and 0.5."""
        invalid_eps_values = [0.0, -0.01, -1.0, 0.5, 0.6, 1.0, float("nan"), float("inf")]
        for eps in invalid_eps_values:
            with self.subTest(eps=eps):
                with self.assertRaises(ValueError):
                    risk_scoring.probability_to_score(0.1, eps=eps)
                with self.assertRaises(ValueError):
                    risk_scoring.probability_to_raw_score(0.1, eps=eps)

        for invalid_type in ["0.01", None, [0.01], True, False]:
            with self.subTest(invalid_type=invalid_type):
                with self.assertRaises(TypeError):
                    risk_scoring.probability_to_score(0.1, eps=invalid_type)  # type: ignore

    def test_base_score_display_bounds_consistency(self):
        """base_score must lie within [min_score, max_score] when display bounding is active."""
        with self.assertRaises(ValueError):
            risk_scoring.probability_to_score(0.1, base_score=900.0, min_score=300, max_score=850)
        with self.assertRaises(ValueError):
            risk_scoring.probability_to_score(0.1, base_score=250.0, min_score=300, max_score=850)
        with self.assertRaises(ValueError):
            risk_scoring.probability_to_score(0.1, min_score=850, max_score=300)
        with self.assertRaises(TypeError):
            risk_scoring.probability_to_score(0.1, min_score=300.5, max_score=850)  # type: ignore

    def test_raw_score_unbounded_by_default_display_bounds(self):
        """probability_to_raw_score evaluates mathematically without enforcing display bounds."""
        raw = risk_scoring.probability_to_raw_score(0.1, base_score=1000.0, base_odds=20.0, pdo=40.0)
        self.assertIsInstance(raw, float)
        self.assertTrue(np.isfinite(raw))

    def test_boundary_probabilities_clipped_and_finite_with_valid_eps(self):
        """p=0.0 and p=1.0 remain finite and bounded with custom epsilon."""
        res_0 = risk_scoring.probability_to_score(0.0, eps=1e-4)
        self.assertTrue(np.isfinite(res_0.raw_score))
        self.assertAlmostEqual(res_0.raw_score, 752.880, places=1)
        self.assertEqual(res_0.display_score, 753)
        self.assertFalse(res_0.is_clamped)

        res_1 = risk_scoring.probability_to_score(1.0, eps=1e-4)
        self.assertTrue(np.isfinite(res_1.raw_score))
        self.assertAlmostEqual(res_1.raw_score, 221.372, places=1)
        self.assertEqual(res_1.display_score, 300)
        self.assertTrue(res_1.is_clamped)


# ---------------------------------------------------------------------------
# Phase 03 Tests: Illustrative Model Risk Bands
# ---------------------------------------------------------------------------

class TestIllustrativeRiskBands(unittest.TestCase):
    """Verify categorical risk-band assignments across probability thresholds."""

    def test_risk_band_boundaries(self):
        """Verify standard cutoffs (<0.10, 0.10-0.20, >=0.20)."""
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(0.0),
            risk_scoring.BAND_LOWER,
        )
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(0.0999),
            risk_scoring.BAND_LOWER,
        )
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(0.1000),
            risk_scoring.BAND_MIDDLE,
        )
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(0.1999),
            risk_scoring.BAND_MIDDLE,
        )
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(0.2000),
            risk_scoring.BAND_HIGHER,
        )
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(0.8500),
            risk_scoring.BAND_HIGHER,
        )
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(1.0),
            risk_scoring.BAND_HIGHER,
        )

    def test_invalid_probability_raises(self):
        """Probabilities outside [0, 1] raise ValueError; non-numeric raises TypeError."""
        with self.assertRaises(ValueError):
            risk_scoring.map_probability_to_risk_band(-0.05)
        with self.assertRaises(ValueError):
            risk_scoring.map_probability_to_risk_band(1.05)
        with self.assertRaises(TypeError):
            risk_scoring.map_probability_to_risk_band("0.15")  # type: ignore

    def test_custom_cutoffs(self):
        """Custom risk band cutoffs function as specified; invalid order raises ValueError."""
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(0.04, lower_cutoff=0.05, middle_cutoff=0.15),
            risk_scoring.BAND_LOWER,
        )
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(0.10, lower_cutoff=0.05, middle_cutoff=0.15),
            risk_scoring.BAND_MIDDLE,
        )
        self.assertEqual(
            risk_scoring.map_probability_to_risk_band(0.20, lower_cutoff=0.05, middle_cutoff=0.15),
            risk_scoring.BAND_HIGHER,
        )
        with self.assertRaises(ValueError):
            risk_scoring.map_probability_to_risk_band(0.10, lower_cutoff=0.30, middle_cutoff=0.20)


# ---------------------------------------------------------------------------
# Phase 03 Tests: Threshold Sensitivity Analysis
# ---------------------------------------------------------------------------

class TestThresholdSensitivityAnalysis(unittest.TestCase):
    """Verify threshold evaluation metrics, zero-division safety, and OOF partition integrity."""

    def test_threshold_metrics_known_matrix(self):
        """Verify metric calculation on a deterministic 4-sample toy example."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        # At threshold 0.5: y_pred = [0, 0, 0, 1]
        # TP=1, TN=2, FP=0, FN=1
        m = train.calculate_threshold_metrics(y_true, y_prob, threshold=0.5)
        self.assertEqual(m["true_positives"], 1)
        self.assertEqual(m["true_negatives"], 2)
        self.assertEqual(m["false_positives"], 0)
        self.assertEqual(m["false_negatives"], 1)
        self.assertAlmostEqual(m["precision"], 1.0, places=6)
        self.assertAlmostEqual(m["recall"], 0.5, places=6)
        self.assertAlmostEqual(m["specificity"], 1.0, places=6)
        self.assertAlmostEqual(m["false_positive_rate"], 0.0, places=6)
        self.assertAlmostEqual(m["false_negative_rate"], 0.5, places=6)
        self.assertAlmostEqual(m["f1_score"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.75, places=6)

    def test_zero_division_safety_all_negative_predictions(self):
        """When threshold exceeds all probabilities, precision/recall return 0.0 instead of NaN."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.05, 0.15])
        m = train.calculate_threshold_metrics(y_true, y_prob, threshold=0.99)
        self.assertEqual(m["true_positives"], 0)
        self.assertEqual(m["false_positives"], 0)
        self.assertEqual(m["precision"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1_score"], 0.0)
        self.assertFalse(np.isnan(m["precision"]))
        self.assertFalse(np.isnan(m["f1_score"]))

    def test_zero_division_safety_all_positive_predictions(self):
        """When threshold is lower than all probabilities, specificity/recall return finite floats."""
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.8, 0.9, 0.75, 0.85])
        m = train.calculate_threshold_metrics(y_true, y_prob, threshold=0.01)
        self.assertEqual(m["true_positives"], 2)
        self.assertEqual(m["false_positives"], 2)
        self.assertEqual(m["true_negatives"], 0)
        self.assertEqual(m["false_negatives"], 0)
        self.assertEqual(m["recall"], 1.0)
        self.assertEqual(m["specificity"], 0.0)
        self.assertEqual(m["false_positive_rate"], 1.0)
        self.assertEqual(m["false_negative_rate"], 0.0)

    def test_threshold_analysis_grid_structure(self):
        """Grid must span 0.05 to 0.95 in steps of 0.05 (19 points)."""
        grid = np.linspace(0.05, 0.95, 19)
        self.assertEqual(len(grid), 19)
        self.assertAlmostEqual(grid[0], 0.05)
        self.assertAlmostEqual(grid[-1], 0.95)
        self.assertAlmostEqual(grid[1] - grid[0], 0.05)

    def test_oof_partition_integrity_and_artifacts(self):
        """Verify generated threshold analysis artifacts reflect training OOF evaluation."""
        csv_path = ROOT / "reports" / "threshold_analysis.csv"
        png_path = ROOT / "reports" / "threshold_analysis.png"
        summary_path = ROOT / "reports" / "run_summary.json"

        self.assertTrue(csv_path.exists(), "threshold_analysis.csv missing")
        self.assertTrue(png_path.exists(), "threshold_analysis.png missing")
        self.assertGreater(png_path.stat().st_size, 10000, "threshold_analysis.png too small")

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 57, f"Expected 57 rows (19 thresholds * 3 models), got {len(df)}")
        self.assertTrue((df["data_partition"] == "training_out_of_fold").all())
        self.assertTrue((df["method"] == "5-fold stratified cross-validation").all())

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertIn("threshold_analysis", summary)
        self.assertEqual(summary["threshold_analysis"]["data_partition"], "training_out_of_fold")
        self.assertEqual(summary["threshold_analysis"]["n_thresholds"], 19)
        self.assertIn("prevent leakage", summary["threshold_analysis"]["note"])

    def test_invalid_thresholds_rejected(self):
        """Thresholds outside [0, 1], non-numeric, or non-finite must raise ValueError or TypeError."""
        y_true = np.array([0, 1])
        probs = np.array([0.3, 0.7])

        for invalid_t in [-0.01, 1.01, -1.0, 2.0, float("nan"), float("inf")]:
            with self.subTest(invalid_t=invalid_t):
                with self.assertRaises(ValueError):
                    train.calculate_threshold_metrics(y_true, probs, threshold=invalid_t)

        for invalid_type in ["0.5", True, False]:
            with self.subTest(invalid_type=invalid_type):
                with self.assertRaises(TypeError):
                    train.calculate_threshold_metrics(y_true, probs, threshold=invalid_type)

        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(y_true, probs, thresholds=[])

        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(y_true, probs, thresholds=[-0.1, 0.5])

    def test_invalid_labels_rejected(self):
        """Labels must be 1D, non-empty, and strictly binary {0, 1} without silent coercion."""
        probs = np.array([0.4, 0.6])

        # Non-binary labels
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(np.array([0, 2]), probs, threshold=0.5)
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(np.array([0.7, 1.0]), probs, threshold=0.5)
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(np.array([-1, 1]), probs, threshold=0.5)

        # Empty array
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(np.array([]), np.array([]), threshold=0.5)

        # Array with NaN
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(np.array([0, np.nan]), probs, threshold=0.5)

    def test_invalid_probabilities_rejected(self):
        """Probabilities must be 1D, non-empty, finite, and in [0.0, 1.0] matching y_true length."""
        y_true = np.array([0, 1])

        # Out of bounds
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(y_true, np.array([-0.05, 0.8]), threshold=0.5)
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(y_true, np.array([0.2, 1.05]), threshold=0.5)

        # NaN / Inf
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(y_true, np.array([np.nan, 0.5]), threshold=0.5)
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(y_true, np.array([np.inf, 0.5]), threshold=0.5)

        # Length mismatch
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(y_true, np.array([0.5]), threshold=0.5)

        # 2D array
        with self.assertRaises(ValueError):
            train.calculate_threshold_metrics(y_true, np.array([[0.5, 0.5]]), threshold=0.5)


# ---------------------------------------------------------------------------
# Phase 03 Tests: Exact Additive Local Feature Explainability
# ---------------------------------------------------------------------------

class TestLocalExplainability(unittest.TestCase):
    """Verify exact linear log-odds decomposition and rejection criteria."""

    @classmethod
    def setUpClass(cls):
        cls.hybrid_lr_path = ROOT / "models" / "hybrid_logistic_regression.joblib"
        cls.hybrid_hgb_path = ROOT / "models" / "hybrid_hist_gradient_boosting.joblib"
        cls.hybrid_lr = joblib.load(cls.hybrid_lr_path)
        cls.hybrid_hgb = joblib.load(cls.hybrid_hgb_path)

    def test_additive_reconstruction_precision_standard_profile(self):
        """Additive feature decomposition must reconstruct decision function within 1e-5."""
        row = pd.DataFrame([{
            "age_years": 30,
            "monthly_income_inr": 40000,
            "employment_tenure_years": 4.0,
            "debt_to_income_ratio": 0.35,
            "existing_loan_count": 1,
            "credit_utilization_ratio": 0.40,
            "prior_delinquencies_24m": 0,
            "bureau_history_available": 1,
            "bureau_score": 650.0,
            "utility_payment_timeliness": 0.82,
            "rental_payment_timeliness": 0.80,
            "digital_payment_regularity": 0.75,
            "cashflow_stability_index": 0.68,
            "alternative_history_months": 15,
        }])
        exp = explainability.explain_logistic_prediction(self.hybrid_lr, row, top_k=3)
        diff = abs(exp.reconstructed_logit - exp.model_decision_function)
        self.assertLess(diff, 1e-5, f"Reconstruction difference {diff} exceeds tolerance 1e-5")
        self.assertEqual(len(exp.all_contributions), 18)

    def test_additive_reconstruction_precision_no_bureau_profile(self):
        """Thin-file / no-bureau profile reconstruction must also be exact."""
        row = pd.DataFrame([{
            "age_years": 24,
            "monthly_income_inr": 25000,
            "employment_tenure_years": 1.0,
            "debt_to_income_ratio": 0.70,
            "existing_loan_count": 0,
            "credit_utilization_ratio": 0.60,
            "prior_delinquencies_24m": 2,
            "bureau_history_available": 0,
            "bureau_score": np.nan,
            "utility_payment_timeliness": 0.60,
            "rental_payment_timeliness": 0.55,
            "digital_payment_regularity": 0.50,
            "cashflow_stability_index": 0.40,
            "alternative_history_months": 8,
        }])
        exp = explainability.explain_logistic_prediction(self.hybrid_lr, row, top_k=3)
        diff = abs(exp.reconstructed_logit - exp.model_decision_function)
        self.assertLess(diff, 1e-5, f"Reconstruction difference {diff} exceeds tolerance 1e-5")

    def test_additive_reconstruction_boundary_profile(self):
        """Extreme supported boundary case reconstruction must also be exact."""
        row = pd.DataFrame([{
            "age_years": 65,
            "monthly_income_inr": 200000,
            "employment_tenure_years": 35.0,
            "debt_to_income_ratio": 1.2,
            "existing_loan_count": 8,
            "credit_utilization_ratio": 1.2,
            "prior_delinquencies_24m": 5,
            "bureau_history_available": 1,
            "bureau_score": 900.0,
            "utility_payment_timeliness": 1.0,
            "rental_payment_timeliness": 1.0,
            "digital_payment_regularity": 1.0,
            "cashflow_stability_index": 1.0,
            "alternative_history_months": 24,
        }])
        exp = explainability.explain_logistic_prediction(self.hybrid_lr, row, top_k=3)
        diff = abs(exp.reconstructed_logit - exp.model_decision_function)
        self.assertLess(diff, 1e-5)

    def test_directional_consistency_of_factors(self):
        """Top risk-increasing must have positive contributions, decreasing negative."""
        row = pd.DataFrame([{
            "age_years": 30,
            "monthly_income_inr": 40000,
            "employment_tenure_years": 4.0,
            "debt_to_income_ratio": 0.35,
            "existing_loan_count": 1,
            "credit_utilization_ratio": 0.40,
            "prior_delinquencies_24m": 0,
            "bureau_history_available": 1,
            "bureau_score": 650.0,
            "utility_payment_timeliness": 0.82,
            "rental_payment_timeliness": 0.80,
            "digital_payment_regularity": 0.75,
            "cashflow_stability_index": 0.68,
            "alternative_history_months": 15,
        }])
        exp = explainability.explain_logistic_prediction(self.hybrid_lr, row, top_k=3)
        for item in exp.top_risk_increasing:
            self.assertGreater(item["contribution"], 0.0)
        for item in exp.top_risk_decreasing:
            self.assertLess(item["contribution"], 0.0)

    def test_non_linear_model_rejection(self):
        """Passing non-linear model (HistGradientBoosting) must raise TypeError."""
        row = pd.DataFrame([{
            "age_years": 30,
            "monthly_income_inr": 40000,
            "employment_tenure_years": 4.0,
            "debt_to_income_ratio": 0.35,
            "existing_loan_count": 1,
            "credit_utilization_ratio": 0.40,
            "prior_delinquencies_24m": 0,
            "bureau_history_available": 1,
            "bureau_score": 650.0,
            "utility_payment_timeliness": 0.82,
            "rental_payment_timeliness": 0.80,
            "digital_payment_regularity": 0.75,
            "cashflow_stability_index": 0.68,
            "alternative_history_months": 15,
        }])
        with self.assertRaises(TypeError) as ctx:
            explainability.explain_logistic_prediction(self.hybrid_hgb, row)
        self.assertIn("logisticregression", str(ctx.exception).lower())

    def test_invalid_input_structure_rejection(self):
        """Passing multi-row DataFrame or non-pipeline object must raise appropriate errors."""
        row2 = pd.DataFrame([
            {"age_years": 30, "monthly_income_inr": 40000},
            {"age_years": 40, "monthly_income_inr": 50000},
        ])
        with self.assertRaises(ValueError) as ctx:
            explainability.explain_logistic_prediction(self.hybrid_lr, row2, validate=False)
        self.assertIn("exactly 1 row", str(ctx.exception))

        with self.assertRaises(TypeError):
            explainability.explain_logistic_prediction("not_a_model", row2)  # type: ignore

    def test_readable_names_coverage(self):
        """All 18 decomposed features must have valid descriptors and field types."""
        row = pd.DataFrame([{
            "age_years": 30,
            "monthly_income_inr": 40000,
            "employment_tenure_years": 4.0,
            "debt_to_income_ratio": 0.35,
            "existing_loan_count": 1,
            "credit_utilization_ratio": 0.40,
            "prior_delinquencies_24m": 0,
            "bureau_history_available": 1,
            "bureau_score": 650.0,
            "utility_payment_timeliness": 0.82,
            "rental_payment_timeliness": 0.80,
            "digital_payment_regularity": 0.75,
            "cashflow_stability_index": 0.68,
            "alternative_history_months": 15,
        }])
        exp = explainability.explain_logistic_prediction(self.hybrid_lr, row)
        for c in exp.all_contributions:
            self.assertIn("readable_name", c)
            self.assertIn("contribution", c)
            self.assertIn("coefficient", c)
            self.assertIn("transformed_value", c)
            self.assertIsInstance(c["readable_name"], str)
            self.assertGreater(len(c["readable_name"]), 0)


# ---------------------------------------------------------------------------
# Phase 03 Tests: CLI Smoke Tests
# ---------------------------------------------------------------------------

class TestPhase03CLISmoke(unittest.TestCase):
    """Verify CLI integration of Phase 03 score, band, and explanations."""

    def test_cli_smoke_contains_phase03_elements(self):
        """CLI predict.py stdout must contain illustrative score, band, and local contributions."""
        res = subprocess.run(
            [sys.executable, str(ROOT / "predict.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Illustrative Credit Score:", res.stdout)
        self.assertIn("Illustrative Risk Band:", res.stdout)
        self.assertIn("Scoring Model Assumptions:", res.stdout)
        self.assertIn("Local Feature Contributions", res.stdout)
        self.assertIn("Top factors increasing modeled default risk", res.stdout)
        self.assertIn("Top factors decreasing modeled default risk", res.stdout)
        self.assertIn("IMPORTANT DISCLAIMER:", res.stdout)

    def test_cli_smoke_no_bureau_contains_phase03_elements(self):
        """CLI predict.py with --no-bureau must execute cleanly and explain properly."""
        res = subprocess.run(
            [
                sys.executable,
                str(ROOT / "predict.py"),
                "--no-bureau",
                "--dti", "0.8",
                "--delinquencies", "3",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Illustrative Credit Score:", res.stdout)
        self.assertIn("Illustrative Risk Band:", res.stdout)
        self.assertIn("Local Feature Contributions", res.stdout)


# ---------------------------------------------------------------------------
# Phase 03.1 Tests: Documentation & Specification Consistency
# ---------------------------------------------------------------------------

class TestDocumentationConsistency(unittest.TestCase):
    """Verify that MODEL_CARD.md and experiment_summary.md match code contracts."""

    def test_model_card_feature_bounds_match_domain_bounds(self):
        """MODEL_CARD.md must accurately document actual validator & generator ranges."""
        model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
        expected_snippets = [
            "`age_years`: integer $\\in [20, 65]$",
            "`monthly_income_inr`: float $\\in [8000.0, 250000.0]$",
            "`employment_tenure_years`: float $\\in [0.0, 35.0]$",
            "`debt_to_income_ratio`: float $\\in [0.0, 1.2]$",
            "`existing_loan_count`: integer $\\in [0, 8]$",
            "`credit_utilization_ratio`: float $\\in [0.0, 1.25]$",
            "`prior_delinquencies_24m`: integer $\\in [0, 5]$",
            "`bureau_history_available`: binary $\\in \\{0, 1\\}$",
            "`bureau_score`: float $\\in [300.0, 900.0]$",
            "`utility_payment_timeliness`: float $\\in [0.05, 1.0]$",
            "`rental_payment_timeliness`: float $\\in [0.05, 1.0]$",
            "`digital_payment_regularity`: float $\\in [0.02, 1.0]$",
            "`cashflow_stability_index`: float $\\in [0.02, 1.0]$",
            "`alternative_history_months`: integer $\\in [0, 24]$",
        ]
        for snippet in expected_snippets:
            self.assertIn(snippet, model_card)

    def test_documentation_does_not_contain_incorrect_score_ranges(self):
        """Risk bands must not contain the incorrect score ranges 552 or 510."""
        model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
        exp_summary = (ROOT / "reports" / "experiment_summary.md").read_text(encoding="utf-8")

        self.assertNotIn("552", model_card)
        self.assertNotIn("510", model_card)
        self.assertNotIn("score \\ge 552", exp_summary.lower())
        self.assertNotIn("score < 510", exp_summary.lower())

    def test_documentation_does_not_contain_unsupported_cost_or_adverse_claims(self):
        """Documentation must not claim arbitrary cost ratios (4x-10x) or adverse action compliance."""
        exp_summary = (ROOT / "reports" / "experiment_summary.md").read_text(encoding="utf-8")
        self.assertNotIn("4x to 10x", exp_summary)
        self.assertNotIn("4\\times to 10\\times", exp_summary)
        self.assertNotIn("4× to 10×", exp_summary)
        self.assertNotIn("satisfy interpretability and adverse-action transparency requirements", exp_summary)
        self.assertNotIn("mathematical foundation for adverse-action notices", exp_summary)


# ---------------------------------------------------------------------------
# Phase 04 Tests: Streamlit Integration & Shared Inference Helper
# ---------------------------------------------------------------------------

class TestStreamlitIntegrationAndInferenceHelper(unittest.TestCase):
    """Verify shared inference helper, applicant row construction, and Streamlit AppTest integration."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = ROOT / "models" / "hybrid_logistic_regression.joblib"
        cls.model = inference_helper.load_saved_model(cls.model_path)

    def test_applicant_row_construction_defaults(self):
        """Default applicant row must construct a valid 1-row DataFrame matching ALL_FEATURES."""
        row = inference_helper.build_applicant_row(
            age_years=30,
            monthly_income_inr=40000.0,
            employment_tenure_years=4.0,
            debt_to_income_ratio=0.35,
            existing_loan_count=1,
            credit_utilization_ratio=0.40,
            prior_delinquencies_24m=0,
            bureau_history_available=1,
            bureau_score=650.0,
            utility_payment_timeliness=0.82,
            rental_payment_timeliness=0.80,
            digital_payment_regularity=0.75,
            cashflow_stability_index=0.68,
            alternative_history_months=15,
        )
        self.assertEqual(row.shape, (1, len(validation.ALL_FEATURES)))
        self.assertListEqual(list(row.columns), validation.ALL_FEATURES)
        validation.validate_feature_frame(row, expected_features=validation.ALL_FEATURES)

    def test_applicant_row_construction_no_bureau(self):
        """Applicants without bureau record must have bureau_score=NaN and pass validation."""
        # Test integer flag 0
        row_int = inference_helper.build_applicant_row(
            age_years=25,
            monthly_income_inr=30000.0,
            employment_tenure_years=2.0,
            debt_to_income_ratio=0.30,
            existing_loan_count=0,
            credit_utilization_ratio=0.20,
            prior_delinquencies_24m=0,
            bureau_history_available=0,
            bureau_score=650.0,  # Should be forced to NaN
        )
        self.assertTrue(np.isnan(row_int["bureau_score"].iloc[0]))
        self.assertEqual(row_int["bureau_history_available"].iloc[0], 0)
        validation.validate_feature_frame(row_int, expected_features=validation.ALL_FEATURES)

        # Test boolean False
        row_bool = inference_helper.build_applicant_row(
            age_years=25,
            monthly_income_inr=30000.0,
            employment_tenure_years=2.0,
            debt_to_income_ratio=0.30,
            existing_loan_count=0,
            credit_utilization_ratio=0.20,
            prior_delinquencies_24m=0,
            bureau_history_available=False,
            bureau_score=None,
        )
        self.assertTrue(np.isnan(row_bool["bureau_score"].iloc[0]))
        self.assertEqual(row_bool["bureau_history_available"].iloc[0], 0)
        validation.validate_feature_frame(row_bool, expected_features=validation.ALL_FEATURES)

    def test_applicant_row_construction_partial_missing_alternative(self):
        """Marking specific alternative fields unavailable assigns NaN and passes validation."""
        unavail = ["utility_payment_timeliness", "cashflow_stability_index"]
        row = inference_helper.build_applicant_row(
            age_years=35,
            monthly_income_inr=50000.0,
            employment_tenure_years=6.0,
            debt_to_income_ratio=0.25,
            existing_loan_count=1,
            credit_utilization_ratio=0.30,
            prior_delinquencies_24m=0,
            bureau_history_available=1,
            bureau_score=700.0,
            unavailable_alternative_features=unavail,
        )
        for col in unavail:
            self.assertTrue(np.isnan(row[col].iloc[0]), f"Expected NaN for {col}")
        # Retained fields must be finite floats
        self.assertFalse(np.isnan(row["rental_payment_timeliness"].iloc[0]))
        self.assertFalse(np.isnan(row["digital_payment_regularity"].iloc[0]))
        self.assertFalse(np.isnan(row["alternative_history_months"].iloc[0]))
        validation.validate_feature_frame(row, expected_features=validation.ALL_FEATURES)

    def test_applicant_row_construction_all_alternative_missing(self):
        """All five alternative features unavailable assigns NaN to all five and passes validation."""
        all_alt = list(validation.ALTERNATIVE_FEATURES)
        row = inference_helper.build_applicant_row(
            age_years=40,
            monthly_income_inr=60000.0,
            employment_tenure_years=8.0,
            debt_to_income_ratio=0.20,
            existing_loan_count=1,
            credit_utilization_ratio=0.25,
            prior_delinquencies_24m=0,
            bureau_history_available=1,
            bureau_score=720.0,
            unavailable_alternative_features=all_alt,
        )
        for col in all_alt:
            self.assertTrue(np.isnan(row[col].iloc[0]), f"Expected NaN for {col}")
        validation.validate_feature_frame(row, expected_features=validation.ALL_FEATURES)

    def test_validation_failure_prevents_inference(self):
        """Malformed or out-of-bounds applicant inputs must fail validation before model invocation."""
        bad_row = inference_helper.build_applicant_row(
            age_years=15,  # Below supported minimum of 20
            monthly_income_inr=40000.0,
            employment_tenure_years=4.0,
            debt_to_income_ratio=0.35,
            existing_loan_count=1,
            credit_utilization_ratio=0.40,
            prior_delinquencies_24m=0,
            bureau_history_available=1,
            bureau_score=650.0,
        )
        with self.assertRaises(validation.ValidationError) as cm:
            validation.validate_feature_frame(bad_row, expected_features=validation.ALL_FEATURES)
        self.assertIn("age_years", str(cm.exception))

        # run_inference_pipeline with validate=True must raise ValidationError
        with self.assertRaises(validation.ValidationError):
            inference_helper.run_inference_pipeline(self.model, bad_row, validate=True)

    def test_inference_pipeline_shared_helpers_integration(self):
        """End-to-end inference helper must produce probability, score, band, and explanation."""
        row = inference_helper.build_applicant_row(
            age_years=30,
            monthly_income_inr=40000.0,
            employment_tenure_years=4.0,
            debt_to_income_ratio=0.35,
            existing_loan_count=1,
            credit_utilization_ratio=0.40,
            prior_delinquencies_24m=0,
            bureau_history_available=1,
            bureau_score=650.0,
        )
        res = inference_helper.run_inference_pipeline(self.model, row, validate=True)
        self.assertIsInstance(res.probability, float)
        self.assertTrue(0.0 <= res.probability <= 1.0)

        # Verify score equals direct probability_to_score
        expected_score = risk_scoring.probability_to_score(res.probability)
        self.assertEqual(res.score_result.display_score, expected_score.display_score)
        self.assertAlmostEqual(res.score_result.raw_score, expected_score.raw_score, places=5)

        # Verify band equals direct map_probability_to_risk_band
        expected_band = risk_scoring.map_probability_to_risk_band(res.probability)
        self.assertEqual(res.risk_band, expected_band)

        # Verify explanation integrity
        self.assertLess(res.explanation.reconstruction_error, 1e-5)
        self.assertEqual(len(res.explanation.top_risk_increasing), 3)
        self.assertEqual(len(res.explanation.top_risk_decreasing), 3)

    def test_local_explanation_reconstruction_across_profiles(self):
        """Local explanation reconstruction must be exact for default, no-bureau, and cold-start profiles."""
        profiles = [
            # Default profile
            inference_helper.build_applicant_row(
                age_years=30, monthly_income_inr=40000.0, employment_tenure_years=4.0,
                debt_to_income_ratio=0.35, existing_loan_count=1, credit_utilization_ratio=0.40,
                prior_delinquencies_24m=0, bureau_history_available=1, bureau_score=650.0,
            ),
            # No-bureau profile
            inference_helper.build_applicant_row(
                age_years=28, monthly_income_inr=35000.0, employment_tenure_years=3.0,
                debt_to_income_ratio=0.45, existing_loan_count=0, credit_utilization_ratio=0.10,
                prior_delinquencies_24m=1, bureau_history_available=0, bureau_score=None,
            ),
            # All alternative missing (cold start)
            inference_helper.build_applicant_row(
                age_years=45, monthly_income_inr=70000.0, employment_tenure_years=10.0,
                debt_to_income_ratio=0.20, existing_loan_count=2, credit_utilization_ratio=0.30,
                prior_delinquencies_24m=0, bureau_history_available=1, bureau_score=750.0,
                unavailable_alternative_features=list(validation.ALTERNATIVE_FEATURES),
            ),
        ]
        for idx, row in enumerate(profiles):
            with self.subTest(profile_index=idx):
                res = inference_helper.run_inference_pipeline(self.model, row, validate=True)
                self.assertLess(res.explanation.reconstruction_error, 1e-5)

    def test_load_saved_model_missing_raises_file_not_found(self):
        """load_saved_model with non-existent path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            inference_helper.load_saved_model(ROOT / "models" / "non_existent_model.joblib")

    def test_load_saved_model_ensures_multi_class_compatibility(self):
        """Loaded model must guarantee multi_class attribute exists on LogisticRegression."""
        model = inference_helper.load_saved_model()
        clf = model.named_steps["classifier"]
        self.assertTrue(hasattr(clf, "multi_class"))

    def test_streamlit_apptest_initial_render(self):
        """Streamlit app must render cleanly without unhandled exceptions."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Exceptions on initial render: {at.exception}")
        self.assertGreater(len(at.button), 0, "Submit button not found")

    def test_streamlit_apptest_form_submission_default(self):
        """Submitting default form in AppTest must run inference and display outputs without error."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        at.run()
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0, f"Exceptions on form submission: {at.exception}")
        # Verify metric components are present in the output
        self.assertGreaterEqual(len(at.metric), 3)

    def test_streamlit_apptest_form_submission_no_bureau(self):
        """Unchecking bureau history in AppTest and submitting must execute cleanly."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        at.run()
        at.checkbox[0].uncheck().run()
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0, f"Exceptions on no-bureau submission: {at.exception}")

    def test_streamlit_apptest_form_submission_missing_alternative(self):
        """Marking alternative fields unavailable in AppTest must execute cleanly."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        at.run()
        at.multiselect[0].select("utility_payment_timeliness").run()
        at.multiselect[0].select("digital_payment_regularity").run()
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0, f"Exceptions on missing-alt submission: {at.exception}")


# ---------------------------------------------------------------------------
# Phase 05 Tests: Inference Hardening & Reliability
# ---------------------------------------------------------------------------

class TestInferenceHardeningAndReliability(unittest.TestCase):
    """Verify hardened integer coercion, boundary validation, and inference safety."""

    @classmethod
    def setUpClass(cls):
        cls.valid_base = {
            "age_years": 30,
            "monthly_income_inr": 40000.0,
            "employment_tenure_years": 4.0,
            "debt_to_income_ratio": 0.35,
            "existing_loan_count": 1,
            "credit_utilization_ratio": 0.40,
            "prior_delinquencies_24m": 0,
            "bureau_history_available": 1,
            "bureau_score": 650.0,
            "utility_payment_timeliness": 0.82,
            "rental_payment_timeliness": 0.80,
            "digital_payment_regularity": 0.75,
            "cashflow_stability_index": 0.68,
            "alternative_history_months": 15,
        }

    def _call_build(self, **overrides):
        args = self.valid_base.copy()
        args.update(overrides)
        return inference_helper.build_applicant_row(**args)

    def test_fractional_integer_fields_rejected(self):
        """Integer fields must reject fractional values without silent truncation."""
        fractional_cases = [
            ("age_years", 30.5),
            ("existing_loan_count", 1.8),
            ("prior_delinquencies_24m", 2.2),
            ("alternative_history_months", 12.5),
        ]
        for field, bad_val in fractional_cases:
            with self.subTest(field=field, bad_val=bad_val):
                with self.assertRaises(validation.ValidationError) as cm:
                    self._call_build(**{field: bad_val})
                self.assertIn(f"Field '{field}'", str(cm.exception))
                self.assertIn("must be an integer", str(cm.exception))

    def test_valid_whole_number_inputs_accepted(self):
        """Float representations of whole numbers (e.g. 1.0, 30.0) must be accepted as integers."""
        row = self._call_build(
            age_years=30.0,
            existing_loan_count=2.0,
            prior_delinquencies_24m=0.0,
            alternative_history_months=15.0,
        )
        self.assertEqual(row["age_years"].iloc[0], 30)
        self.assertEqual(row["existing_loan_count"].iloc[0], 2)
        self.assertEqual(row["prior_delinquencies_24m"].iloc[0], 0)
        self.assertEqual(row["alternative_history_months"].iloc[0], 15.0)
        validation.validate_feature_frame(row, expected_features=validation.ALL_FEATURES)

    def test_boolean_inputs_rejected_for_numeric_fields(self):
        """Booleans must not be silently treated as integers or floats (e.g. True as 1)."""
        bad_bool_cases = [
            ("age_years", True),
            ("monthly_income_inr", False),
            ("existing_loan_count", True),
            ("prior_delinquencies_24m", False),
            ("credit_utilization_ratio", True),
            ("alternative_history_months", True),
        ]
        for field, bad_val in bad_bool_cases:
            with self.subTest(field=field, bad_val=bad_val):
                with self.assertRaises(validation.ValidationError) as cm:
                    self._call_build(**{field: bad_val})
                self.assertIn(f"Field '{field}'", str(cm.exception))

    def test_non_finite_required_fields_rejected(self):
        """Required traditional fields must reject NaN, inf, and -inf."""
        non_finite_cases = [
            ("monthly_income_inr", np.nan),
            ("monthly_income_inr", np.inf),
            ("age_years", np.nan),
            ("age_years", np.inf),
            ("employment_tenure_years", -np.inf),
            ("debt_to_income_ratio", np.nan),
            ("credit_utilization_ratio", np.nan),
            ("existing_loan_count", np.nan),
            ("prior_delinquencies_24m", np.nan),
            ("bureau_score", np.nan),  # when bureau_history_available=1
        ]
        for field, bad_val in non_finite_cases:
            with self.subTest(field=field, bad_val=bad_val):
                with self.assertRaises(validation.ValidationError) as cm:
                    self._call_build(**{field: bad_val})
                self.assertIn(f"'{field}'", str(cm.exception))

    def test_invalid_unavailable_alternative_features_rejected(self):
        """Passing unknown feature names to unavailable_alternative_features must raise ValidationError."""
        bad_names = ["non_existent_feature", "crypto_activity"]
        with self.assertRaises(validation.ValidationError) as cm:
            self._call_build(unavailable_alternative_features=bad_names)
        self.assertIn("Invalid unavailable alternative feature(s)", str(cm.exception))
        self.assertIn("non_existent_feature", str(cm.exception))

    def test_legitimate_optional_missingness_preserved_as_nan(self):
        """Missing alternative features must remain real NaN values in the assembled row."""
        row = self._call_build(
            utility_payment_timeliness=None,
            rental_payment_timeliness=np.nan,
            unavailable_alternative_features=["digital_payment_regularity"],
        )
        self.assertTrue(np.isnan(row["utility_payment_timeliness"].iloc[0]))
        self.assertTrue(np.isnan(row["rental_payment_timeliness"].iloc[0]))
        self.assertTrue(np.isnan(row["digital_payment_regularity"].iloc[0]))
        self.assertFalse(np.isnan(row["cashflow_stability_index"].iloc[0]))
        validation.validate_feature_frame(row, expected_features=validation.ALL_FEATURES)

    def test_unavailable_bureau_history_forces_nan_score(self):
        """Setting bureau_history_available=0 forces bureau_score to NaN regardless of input."""
        row_zero = self._call_build(bureau_history_available=0, bureau_score=750.0)
        self.assertTrue(np.isnan(row_zero["bureau_score"].iloc[0]))
        self.assertEqual(row_zero["bureau_history_available"].iloc[0], 0)
        validation.validate_feature_frame(row_zero, expected_features=validation.ALL_FEATURES)

        row_false = self._call_build(bureau_history_available=False, bureau_score=750.0)
        self.assertTrue(np.isnan(row_false["bureau_score"].iloc[0]))
        self.assertEqual(row_false["bureau_history_available"].iloc[0], 0)
        validation.validate_feature_frame(row_false, expected_features=validation.ALL_FEATURES)

    def test_predict_proba_not_called_on_invalid_row(self):
        """Model predict_proba must NOT be invoked when input validation fails."""
        mock_model = MagicMock()
        bad_row = pd.DataFrame([{
            "age_years": 15,  # Out of domain bounds [20, 65]
            "monthly_income_inr": 40000.0,
            "employment_tenure_years": 4.0,
            "debt_to_income_ratio": 0.35,
            "existing_loan_count": 1,
            "credit_utilization_ratio": 0.40,
            "prior_delinquencies_24m": 0,
            "bureau_history_available": 1,
            "bureau_score": 650.0,
            "utility_payment_timeliness": 0.82,
            "rental_payment_timeliness": 0.80,
            "digital_payment_regularity": 0.75,
            "cashflow_stability_index": 0.68,
            "alternative_history_months": 15,
        }])
        with self.assertRaises(validation.ValidationError):
            inference_helper.run_inference_pipeline(mock_model, bad_row, validate=True)
        mock_model.predict_proba.assert_not_called()

    def test_run_inference_pipeline_rejects_malformed_probabilities(self):
        """Inference pipeline must raise controlled ValueError for malformed or out-of-range probabilities."""
        valid_row = self._call_build()

        malformed_cases = [
            ("out of range high", np.array([[0.1, 1.2]])),
            ("out of range negative", np.array([[1.1, -0.1]])),
            ("NaN probability", np.array([[np.nan, np.nan]])),
            ("infinite probability", np.array([[0.0, np.inf]])),
            ("1D array", np.array([0.5, 0.5])),
            ("multi-row output", np.array([[0.1, 0.9], [0.2, 0.8]])),
        ]
        for desc, bad_proba in malformed_cases:
            with self.subTest(case=desc):
                mock_model = MagicMock()
                mock_model.predict_proba.return_value = bad_proba
                with self.assertRaises(ValueError):
                    inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)

    def test_run_inference_pipeline_rejects_three_column_multiclass_output(self):
        """Model returning a 3-column array must be rejected as malformed binary output."""
        valid_row = self._call_build()
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.7, 0.1]])
        with self.assertRaises(ValueError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("expected exactly (1, 2)", str(cm.exception))

    def test_near_integer_fractional_values_strictly_rejected(self):
        """Values extremely close to integers must be rejected rather than rounded via tolerance."""
        near_integer_cases = [
            ("age_years", 30.0001),
            ("existing_loan_count", 1.00001),
            ("prior_delinquencies_24m", 1.000001),
            ("alternative_history_months", 12.0001),
        ]
        for field, bad_val in near_integer_cases:
            with self.subTest(field=field, bad_val=bad_val):
                with self.assertRaises(validation.ValidationError) as cm:
                    self._call_build(**{field: bad_val})
                self.assertIn(f"Field '{field}'", str(cm.exception))
                self.assertIn("must be an integer", str(cm.exception))

    def test_run_inference_pipeline_rejects_multi_row_or_non_dataframe(self):
        """Inference pipeline expects exactly 1 DataFrame row; rejects non-DataFrame or multi-row inputs."""
        mock_model = MagicMock()
        # Non-DataFrame
        with self.assertRaises(validation.ValidationError) as cm:
            inference_helper.run_inference_pipeline(mock_model, {"age_years": 30}, validate=False)  # type: ignore
        self.assertIn("pandas DataFrame", str(cm.exception))

        # Multi-row DataFrame
        two_rows = pd.concat([self._call_build(), self._call_build()], ignore_index=True)
        with self.assertRaises(validation.ValidationError) as cm:
            inference_helper.run_inference_pipeline(mock_model, two_rows, validate=False)
        self.assertIn("exactly 1 applicant row", str(cm.exception))

    def test_run_inference_pipeline_validates_complete_probability_vector(self):
        """Inference pipeline validates both classes, non-boolean/non-complex types, finiteness, [0, 1] range, and unit sum."""
        valid_row = self._call_build()

        # 1. First class probability greater than one while second is valid: [[1.2, 0.7]]
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[1.2, 0.7]])
        with self.assertRaises(ValueError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("must lie in [0.0, 1.0]", str(cm.exception))

        # 2. First class probability below zero while second is valid: [[-0.2, 0.7]]
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[-0.2, 0.7]])
        with self.assertRaises(ValueError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("must lie in [0.0, 1.0]", str(cm.exception))

        # 3. NaN in the first class probability while the second is valid: [[np.nan, 0.7]]
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[np.nan, 0.7]])
        with self.assertRaises(ValueError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("must be finite", str(cm.exception))

        # 4. Infinity in the first class probability while the second is valid: [[np.inf, 0.7]]
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[np.inf, 0.7]])
        with self.assertRaises(ValueError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("must be finite", str(cm.exception))

        # 5. Two individually valid values that do not sum to 1.0: [[0.1, 0.6]]
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.6]])
        with self.assertRaises(ValueError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("must sum to 1.0", str(cm.exception))

        # Another unnormalized pair that exceeds 1.0: [[0.8, 0.8]]
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.8]])
        with self.assertRaises(ValueError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("must sum to 1.0", str(cm.exception))

        # 6. Boolean array rejected cleanly with TypeError
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[True, False]])
        with self.assertRaises(TypeError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("must be a real numeric value; got boolean", str(cm.exception))

        # 7. Non-numeric / string array rejected cleanly with TypeError
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([["0.3", "0.7"]])
        with self.assertRaises(TypeError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("must be a real numeric value", str(cm.exception))

        # 8. Complex probability array rejected cleanly with TypeError
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3 + 0.1j, 0.7 - 0.1j]])
        with self.assertRaises(TypeError) as cm:
            inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
        self.assertIn("must be a real numeric value", str(cm.exception))

        # 9. Valid normalized binary pair is accepted
        real_model = inference_helper.load_saved_model()
        with patch.object(real_model, "predict_proba", return_value=np.array([[0.85, 0.15]])):
            res = inference_helper.run_inference_pipeline(real_model, valid_row, validate=False)
            self.assertEqual(res.probability, 0.15)
            self.assertIsInstance(res.score_result, risk_scoring.ScoreResult)
            self.assertEqual(res.risk_band, "Middle model-estimated risk")

    def test_malformed_probabilities_rejected_before_score_or_explanation_calls(self):
        """Malformed output must fail at probability validation before scoring or explanation functions run."""
        valid_row = self._call_build()
        malformed_vectors = [
            np.array([[1.2, 0.7]]),
            np.array([[-0.2, 0.7]]),
            np.array([[np.nan, 0.7]]),
            np.array([[0.1, 0.6]]),
            np.array([[True, False]]),
        ]
        for bad_proba in malformed_vectors:
            mock_model = MagicMock()
            mock_model.predict_proba.return_value = bad_proba
            with patch("inference_helper.probability_to_score") as mock_score, \
                 patch("inference_helper.explain_logistic_prediction") as mock_explain:
                with self.assertRaises((ValueError, TypeError)):
                    inference_helper.run_inference_pipeline(mock_model, valid_row, validate=False)
                mock_score.assert_not_called()
                mock_explain.assert_not_called()

class TestAlternativeFeatureAblationExperiment(unittest.TestCase):
    """Automated test suite for the multi-seed alternative feature ablation experiment."""

    def test_scenario_definitions_exact_seven(self):
        """Scenarios dictionary must contain exactly the seven required configurations."""
        scenarios = ablation_exp.get_ablation_scenarios()
        self.assertEqual(len(scenarios), 7)
        expected_keys = {
            "traditional_baseline",
            "full_hybrid",
            "without_utility_payment_timeliness",
            "without_rental_payment_timeliness",
            "without_digital_payment_regularity",
            "without_cashflow_stability_index",
            "without_alternative_history_months",
        }
        self.assertEqual(set(scenarios.keys()), expected_keys)

    def test_scenario_feature_counts_and_ablation_exactness(self):
        """Each ablation scenario must remove exactly one named feature and retain all others."""
        scenarios = ablation_exp.get_ablation_scenarios()

        # Traditional baseline has exactly the 9 traditional features
        self.assertEqual(scenarios["traditional_baseline"]["features"], list(validation.TRADITIONAL_FEATURES))
        self.assertEqual(len(scenarios["traditional_baseline"]["features"]), 9)

        # Full hybrid has all 14 features
        self.assertEqual(scenarios["full_hybrid"]["features"], list(validation.ALL_FEATURES))
        self.assertEqual(len(scenarios["full_hybrid"]["features"]), 14)

        # Ablation scenarios remove exactly 1 alternative feature
        all_features_set = set(validation.ALL_FEATURES)
        for sc_name, sc_info in scenarios.items():
            if sc_name in ("traditional_baseline", "full_hybrid"):
                continue
            feats = sc_info["features"]
            ablated = sc_info["ablated_feature"]
            self.assertIn(ablated, validation.ALTERNATIVE_FEATURES)
            self.assertEqual(len(feats), 13)
            self.assertNotIn(ablated, feats)
            self.assertEqual(set(feats) | {ablated}, all_features_set)

    def test_single_seed_execution_and_schema_compatibility(self):
        """Executing one small seed must produce all 7 scenarios, valid metrics, and 6 paired deltas."""
        res_df, deltas_df = ablation_exp.run_single_seed_ablation(
            seed=2026,
            n_samples=250,
            test_size=0.20,
            n_bins=10,
        )
        self.assertEqual(len(res_df), 7)
        self.assertEqual(len(deltas_df), 6)

        # Check sample sizes
        self.assertTrue((res_df["train_size"] == 200).all())
        self.assertTrue((res_df["test_size"] == 50).all())

        # Check metric ranges
        for m in ablation_exp.METRIC_NAMES:
            self.assertIn(m, res_df.columns)
            self.assertTrue(np.all(np.isfinite(res_df[m])))

        self.assertTrue((res_df["roc_auc"] >= 0.0).all() and (res_df["roc_auc"] <= 1.0).all())
        self.assertTrue((res_df["gini_coefficient"] >= -1.0).all() and (res_df["gini_coefficient"] <= 1.0).all())
        self.assertTrue((res_df["ks_statistic"] >= 0.0).all() and (res_df["ks_statistic"] <= 1.0).all())
        self.assertTrue((res_df["average_precision"] >= 0.0).all() and (res_df["average_precision"] <= 1.0).all())
        self.assertTrue((res_df["brier_score"] >= 0.0).all() and (res_df["brier_score"] <= 1.0).all())
        self.assertTrue((res_df["log_loss"] >= 0.0).all())
        self.assertTrue((res_df["ece_10_bins"] >= 0.0).all() and (res_df["ece_10_bins"] <= 1.0).all())

        # Subtraction direction: ablated - full_hybrid
        full_auc = res_df.loc[res_df["scenario"] == "full_hybrid", "roc_auc"].iloc[0]
        for sc in deltas_df["scenario"].unique():
            sc_auc = res_df.loc[res_df["scenario"] == sc, "roc_auc"].iloc[0]
            expected_delta = sc_auc - full_auc
            actual_delta = deltas_df.loc[deltas_df["scenario"] == sc, "delta_roc_auc"].iloc[0]
            self.assertAlmostEqual(actual_delta, expected_delta, places=6)

    def test_deterministic_reproducibility(self):
        """Repeated execution with identical seed and sample count must be exactly deterministic."""
        res_a, deltas_a = ablation_exp.run_single_seed_ablation(seed=2030, n_samples=200, test_size=0.20)
        res_b, deltas_b = ablation_exp.run_single_seed_ablation(seed=2030, n_samples=200, test_size=0.20)
        pd.testing.assert_frame_equal(res_a, res_b)
        pd.testing.assert_frame_equal(deltas_a, deltas_b)

    def test_summary_computation_structure_and_quantiles(self):
        """Summary computation must compute expected quantiles and maintain monotonicity."""
        res_df, deltas_df, summary_df = ablation_exp.run_multi_seed_ablation(
            seeds=[2026, 2027],
            n_samples=200,
            test_size=0.20,
            verbose=False,
        )
        self.assertEqual(len(res_df), 14)  # 2 seeds * 7 scenarios
        self.assertEqual(len(deltas_df), 12)  # 2 seeds * 6 deltas

        # 7 scenarios * 7 metrics (abs) + 6 scenarios * 7 metrics (deltas) = 91 rows
        self.assertEqual(len(summary_df), 91)

        expected_cols = {
            "statistic_type", "scenario", "scenario_label", "ablated_feature",
            "metric", "mean", "std", "median", "min", "max", "q25", "q75",
            "fraction_negative", "fraction_positive", "fraction_negligible",
        }
        self.assertTrue(expected_cols.issubset(set(summary_df.columns)))
        self.assertNotIn("fraction_zero", summary_df.columns)

        # Quantile monotonicity: min <= q25 <= median <= q75 <= max
        self.assertTrue((summary_df["min"] <= summary_df["q25"] + 1e-9).all())
        self.assertTrue((summary_df["q25"] <= summary_df["median"] + 1e-9).all())
        self.assertTrue((summary_df["median"] <= summary_df["q75"] + 1e-9).all())
        self.assertTrue((summary_df["q75"] <= summary_df["max"] + 1e-9).all())

        # Absolute metrics must have NaN / blank direction fractions
        abs_mask = summary_df["statistic_type"] == "absolute_metric"
        self.assertTrue(summary_df.loc[abs_mask, "fraction_negative"].isna().all())
        self.assertTrue(summary_df.loc[abs_mask, "fraction_positive"].isna().all())
        self.assertTrue(summary_df.loc[abs_mask, "fraction_negligible"].isna().all())

        # Paired deltas must have finite fractions summing to 1.0
        delta_mask = summary_df["statistic_type"] == "paired_delta"
        delta_rows = summary_df[delta_mask]
        self.assertTrue(delta_rows["fraction_negative"].notna().all())
        self.assertTrue(delta_rows["fraction_positive"].notna().all())
        self.assertTrue(delta_rows["fraction_negligible"].notna().all())
        self.assertTrue((delta_rows["fraction_negative"] >= 0.0).all() and (delta_rows["fraction_negative"] <= 1.0).all())
        self.assertTrue((delta_rows["fraction_positive"] >= 0.0).all() and (delta_rows["fraction_positive"] <= 1.0).all())
        self.assertTrue((delta_rows["fraction_negligible"] >= 0.0).all() and (delta_rows["fraction_negligible"] <= 1.0).all())
        fraction_sums = delta_rows["fraction_negative"] + delta_rows["fraction_positive"] + delta_rows["fraction_negligible"]
        np.testing.assert_allclose(fraction_sums.to_numpy(), 1.0, atol=1e-6)

    def test_experiment_does_not_touch_production_artifacts_or_reports(self):
        """Experiment artifact generation must not overwrite existing models or primary reports."""
        prod_model = ROOT / "models" / "hybrid_logistic_regression.joblib"
        primary_report = ROOT / "reports" / "test_metrics.csv"

        model_mtime_before = prod_model.stat().st_mtime
        report_mtime_before = primary_report.stat().st_mtime

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            res_df, deltas_df, summary_df = ablation_exp.run_multi_seed_ablation(
                seeds=[2026],
                n_samples=150,
                verbose=False,
            )
            metadata = {
                "experiment_name": "test_isolation",
                "n_seeds": 1,
                "n_samples_per_seed": 150,
            }
            ablation_exp.save_experiment_artifacts(
                results_df=res_df,
                deltas_df=deltas_df,
                summary_df=summary_df,
                metadata=metadata,
                output_dir=tmp_path,
            )
            self.assertTrue((tmp_path / "results_by_seed.csv").exists())
            self.assertTrue((tmp_path / "paired_deltas_by_seed.csv").exists())
            self.assertTrue((tmp_path / "summary.csv").exists())
            self.assertTrue((tmp_path / "ranking_distribution.csv").exists())
            self.assertTrue((tmp_path / "feature_ablation_plot.png").exists())
            self.assertTrue((tmp_path / "experiment_report.md").exists())
            self.assertTrue((tmp_path / "run_metadata.json").exists())

        # Verify production files remain untouched
        self.assertEqual(prod_model.stat().st_mtime, model_mtime_before)
        self.assertEqual(primary_report.stat().st_mtime, report_mtime_before)

    def test_feature_ranking_statistics_derivation_and_robustness_expectations(self):
        """Derives per-feature ranking statistics from paired deltas and checks regression expectations."""
        deltas_csv_path = ROOT / "reports" / "ablation_robustness" / "paired_deltas_by_seed.csv"
        self.assertTrue(deltas_csv_path.exists(), "paired_deltas_by_seed.csv must exist in reports.")

        deltas_df = pd.read_csv(deltas_csv_path)
        ranking_df, ranking_meta = ablation_exp.compute_feature_ranking_statistics(deltas_df)

        # 1. Scope & filtering: exactly 5 single-feature ablations, traditional_baseline excluded
        self.assertEqual(len(ranking_df), 5)
        self.assertNotIn("traditional_baseline", ranking_df["scenario"].tolist())
        self.assertNotIn("all_alternative", ranking_df["ablated_feature"].tolist())
        self.assertEqual(set(ranking_df["ablated_feature"]), set(validation.ALTERNATIVE_FEATURES))

        # 2. Regression expectations against committed 20-seed dataset
        # Table counts: [rank_1, rank_2, rank_3, rank_4, rank_5], mean_rank, min_rank, max_rank
        expected_records = {
            "utility_payment_timeliness": {
                "counts": [15, 4, 0, 1, 0],
                "mean_rank": 1.35,
                "min_rank": 1,
                "max_rank": 4,
            },
            "cashflow_stability_index": {
                "counts": [5, 12, 3, 0, 0],
                "mean_rank": 1.90,
                "min_rank": 1,
                "max_rank": 3,
            },
            "digital_payment_regularity": {
                "counts": [0, 4, 10, 6, 0],
                "mean_rank": 3.10,
                "min_rank": 2,
                "max_rank": 4,
            },
            "rental_payment_timeliness": {
                "counts": [0, 0, 7, 13, 0],
                "mean_rank": 3.65,
                "min_rank": 3,
                "max_rank": 4,
            },
            "alternative_history_months": {
                "counts": [0, 0, 0, 0, 20],
                "mean_rank": 5.00,
                "min_rank": 5,
                "max_rank": 5,
            },
        }

        total_seeds = ranking_meta["total_seeds"]
        self.assertEqual(total_seeds, 20)

        for feat, exp in expected_records.items():
            row = ranking_df[ranking_df["ablated_feature"] == feat].iloc[0]
            for r in range(1, 6):
                exp_count = exp["counts"][r - 1]
                exp_prop = exp_count / total_seeds
                self.assertEqual(row[f"rank_{r}_count"], exp_count, f"{feat} rank {r} count mismatch")
                self.assertAlmostEqual(row[f"rank_{r}_prop"], exp_prop, places=6, msg=f"{feat} rank {r} prop mismatch")

            self.assertAlmostEqual(row["mean_rank"], exp["mean_rank"], places=4, msg=f"{feat} mean rank mismatch")
            self.assertEqual(row["min_rank"], exp["min_rank"], f"{feat} min rank mismatch")
            self.assertEqual(row["max_rank"], exp["max_rank"], f"{feat} max rank mismatch")

        # Proportions sum to 1.0 for each feature
        for _, row in ranking_df.iterrows():
            total_prop = sum(row[f"rank_{r}_prop"] for r in range(1, 6))
            self.assertAlmostEqual(total_prop, 1.0, places=6)

        # 3. Hierarchy ordering metadata
        expected_order = [
            "utility_payment_timeliness",
            "cashflow_stability_index",
            "digital_payment_regularity",
            "rental_payment_timeliness",
            "alternative_history_months",
        ]
        self.assertEqual(ranking_meta["mean_hierarchy"], expected_order)
        self.assertEqual(ranking_meta["exact_hierarchy_matches"], 7)
        self.assertAlmostEqual(ranking_meta["exact_hierarchy_proportion"], 7 / 20, places=6)

        # 4. Error handling
        bad_df = pd.DataFrame({"seed": [1], "scenario": ["traditional_baseline"], "ablated_feature": ["all_alternative"]})
        with self.assertRaises(ValueError):
            ablation_exp.compute_feature_ranking_statistics(bad_df)

    def test_paired_delta_classification_boundaries_and_exact_semantics(self):
        """Verify exact boundary semantics and mutually exclusive 1.0 sum for paired deltas."""
        # Using roc_auc where tolerance is exactly 0.001
        tol = ablation_exp.PAIRED_DELTA_TOLERANCES["roc_auc"]
        self.assertEqual(tol, 0.001)

        # 1. Negative delta below negative tolerance (delta < -tol)
        res_neg = ablation_exp.classify_paired_deltas([-0.002, -0.0010001], "roc_auc")
        self.assertEqual(res_neg["fraction_negative"], 1.0)
        self.assertEqual(res_neg["fraction_positive"], 0.0)
        self.assertEqual(res_neg["fraction_negligible"], 0.0)

        # 2. Positive delta above positive tolerance (delta > tol)
        res_pos = ablation_exp.classify_paired_deltas([0.002, 0.0010001], "roc_auc")
        self.assertEqual(res_pos["fraction_negative"], 0.0)
        self.assertEqual(res_pos["fraction_positive"], 1.0)
        self.assertEqual(res_pos["fraction_negligible"], 0.0)

        # 3. Delta exactly at positive tolerance (delta == tol -> negligible)
        res_exact_pos = ablation_exp.classify_paired_deltas([0.001], "roc_auc")
        self.assertEqual(res_exact_pos["fraction_negative"], 0.0)
        self.assertEqual(res_exact_pos["fraction_positive"], 0.0)
        self.assertEqual(res_exact_pos["fraction_negligible"], 1.0)

        # 4. Delta exactly at negative tolerance (delta == -tol -> negligible)
        res_exact_neg = ablation_exp.classify_paired_deltas([-0.001], "roc_auc")
        self.assertEqual(res_exact_neg["fraction_negative"], 0.0)
        self.assertEqual(res_exact_neg["fraction_positive"], 0.0)
        self.assertEqual(res_exact_neg["fraction_negligible"], 1.0)

        # 5. Delta between boundaries, including exact zero (-tol < delta < tol)
        res_between = ablation_exp.classify_paired_deltas([-0.0005, 0.0, 0.0005], "roc_auc")
        self.assertEqual(res_between["fraction_negative"], 0.0)
        self.assertEqual(res_between["fraction_positive"], 0.0)
        self.assertEqual(res_between["fraction_negligible"], 1.0)

        # 6. Combined mixed vector verifying mutual exclusivity and exact 1.0 sum
        mixed = [-0.005, -0.001, -0.0002, 0.0, 0.0004, 0.001, 0.003]  # 7 items: 1 neg, 5 negligible, 1 pos
        res_mixed = ablation_exp.classify_paired_deltas(mixed, "roc_auc")
        self.assertAlmostEqual(res_mixed["fraction_negative"], 1 / 7)
        self.assertAlmostEqual(res_mixed["fraction_negligible"], 5 / 7)
        self.assertAlmostEqual(res_mixed["fraction_positive"], 1 / 7)
        self.assertAlmostEqual(
            res_mixed["fraction_negative"] + res_mixed["fraction_positive"] + res_mixed["fraction_negligible"],
            1.0,
            places=6,
        )

        # Empty and all-NaN handling
        empty_res = ablation_exp.classify_paired_deltas([], "roc_auc")
        self.assertEqual(empty_res, {"fraction_negative": 0.0, "fraction_positive": 0.0, "fraction_negligible": 0.0})
        nan_res = ablation_exp.classify_paired_deltas([np.nan, np.inf], "roc_auc")
        self.assertEqual(nan_res, {"fraction_negative": 0.0, "fraction_positive": 0.0, "fraction_negligible": 0.0})

    def test_all_seven_configured_metrics_and_tolerances(self):
        """Verify that all 7 configured metrics have explicit tolerances and partition correctly."""
        expected_metrics = {
            "roc_auc",
            "gini_coefficient",
            "ks_statistic",
            "average_precision",
            "brier_score",
            "log_loss",
            "ece_10_bins",
        }
        self.assertEqual(set(ablation_exp.PAIRED_DELTA_TOLERANCES.keys()), expected_metrics)
        self.assertEqual(ablation_exp.PAIRED_DELTA_TOLERANCES["gini_coefficient"], 0.002)

        for m in expected_metrics:
            tol = ablation_exp.PAIRED_DELTA_TOLERANCES[m]
            self.assertGreater(tol, 0.0)

            # Test vector with: 1 strictly below, 1 at -tol, 1 at 0, 1 at +tol, 1 strictly above
            sample_deltas = [-2 * tol, -tol, 0.0, tol, 2 * tol]
            # Test with both base metric name and delta_ prefixed name
            for name in (m, f"delta_{m}"):
                res = ablation_exp.classify_paired_deltas(sample_deltas, name)
                self.assertAlmostEqual(res["fraction_negative"], 0.20, places=6)
                self.assertAlmostEqual(res["fraction_negligible"], 0.60, places=6)
                self.assertAlmostEqual(res["fraction_positive"], 0.20, places=6)
                self.assertAlmostEqual(
                    res["fraction_negative"] + res["fraction_positive"] + res["fraction_negligible"],
                    1.0,
                    places=6,
                )

    def test_invalid_or_unsupported_metric_tolerance_handling(self):
        """Unsupported metric or invalid tolerance must raise appropriate ValueError."""
        # 1. Unsupported metric or missing tolerance mapping
        with self.assertRaises(ValueError):
            ablation_exp.classify_paired_deltas([0.01], "unsupported_metric")

        with self.assertRaises(ValueError):
            ablation_exp.classify_paired_deltas([0.01], "roc_auc", tolerances={"gini_coefficient": 0.002})

        with self.assertRaises(ValueError):
            ablation_exp.classify_paired_deltas([0.01], "roc_auc", tolerances=None)  # type: ignore

        # 2. Invalid tolerance values: NaN, +inf, -inf, negatives, None, strings, Booleans
        invalid_tolerances = [
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.001,
            -1.0,
            -5,
            None,
            "0.001",
            "none",
            True,
            False,
        ]
        for inv in invalid_tolerances:
            with self.subTest(invalid_tolerance=inv):
                with self.assertRaises(ValueError):
                    ablation_exp.classify_paired_deltas([0.01], "roc_auc", tolerances={"roc_auc": inv})  # type: ignore

        # 3. Valid non-negative zero tolerance is accepted
        res_zero = ablation_exp.classify_paired_deltas([-0.01, 0.0, 0.01], "roc_auc", tolerances={"roc_auc": 0.0})
        self.assertAlmostEqual(res_zero["fraction_negative"], 1 / 3)
        self.assertAlmostEqual(res_zero["fraction_negligible"], 1 / 3)
        self.assertAlmostEqual(res_zero["fraction_positive"], 1 / 3)

        res_zero_int = ablation_exp.classify_paired_deltas([0.0], "roc_auc", tolerances={"roc_auc": 0})
        self.assertEqual(res_zero_int["fraction_negligible"], 1.0)

    def test_installed_dependency_versions_in_metadata(self):
        """Metadata must dynamically record installed versions of Python, NumPy, pandas, sklearn, matplotlib."""
        versions = ablation_exp.get_installed_dependency_versions()
        required_keys = {
            "python_version",
            "numpy_version",
            "pandas_version",
            "scikit_learn_version",
            "matplotlib_version",
        }
        self.assertTrue(required_keys.issubset(set(versions.keys())))
        for k in required_keys:
            val = versions[k]
            self.assertIsInstance(val, str)
            self.assertNotEqual(val, "not_installed")
            self.assertFalse(val.startswith("error:"))

        # Also inspect committed run_metadata.json
        meta_path = ROOT / "reports" / "ablation_robustness" / "run_metadata.json"
        self.assertTrue(meta_path.exists())
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for k in required_keys:
            self.assertIn(k, meta)
            self.assertIsInstance(meta[k], str)

        self.assertIn("paired_delta_tolerances", meta)
        self.assertEqual(meta["paired_delta_tolerances"], ablation_exp.PAIRED_DELTA_TOLERANCES)
        self.assertIn("delta_classification_policy", meta)


class TestBootstrapUncertaintyAnalysis(unittest.TestCase):
    """Regression and contract tests for Phase 07 bootstrap uncertainty analysis."""

    def setUp(self):
        # Create deterministic toy binary labels and model predictions (N=60)
        rng = np.random.default_rng(42)
        self.n_samples = 60
        # Ensure balanced classes: 20 ones, 40 zeros
        self.y_true = np.array([1] * 20 + [0] * 40)
        # Model A: moderate accuracy
        self.p_a = np.clip(
            np.where(
                self.y_true == 1,
                rng.uniform(0.4, 0.9, size=60),
                rng.uniform(0.1, 0.6, size=60),
            ),
            0.0,
            1.0,
        )
        # Model B: slightly higher discrimination
        self.p_b = np.clip(
            np.where(
                self.y_true == 1,
                rng.uniform(0.6, 0.95, size=60),
                rng.uniform(0.05, 0.4, size=60),
            ),
            0.0,
            1.0,
        )
        self.model_preds = {
            "ModelA": self.p_a,
            "ModelB": self.p_b,
        }
        self.pairs = [
            ("ModelB", "ModelA"),
            ("ModelA", "ModelA"),  # Identical model comparison
        ]

    def test_bootstrap_output_shape_and_column_schema(self):
        """Bootstrap cohort analysis must produce DataFrames with required column schemas."""
        m_df, p_df, req, val = uncertainty_analysis.bootstrap_cohort_analysis(
            cohort_name="toy_cohort",
            y_true=self.y_true,
            model_predictions=self.model_preds,
            paired_comparisons=self.pairs,
            n_bootstrap=25,
            seed=2027,
            confidence_level=0.95,
        )
        self.assertEqual(req, 25)
        self.assertEqual(val, 25)

        required_model_cols = [
            "cohort",
            "n_samples",
            "n_positive",
            "n_negative",
            "model",
            "metric",
            "point_estimate",
            "ci_lower",
            "ci_upper",
            "confidence_level",
            "method",
            "n_bootstrap_requested",
            "n_bootstrap_valid",
            "bootstrap_seed",
        ]
        self.assertEqual(list(m_df.columns), required_model_cols)
        # 2 models * 7 metrics = 14 rows
        self.assertEqual(len(m_df), 2 * 7)

        required_paired_cols = [
            "cohort",
            "n_samples",
            "model_a",
            "model_b",
            "metric",
            "metric_direction",
            "point_delta",
            "ci_lower",
            "ci_upper",
            "confidence_level",
            "method",
            "n_bootstrap_requested",
            "n_bootstrap_valid",
            "bootstrap_seed",
            "interval_excludes_zero",
        ]
        self.assertEqual(list(p_df.columns), required_paired_cols)
        # 2 pairs * 7 metrics = 14 rows
        self.assertEqual(len(p_df), 2 * 7)

    def test_finite_ordered_interval_bounds(self):
        """All confidence intervals must be finite and satisfy ci_lower <= ci_upper."""
        m_df, p_df, _, _ = uncertainty_analysis.bootstrap_cohort_analysis(
            cohort_name="toy_cohort",
            y_true=self.y_true,
            model_predictions=self.model_preds,
            paired_comparisons=self.pairs,
            n_bootstrap=25,
            seed=2027,
            confidence_level=0.95,
        )
        for _, row in m_df.iterrows():
            self.assertTrue(np.isfinite(row["ci_lower"]))
            self.assertTrue(np.isfinite(row["ci_upper"]))
            self.assertLessEqual(row["ci_lower"], row["ci_upper"])

        for _, row in p_df.iterrows():
            self.assertTrue(np.isfinite(row["ci_lower"]))
            self.assertTrue(np.isfinite(row["ci_upper"]))
            self.assertLessEqual(row["ci_lower"], row["ci_upper"])

    def test_point_estimate_computation_original_observations(self):
        """Point estimates must exactly match metric calculations on unresampled original cohort."""
        m_df, p_df, _, _ = uncertainty_analysis.bootstrap_cohort_analysis(
            cohort_name="toy_cohort",
            y_true=self.y_true,
            model_predictions=self.model_preds,
            paired_comparisons=self.pairs,
            n_bootstrap=25,
            seed=2027,
            confidence_level=0.95,
        )
        expected_metrics_a = uncertainty_analysis.compute_model_metrics(self.y_true, self.p_a)
        expected_metrics_b = uncertainty_analysis.compute_model_metrics(self.y_true, self.p_b)

        for _, row in m_df[m_df["model"] == "ModelA"].iterrows():
            self.assertAlmostEqual(row["point_estimate"], expected_metrics_a[row["metric"]], places=9)

        for _, row in p_df[p_df["model_a"] == "ModelB"].iterrows():
            expected_delta = expected_metrics_b[row["metric"]] - expected_metrics_a[row["metric"]]
            self.assertAlmostEqual(row["point_delta"], expected_delta, places=9)

    def test_reproducibility_and_seed_sensitivity(self):
        """Identical seeds must yield identical intervals; different seeds must produce different resamples."""
        m_df1, p_df1, _, _ = uncertainty_analysis.bootstrap_cohort_analysis(
            cohort_name="toy_cohort",
            y_true=self.y_true,
            model_predictions=self.model_preds,
            paired_comparisons=self.pairs,
            n_bootstrap=30,
            seed=2027,
        )
        m_df2, p_df2, _, _ = uncertainty_analysis.bootstrap_cohort_analysis(
            cohort_name="toy_cohort",
            y_true=self.y_true,
            model_predictions=self.model_preds,
            paired_comparisons=self.pairs,
            n_bootstrap=30,
            seed=2027,
        )
        pd.testing.assert_frame_equal(m_df1, m_df2)
        pd.testing.assert_frame_equal(p_df1, p_df2)

        m_df_diff, _, _, _ = uncertainty_analysis.bootstrap_cohort_analysis(
            cohort_name="toy_cohort",
            y_true=self.y_true,
            model_predictions=self.model_preds,
            paired_comparisons=self.pairs,
            n_bootstrap=30,
            seed=9999,
        )
        # At least one bound should differ between seed 2027 and seed 9999
        self.assertFalse(np.allclose(m_df1["ci_lower"].values, m_df_diff["ci_lower"].values))

    def test_paired_deltas_on_shared_indices_and_identical_model(self):
        """Paired deltas on shared indices with identical models must yield zero delta and zero width."""
        _, p_df, _, _ = uncertainty_analysis.bootstrap_cohort_analysis(
            cohort_name="toy_cohort",
            y_true=self.y_true,
            model_predictions=self.model_preds,
            paired_comparisons=[("ModelA", "ModelA")],
            n_bootstrap=25,
            seed=2027,
        )
        for _, row in p_df.iterrows():
            self.assertEqual(row["point_delta"], 0.0)
            self.assertEqual(row["ci_lower"], 0.0)
            self.assertEqual(row["ci_upper"], 0.0)
            self.assertFalse(row["interval_excludes_zero"])

    def test_metric_direction_and_interpretation(self):
        """Classification metrics must use 'higher' direction; error/calibration must use 'lower'."""
        _, p_df, _, _ = uncertainty_analysis.bootstrap_cohort_analysis(
            cohort_name="toy_cohort",
            y_true=self.y_true,
            model_predictions=self.model_preds,
            paired_comparisons=[("ModelB", "ModelA")],
            n_bootstrap=25,
            seed=2027,
        )
        for _, row in p_df.iterrows():
            if row["metric"] in {"roc_auc", "gini_coefficient", "ks_statistic", "average_precision"}:
                self.assertEqual(row["metric_direction"], "higher")
            else:
                self.assertEqual(row["metric_direction"], "lower")

    def test_one_class_replicates_handled_explicitly(self):
        """Replicates containing only one class must be detected and discarded without raising unhandled errors."""
        # Single-class dataset must raise ValueError in compute_model_metrics
        with self.assertRaises(ValueError):
            uncertainty_analysis.compute_model_metrics([0, 0, 0], [0.1, 0.2, 0.3])

        # If a cohort has only one class, bootstrap_cohort_analysis must fail clearly
        with self.assertRaises(ValueError):
            uncertainty_analysis.bootstrap_cohort_analysis(
                cohort_name="single_class",
                y_true=[1, 1, 1],
                model_predictions={"m": [0.5, 0.5, 0.5]},
                paired_comparisons=[("m", "m")],
            )

    def test_invalid_arguments_and_inputs_fail_clearly(self):
        """Invalid bootstrap arguments, probabilities, or shapes must raise ValueError."""
        # Argument validation
        for inv_n in [0, -10, None, 2.5, "100", True]:
            with self.subTest(invalid_n=inv_n):
                with self.assertRaises(ValueError):
                    uncertainty_analysis.validate_arguments(inv_n, 2027, 0.95)  # type: ignore

        for inv_seed in [-1, None, "2027", True]:
            with self.subTest(invalid_seed=inv_seed):
                with self.assertRaises(ValueError):
                    uncertainty_analysis.validate_arguments(100, inv_seed, 0.95)  # type: ignore

        for inv_conf in [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf"), None, "0.95", True]:
            with self.subTest(invalid_conf=inv_conf):
                with self.assertRaises(ValueError):
                    uncertainty_analysis.validate_arguments(100, 2027, inv_conf)  # type: ignore

        # Array validation
        with self.assertRaises(ValueError):
            uncertainty_analysis.compute_model_metrics([0, 1], [0.2, 1.5])  # > 1.0
        with self.assertRaises(ValueError):
            uncertainty_analysis.compute_model_metrics([0, 1], [0.2, float("nan")])  # NaN
        with self.assertRaises(ValueError):
            uncertainty_analysis.compute_model_metrics([0, 1, 0], [0.2, 0.8])  # Length mismatch

    def test_metadata_generation_and_package_versions(self):
        """Metadata helper and committed run_metadata.json must include required keys and packages."""
        versions = uncertainty_analysis.get_installed_dependency_versions()
        required_pkgs = [
            "python_version",
            "numpy_version",
            "pandas_version",
            "scikit_learn_version",
            "matplotlib_version",
        ]
        for p in required_pkgs:
            self.assertIn(p, versions)
            self.assertIsInstance(versions[p], str)
            self.assertNotEqual(versions[p], "not_installed")

        meta_file = ROOT / "reports" / "uncertainty" / "run_metadata.json"
        self.assertTrue(meta_file.exists())
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self.assertIn("cohorts", meta)
        self.assertIn("model_pair_definitions", meta)
        self.assertIn("conditioning_scope_and_limitations", meta)
        self.assertIn("synthetic_data_disclaimer", meta)

    def test_no_bureau_mask_alignment_and_cohort_counts(self):
        """The no_bureau_history cohort mask must align accurately with test observations."""
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(ROOT / "data" / "synthetic_applicants.csv")
        train_df, test_df = train_test_split(
            df, test_size=0.20, random_state=2026, stratify=df[train.TARGET]
        )

        sub_mask = test_df["bureau_history_available"].eq(0)
        self.assertEqual(sub_mask.sum(), 926)

        m_file = ROOT / "reports" / "uncertainty" / "model_metric_confidence_intervals.csv"
        self.assertTrue(m_file.exists())
        m_df = pd.read_csv(m_file)
        sub_rows = m_df[m_df["cohort"] == "no_bureau_history"]
        self.assertTrue((sub_rows["n_samples"] == 926).all())
        self.assertTrue((sub_rows["n_positive"] == 259).all())
        self.assertTrue((sub_rows["n_negative"] == 667).all())

    def test_dynamic_confidence_level_in_markdown_report(self):
        """Report table headings and interval text must dynamically reflect configured confidence level."""
        m_df, p_df, req, val = uncertainty_analysis.bootstrap_cohort_analysis(
            cohort_name="toy_cohort",
            y_true=self.y_true,
            model_predictions=self.model_preds,
            paired_comparisons=self.pairs,
            n_bootstrap=20,
            seed=2027,
            confidence_level=0.90,
        )
        metadata = {
            "dataset_source_path": "data/synthetic_applicants.csv",
            "dataset_generation_seed": 2026,
            "test_split_configuration": {
                "test_size": 0.20,
                "random_state": 2026,
                "stratify_column": "default_within_12_months",
                "n_test_observations": 2000,
            },
            "cohorts": {
                "toy_cohort": {
                    "definition": "Toy cohort for testing",
                    "n_samples": self.n_samples,
                    "n_positive": 20,
                    "n_negative": 40,
                    "synthetic_default_rate": 0.3333,
                }
            },
            "bootstrap_configuration": {
                "n_bootstrap_requested": 20,
                "n_bootstrap_valid": {"toy_cohort": val},
                "confidence_level": 0.90,
                "seed": 2027,
            },
            "environment_versions": {
                "python_version": "3.13.7",
                "numpy_version": "2.5.4",
                "pandas_version": "3.0.6",
                "scikit_learn_version": "1.7.2",
                "matplotlib_version": "3.11.2",
            },
        }
        report = uncertainty_analysis.generate_markdown_report(m_df, p_df, metadata)

        # Headings and descriptions must state 90%, NOT 95%
        self.assertIn("90% CI Lower", report)
        self.assertIn("90% CI Upper", report)
        self.assertIn("90% two-sided percentile intervals", report)
        self.assertIn("90% percentile interval", report)
        self.assertNotIn("95% CI", report)

    def test_atomic_replicate_failure_handling(self):
        """Injected failure on one model must atomically discard the replicate without leaving partial data."""
        original_compute = uncertainty_analysis.compute_model_metrics
        call_count = 0

        def flaky_compute(y_true, probabilities, n_bins=10):
            nonlocal call_count
            call_count += 1
            # Calls 1 & 2 are point estimates. Call 3 & 4 are rep 1. Call 5 & 6 are rep 2.
            # Inject a controlled ValueError on ModelB of rep 2 (call 6):
            if call_count == 6:
                raise ValueError("Controlled simulated failure during ModelB evaluation")
            return original_compute(y_true, probabilities, n_bins=n_bins)

        with patch("uncertainty_analysis.compute_model_metrics", side_effect=flaky_compute):
            m_df, p_df, req, val = uncertainty_analysis.bootstrap_cohort_analysis(
                cohort_name="toy_cohort",
                y_true=self.y_true,
                model_predictions=self.model_preds,
                paired_comparisons=self.pairs,
                n_bootstrap=10,
                seed=2027,
                min_valid_replicates=5,
            )

        self.assertEqual(req, 10)
        self.assertEqual(val, 9)  # Exactly 1 replicate failed and was discarded
        self.assertTrue((m_df["n_bootstrap_valid"] == 9).all())
        self.assertTrue((p_df["n_bootstrap_valid"] == 9).all())
        self.assertTrue(np.all(np.isfinite(m_df["ci_lower"])))
        self.assertTrue(np.all(np.isfinite(m_df["ci_upper"])))

    def test_unexpected_programming_error_surfaces(self):
        """Unexpected programming exceptions (e.g., TypeError) must not be silently swallowed."""
        with patch("uncertainty_analysis.compute_model_metrics", side_effect=TypeError("Unexpected code bug")):
            with self.assertRaises(TypeError):
                uncertainty_analysis.bootstrap_cohort_analysis(
                    cohort_name="toy_cohort",
                    y_true=self.y_true,
                    model_predictions=self.model_preds,
                    paired_comparisons=self.pairs,
                    n_bootstrap=5,
                    seed=2027,
                )


# ---------------------------------------------------------------------------
# Phase 08: Controlled Input-Degradation Stress Testing & Sensitivity Analysis
# ---------------------------------------------------------------------------

class TestStressTestingSensitivityAnalysis(unittest.TestCase):
    """Automated tests for Phase 08 input-degradation stress testing and sensitivity analysis."""

    @classmethod
    def setUpClass(cls):
        df = pd.read_csv(stress_testing.DATA_PATH)
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df,
            test_size=stress_testing.DEFAULT_TEST_SIZE,
            random_state=stress_testing.DEFAULT_SPLIT_SEED,
            stratify=df[stress_testing.TARGET],
        )
        cls.test_df = test_df
        cls.y_test = test_df[stress_testing.TARGET].to_numpy()

        # Run one deterministic execution of the experiment for fast verification
        cls.metrics_df, cls.deltas_df, cls.metadata, cls.report = (
            stress_testing.run_stress_testing_experiment(
                base_seed=2028,
                execution_timestamp="2026-01-01T00:00:00+00:00",
            )
        )

    def test_scenario_definitions_count_and_composition(self):
        """All expected 35 scenario definitions are produced with correct families and counts."""
        scenarios = stress_testing.build_scenario_definitions()
        self.assertEqual(len(scenarios), 35)

        families = [s["scenario_family"] for s in scenarios]
        self.assertEqual(families.count("clean_reference"), 1)
        self.assertEqual(families.count("joint_missingness"), 4)
        self.assertEqual(families.count("individual_missingness"), 15)
        self.assertEqual(families.count("measurement_noise"), 15)

        # All 5 alternative features represented in individual missingness
        ind_features = {s["perturbed_feature"] for s in scenarios if s["scenario_family"] == "individual_missingness"}
        self.assertEqual(ind_features, set(stress_testing.ALTERNATIVE_FEATURES))

        # 4 continuous features in continuous noise (3 levels each)
        cont_features = {
            s["perturbed_feature"]
            for s in scenarios
            if s["scenario_type"] == "measurement_noise_continuous"
        }
        self.assertEqual(cont_features, set(stress_testing.CONTINUOUS_ALTERNATIVE_FEATURES))

        # alternative_history_months in ordinal noise (3 levels)
        ord_features = [
            s for s in scenarios if s["scenario_type"] == "measurement_noise_ordinal"
        ]
        self.assertEqual(len(ord_features), 3)
        self.assertEqual({s["configured_intensity"] for s in ord_features}, {1.0, 3.0, 6.0})

    def test_clean_baseline_unchanged_and_matches_phase_07(self):
        """Clean baseline data and predictions must be unchanged and match Phase 07 values."""
        clean_def = {
            "scenario_id": "clean_reference",
            "scenario_family": "clean_reference",
            "scenario_type": "clean",
            "scenario_description": "Clean reference",
            "perturbed_feature": "none",
            "configured_intensity": 0.0,
        }
        df_clean_gen, meta = stress_testing.generate_scenario_data(self.test_df, clean_def, scenario_seed=2028)
        pd.testing.assert_frame_equal(df_clean_gen, self.test_df)
        self.assertEqual(meta["effective_additional_rate"], 0.0)

        clean_metrics = self.metadata["clean_baseline_metrics"]
        # Assert exact agreement with Phase 07 held-out baseline point estimates
        self.assertAlmostEqual(clean_metrics["Traditional baseline"]["roc_auc"], 0.80709467, places=6)
        self.assertAlmostEqual(clean_metrics["Hybrid (traditional + alternative)"]["roc_auc"], 0.90978432, places=6)
        self.assertAlmostEqual(clean_metrics["Hybrid (HistGradientBoosting)"]["roc_auc"], 0.90632351, places=6)

    def test_scenario_generation_deterministic_for_fixed_seeds(self):
        """Scenario generation must be completely deterministic for identical seeds and differ for different seeds."""
        sc_def = {
            "scenario_id": "joint_test",
            "scenario_family": "joint_missingness",
            "scenario_type": "missingness_joint",
            "scenario_description": "Joint 50%",
            "perturbed_feature": "all_alternative",
            "configured_intensity": 0.50,
        }
        df1, meta1 = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=12345)
        df2, meta2 = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=12345)
        df3, _ = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=54321)

        pd.testing.assert_frame_equal(df1, df2)
        self.assertEqual(meta1, meta2)
        self.assertFalse(df1.equals(df3))

    def test_source_dataframe_not_mutated(self):
        """The source DataFrame must not be mutated in any scenario generation call."""
        df_copy = self.test_df.copy(deep=True)
        for sc_def in stress_testing.build_scenario_definitions()[:5]:
            _ = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=999)
        pd.testing.assert_frame_equal(self.test_df, df_copy)

    def test_row_order_and_target_alignment_preserved(self):
        """Original row order, indices, and alignment must be strictly preserved."""
        for sc_def in stress_testing.build_scenario_definitions()[::7]:
            df_sc, _ = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=100)
            self.assertTrue(df_sc.index.equals(self.test_df.index))
            self.assertEqual(len(df_sc), len(self.test_df))

    def test_mcar_masks_independent_of_target_and_predictions(self):
        """MCAR dropout selection must be purely random and independent of target values or predictions."""
        sc_def = {
            "scenario_id": "joint_test",
            "scenario_family": "joint_missingness",
            "scenario_type": "missingness_joint",
            "scenario_description": "Joint 50%",
            "perturbed_feature": "all_alternative",
            "configured_intensity": 0.50,
        }
        df_sc, _ = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=42)
        degraded_mask = df_sc["utility_payment_timeliness"].isna() & self.test_df["utility_payment_timeliness"].notna()

        # The proportion of positives among degraded rows should be close to test set base rate (~27.5%)
        positive_rate_degraded = self.y_test[degraded_mask].mean()
        overall_positive_rate = self.y_test.mean()
        self.assertAlmostEqual(positive_rate_degraded, overall_positive_rate, delta=0.08)

    def test_preexisting_missingness_preserved_and_effective_rates_accurate(self):
        """Pre-existing missingness must remain NaN and observed rates must reflect total missingness."""
        sc_def = {
            "scenario_id": "ind_missing_util",
            "scenario_family": "individual_missingness",
            "scenario_type": "missingness_individual",
            "scenario_description": "Util 25%",
            "perturbed_feature": "utility_payment_timeliness",
            "configured_intensity": 0.25,
        }
        df_sc, meta = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=77)
        preexisting_nans = self.test_df["utility_payment_timeliness"].isna()
        self.assertTrue(df_sc.loc[preexisting_nans, "utility_payment_timeliness"].isna().all())

        # Effective additional rate was 250 / 2000 = 0.125
        self.assertEqual(meta["effective_additional_rate"], 0.25)
        # Observed rate reflects pre-existing ~7% plus additional ~25%
        self.assertAlmostEqual(
            meta["observed_missingness_rates"]["utility_payment_timeliness"],
            df_sc["utility_payment_timeliness"].isna().mean(),
        )

    def test_joint_dropout_applies_same_row_mask_across_all_alternative_features(self):
        """Joint dropout must apply the exact same row degradation mask to all 5 alternative features."""
        sc_def = {
            "scenario_id": "joint_50",
            "scenario_family": "joint_missingness",
            "scenario_type": "missingness_joint",
            "scenario_description": "Joint 50%",
            "perturbed_feature": "all_alternative",
            "configured_intensity": 0.50,
        }
        df_sc, meta = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=88)

        # In alternative_history_months, which had 0 pre-existing missingness, missing rows show the exact mask:
        selected_mask = df_sc["alternative_history_months"].isna()
        self.assertEqual(selected_mask.sum(), meta["rows_selected_for_degradation"])

        # Every other alternative feature must be NaN on that exact mask:
        for alt_col in stress_testing.ALTERNATIVE_FEATURES:
            self.assertTrue(df_sc.loc[selected_mask, alt_col].isna().all())

    def test_individual_dropout_does_not_alter_other_columns(self):
        """Single-feature dropout must not alter any other predictor columns."""
        sc_def = {
            "scenario_id": "ind_missing_rental",
            "scenario_family": "individual_missingness",
            "scenario_type": "missingness_individual",
            "scenario_description": "Rental 50%",
            "perturbed_feature": "rental_payment_timeliness",
            "configured_intensity": 0.50,
        }
        df_sc, _ = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=99)
        other_features = [c for c in validation.ALL_FEATURES if c != "rental_payment_timeliness"]
        pd.testing.assert_frame_equal(df_sc[other_features], self.test_df[other_features])

    def test_measurement_noise_preserves_nans(self):
        """Additive measurement noise must never perturb existing NaN values."""
        sc_def = {
            "scenario_id": "noise_rental",
            "scenario_family": "measurement_noise",
            "scenario_type": "measurement_noise_continuous",
            "scenario_description": "Rental noise 20%",
            "perturbed_feature": "rental_payment_timeliness",
            "configured_intensity": 0.20,
        }
        df_sc, _ = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=111)
        clean_nan_mask = self.test_df["rental_payment_timeliness"].isna()
        scenario_nan_mask = df_sc["rental_payment_timeliness"].isna()
        self.assertTrue(clean_nan_mask.equals(scenario_nan_mask))

    def test_continuous_features_respect_domain_bounds(self):
        """Continuous features perturbed by noise must strictly respect synthetic domain bounds."""
        for feat in stress_testing.CONTINUOUS_ALTERNATIVE_FEATURES:
            sc_def = {
                "scenario_id": f"noise_{feat}_test",
                "scenario_family": "measurement_noise",
                "scenario_type": "measurement_noise_continuous",
                "scenario_description": f"{feat} noise 20%",
                "perturbed_feature": feat,
                "configured_intensity": 0.20,
            }
            df_sc, meta = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=222)
            vals = df_sc[feat].dropna().to_numpy()
            dmin = validation.DOMAIN_BOUNDS[feat]["min"]
            dmax = validation.DOMAIN_BOUNDS[feat]["max"]
            self.assertTrue(np.all(vals >= dmin))
            self.assertTrue(np.all(vals <= dmax))
            self.assertGreater(meta["pct_clipped"], 0.0)

    def test_alternative_history_months_remains_valid_bounded_integer(self):
        """alternative_history_months after ordinal noise must be integers bounded in [0, 24]."""
        sc_def = {
            "scenario_id": "noise_alt_months",
            "scenario_family": "measurement_noise",
            "scenario_type": "measurement_noise_ordinal",
            "scenario_description": "Alt months noise 6m",
            "perturbed_feature": "alternative_history_months",
            "configured_intensity": 6.0,
        }
        df_sc, meta = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=333)
        vals = df_sc["alternative_history_months"].dropna().to_numpy()
        self.assertTrue(np.all(vals >= 0))
        self.assertTrue(np.all(vals <= 24))
        self.assertTrue(np.all(np.isclose(vals, np.round(vals))))
        self.assertGreater(meta["pct_clipped"], 0.0)

    def test_each_scenario_passes_inference_validation_contract(self):
        """Every generated scenario DataFrame must pass validate_feature_frame without errors."""
        for sc_def in stress_testing.build_scenario_definitions():
            df_sc, _ = stress_testing.generate_scenario_data(self.test_df, sc_def, scenario_seed=444)
            validation.validate_feature_frame(df_sc[validation.ALL_FEATURES])

    def test_metrics_and_deltas_coverage_and_finiteness(self):
        """Scenario metrics and deltas DataFrames must have 735 rows and all finite values."""
        self.assertEqual(len(self.metrics_df), 735)
        self.assertEqual(len(self.deltas_df), 735)

        self.assertTrue(np.all(np.isfinite(self.metrics_df["metric_value"])))
        self.assertTrue(np.all(np.isfinite(self.metrics_df["mean_abs_prob_delta"])))
        self.assertTrue(np.all(np.isfinite(self.metrics_df["p95_abs_prob_delta"])))

        self.assertTrue(np.all(np.isfinite(self.deltas_df["delta_vs_clean"])))
        self.assertTrue(np.all(np.isfinite(self.deltas_df["direction_adjusted_degradation"])))

    def test_direction_adjusted_degradation_calculation(self):
        """Degradation calculation signs must correctly reflect metric optimization direction."""
        calc = stress_testing.calculate_direction_adjusted_degradation
        # Higher is better: clean 0.90, scenario 0.85 -> degradation +0.05 (worse)
        self.assertAlmostEqual(calc(0.90, 0.85, "higher"), 0.05)
        # Higher is better: clean 0.90, scenario 0.92 -> degradation -0.02 (improved)
        self.assertAlmostEqual(calc(0.90, 0.92, "higher"), -0.02)

        # Lower is better: clean 0.10, scenario 0.12 -> degradation +0.02 (worse)
        self.assertAlmostEqual(calc(0.10, 0.12, "lower"), 0.02)
        # Lower is better: clean 0.10, scenario 0.08 -> degradation -0.02 (improved)
        self.assertAlmostEqual(calc(0.10, 0.08, "lower"), -0.02)

        with self.assertRaises(ValueError):
            calc(0.5, 0.5, "invalid_direction")

    def test_traditional_baseline_negative_control(self):
        """Traditional baseline must have 0.0 delta and 0.0 probability shift across all 35 scenarios."""
        trad_deltas = self.deltas_df[self.deltas_df["model_name"] == "Traditional baseline"]
        self.assertEqual(len(trad_deltas), 35 * 7)
        self.assertTrue((trad_deltas["delta_vs_clean"] == 0.0).all())
        self.assertTrue((trad_deltas["direction_adjusted_degradation"] == 0.0).all())
        self.assertTrue((trad_deltas["mean_abs_prob_delta"] == 0.0).all())
        self.assertTrue((trad_deltas["p95_abs_prob_delta"] == 0.0).all())

    def test_argument_validation_and_missing_files(self):
        """CLI arguments and input file paths must be rigorously validated."""
        val_args = stress_testing.validate_arguments
        with self.assertRaises(ValueError):
            val_args(seed=-1)
        with self.assertRaises(ValueError):
            val_args(seed=True)
        with self.assertRaises(ValueError):
            val_args(seed=2028, test_size=0.0)
        with self.assertRaises(ValueError):
            val_args(seed=2028, test_size=1.0)
        with self.assertRaises(ValueError):
            val_args(seed=2028, test_size=float("nan"))
        with self.assertRaises(ValueError):
            val_args(seed=2028, split_seed=-5)

        with self.assertRaises(FileNotFoundError):
            stress_testing.run_stress_testing_experiment(data_path=Path("non_existent_data.csv"))

        with self.assertRaises(FileNotFoundError):
            fake_paths = {"Bad Model": Path("non_existent_model.joblib")}
            stress_testing.run_stress_testing_experiment(model_paths=fake_paths)

    def test_synthetic_use_limitations_in_report(self):
        """The generated audit report must contain prominent educational and synthetic use disclaimers."""
        self.assertIn("SYNTHETIC AND EDUCATIONAL PURPOSE ONLY", self.report)
        self.assertIn("Label-Preserving Input Corruption", self.report)
        self.assertIn("Negative Control", self.report)

    def test_repeated_execution_reproducibility(self):
        """Repeated execution produces identical CSVs, report markdown, and metadata excluding generated_at_utc."""
        # 1. Normal executions without injected timestamp produce identical outputs excluding audit timestamp
        m_a, d_a, meta_a, rep_a = stress_testing.run_stress_testing_experiment(base_seed=2028)
        m_b, d_b, meta_b, rep_b = stress_testing.run_stress_testing_experiment(base_seed=2028)

        pd.testing.assert_frame_equal(m_a, m_b)
        pd.testing.assert_frame_equal(d_a, d_b)
        self.assertEqual(rep_a, rep_b)
        self.assertIn("generated_at_utc", meta_a)
        self.assertIn("generated_at_utc", meta_b)

        meta_a_no_ts = {k: v for k, v in meta_a.items() if k != "generated_at_utc"}
        meta_b_no_ts = {k: v for k, v in meta_b.items() if k != "generated_at_utc"}
        self.assertEqual(meta_a_no_ts, meta_b_no_ts)

        # 2. Injecting a fixed timestamp produces full metadata equality
        m_fixed, d_fixed, meta_fixed, rep_fixed = stress_testing.run_stress_testing_experiment(
            base_seed=2028,
            execution_timestamp="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(self.metadata, meta_fixed)
        self.assertEqual(self.report, rep_fixed)

    def test_total_dropout_interpretation_accuracy(self):
        """Total alternative dropout ROC-AUC must be lower than standalone traditional baseline and accurately reported."""
        clean_trad_auc = self.metadata["clean_baseline_metrics"]["Traditional baseline"]["roc_auc"]

        lr_rows = self.metrics_df[
            (self.metrics_df["model_name"] == "Hybrid (traditional + alternative)")
            & (self.metrics_df["scenario_id"] == "joint_missingness_rate_1.00")
            & (self.metrics_df["metric"] == "roc_auc")
        ]
        self.assertEqual(len(lr_rows), 1)
        dropout_lr_auc = lr_rows["metric_value"].iloc[0]

        hgb_rows = self.metrics_df[
            (self.metrics_df["model_name"] == "Hybrid (HistGradientBoosting)")
            & (self.metrics_df["scenario_id"] == "joint_missingness_rate_1.00")
            & (self.metrics_df["metric"] == "roc_auc")
        ]
        self.assertEqual(len(hgb_rows), 1)
        dropout_hgb_auc = hgb_rows["metric_value"].iloc[0]

        # Empirical reality: Under total alternative dropout, both hybrid models perform worse than dedicated baseline
        self.assertLess(dropout_lr_auc, clean_trad_auc)
        self.assertLess(dropout_hgb_auc, clean_trad_auc)

        # Audit report text must accurately convey that hybrid discrimination drops below traditional baseline
        self.assertIn("falls below the standalone traditional model's clean ROC-AUC", self.report)
        self.assertIn("not a model-selection fallback", self.report)
        # Inaccurate claims of baseline recovery or comparable discrimination must be absent
        self.assertNotIn("maintained discrimination comparable to the traditional baseline", self.report)
        self.assertNotIn("fallback to baseline-level performance", self.report)

    def test_cli_execution_smoke(self):
        """CLI main entrypoint must execute cleanly and return exit code 0."""
        ret = stress_testing.main(["--seed", "2028"])
        self.assertEqual(ret, 0)


if __name__ == "__main__":
    unittest.main()
