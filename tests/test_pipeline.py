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
from unittest.mock import MagicMock

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


if __name__ == "__main__":
    unittest.main()
