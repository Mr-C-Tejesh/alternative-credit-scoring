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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train  # noqa: E402
import validation  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
