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
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train  # noqa: E402  (must come after sys.path manipulation)


# ---------------------------------------------------------------------------
# Shared test fixtures (small, deterministic)
# ---------------------------------------------------------------------------

def _small_dataset(n: int = 300, seed: int = 7) -> "pd.DataFrame":
    import pandas as pd  # imported locally to keep the import footprint minimal
    return train.generate_synthetic_dataset(n=n, seed=seed)


def _train_test_split(df: "pd.DataFrame"):
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
        # With n=300 and rates 5-12%, each column should have at least one NaN.
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

    def _assert_estimator_type(self, pipe, expected_cls):
        from sklearn.pipeline import Pipeline
        self.assertIsInstance(pipe, Pipeline)
        self.assertIn("preprocess", pipe.named_steps)
        self.assertIn("classifier", pipe.named_steps)
        self.assertIsInstance(pipe.named_steps["classifier"], expected_cls)

    def test_build_traditional_lr_pipeline_structure(self):
        from sklearn.linear_model import LogisticRegression
        pipe = train.build_traditional_lr_pipeline()
        self._assert_estimator_type(pipe, LogisticRegression)

    def test_build_hybrid_lr_pipeline_structure(self):
        from sklearn.linear_model import LogisticRegression
        pipe = train.build_hybrid_lr_pipeline()
        self._assert_estimator_type(pipe, LogisticRegression)

    def test_build_hybrid_hgb_pipeline_structure(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        pipe = train.build_hybrid_hgb_pipeline()
        self._assert_estimator_type(pipe, HistGradientBoostingClassifier)

    def test_pipeline_builders_return_new_instances(self):
        """Each call to a builder must return a fresh, unfitted instance."""
        p1 = train.build_traditional_lr_pipeline()
        p2 = train.build_traditional_lr_pipeline()
        self.assertIsNot(p1, p2)


# ---------------------------------------------------------------------------
# 3. Pipeline fit and prediction
# ---------------------------------------------------------------------------

class TestPipelineFitPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        df = _small_dataset(n=400, seed=5)
        cls.train_df, cls.test_df = _train_test_split(df)
        cls.y_train = cls.train_df[train.TARGET]
        cls.y_test = cls.test_df[train.TARGET]

    def _fit_and_predict(self, pipe, feature_cols):
        pipe.fit(self.train_df[feature_cols], self.y_train)
        probs = pipe.predict_proba(self.test_df[feature_cols])[:, 1]
        return probs

    def test_traditional_lr_fit_predict_shape(self):
        probs = self._fit_and_predict(
            train.build_traditional_lr_pipeline(), train.TRADITIONAL_FEATURES
        )
        self.assertEqual(len(probs), len(self.test_df))

    def test_traditional_lr_probabilities_valid(self):
        probs = self._fit_and_predict(
            train.build_traditional_lr_pipeline(), train.TRADITIONAL_FEATURES
        )
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_hybrid_lr_fit_predict_shape(self):
        probs = self._fit_and_predict(
            train.build_hybrid_lr_pipeline(), train.ALL_FEATURES
        )
        self.assertEqual(len(probs), len(self.test_df))

    def test_hybrid_lr_probabilities_valid(self):
        probs = self._fit_and_predict(
            train.build_hybrid_lr_pipeline(), train.ALL_FEATURES
        )
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

    def test_hybrid_hgb_fit_predict_shape(self):
        probs = self._fit_and_predict(
            train.build_hybrid_hgb_pipeline(), train.ALL_FEATURES
        )
        self.assertEqual(len(probs), len(self.test_df))

    def test_hybrid_hgb_probabilities_valid(self):
        probs = self._fit_and_predict(
            train.build_hybrid_hgb_pipeline(), train.ALL_FEATURES
        )
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))


# ---------------------------------------------------------------------------
# 4. Redundant-indicator correction (Objective B)
# ---------------------------------------------------------------------------

class TestRedundantIndicatorCorrection(unittest.TestCase):
    """Verify that fitted pipelines do NOT expose missingindicator_bureau_score."""

    @classmethod
    def setUpClass(cls):
        df = _small_dataset(n=400, seed=3)
        train_df, _ = _train_test_split(df)
        y_train = train_df[train.TARGET]

        cls.trad_pipe = train.build_traditional_lr_pipeline()
        cls.trad_pipe.fit(train_df[train.TRADITIONAL_FEATURES], y_train)

        cls.hybrid_lr_pipe = train.build_hybrid_lr_pipeline()
        cls.hybrid_lr_pipe.fit(train_df[train.ALL_FEATURES], y_train)

        cls.hybrid_hgb_pipe = train.build_hybrid_hgb_pipeline()
        cls.hybrid_hgb_pipe.fit(train_df[train.ALL_FEATURES], y_train)

    def _feature_names(self, pipe):
        return list(pipe.named_steps["preprocess"].get_feature_names_out())

    def test_traditional_pipeline_no_bureau_score_indicator(self):
        names = self._feature_names(self.trad_pipe)
        redundant = [n for n in names if "missingindicator_bureau_score" in n]
        self.assertEqual(
            redundant, [],
            msg=f"Redundant bureau-score indicator found: {redundant}",
        )

    def test_hybrid_lr_pipeline_no_bureau_score_indicator(self):
        names = self._feature_names(self.hybrid_lr_pipe)
        redundant = [n for n in names if "missingindicator_bureau_score" in n]
        self.assertEqual(
            redundant, [],
            msg=f"Redundant bureau-score indicator found: {redundant}",
        )

    def test_hybrid_hgb_pipeline_no_bureau_score_indicator(self):
        names = self._feature_names(self.hybrid_hgb_pipe)
        redundant = [n for n in names if "missingindicator_bureau_score" in n]
        self.assertEqual(
            redundant, [],
            msg=f"Redundant bureau-score indicator found: {redundant}",
        )

    def test_hybrid_lr_pipeline_retains_alt_indicators(self):
        """The four genuine alt-feature missingness indicators must still be present."""
        names = self._feature_names(self.hybrid_lr_pipe)
        for col in train._ALT_FEATURES_WITH_MISSINGNESS:
            expected = f"alt_ind__missingindicator_{col}"
            self.assertIn(expected, names, msg=f"Missing expected indicator: {expected}")

    def test_hybrid_hgb_pipeline_retains_alt_indicators(self):
        names = self._feature_names(self.hybrid_hgb_pipe)
        for col in train._ALT_FEATURES_WITH_MISSINGNESS:
            expected = f"alt_ind__missingindicator_{col}"
            self.assertIn(expected, names, msg=f"Missing expected indicator: {expected}")

    def test_traditional_pipeline_retains_bureau_history_feature(self):
        """bureau_history_available must still appear as a predictor."""
        names = self._feature_names(self.trad_pipe)
        self.assertTrue(
            any("bureau_history_available" in n for n in names),
            msg="bureau_history_available missing from traditional pipeline",
        )


# ---------------------------------------------------------------------------
# 5. Missing alternative data (inference robustness)
# ---------------------------------------------------------------------------

class TestMissingAlternativeData(unittest.TestCase):
    """Hybrid models must handle entirely missing alternative features at inference."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd
        df = _small_dataset(n=400, seed=11)
        train_df, cls.test_df = _train_test_split(df)
        y_train = train_df[train.TARGET]

        cls.hybrid_lr_pipe = train.build_hybrid_lr_pipeline()
        cls.hybrid_lr_pipe.fit(train_df[train.ALL_FEATURES], y_train)

        cls.hybrid_hgb_pipe = train.build_hybrid_hgb_pipeline()
        cls.hybrid_hgb_pipe.fit(train_df[train.ALL_FEATURES], y_train)

        # Build a single-row test case where ALL alternative features are NaN.
        base_row = train_df[train.ALL_FEATURES].iloc[[0]].copy()
        for col in train.ALTERNATIVE_FEATURES:
            base_row[col] = float("nan")
        cls.all_alt_missing_row = base_row

        # Build a row with bureau_history_available==0 and bureau_score NaN.
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
    """All three saved .joblib models must load and predict without error."""

    ARTIFACT_SPECS = [
        ("traditional_baseline.joblib",       train.TRADITIONAL_FEATURES),
        ("hybrid_logistic_regression.joblib",  train.ALL_FEATURES),
        ("hybrid_hist_gradient_boosting.joblib", train.ALL_FEATURES),
    ]

    @classmethod
    def setUpClass(cls):
        import joblib
        cls.models = {}
        cls.load_errors = {}
        for filename, _ in cls.ARTIFACT_SPECS:
            path = ROOT / "models" / filename
            if path.exists():
                try:
                    cls.models[filename] = joblib.load(path)
                except Exception as exc:
                    cls.load_errors[filename] = exc
            else:
                cls.load_errors[filename] = FileNotFoundError(f"{path} not found")

        # One fixed synthetic row for inference smoke tests.
        cls.sample_row = train.generate_synthetic_dataset(n=1, seed=99)[train.ALL_FEATURES]

    def _check_model_available(self, filename):
        if filename in self.load_errors:
            self.fail(f"Could not load {filename}: {self.load_errors[filename]}")

    def _run_inference(self, filename, feature_cols):
        self._check_model_available(filename)
        model = self.models[filename]
        row = self.sample_row[feature_cols]
        prob = model.predict_proba(row)[0, 1]
        self.assertTrue(np.isfinite(prob), f"{filename}: non-finite probability")
        self.assertTrue(0.0 <= prob <= 1.0, f"{filename}: probability out of [0,1]")

    def test_traditional_baseline_loads(self):
        self._check_model_available("traditional_baseline.joblib")

    def test_traditional_baseline_inference(self):
        self._run_inference("traditional_baseline.joblib", train.TRADITIONAL_FEATURES)

    def test_hybrid_lr_loads(self):
        self._check_model_available("hybrid_logistic_regression.joblib")

    def test_hybrid_lr_inference(self):
        self._run_inference("hybrid_logistic_regression.joblib", train.ALL_FEATURES)

    def test_hybrid_hgb_loads(self):
        self._check_model_available("hybrid_hist_gradient_boosting.joblib")

    def test_hybrid_hgb_inference(self):
        self._run_inference("hybrid_hist_gradient_boosting.joblib", train.ALL_FEATURES)

    def test_hybrid_lr_probability_in_range(self):
        """Explicit range assertion for the primary inference model."""
        self._check_model_available("hybrid_logistic_regression.joblib")
        model = self.models["hybrid_logistic_regression.joblib"]
        row = self.sample_row[train.ALL_FEATURES]
        prob = float(model.predict_proba(row)[0, 1])
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertTrue(np.isfinite(prob))


# ---------------------------------------------------------------------------
# 7. Backward-compatibility check for serialized models
# ---------------------------------------------------------------------------

class TestSerializedArtifactCompatibility(unittest.TestCase):
    """Report incompatibilities between old serialized models and corrected pipelines.

    Old models were trained with the redundant bureau-score indicator.  After
    retraining the saved artifacts, this test checks that the saved models use the
    corrected preprocessor.  If the models on disk still carry the old schema, the
    test will FAIL with a descriptive message rather than silently passing.
    """

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


if __name__ == "__main__":
    unittest.main()
