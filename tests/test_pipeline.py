from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train

class SyntheticPipelineTests(unittest.TestCase):
    def test_generator_reproducibility(self):
        a = train.generate_synthetic_dataset(n=100, seed=7)
        b = train.generate_synthetic_dataset(n=100, seed=7)
        self.assertTrue(a.equals(b))

    def test_schema_and_binary_target(self):
        df = train.generate_synthetic_dataset(n=200, seed=4)
        self.assertTrue(set(train.ALL_FEATURES).issubset(df.columns))
        self.assertEqual(set(df[train.TARGET].unique()), {0, 1})
        self.assertTrue(df["bureau_score"].isna().any())
        self.assertTrue(df["utility_payment_timeliness"].isna().any())

    def test_saved_hybrid_model_predicts_probability(self):
        import joblib
        model_path = ROOT / "models" / "hybrid_logistic_regression.joblib"
        self.assertTrue(model_path.exists(), "Run `python train.py` before tests")
        model = joblib.load(model_path)
        row = train.generate_synthetic_dataset(n=1, seed=99)[train.ALL_FEATURES]
        prob = model.predict_proba(row)[0, 1]
        self.assertTrue(0.0 <= prob <= 1.0)
        self.assertTrue(np.isfinite(prob))

if __name__ == "__main__":
    unittest.main()
