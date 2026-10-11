"""Short model-fit and serialization smoke checks on synthetic data."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier

from src.alternative_models import create_model, fit_model
from src.common import ROOT


class AlternativeModelChecks(unittest.TestCase):
    def test_missing_values_weighted_fit_and_checkpoint_roundtrip(self) -> None:
        rng = np.random.default_rng(12)
        features = rng.normal(size=(160, 8)).astype("float32")
        features[::7, 2] = np.nan
        labels = np.arange(len(features), dtype="int32") % 4
        for name in ["catboost", "extra_trees"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=ROOT / "outputs/cache") as directory:
                model = create_model(name, threads=1, smoke=True)
                fit_model(name, model, features[:120], labels[:120])
                expected = model.predict_proba(features[120:])
                self.assertEqual(expected.shape, (40, 4))
                self.assertTrue(np.isfinite(expected).all())
                np.testing.assert_allclose(expected.sum(axis=1), 1)
                path = Path(directory) / "model"
                if name == "catboost":
                    model.save_model(str(path))
                    restored = CatBoostClassifier().load_model(str(path))
                else:
                    joblib.dump(model, path)
                    restored = joblib.load(path)
                np.testing.assert_allclose(restored.predict_proba(features[120:]), expected)


if __name__ == "__main__":
    unittest.main()
