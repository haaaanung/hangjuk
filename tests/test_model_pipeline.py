import importlib.util
from pathlib import Path
import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS

spec = importlib.util.spec_from_file_location('improve', Path(__file__).resolve().parents[1] / '04_improve.py')
improve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(improve)


class ModelPipelineChecks(unittest.TestCase):
    def test_sampling_keeps_training_membership_and_caps_only_training(self):
        vessels = np.repeat([10, 20, 30], [6, 4, 2])
        train = np.arange(10)
        sampled = improve.sample_training(train, vessels, cap=3)
        self.assertTrue(set(sampled) <= set(train))
        self.assertEqual(pd.Series(vessels[sampled]).value_counts().to_dict(), {10: 3, 20: 3})
        np.testing.assert_array_equal(sampled, improve.sample_training(train, vessels, cap=3))
        np.testing.assert_array_equal(train, improve.sample_training(train, vessels, cap=0))

    def test_equal_vessel_metric_is_not_row_accuracy(self):
        truth = np.array([0, 0, 0, 1])
        probabilities = np.array([[1., 0.]] * 4)
        result = improve.scores(truth, probabilities, np.array([10, 10, 10, 20]))
        self.assertEqual(result['accuracy'], .75)
        self.assertEqual(result['vessel_equal_accuracy'], .5)

    def test_saved_model_preserves_probabilities_and_feature_order(self):
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(240, len(FEATURE_COLUMNS))).astype('float32')
        truth = np.digitize(matrix[:, 0], [-.6, 0., .6]).astype('int32')
        config = {'features': 'baseline', 'weight_power': .5, 'rounds': 3}
        train, valid = np.arange(200), np.arange(200, 240)
        model, _ = improve.fit(config, matrix, FEATURE_COLUMNS, train, truth, threads=1)
        expected = improve.predict(model, config, matrix, FEATURE_COLUMNS, valid)
        restored = lgb.Booster(model_str=model.booster_.model_to_string())
        actual = restored.predict(pd.DataFrame(matrix[valid], columns=FEATURE_COLUMNS), num_threads=1)
        np.testing.assert_allclose(expected, actual, atol=1e-12)
        self.assertEqual(restored.feature_name(), FEATURE_COLUMNS)


if __name__ == '__main__':
    unittest.main()
