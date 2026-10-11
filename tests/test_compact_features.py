"""Targeted selection checks and one short named-feature model roundtrip."""

import importlib.util
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.compact_features import choose_features, describe_feature, rank_groups, redundant_groups


class CompactFeatureChecks(unittest.TestCase):
    def test_redundancy_and_missingness_are_both_considered(self) -> None:
        values = np.arange(1000, dtype="float32")
        matrix = np.column_stack([values, values * 2, values * 3, np.sin(values)])
        matrix[:200, 2] = np.nan
        groups = redundant_groups(matrix, np.array([4, 3, 2, 1]))
        self.assertIn([0, 1], groups)
        self.assertIn([2], groups)
        self.assertIn([3], groups)

    def test_permutation_keeps_group_members_together(self) -> None:
        values = np.linspace(-1, 1, 400).astype("float32")
        matrix = np.column_stack([values, values * 2, np.ones(400)])

        class Teacher:
            feature_importances_ = np.array([4, 3, 1])

            @property
            def booster_(self):
                return self

            def predict(self, features, num_threads):
                np.testing.assert_allclose(features[:, 1], 2 * features[:, 0])
                return np.eye(2)[(features[:, 0] > 0).astype(int)]

        result = rank_groups(Teacher(), matrix, (values > 0).astype(int), ["x", "double_x", "constant"], [[0, 1], [2]])
        self.assertEqual(result.feature.iloc[0], "x")
        self.assertGreater(result.permutation_drop.iloc[0], .1)
        self.assertEqual(result.permutation_drop.iloc[1], 0)

    def test_budget_uses_only_positive_groups(self) -> None:
        ranking = pd.DataFrame({"feature": list("abcdef"), "permutation_drop": [.6, .5, .4, .3, 0, -.1]})
        self.assertEqual(choose_features(ranking, 12), list("abcd"))
        with self.assertRaises(ValueError):
            choose_features(ranking.iloc[:3], 12)
        self.assertIn("위도", describe_feature("behavior_past_std_active_lat"))

    def test_compact_checkpoint_preserves_names_and_predictions(self) -> None:
        path = Path(__file__).resolve().parents[1] / "14_compact_model.py"
        spec = importlib.util.spec_from_file_location("compact", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(1600, 4)).astype("float32")
        truth = np.digitize(matrix[:, 0], [-.6, 0, .6]).astype("int32")
        names = ["a", "b", "c", "d"]
        model = runner.fit_model(matrix, truth, np.arange(1200), 1, feature_names=names)
        restored = lgb.Booster(model_str=model.booster_.model_to_string())
        self.assertEqual(restored.feature_name(), names)
        rows = np.arange(1200, 1600)
        np.testing.assert_allclose(runner.probabilities(model, matrix, rows, 1), runner.probabilities(restored, matrix, rows, 1))


if __name__ == "__main__":
    unittest.main()
