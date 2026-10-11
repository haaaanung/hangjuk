"""Correctness checks only: no model fitting and no production data."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.improved_training import row_scores, save_json, select_recipe, stopping_split, training_weights
from src.prediction_history import blend_history, prior_prediction_history
from src.training_features import extra_features, feature_batch, feature_names


class PredictionHistoryChecks(unittest.TestCase):
    def setUp(self):
        self.keys = pd.DataFrame({
            "rfid_id": [1, 1, 1, 1, 2, 2],
            "part_dt": [20260101, 20260101, 20260103, 20260104, 20260102, 20260104],
        })
        self.probability = np.array([
            [.8, .2], [.6, .4], [.2, .8], [.4, .6], [.1, .9], [.3, .7]
        ])

    def test_prior_days_have_equal_votes_and_vessels_are_separate(self):
        prior, days = prior_prediction_history(self.keys, self.probability)
        np.testing.assert_array_equal(days, [0, 0, 1, 2, 0, 1])
        np.testing.assert_allclose(prior[[2, 3, 5]], [[.7, .3], [.45, .55], [.1, .9]])

    def test_future_and_current_day_peers_cannot_change_earlier_predictions(self):
        prior, days = prior_prediction_history(self.keys, self.probability)
        expected = blend_history(self.probability, prior, days, .8)
        changed = self.probability.copy()
        changed[[1, 2, 3]] = [1, 0]
        prior2, days2 = prior_prediction_history(self.keys, changed)
        actual = blend_history(changed, prior2, days2, .8)
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[4:], expected[4:])

    def test_same_day_peers_excluded_even_when_history_exists(self):
        keys = pd.concat([self.keys, self.keys.iloc[[2]]], ignore_index=True)
        probability = np.vstack([self.probability, [.99, .01]])
        prior, days = prior_prediction_history(keys, probability)
        np.testing.assert_allclose(prior[2], [.7, .3])
        self.assertEqual(days[2], 1)

    def test_first_day_and_zero_strength_preserve_current_prediction(self):
        prior, days = prior_prediction_history(self.keys, self.probability)
        mixed = blend_history(self.probability, prior, days, .8)
        np.testing.assert_allclose(mixed[days == 0], self.probability[days == 0])
        np.testing.assert_allclose(mixed.sum(axis=1), 1)
        np.testing.assert_allclose(blend_history(self.probability, prior, days, 0), self.probability)

    def test_input_order_and_labels_do_not_affect_history(self):
        expected, expected_days = prior_prediction_history(self.keys, self.probability)
        order = np.array([4, 2, 0, 5, 3, 1])
        keys = self.keys.iloc[order].assign(fishing_code="unused label")
        actual, days = prior_prediction_history(keys, self.probability[order])
        np.testing.assert_allclose(actual, expected[order])
        np.testing.assert_array_equal(days, expected_days[order])

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            prior_prediction_history(self.keys, self.probability * 2)


class SelectionChecks(unittest.TestCase):
    def test_accuracy_or_f1_regression_is_ineligible(self):
        table = pd.DataFrame([
            ("control", 0, .6, .58), ("f1_only", .8, .59, .7),
            ("accuracy_only", 0, .7, .57), ("both", .5, .62, .60),
        ], columns=["members", "history_strength", "accuracy_mean", "macro_f1_mean"])
        summary, selected = select_recipe(table)
        self.assertEqual(selected["members"], "both")
        self.assertFalse(summary.set_index("members").loc["f1_only", "eligible"])
        self.assertFalse(summary.set_index("members").loc["accuracy_only", "eligible"])

    def test_ties_prefer_fewer_models_and_no_history(self):
        table = pd.DataFrame([
            ("control+profile", 0, .6, .58), ("control", .8, .6, .58), ("control", 0, .6, .58),
        ], columns=["members", "history_strength", "accuracy_mean", "macro_f1_mean"])
        _, selected = select_recipe(table)
        self.assertEqual(selected["members"], "control")
        self.assertEqual(selected["history_strength"], 0)

    def test_scores_include_absent_classes(self):
        result = row_scores(np.array([0, 0]), np.array([[1, 0], [1, 0]]))
        self.assertEqual(result, {"accuracy": 1.0, "macro_f1": .5})

    def test_weights_balance_effective_class_mass(self):
        truth = np.array([0, 0, 0, 0, 1, 1])
        vessels = np.array([10, 10, 10, 11, 20, 21])
        weights = training_weights(truth, vessels, 1, 1, 2)
        self.assertAlmostEqual(weights[truth == 0].sum(), weights[truth == 1].sum())
        self.assertAlmostEqual(weights[vessels == 10].sum(), weights[vessels == 11].sum())
        self.assertAlmostEqual(weights.mean(), 1)

    def test_stopping_split_excludes_outer_validation_vessels(self):
        vessels = np.repeat(np.arange(24), 3)
        truth = np.repeat(np.arange(24) % 2, 3)
        rows = np.flatnonzero(vessels < 20)
        fit, stop = stopping_split(rows, vessels, truth, .2, 42)
        self.assertFalse(set(vessels[fit]) & set(vessels[stop]))
        self.assertEqual(set(fit) | set(stop), set(rows))
        self.assertTrue((vessels[stop] < 20).all())


class FeatureChecks(unittest.TestCase):
    def test_relative_features_and_feature_order_preserve_missing_values(self):
        columns = []
        for name in ["total_distance_m", "spatial_spread_m", "num_points", "straightness", "turn_rate_45", "duplicate_ratio"]:
            columns.extend([name, f"past_mean_{name}"])
        for name in ["log_total_distance_m", "log_spatial_spread_m", "active_fraction"]:
            columns.extend([f"recent7_{name}", f"recent30_{name}"])
        columns += ["center_lon", "center_lat"]
        columns += [f"behavior_past_mean_{state}_{axis}" for state in ["quiet", "active"] for axis in ["lon", "lat"]]
        matrix = np.ones((2, len(columns)), dtype="float32")
        matrix[1, columns.index("past_mean_total_distance_m")] = np.nan
        result = extra_features(matrix, columns, np.arange(2))
        self.assertEqual(result["relative_log_total_distance_m"][0], 0)
        self.assertTrue(np.isnan(result["relative_log_total_distance_m"][1]))
        self.assertEqual(result["distance_from_prior_quiet_m"][0], 0)
        batch = feature_batch(matrix, columns, np.array([1, 0]), "enriched")
        self.assertEqual(batch.columns.tolist(), feature_names(columns, "enriched"))
        self.assertTrue(np.isnan(batch.relative_log_total_distance_m.iloc[0]))
        self.assertNotIn("total_distance_m", feature_names(columns, "profile"))


class CachedPipelineChecks(unittest.TestCase):
    def test_cv_resume_final_reports_and_completed_resume_without_fitting(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("new_runner", root / "10_train_improved.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        behavior = runner.load_behavior_pipeline()
        config = runner.read_json(root / "configs/improved_training.json")
        config["candidates"] = {"control": config["candidates"]["control"]}
        config["history_strengths"] = [0, .8]
        df = pd.DataFrame({
            "rfid_id": np.repeat(np.arange(8), 2),
            "part_dt": [20260101, 20260102] * 8,
            "rfid_time": 0, "source_file_id": 1, "source_row": np.arange(16),
            "fishing_code": np.repeat(["A", "B"] * 4, 2),
        })
        truth = np.repeat([0, 1] * 4, 2)
        row_folds = np.repeat([1, 1, 2, 2, 3, 3, 0, 0], 2)
        probability = np.eye(2, dtype="float32")[truth] * .8 + .1
        matrix = np.zeros((len(df), 1), dtype="float32")
        # All model fitting is forbidden in this integration check.
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            out, cache = Path(temporary) / "out", Path(temporary) / "cache"
            out.mkdir()
            cache.mkdir()
            for fold in [1, 2, 3]:
                np.save(cache / f"control_fold{fold}.npy", probability[row_folds == fold])
                save_json(cache / f"control_fold{fold}.json", {
                    "candidate": "control", "fold": fold, "rounds": 350,
                    "accuracy": 1.0, "macro_f1": 1.0,
                })
            with patch.object(runner, "fit_fold", side_effect=AssertionError("No training")):
                cv = runner.run_cv(behavior, df, matrix, ["x"], truth, row_folds, config, out, cache)
                self.assertEqual(len(cv), 3)
            selection = runner.choose_recipe(df, truth, row_folds, config, out, cache)
            self.assertEqual(selection["history_strength"], 0)
            self.assertEqual(selection["members"], ["control"])
            # Changing holdout truth must not alter CV recipe selection.
            changed_truth = truth.copy()
            changed_truth[row_folds == 0] = 1 - changed_truth[row_folds == 0]
            self.assertEqual(selection, runner.choose_recipe(df, changed_truth, row_folds, config, out, cache))
            (out / "model_control.txt").write_text("mock saved model", encoding="utf-8")
            save_json(out / "model_control.json", {"candidate": "control", "rounds": 350, "train_rows": 12})
            with (
                patch.object(runner, "fit_model", side_effect=AssertionError("No training")),
                patch.object(runner.lgb, "Booster", return_value=object()),
                patch.object(runner, "predict", return_value=probability[row_folds == 0]) as prediction,
            ):
                args = (behavior, df, matrix, ["x"], truth, ["A", "B"], row_folds, config, out)
                runner.run_final(*args)
                self.assertEqual(runner.read_json(out / "metrics.json")["selected"]["accuracy"], 1)
                self.assertTrue((out / "RESULTS.md").exists())
                self.assertTrue((out / "model_metadata.json").exists())
                runner.run_final(*args)
                self.assertEqual(prediction.call_count, 1)


if __name__ == "__main__":
    unittest.main()
