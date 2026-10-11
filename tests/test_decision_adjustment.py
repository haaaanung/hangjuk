"""Small decision-layer checks; no base-model fitting."""

import unittest

import numpy as np

from src.decision_adjustment import apply_class_weights, fit_class_weights
from src.improved_training import row_scores


class DecisionChecks(unittest.TestCase):
    def test_search_is_deterministic_bounded_and_preserves_both_fit_metrics(self) -> None:
        probability = np.array([[.54, .46], [.58, .42], [.7, .3], [.8, .2]])
        truth = np.array([1, 1, 0, 0])
        weights = fit_class_weights(truth, probability)
        np.testing.assert_array_equal(weights, fit_class_weights(truth, probability))
        self.assertTrue(((weights >= .7) & (weights <= 1.4)).all())
        before = row_scores(truth, probability)
        adjusted = apply_class_weights(probability, weights)
        after = row_scores(truth, adjusted)
        self.assertTrue(all(after[key] >= before[key] for key in before))
        self.assertGreater(after["macro_f1"], before["macro_f1"])
        np.testing.assert_allclose(adjusted.sum(axis=1), 1)

    def test_invalid_or_misaligned_weights_rejected(self) -> None:
        for weights in [[1], [0, 1], [-1, 1], [np.nan, 1]]:
            with self.assertRaises(ValueError):
                apply_class_weights(np.array([[.5, .5]]), np.asarray(weights))


if __name__ == "__main__":
    unittest.main()
