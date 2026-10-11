"""Five-fold evaluation must keep whole vessels and reserve the holdout."""

import unittest

import numpy as np

from src.compact_features import vessel_folds


class VesselFoldChecks(unittest.TestCase):
    def test_vessels_are_disjoint_and_each_training_row_is_evaluated_once(self) -> None:
        vessels = np.repeat(np.arange(44), 3)
        truth = np.repeat(np.arange(44) % 4, 3)
        train = np.flatnonzero(vessels < 40)
        folds = vessel_folds(vessels, truth, train)
        self.assertEqual(set(folds[train]), {1, 2, 3, 4, 5})
        self.assertTrue((folds[vessels >= 40] == 0).all())
        for fold in range(1, 6):
            evaluation = set(vessels[folds == fold])
            fitting = set(vessels[(folds > 0) & (folds != fold)])
            self.assertFalse(evaluation & fitting)
            self.assertEqual(len(evaluation), 8)
            self.assertEqual(set(truth[folds == fold]), {0, 1, 2, 3})
        np.testing.assert_array_equal(folds, vessel_folds(vessels, truth, train))

    def test_partial_vessels_and_changing_labels_are_rejected(self) -> None:
        vessels = np.repeat(np.arange(12), 2)
        truth = np.repeat(np.arange(12) % 2, 2)
        with self.assertRaises(ValueError):
            vessel_folds(vessels, truth, np.arange(21))
        truth[0] = 1
        with self.assertRaises(ValueError):
            vessel_folds(vessels, truth, np.arange(20))


if __name__ == "__main__":
    unittest.main()
