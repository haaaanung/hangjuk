import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.spatial_prior import SpatialPrior, cross_fit_prior


class SpatialPriorChecks(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({'lon': [127., 127., 127., 128.], 'lat': [35., 35., 35., 36.]})
        self.vessels = np.array([1, 1, 2, 3])
        self.y = np.array([0, 0, 1, 2])
        self.anchors = [('center', 'lon', 'lat')]

    def test_own_label_cannot_change_own_training_features(self):
        a = SpatialPrior(self.anchors).fit(self.x, self.vessels, self.y)
        changed = self.y.copy()
        changed[self.vessels == 1] = 3
        b = SpatialPrior(self.anchors).fit(self.x, self.vessels, changed)
        own = self.vessels == 1
        xa = a.transform(self.x, self.vessels, True)
        xb = b.transform(self.x, self.vessels, True)
        assert_frame_equal(xa.loc[own], xb.loc[own])

    def test_repeated_observations_do_not_increase_votes(self):
        a = SpatialPrior(self.anchors).fit(self.x, self.vessels, self.y)
        b = SpatialPrior(self.anchors).fit(self.x.iloc[[0, 2, 3]], self.vessels[[0, 2, 3]], self.y[[0, 2, 3]])
        assert_frame_equal(a.transform(self.x), b.transform(self.x))
        result = a.transform(self.x, self.vessels, True)
        self.assertAlmostEqual(result.iloc[0].filter(like='support').iloc[0], np.log(2), places=6)

    def test_unknown_and_missing_locations_and_saved_state(self):
        model = SpatialPrior(self.anchors).fit(self.x, self.vessels, self.y)
        query = pd.DataFrame({'lon': [130., np.nan], 'lat': [34., np.nan]})
        result = model.transform(query)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        np.testing.assert_allclose(result.iloc[0, :4], [1/3, 1/3, 1/3, 0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'priors.npz'
            model.save(path)
            loaded = SpatialPrior.load(path)
            assert_frame_equal(result, loaded.transform(query))
            with self.assertRaises(ValueError):
                loaded.transform(self.x, self.vessels, True)

    def test_single_training_vessel_leaves_uniform_prior(self):
        model = SpatialPrior(self.anchors).fit(self.x.iloc[:2], self.vessels[:2], self.y[:2])
        result = model.transform(self.x.iloc[:2], self.vessels[:2], True)
        np.testing.assert_allclose(result.iloc[:, :4], .25)

    def test_crossfit_own_labels_and_duplicates_do_not_leak(self):
        original, _ = cross_fit_prior(self.x, self.vessels, self.y, self.anchors)
        changed_y = self.y.copy()
        changed_y[self.vessels == 1] = 3
        changed, _ = cross_fit_prior(self.x, self.vessels, changed_y, self.anchors)
        assert_frame_equal(original.loc[self.vessels == 1], changed.loc[self.vessels == 1])
        np.testing.assert_array_equal(original.iloc[0], original.iloc[1])


if __name__ == '__main__':
    unittest.main()
