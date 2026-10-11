import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.spatial_embedding import ANCHORS, SpatialEmbedding, append_embedding


class SpatialEmbeddingTest(unittest.TestCase):
    def test_roundtrip_missing_and_fixed_transform(self):
        coordinates = np.random.default_rng(4).normal([128., 35.], [.5, .5], size=(500, 2))
        model = SpatialEmbedding().fit(coordinates)
        centers = model.centers.copy()
        query = np.array([[128., 35.], [np.nan, 35.], [181., 35.]])
        result = model.transform(query)
        self.assertEqual(result.shape, (3, 9))
        self.assertEqual(len(model.feature_names), 18)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[1:, :-1], 0)
        np.testing.assert_array_equal(result[:, -1], [0, 1, 1])
        np.testing.assert_array_equal(model.centers, centers)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.npz"
            model.save(path)
            np.testing.assert_array_equal(SpatialEmbedding.load(path).transform(query), result)

    def test_sphere_wraps_longitude(self):
        xyz, valid = SpatialEmbedding.sphere(np.array([[180., 10.], [-180., 10.]]))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(xyz[0], xyz[1], atol=1e-12)

    def test_append_fits_training_coordinates_only(self):
        rng = np.random.default_rng(2)
        matrix = rng.normal([128., 35., 128., 35.], .3, size=(500, 4)).astype("float32")
        columns = [name for _, lon, lat in ANCHORS for name in [lon, lat]]
        training = np.arange(400)
        compact = matrix[:, :2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial.npz"
            result, names = append_embedding(matrix, columns, compact, training, path)
            self.assertEqual(result.shape, (500, 20))
            self.assertEqual(len(names), 18)
            changed = matrix.copy()
            changed[400:] = [0., 0., 0., 0.]
            second = Path(directory) / "second.npz"
            repeated, _ = append_embedding(changed, columns, compact, training, second)
            np.testing.assert_array_equal(result[:400], repeated[:400])
            np.testing.assert_array_equal(SpatialEmbedding.load(path).centers,
                                          SpatialEmbedding.load(second).centers)


if __name__ == "__main__":
    unittest.main()
