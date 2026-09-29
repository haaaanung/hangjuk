"""거리·선분 경계·정지·중복을 손으로 예측할 수 있는 최소 회귀 검산."""
import unittest
import numpy as np
from src.features import extract_features


class FeatureChecks(unittest.TestCase):
    def test_disconnected_straight_segments(self):
        row = extract_features(['MULTILINESTRING ((0 0, 1 0), (100 0, 101 0))']).iloc[0]
        self.assertAlmostEqual(row.total_distance_m, 222390.160, delta=1)
        self.assertAlmostEqual(row.straightness, 1)
        self.assertEqual(row.num_segments, 2)
        self.assertEqual(row.num_direction_changes, 0)

    def test_stationary_and_missing(self):
        df = extract_features(['MULTILINESTRING ((127 35, 127 35, 127 35))', None, 'broken'])
        self.assertEqual(df.iloc[0].total_distance_m, 0)
        self.assertTrue(np.isnan(df.iloc[0].straightness))
        self.assertAlmostEqual(df.iloc[0].duplicate_ratio, 2 / 3)
        self.assertEqual(df.geometry_status.tolist(), ['ok', 'missing_geometry', 'invalid_geometry'])

    def test_right_angle_with_duplicate(self):
        row = extract_features(['LINESTRING (0 0, 1 0, 1 0, 1 1)']).iloc[0]
        self.assertAlmostEqual(row.mean_direction_change_deg, 90)
        self.assertEqual(row.turn_count_45, 1)
        self.assertEqual(row.num_direction_changes, 1)
        self.assertEqual(row.duplicate_ratio, .25)


if __name__ == '__main__':
    unittest.main()
