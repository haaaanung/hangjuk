import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.behavior_features import robust_geometry, prior_window_means, behavior_context
from src.context_features import coordinates
from src.features import extract_features


class BehaviorChecks(unittest.TestCase):
    def test_jitter_is_not_large_scale_turning(self):
        jitter = 'LINESTRING (127 35,127.00001 35.00001,127 35.00001,127.00001 35,127 35)'
        result = robust_geometry([jitter]).iloc[0]
        self.assertAlmostEqual(result.smooth_distance_m, 0)
        self.assertEqual(result.smooth_direction_changes, 0)
        self.assertTrue(np.isnan(result.smooth_mean_turn_deg))
        self.assertAlmostEqual(result.jitter_distance_fraction, 1)

    def test_real_turn_and_disconnected_parts(self):
        wkts = ['LINESTRING (127 35,127.01 35,127.01 35.01)',
                'MULTILINESTRING ((127 35,127.01 35),(129 37,129.01 37))']
        result = robust_geometry(wkts)
        self.assertAlmostEqual(result.loc[0, 'smooth_mean_turn_deg'], 90, places=4)
        self.assertEqual(result.loc[1, 'smooth_direction_changes'], 0)
        self.assertLess(result.loc[1, 'smooth_distance_m'], 2000)
        self.assertAlmostEqual(result.loc[1, 'smooth_straightness'], 1, places=6)

    def test_actual_calendar_window_and_no_current_day(self):
        df = pd.DataFrame({'rfid_id': [1, 2, 1, 1, 1],
                           'part_dt': [20260101, 20260101, 20260108, 20260108, 20260110],
                           'x': [2., 999., 4., 8., 10.]})
        got = prior_window_means(df, ['x'], 7)
        self.assertTrue(got.loc[[0, 1], 'recent7_x'].isna().all())
        self.assertEqual(got.loc[2, 'recent7_x'], 2.)
        self.assertEqual(got.loc[3, 'recent7_x'], 2.)
        self.assertEqual(got.loc[4, 'recent7_x'], 6.)
        assert_frame_equal(got, prior_window_means(df.sample(frac=1, random_state=2), ['x'], 7).sort_index())

    def test_full_profile_is_prefix_invariant_and_label_free(self):
        wkts = ['LINESTRING (127 35,127.01 35)', 'LINESTRING (127 35,127 35)',
                'LINESTRING (129 36,129.01 36)', 'LINESTRING (127 35,127.02 35)']
        df = extract_features(wkts)
        df['rfid_id'] = [1, 1, 2, 1]
        df['part_dt'] = [20260131, 20260201, 20260201, 20260210]
        df['rfid_time'] = [23, 1, 3, 22]
        loc, geo = coordinates(wkts), robust_geometry(wkts).to_numpy()
        all_features = behavior_context(df, loc, geo)
        prefix = behavior_context(df.iloc[:3], loc[:3], geo[:3])
        assert_frame_equal(all_features.iloc[:3], prefix)
        changed = df.assign(fishing_code='unused', rfid_id=df.rfid_id + 500)
        assert_frame_equal(all_features, behavior_context(changed, loc, geo))
        self.assertFalse(np.isinf(all_features.to_numpy()).any())
        self.assertTrue(all_features.loc[0].filter(like='past_').isna().all())


if __name__ == '__main__':
    unittest.main()
