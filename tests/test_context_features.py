import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.context_features import build_context, coordinates, load_locations, prior_day_history, safe_ratio
from src.features import extract_features


class ContextChecks(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'rfid_id': [1, 2, 1, 1, 1, 2],
            'part_dt': [20260131, 20260131, 20260131, 20260202, 20260203, 20260202],
            'value': [2., 100., 4., 8., np.nan, 200.],
        })

    def test_strict_prior_dates_and_vessel_isolation(self):
        result = prior_day_history(self.df, ['value'])
        self.assertTrue(result.loc[[0, 1, 2], 'past_mean_value'].isna().all())
        self.assertEqual(result.loc[3, 'past_mean_value'], 3.)
        self.assertEqual(result.loc[3, 'past_std_value'], 1.)
        self.assertEqual(result.loc[3, 'past_observations'], 2)
        self.assertEqual(result.loc[3, 'days_since_previous_date'], 2)
        self.assertEqual(result.loc[5, 'past_mean_value'], 100.)
        self.assertAlmostEqual(result.loc[4, 'past_mean_value'], 14 / 3, places=6)

    def test_future_and_current_day_cannot_change_past(self):
        before = prior_day_history(self.df, ['value'])
        changed = self.df.copy()
        changed.loc[changed.part_dt >= 20260202, 'value'] = 99999.
        after = prior_day_history(changed, ['value'])
        mask = self.df.part_dt <= 20260202
        assert_frame_equal(before.loc[mask], after.loc[mask])

    def test_order_labels_and_identifiers_are_not_inputs(self):
        expected = prior_day_history(self.df, ['value'])
        shuffled = self.df.sample(frac=1, random_state=7)
        shuffled['fishing_code'] = 'ignored'
        shuffled.rfid_id += 999
        actual = prior_day_history(shuffled, ['value']).sort_index()
        assert_frame_equal(expected, actual)

    def test_prefix_and_missing_values(self):
        expected = prior_day_history(self.df, ['value'])
        prefix = self.df[self.df.part_dt <= 20260202]
        assert_frame_equal(expected.loc[prefix.index], prior_day_history(prefix, ['value']))
        missing = self.df.copy()
        missing['value'] = np.nan
        result = prior_day_history(missing, ['value'])
        self.assertTrue(result.past_mean_value.isna().all())
        self.assertFalse(np.isinf(result.to_numpy()).any())

    def test_geometry_and_zero_denominator(self):
        result = coordinates(['MULTILINESTRING ((0 0, 2 0), (10 4, 12 4))'])
        np.testing.assert_allclose(result, [[6, 2]])
        values = safe_ratio([1, 4], [0, 2])
        self.assertTrue(np.isnan(values[0]))
        self.assertEqual(values[1], 2)

    def test_raw_alignment_exclusions_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.csv'
            pd.DataFrame({'geom': ['LINESTRING (10 2, 12 4)', 'broken',
                                   'LINESTRING (20 4, 22 6)']}).to_csv(source, index=False)
            # Filtered table is deliberately reordered and skips an invalid raw row.
            df = pd.DataFrame({'source_file_id': [1, 1], 'source_row': [2, 0]})
            expected = [[21, 5], [11, 3]]
            np.testing.assert_allclose(load_locations(df, [source], root / 'cache'), expected)
            np.testing.assert_allclose(load_locations(df, [source], root / 'cache'), expected)
            reordered = df.iloc[::-1].reset_index(drop=True)
            np.testing.assert_allclose(load_locations(reordered, [source], root / 'cache'), expected[::-1])

    def test_complete_feature_contract_with_stationary_track(self):
        wkts = ['LINESTRING (127 35, 127 35)', 'LINESTRING (127 35, 128 36)']
        df = extract_features(wkts)
        df['rfid_id'] = [10, 10]
        df['part_dt'] = [20260131, 20260202]
        df['rfid_time'] = [23, 0]
        df['fishing_code'] = ['private_label', 'private_label']
        result = build_context(df, coordinates(wkts))
        self.assertEqual(result.shape, (2, 71))
        self.assertFalse(set(['rfid_id', 'fishing_code', 'part_dt', 'source_row']) & set(result.columns))
        self.assertFalse(np.isinf(result.to_numpy()).any())
        self.assertEqual(result.loc[1, 'past_mean_total_distance_m'], 0.)
        changed = df.assign(rfid_id=999, fishing_code='different_label')
        assert_frame_equal(result, build_context(changed, coordinates(wkts)))


if __name__ == '__main__':
    unittest.main()
