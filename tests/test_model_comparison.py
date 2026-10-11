import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

spec = importlib.util.spec_from_file_location('compare', Path(__file__).resolve().parents[1] / '09_compare_models.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.old = pd.DataFrame({'source_file_id': [1]*6, 'source_row': range(6),
            'rfid_id': [1, 1, 1, 2, 2, 3], 'fishing_code': ['a', 'a', 'a', 'b', 'b', 'c'],
            'prediction': ['a', 'b', 'b', 'b', 'a', 'c']})

    def test_identical_predictions_have_zero_paired_uncertainty(self):
        result = compare.paired_bootstrap(self.old, self.old.sample(frac=1, random_state=5), list('abc'), 50)
        for metric in ['accuracy', 'macro_f1']:
            self.assertEqual(result[metric]['delta'], 0)
            self.assertEqual(result[metric]['paired_vessel_95pct_interval'], [0, 0])
        self.assertAlmostEqual(result['accuracy']['previous'], accuracy_score(self.old.fishing_code, self.old.prediction))
        self.assertAlmostEqual(result['macro_f1']['previous'], f1_score(self.old.fishing_code, self.old.prediction, average='macro'))

    def test_changed_rows_or_labels_fail(self):
        with self.assertRaises(ValueError):
            compare.paired_bootstrap(self.old, self.old.iloc[:-1], list('abc'), 10)
        changed = self.old.copy()
        changed.loc[0, 'fishing_code'] = 'b'
        with self.assertRaises(ValueError):
            compare.paired_bootstrap(self.old, changed, list('abc'), 10)


if __name__ == '__main__':
    unittest.main()
