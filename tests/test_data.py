"""Check repeatable patient splits and inference exports without using private data."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import _bootstrap
from polarity_d.data import make_splits, validate_splits
from polarity_d.export import inference_metadata
from fixtures import synthetic_plans, synthetic_dataset


class DataTests(unittest.TestCase):
    def test_group_split_repeats_without_patient_leakage(self):
        groups = {f'scan_{i}_{j}': f'patient_{i}' for i in range(10) for j in range(2)}
        cases = sorted(groups)
        splits = make_splits(cases, 20261017, groups)
        self.assertEqual(splits, make_splits(list(reversed(cases)), 20261017, groups))
        self.assertNotEqual(splits, make_splits(cases, 20261018, groups))
        self.assertEqual(sorted(c for fold in splits for c in fold['val']), cases)
        for split in splits:
            self.assertFalse({groups[c] for c in split['train']} & {groups[c] for c in split['val']})

    def test_invalid_split_rejected(self):
        cases = [f'case_{i}' for i in range(10)]
        splits = make_splits(cases, 17)
        splits[0]['train'].append(splits[0]['val'][0])
        with self.assertRaises(ValueError):
            validate_splits(splits, cases)
        with self.assertRaises(ValueError):
            make_splits(cases, 17, {c: 'one_patient' for c in cases})

    def test_export_keeps_inference_settings_not_dataset_statistics(self):
        plans = synthetic_plans()
        plans['dataset_name'] = 'DatasetSyntheticPrivate'
        plans['original_median_shape_after_transp'] = [91, 92, 93]
        plans['foreground_intensity_properties_per_channel'] = {'0': {'mean': 12345.}}
        original = copy.deepcopy(plans)
        minimized, metadata = inference_metadata(plans, synthetic_dataset())
        self.assertEqual(plans, original)
        self.assertNotIn('dataset_name', minimized)
        self.assertNotIn('original_median_shape_after_transp', minimized)
        self.assertEqual(minimized['foreground_intensity_properties_per_channel'], {'0': {}})
        self.assertNotIn('numTraining', metadata)
        self.assertEqual(minimized['configurations']['3d_fullres']['spacing'], [1., 1., 1.])
        self.assertEqual(metadata['labels'], {'background': 0, 'foreground': 1})


if __name__ == '__main__':
    unittest.main()
