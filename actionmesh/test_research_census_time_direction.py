"""CPU-only mapping and identity controls; never imports torch or loads a model."""
import unittest
import numpy as np
from research_census_time_direction import arm_mapping, bitwise_equal, discrepancy


class TimeDirectionMappingTests(unittest.TestCase):
    def test_permutation_retains_every_physical_timestamp(self):
        mapping = arm_mapping('row_permutation')
        self.assertEqual(mapping['source_row_index'], 15)
        self.assertEqual(mapping['source_clock_time'], 0)
        np.testing.assert_array_equal(mapping['latent_row_clock_times'], mapping['latent_row_physical_ids'])
        np.testing.assert_array_equal(mapping['target_clock_times'], np.arange(1, 16))

    def test_true_reversal_retains_physical_anchor_and_every_target(self):
        mapping = arm_mapping('time_reversal')
        self.assertEqual(mapping['source_row_index'], 0)
        self.assertEqual(mapping['source_clock_time'], 15)
        np.testing.assert_array_equal(mapping['latent_row_physical_ids'], np.arange(16))
        np.testing.assert_array_equal(mapping['target_clock_times'], np.arange(14, -1, -1))
        self.assertEqual(len(set(mapping['output_clock_times'])), 16)
        self.assertNotIn(mapping['source_clock_time'], mapping['target_clock_times'])

    def test_bitwise_control_and_zero_disagreement(self):
        reference = np.zeros((16, 3, 3), dtype=np.float32)
        self.assertTrue(bitwise_equal(reference, reference.copy()))
        self.assertFalse(bitwise_equal(reference, reference.astype(np.float64)))
        changed = reference.copy()
        changed[0, 0, 0] = -0.0
        self.assertFalse(bitwise_equal(reference, changed))
        self.assertEqual(discrepancy(reference, reference.copy(), 1)['moving_rms_xyz_over_anchor_diagonal'], 0)


if __name__ == '__main__':
    unittest.main()
