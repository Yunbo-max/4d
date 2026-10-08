"""Local engineering contracts only; no fixture is native scientific evidence.

Run through the repository software-acceptance harness, never as Web evidence.
The native retained-bundle acceptance is separate from these algebra fixtures.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math import self_map_candidate as candidate


class SelfMapAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.anchor = np.array([[0., .25, -.25], [.5, .25, 0.], [-.25, .5, .25]], dtype=np.float32)
        self.bias = np.array([[.125, -.0625, .125], [-.125, .0625, .25], [.25, .125, -.125]])
        self.motion = np.arange(1, 16)[:, None, None] * np.array([.0078125, -.00390625, .015625])
        self.target = self.anchor[None] + self.bias[None] + self.motion
        self.source = self.anchor + self.bias

    def solve(self, anchor=None, target=None, source=None, **options):
        return candidate.correction_arrays(
            self.anchor if anchor is None else anchor,
            self.target if target is None else target,
            self.source if source is None else source,
            coordinate_bounds=options.get('coordinate_bounds', [-1., 1.]),
            bounds_policy=options.get('bounds_policy', 'preserve_and_report'))

    def test_shared_pointwise_nuisance_cancels_full_16_frame_field(self):
        arms, diagnostics = self.solve()
        expected = (self.anchor[None] + self.motion).astype(np.float32)
        np.testing.assert_array_equal(arms['self_map_subtraction'][1:], expected)
        for arm in arms.values():
            self.assertEqual(arm.shape, (16, 3, 3))
            self.assertEqual(arm.dtype, np.float32)
            np.testing.assert_array_equal(arm[0], self.anchor)
        nuisance = np.array([[.25, 0., -.125], [.125, -.25, 0.], [-.5, .5, .25]])
        shifted, _ = self.solve(target=self.target + nuisance[None], source=self.source + nuisance)
        np.testing.assert_array_equal(shifted['self_map_subtraction'], arms['self_map_subtraction'])
        self.assertFalse(diagnostics['arms']['self_map_subtraction']['clipped'])

    def test_mean_control_is_only_spatial_mean_not_pointwise_correction(self):
        arms, _ = self.solve()
        np.testing.assert_array_equal(arms['mean_bias'][1:],
            (self.target - self.bias.mean(axis=0)[None, None]).astype(np.float32))
        self.assertFalse(np.array_equal(arms['mean_bias'], arms['self_map_subtraction']))
        np.testing.assert_array_equal(arms['raw_uncorrected'][1:], self.target.astype(np.float32))

    def test_zero_residual_changes_nothing_and_preserves_anchor(self):
        arms, diagnostics = self.solve(source=self.anchor)
        for role in candidate.ROLES:
            np.testing.assert_array_equal(arms[role], arms['raw_uncorrected'])
        self.assertEqual(diagnostics['self_residual_l2'], 0.)

    def test_bounds_preserve_values_or_reject_per_arm_never_clip(self):
        target = self.target + 2.
        kept, diagnostics = self.solve(target=target)
        self.assertGreater(float(kept['self_map_subtraction'].max()), 1.)
        self.assertFalse(diagnostics['arms']['self_map_subtraction']['rejected'])
        rejected, flags = self.solve(target=target, bounds_policy='reject')
        np.testing.assert_array_equal(kept['self_map_subtraction'], rejected['self_map_subtraction'])
        self.assertTrue(flags['arms']['self_map_subtraction']['rejected'])
        self.assertGreater(flags['arms']['self_map_subtraction']['outside_coordinate_count'], 0)
        with self.assertRaises(ValueError):
            self.solve(bounds_policy='clip')

    def test_float16_decoder_values_keep_original_float32_anchor(self):
        anchor = self.anchor.copy(); anchor[0, 0] = np.float32(.1234567)
        source = anchor.astype(np.float16)
        targets = np.repeat(source[None], 15, axis=0)
        arms, _ = self.solve(anchor=anchor, target=targets, source=source)
        np.testing.assert_array_equal(arms['self_map_subtraction'], np.repeat(anchor[None], 16, axis=0))

    def test_cross_term_kept_in_risk_identity_not_automatic_improvement(self):
        downstream_error = np.repeat(self.bias[None], 15, axis=0)
        source_error = -self.bias
        before = np.square(downstream_error).sum()
        after = np.square(downstream_error - source_error).sum()
        expected = 15 * np.square(source_error).sum() - 2 * np.sum(downstream_error * source_error)
        self.assertAlmostEqual(after - before, expected)
        self.assertGreater(after, before)

    def test_incomplete_or_nonfinite_inputs_fail_closed(self):
        for value in (self.target[:14], np.full_like(self.target, np.nan)):
            with self.assertRaises(ValueError):
                self.solve(target=value)
        with self.assertRaises(ValueError):
            self.solve(anchor=self.anchor.astype(np.float64))
        with self.assertRaises(ValueError):
            self.solve(coordinate_bounds=[1., -1.])

    def test_snapshot_rejects_truncation_growth_and_hash_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source'; source.write_bytes(b'abc')
            for size in (2, 4):
                with self.assertRaises(ValueError):
                    candidate._copy_file(source, root / ('out' + str(size)), candidate.digest(source), size)
            with self.assertRaises(ValueError):
                candidate._copy_file(source, root / 'wrong-hash', '0' * 64, 3)

    def test_native_metadata_validation_keeps_original_identity(self):
        arrays = {'vertices': np.repeat(self.anchor[None], 16, axis=0),
                  'faces': np.array([[0, 1, 2]], dtype=np.int64),
                  'frame_indices': np.arange(16), 'timesteps': np.arange(16, dtype=np.float32),
                  'query_vertex_ids': np.arange(3)}
        candidate.validate_native_arrays(arrays)
        arrays['query_vertex_ids'] = np.array([2, 1, 0])
        with self.assertRaises(ValueError):
            candidate.validate_native_arrays(arrays)


if __name__ == '__main__':
    unittest.main()
