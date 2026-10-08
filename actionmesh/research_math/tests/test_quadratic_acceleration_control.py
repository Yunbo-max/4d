"""Engineering contracts for C13's quadratic control, not native evidence."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    control = importlib.import_module('research_math.quadratic_acceleration_control')
except ModuleNotFoundError:
    control = None


class QuadraticAccelerationControlTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(control, 'Quadratic acceleration control is missing')
        self.times = np.arange(16, dtype=np.float64)
        anchor = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]], dtype=np.float32)
        self.vertices = np.repeat(anchor[None], 16, axis=0)
        self.faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)

    def test_timestamp_second_difference_annihilates_affine_motion(self):
        times = np.array([0., .25, 1., 2.5, 5.])
        operator = control.timestamp_second_difference(times)
        np.testing.assert_allclose(operator @ np.ones(len(times)), 0., atol=1e-14)
        np.testing.assert_allclose(operator @ times, 0., atol=1e-14)

    def test_affine_trajectory_is_fixed_point_and_anchor_is_exact(self):
        velocity = np.array([.2, -.1, .3], dtype=np.float32)
        source = self.vertices + self.times[:, None, None].astype(np.float32) * velocity
        repaired, diagnostics = control.repair_quadratic_acceleration(source, self.times, 3.)
        np.testing.assert_allclose(repaired, source, atol=2e-6)
        np.testing.assert_array_equal(repaired[0], source[0])
        self.assertLess(diagnostics['linear_residual_linf'], 1e-10)
        self.assertLessEqual(diagnostics['relative_backward_error'],
                             diagnostics['stationarity_tolerance'])
        self.assertLessEqual(diagnostics['free_system_condition_2'],
                             diagnostics['condition_2_max'])

    def test_isolated_spike_is_reduced_with_lower_timestamp_acceleration(self):
        source = self.vertices.copy(); source[8, 3, 2] += 2.
        repaired, diagnostics = control.repair_quadratic_acceleration(source, self.times, 2.)
        np.testing.assert_array_equal(repaired[0], source[0])
        self.assertLess(float(repaired[8, 3, 2] - self.vertices[8, 3, 2]), 1.)
        self.assertLess(diagnostics['acceleration_l2_after'], diagnostics['acceleration_l2_before'])

    def test_invalid_parameters_are_rejected(self):
        for weight in (True, 0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                control.repair_quadratic_acceleration(self.vertices, self.times, weight)
        with self.assertRaises(ValueError):
            control.timestamp_second_difference(np.array([0., 1., 1.]))
        broken = self.vertices.copy(); broken[2, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            control.repair_quadratic_acceleration(broken, self.times, 1.)

    def test_non_float32_sequence_is_rejected_for_official_adapter_compatibility(self):
        with self.assertRaises(ValueError):
            control.repair_quadratic_acceleration(self.vertices.astype(np.float64), self.times, 1.)

    def test_ill_conditioned_weight_is_rejected_with_observed_diagnostics(self):
        with self.assertRaisesRegex(
                ValueError,
                r'backward_error=.*stationarity_tolerance=.*condition_2=.*condition_2_max='):
            control.repair_quadratic_acceleration(self.vertices, self.times, 1e16)

    def source(self, root):
        case = root/'source'; case.mkdir()
        self.vertices[8, 3, 2] += 2.
        np.savez_compressed(case/'sequence.npz', vertices=self.vertices, faces=self.faces,
                            frame_indices=np.arange(16), timesteps=self.times.astype(np.float32),
                            query_vertex_ids=np.arange(4))
        digest = hashlib.sha256((case/'sequence.npz').read_bytes()).hexdigest()
        (case/'report.json').write_text(json.dumps({'status': 'completed', 'uid': 'fixture', 'seed': 42,
            'sha256': {'sequence.npz': digest}}))
        return case, digest

    def test_export_is_one_real_full_sequence_arm_with_no_candidate_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, digest = self.source(root)
            result = control.export_quadratic_control(case, root/'out', uid='fixture',
                expected_sequence_sha256=digest, weight=2.)
            self.assertEqual(result['status'], 'completed')
            self.assertFalse(result['candidate_methods_tested'])
            self.assertFalse(result['native_qualified'])
            manifest = json.loads((root/'out/manifest.json').read_text())
            self.assertEqual(manifest['cases'], [{'case_id': 'fixture-quadratic_acceleration',
                'uid': 'fixture', 'case_dir': 'quadratic_acceleration'}])
            with np.load(root/'out/quadratic_acceleration/sequence.npz', allow_pickle=False) as data:
                np.testing.assert_array_equal(data['faces'], self.faces)
                np.testing.assert_array_equal(data['frame_indices'], np.arange(16))
                np.testing.assert_array_equal(data['query_vertex_ids'], np.arange(4))
                np.testing.assert_array_equal(data['timesteps'], self.times)
                np.testing.assert_array_equal(data['vertices'][0], self.vertices[0])
                self.assertEqual(data['vertices'].dtype, self.vertices.dtype)
            arm = json.loads((root/'out/quadratic_acceleration/report.json').read_text())
            self.assertEqual(arm['objective'], '0.5||Y-Yhat||_F^2 + 0.5*weight||D2_timestamp Y||_F^2; Y[0]=Yhat[0]')
            self.assertEqual(arm['observation_metric'], 'identity')
            self.assertEqual(arm['timestamp_units'], 'supplied_sequence_units')
            self.assertEqual(arm['weight_units'], 'supplied_sequence_units^4')
            self.assertIn('linear_residual_linf', arm['diagnostics'])

    def test_stale_hash_and_existing_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, digest = self.source(root)
            with self.assertRaises(ValueError):
                control.export_quadratic_control(case, root/'out', uid='fixture',
                    expected_sequence_sha256='0'*64, weight=2.)
            self.assertFalse((root/'out').exists())
            control.export_quadratic_control(case, root/'out', uid='fixture',
                expected_sequence_sha256=digest, weight=2.)
            with self.assertRaises(FileExistsError):
                control.export_quadratic_control(case, root/'out', uid='fixture',
                    expected_sequence_sha256=digest, weight=2.)


if __name__ == '__main__': unittest.main()
