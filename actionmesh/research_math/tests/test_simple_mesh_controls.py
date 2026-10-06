"""Engineering fixtures only: never native benchmark scores or qualification."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np

try:
    controls = importlib.import_module('research_math.simple_mesh_controls')
except ModuleNotFoundError:
    controls = None


class SimpleMeshControlsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(controls, 'Simple mesh baseline adapter is missing')
        self.anchor = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]], dtype=np.float32)
        self.faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        self.sequence = np.repeat(self.anchor[None], 16, axis=0)

    def test_gaussian_boundary_renormalizes_without_wrapping(self):
        weights = controls.gaussian_weights(3, 1.)
        np.testing.assert_allclose(weights[0], [0.57409699297, 0.34820742788, 0.07769557915], atol=1e-10)
        np.testing.assert_allclose(weights.sum(axis=1), 1.)

    def test_world_smoothing_damps_isolated_displacement_and_pins_anchor(self):
        self.sequence[8] += 1.
        out = controls.smooth_world(self.sequence, 1.)
        np.testing.assert_array_equal(out[0], self.sequence[0])
        self.assertGreater(float(out[7, 0, 0]), .2)
        self.assertLess(float(out[8, 0, 0]), .41)
        self.assertEqual(out.dtype, self.sequence.dtype)

    def test_body_smoothing_preserves_time_varying_rigid_motion(self):
        for t in range(16):
            angle = .2*t
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
            self.sequence[t] = self.anchor @ rotation.T + [t*.2, np.sin(t), 0.]
        out, pose = controls.smooth_body(self.sequence, 1.)
        np.testing.assert_allclose(out, self.sequence, atol=2e-6)
        np.testing.assert_array_equal(out[0], self.sequence[0])
        np.testing.assert_allclose(np.linalg.det(pose['rotation_rows']), 1., atol=1e-10)
        self.assertGreater(float(np.max(np.abs(controls.smooth_world(self.sequence, 1.)-self.sequence))), .1)

    def test_body_smoothing_removes_nonrigid_jitter(self):
        self.sequence[8, 3, 2] += 1.
        out, _ = controls.smooth_body(self.sequence, 1.)
        # Edge length is invariant to the fitted pose; the local spike must shrink.
        spike = np.linalg.norm(out[8, 3]-out[8, 0])-4.
        self.assertGreater(float(spike), 0.)
        self.assertLess(float(spike), .8)

    def test_body_smoothing_is_equivariant_to_global_translation(self):
        self.sequence[8, 3, 2] += 1.
        a, _ = controls.smooth_body(self.sequence, 1.)
        b, _ = controls.smooth_body(self.sequence+[9., -4., 2.], 1.)
        np.testing.assert_allclose(b, a+[9., -4., 2.], atol=2e-6)

    def test_collinear_pose_is_rejected_instead_of_arbitrary_rotation(self):
        line = np.zeros((16, 4, 3)); line[:, :, 0] = np.arange(4)
        with self.assertRaises(ValueError):
            controls.smooth_body(line, 1.)

    def test_ambiguous_reflection_pose_is_rejected(self):
        anchor = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])
        sequence = np.repeat(anchor[None], 16, axis=0)
        sequence[8, :, 0] *= -1.
        # Covariance diag(-4,4,4): correcting the reflection has no unique axis.
        with self.assertRaises(ValueError): controls.smooth_body(sequence, 1.)

    def test_invalid_parameters_or_nonfinite_vertices_rejected(self):
        for sigma in [0., -1., np.nan, np.inf]:
            with self.assertRaises(ValueError): controls.smooth_world(self.sequence, sigma)
        self.sequence[1, 0, 0] = np.nan
        with self.assertRaises(ValueError): controls.smooth_body(self.sequence, 1.)

    def test_finite_extreme_input_cannot_emit_nonfinite_world_control(self):
        huge = self.sequence.astype(np.float64)
        huge[0, :, 0] = -1e308; huge[1:, :, 0] = 1e308
        with self.assertRaises(ValueError): controls.smooth_world(huge, 1.)

    def source(self, root):
        case = root/'source'; case.mkdir()
        np.savez_compressed(case/'sequence.npz', vertices=self.sequence, faces=self.faces,
                            frame_indices=np.arange(16), timesteps=np.arange(16, dtype=np.float32), query_vertex_ids=np.arange(4))
        sha = hashlib.sha256((case/'sequence.npz').read_bytes()).hexdigest()
        (case/'report.json').write_text(json.dumps({'status': 'completed', 'uid': 'fixture', 'seed': 42, 'sha256': {'sequence.npz': sha}}))
        return case, sha

    def test_export_preserves_native_arrays_and_reports_full_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, sha = self.source(root)
            result = controls.export_controls(case, root/'out', uid='fixture', expected_sequence_sha256=sha, sigma=1.)
            self.assertEqual(result['status'], 'completed')
            manifest = json.loads((root/'out/manifest.json').read_text())
            self.assertEqual(len(manifest['cases']), 3)
            self.assertEqual({x['uid'] for x in manifest['cases']}, {'fixture'})
            for entry in manifest['cases']:
                with np.load(root/'out'/entry['case_dir']/'sequence.npz', allow_pickle=False) as data:
                    for key, expected in [('faces', self.faces), ('frame_indices', np.arange(16)), ('query_vertex_ids', np.arange(4)), ('timesteps', np.arange(16))]:
                        np.testing.assert_array_equal(data[key], expected)
                    np.testing.assert_array_equal(data['vertices'][0], self.anchor)
            self.assertEqual(hashlib.sha256((root/'out/native/sequence.npz').read_bytes()).hexdigest(), sha)
            self.assertFalse(result['native_qualified'])
            with self.assertRaises(FileExistsError):
                controls.export_controls(case, root/'out', uid='fixture', expected_sequence_sha256=sha, sigma=1.)

    def test_stale_source_hash_rejected_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, _ = self.source(root)
            with self.assertRaises(ValueError):
                controls.export_controls(case, root/'out', uid='fixture', expected_sequence_sha256='0'*64, sigma=1.)
            self.assertFalse((root/'out').exists())

    def test_noninteger_identity_metadata_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, _ = self.source(root)
            with np.load(case/'sequence.npz', allow_pickle=False) as data:
                arrays = {k: data[k].copy() for k in data.files}
            arrays['frame_indices'] = arrays['frame_indices'].astype(np.float64)
            np.savez_compressed(case/'sequence.npz', **arrays)
            sha = hashlib.sha256((case/'sequence.npz').read_bytes()).hexdigest()
            (case/'report.json').write_text(json.dumps({'status': 'completed', 'uid': 'fixture', 'seed': 42, 'sha256': {'sequence.npz': sha}}))
            with self.assertRaises(ValueError):
                controls.export_controls(case, root/'out', uid='fixture', expected_sequence_sha256=sha, sigma=1.)
            self.assertFalse((root/'out').exists())

    def test_failed_body_arm_stays_in_manifest_and_prevents_completion(self):
        self.sequence[:] = 0.; self.sequence[:, :, 0] = np.arange(4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, sha = self.source(root)
            result = controls.export_controls(case, root/'out', uid='fixture', expected_sequence_sha256=sha, sigma=1.)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(len(json.loads((root/'out/manifest.json').read_text())['cases']), 3)
            failure = json.loads((root/'out/body_gaussian/report.json').read_text())
            self.assertEqual(failure['status'], 'error')
            self.assertFalse((root/'out/body_gaussian/sequence.npz').exists())

    def test_source_sequence_cli_works_from_a_staged_file_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, sha = self.source(root)
            process = subprocess.run([sys.executable, '-m', 'research_math.simple_mesh_controls',
                '--source-sequence', str(case/'sequence.npz'), '--output', str(root/'out'), '--uid', 'fixture',
                '--expected-sequence-sha256', sha, '--sigma', '1'], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertFalse(json.loads(process.stdout)['native_qualified'])

    def test_existing_evaluator_keeps_failed_arm_in_declared_inventory(self):
        from research_census_eval import discover_cases
        self.sequence[:] = 0.; self.sequence[:, :, 0] = np.arange(4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, sha = self.source(root)
            controls.export_controls(case, root/'out', uid='fixture', expected_sequence_sha256=sha, sigma=1.)
            cases, denominator = discover_cases(root/'out', root/'out/manifest.json')
            self.assertEqual(len(cases), 3)
            self.assertEqual(denominator['n_declared'], 3)
            self.assertTrue(any(Path(x['case_dir']).name == 'body_gaussian' for x in cases))


if __name__ == '__main__':
    unittest.main()
