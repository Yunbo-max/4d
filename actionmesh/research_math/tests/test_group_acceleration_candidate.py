"""Engineering contracts for the C13 group-acceleration source draft.

These fixtures exercise the mathematical/native interface only.  They are not
ActionBench evidence, candidate admission, or a scientific result.
"""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


try:
    candidate = importlib.import_module('research_math.group_acceleration_candidate')
except ModuleNotFoundError:
    candidate = None


class GroupAccelerationCandidateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(candidate, 'C13 group-acceleration candidate is missing')
        self.times = np.arange(16, dtype=np.float64)
        anchor = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.],
                           [0., 0., 4.]], dtype=np.float32)
        self.vertices = np.repeat(anchor[None], 16, axis=0)
        self.faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)

    def solve(self, vertices, metric=None, **overrides):
        options = dict(group_weight=.25, rho=1., absolute_tolerance=1e-8,
                       relative_tolerance=1e-7, gap_tolerance=1e-7,
                       max_iterations=10000)
        options.update(overrides)
        if metric is None:
            metric = np.eye(len(vertices), dtype=np.float64)
        return candidate.repair_group_acceleration(
            vertices, self.times, metric, pinned_frames=[0], **options)

    def solve_float64(self, vertices, metric=None):
        # Mathematical symmetry and SPD behavior are properties of the solve.
        # Certification of the rounded native export is tested separately.
        if metric is None:
            metric = np.eye(len(vertices), dtype=np.float64)
        problem = candidate.form_anchored_group_trend_problem(
            vertices, self.times, metric, pinned_frames=[0])
        return candidate.solve_affine_anchor_stationarity(
            problem, group_weight=.25, rho=1., absolute_tolerance=1e-8,
            relative_tolerance=1e-7, gap_tolerance=1e-7, max_iterations=10000)

    def test_nonuniform_timestamp_operator_annihilates_affine_motion(self):
        times = np.array([0., .25, 1., 2.5, 5.])
        operator = candidate.timestamp_second_difference(times)
        np.testing.assert_allclose(operator @ np.ones(len(times)), 0., atol=1e-14)
        np.testing.assert_allclose(operator @ times, 0., atol=1e-14)

    def test_affine_motion_is_fixed_point_with_exact_anchor_and_certificate(self):
        velocity = np.array([.25, -.125, .5], dtype=np.float32)
        source = self.vertices + self.times[:, None, None].astype(np.float32) * velocity
        repaired, certificate, diagnostics = self.solve(source)
        np.testing.assert_allclose(repaired, source, atol=2e-6)
        np.testing.assert_array_equal(repaired[0], source[0])
        self.assertEqual(diagnostics['termination_reason'], 'converged')
        self.assertLessEqual(diagnostics['primal_dual_gap_relative'], 1e-7)
        self.assertLessEqual(diagnostics['dual_group_norm_max'], .25 + 1e-12)
        self.assertLessEqual(diagnostics['anchor_residual_linf'], 0.)
        self.assertEqual(certificate['dual'].shape, (14, 4, 3))

    def test_three_dimensional_group_prior_is_rotation_equivariant(self):
        source = self.vertices.copy()
        source[5, 3] += np.array([.7, -.2, .4], dtype=np.float32)
        source[10, 1] += np.array([-.3, .6, .1], dtype=np.float32)
        angle = .61
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.],
                             [0., 0., 1.]], dtype=np.float64)
        repaired, _, _ = self.solve_float64(source)
        rotated, _, _ = self.solve_float64((source.astype(np.float64) @ rotation).astype(np.float32))
        np.testing.assert_allclose(rotated, repaired.astype(np.float64) @ rotation,
                                   atol=3e-5, rtol=3e-5)

    def test_off_diagonal_metric_keeps_anchor_cross_term(self):
        source = self.vertices.copy()
        source[6, 3, 2] += 1.
        metric = np.eye(16, dtype=np.float64)
        metric += .15 * (np.eye(16, k=1) + np.eye(16, k=-1))
        repaired, _, diagnostics = self.solve_float64(source, metric)
        identity, _, _ = self.solve_float64(source)
        self.assertGreater(float(np.max(np.abs(repaired - identity))), 1e-5)
        self.assertEqual(diagnostics['metric_scope'], 'shared_temporal_spd_kron_identity_vertex_xyz')
        self.assertGreater(diagnostics['metric_offdiagonal_linf'], 0.)
        np.testing.assert_array_equal(repaired[0], source[0])

    def test_explicit_anchor_value_enters_free_linear_term(self):
        source = self.vertices[:4].copy()
        times = np.arange(4, dtype=np.float64)
        metric = np.eye(4, dtype=np.float64)
        metric[0, 1] = metric[1, 0] = .2
        anchors = source[:1].copy()
        anchors[0, 0, 0] += 1.
        problem = candidate.form_anchored_group_trend_problem(
            source, times, metric, pinned_frames=[0], anchor_values=anchors)
        # With Yhat fixed and y_p nonzero only at frame 0, the first free
        # linear coefficient is M[1,0] * (X-Yhat[0]) = 0.2 exactly.
        self.assertAlmostEqual(float(problem['reduced_linear'][0, 0, 0]), .2)
        np.testing.assert_array_equal(problem['anchor_values'], anchors)
        repaired, _, diagnostics = candidate.repair_group_acceleration(
            source, times, metric, pinned_frames=[0], anchor_values=anchors,
            group_weight=.25, rho=1., absolute_tolerance=1e-8,
            relative_tolerance=1e-7, gap_tolerance=1e-7,
            max_iterations=10000)
        np.testing.assert_array_equal(repaired[0], anchors[0].astype(np.float32))
        self.assertTrue(diagnostics['anchor_exact_float32'])

    def test_bad_metric_parameters_and_inputs_fail_closed(self):
        nonsymmetric = np.eye(16); nonsymmetric[0, 1] = .2
        indefinite = np.eye(16); indefinite[3, 3] = -1.
        for metric in (nonsymmetric, indefinite, np.eye(15), np.full((16, 16), np.nan)):
            with self.assertRaises(ValueError):
                self.solve(self.vertices, metric)
        for key, value in (('group_weight', 0.), ('rho', np.inf),
                           ('absolute_tolerance', 0.), ('relative_tolerance', -1.),
                           ('gap_tolerance', True), ('max_iterations', 0)):
            with self.assertRaises(ValueError):
                self.solve(self.vertices, **{key: value})
        with self.assertRaises(ValueError):
            self.solve(self.vertices.astype(np.float64))
        anchors = self.vertices[[3, 0]].copy()
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            candidate.form_anchored_group_trend_problem(
                self.vertices, self.times, np.eye(16),
                pinned_frames=[3, 0], anchor_values=anchors)

    def source(self, root, *, spike=True):
        case = root / 'source'; case.mkdir()
        if spike:
            self.vertices[8, 3, 2] += 1.
        np.savez_compressed(case / 'sequence.npz', vertices=self.vertices,
                            faces=self.faces, frame_indices=np.arange(16),
                            timesteps=self.times.astype(np.float32),
                            query_vertex_ids=np.arange(4))
        digest = hashlib.sha256((case / 'sequence.npz').read_bytes()).hexdigest()
        (case / 'report.json').write_text(json.dumps({
            'status': 'completed', 'uid': 'fixture', 'seed': 42,
            'sha256': {'sequence.npz': digest}}))
        return case, digest

    def test_export_is_complete_identity_preserving_candidate_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, digest = self.source(root, spike=False)
            result = candidate.export_group_candidate(
                case, root / 'out', uid='fixture',
                expected_sequence_sha256=digest, metric_path=None,
                expected_metric_sha256=None, group_weight=.25, rho=1.,
                absolute_tolerance=1e-8, relative_tolerance=1e-7,
                gap_tolerance=1e-7, max_iterations=10000)
            self.assertEqual(result['status'], 'completed')
            self.assertFalse(result['native_qualified'])
            self.assertFalse(result['scientific_admission'])
            manifest = json.loads((root / 'out/manifest.json').read_text())
            self.assertEqual(manifest['cases'], [{
                'case_id': 'fixture-group_acceleration', 'uid': 'fixture',
                'case_dir': 'group_acceleration'}])
            with np.load(root / 'out/group_acceleration/sequence.npz', allow_pickle=False) as data:
                np.testing.assert_array_equal(data['faces'], self.faces)
                np.testing.assert_array_equal(data['frame_indices'], np.arange(16))
                np.testing.assert_array_equal(data['query_vertex_ids'], np.arange(4))
                np.testing.assert_array_equal(data['timesteps'], self.times)
                np.testing.assert_array_equal(data['vertices'][0], self.vertices[0])
                self.assertEqual(data['vertices'].dtype, np.dtype(np.float32))
            report = json.loads((root / 'out/group_acceleration/report.json').read_text())
            self.assertEqual(report['candidate_id'], '4d-math-20261006-c13')
            self.assertEqual(report['source_delivery_status'],
                             'generated_unexecuted_at_authoring')
            self.assertEqual(report['observation_metric']['mode'], 'identity')
            self.assertIn('primal_dual_gap', report['diagnostics'])
            self.assertIn('certificate.npz', report['sha256'])
            with np.load(root / 'out/group_acceleration/certificate.npz',
                         allow_pickle=False) as certificate:
                with np.load(root / 'out/group_acceleration/sequence.npz',
                             allow_pickle=False) as sequence:
                    operator = candidate.timestamp_second_difference(sequence['timesteps'])
                    expected = np.einsum('rt,tvc->rvc', operator,
                                         sequence['vertices'].astype(np.float64))
                np.testing.assert_array_equal(certificate['second_difference'], expected)
                np.testing.assert_array_equal(certificate['anchor_values'],
                                              self.vertices[:1].astype(np.float64))

    def test_explicit_metric_is_hash_bound_and_stale_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, digest = self.source(root)
            metric_path = root / 'metric.npz'
            metric = np.eye(16); metric += .1 * (
                np.eye(16, k=1) + np.eye(16, k=-1))
            np.savez(metric_path, observation_metric=metric)
            metric_digest = hashlib.sha256(metric_path.read_bytes()).hexdigest()
            candidate.export_group_candidate(
                case, root / 'out', uid='fixture', expected_sequence_sha256=digest,
                metric_path=metric_path, expected_metric_sha256=metric_digest,
                group_weight=.25, rho=1., absolute_tolerance=1e-8,
                relative_tolerance=1e-7, gap_tolerance=1e-7,
                max_iterations=10000)
            report = json.loads((root / 'out/group_acceleration/report.json').read_text())
            self.assertEqual(report['observation_metric']['sha256'], metric_digest)
            with self.assertRaises(FileExistsError):
                candidate.export_group_candidate(
                    case, root / 'out', uid='fixture', expected_sequence_sha256=digest,
                    metric_path=metric_path, expected_metric_sha256=metric_digest,
                    group_weight=.25, rho=1., absolute_tolerance=1e-8,
                    relative_tolerance=1e-7, gap_tolerance=1e-7,
                    max_iterations=10000)
            with self.assertRaises(ValueError):
                candidate.export_group_candidate(
                    case, root / 'other', uid='fixture', expected_sequence_sha256=digest,
                    metric_path=metric_path, expected_metric_sha256='0' * 64,
                    group_weight=.25, rho=1., absolute_tolerance=1e-8,
                    relative_tolerance=1e-7, gap_tolerance=1e-7,
                    max_iterations=10000)
            self.assertFalse((root / 'other').exists())

    def test_float32_quantization_failure_retains_no_scoreable_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, digest = self.source(root)
            result = candidate.export_group_candidate(
                case, root / 'out', uid='fixture',
                expected_sequence_sha256=digest, metric_path=None,
                expected_metric_sha256=None, group_weight=.25, rho=1.,
                absolute_tolerance=1e-8, relative_tolerance=1e-7,
                gap_tolerance=1e-7, max_iterations=10000)
            self.assertEqual(result['status'], 'incomplete')
            report = json.loads((root / 'out/group_acceleration/report.json').read_text())
            self.assertIn('Exported float32 sequence failed', report['error'])
            self.assertFalse((root / 'out/group_acceleration/sequence.npz').exists())
            self.assertFalse((root / 'out/group_acceleration/certificate.npz').exists())

    def test_nonconvergence_retains_error_report_without_fake_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); case, digest = self.source(root)
            result = candidate.export_group_candidate(
                case, root / 'out', uid='fixture',
                expected_sequence_sha256=digest, metric_path=None,
                expected_metric_sha256=None, group_weight=.25, rho=1.,
                absolute_tolerance=1e-15, relative_tolerance=1e-15,
                gap_tolerance=1e-15, max_iterations=1)
            self.assertEqual(result['status'], 'incomplete')
            report = json.loads((root / 'out/group_acceleration/report.json').read_text())
            self.assertEqual(report['status'], 'error')
            self.assertEqual(report['exception_type'], 'RuntimeError')
            self.assertFalse((root / 'out/group_acceleration/sequence.npz').exists())
            self.assertFalse((root / 'out/group_acceleration/certificate.npz').exists())


if __name__ == '__main__':
    unittest.main()
