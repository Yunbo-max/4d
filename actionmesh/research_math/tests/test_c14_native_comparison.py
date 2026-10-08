"""Engineering contracts for C14's frozen five-role native comparison.

These tests use tiny local byte/array fixtures. They never invoke ActionBench,
model inference, a scientific metric, or a GPU.
"""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


try:
    comparison = importlib.import_module('research_math.c14_native_comparison')
except ModuleNotFoundError:
    comparison = None


class C14NativeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(comparison, 'C14 native comparison implementation is missing')
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.uid = 'fixture'
        self.seed = 42
        base = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [0., 0., 1.]], dtype=np.float32)
        vertices = np.repeat(base[None], 16, axis=0)
        vertices[:, 3, 0] += np.linspace(0., .15, 16, dtype=np.float32)
        self.arrays = {
            'vertices': vertices,
            'faces': np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            'timesteps': np.arange(16, dtype=np.float32),
            'frame_indices': np.arange(16, dtype=np.int64),
            'query_vertex_ids': np.arange(4, dtype=np.int64),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def digest(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write_arm(self, name, *, role_name=None, status='completed', mutate=None):
        directory = self.root / 'arms' / name
        directory.mkdir(parents=True)
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        if mutate is not None:
            mutate(arrays)
        if status == 'completed':
            np.savez_compressed(directory / 'sequence.npz', **arrays)
        report = {
            'status': status, 'uid': self.uid, 'seed': self.seed,
            'baseline_arm': role_name or name,
            'source_sequence_sha256': self.source_sequence_sha,
            'source_report_sha256': self.source_report_sha,
            'implementation_sha256': self.digest(
                Path(comparison.controls_module.__file__)),
            'sigma_frames': 1.0,
        }
        if name == 'corotational_residual':
            factors = comparison.candidate_module.fit_frozen_rigid_factors(
                self.arrays['vertices'])
            parameters = {
                'weight': .1, 'rho': 1.0,
                'absolute_tolerance': 1e-7,
                'relative_tolerance': 1e-7,
                'max_iterations': 10000,
                'timestamp_units': ('supplied native sequence units; '
                                    'weight scales in those units')}
            repaired, solver = comparison.candidate_module.repair_body_residual(
                factors['body_residual'], self.arrays['timesteps'],
                weight=parameters['weight'], rho=parameters['rho'],
                absolute_tolerance=parameters['absolute_tolerance'],
                relative_tolerance=parameters['relative_tolerance'],
                max_iterations=parameters['max_iterations'])
            rebuilt = comparison.candidate_module.reconstruct_with_fixed_pose(
                factors['anchor'], repaired, factors['rotation_rows'],
                factors['centroids']).astype(np.float32)
            arrays['vertices'] = rebuilt
            np.savez_compressed(directory / 'sequence.npz', **arrays)
            report.update(
                candidate_id=comparison.CANDIDATE_ID,
                method_id=comparison.CANDIDATE_ID,
                arm_role=comparison.candidate_module.ARM,
                pose_source=comparison.POSE_SOURCE,
                pose_fit=comparison.POSE_FIT,
                repair_operator=comparison.REPAIR_OPERATOR,
                implementation_sha256=self.digest(
                    Path(comparison.candidate_module.__file__)),
                parameters=parameters, solver=solver)
            np.savez_compressed(
                directory / 'certificate.npz',
                anchor=factors['anchor'], rotation_rows=factors['rotation_rows'],
                centroids=factors['centroids'],
                singular_values=factors['singular_values'],
                fit_rms=factors['fit_rms'],
                observed_body_residual=factors['body_residual'],
                repaired_body_residual=repaired)
            report.setdefault('sha256', {})['certificate.npz'] = self.digest(
                directory / 'certificate.npz')
        if name == 'b_star':
            report['method_id'] = 'b_star'
        if name == 'body_gaussian' and status == 'completed':
            arrays['vertices'], poses = comparison.controls_module.smooth_body(
                self.arrays['vertices'], 1.0)
            np.savez_compressed(directory / 'sequence.npz', **arrays)
            np.savez_compressed(directory / 'poses.npz', **poses)
            report.setdefault('sha256', {})['poses.npz'] = self.digest(
                directory / 'poses.npz')
        if name == 'world_gaussian' and status == 'completed':
            arrays['vertices'] = comparison.controls_module.smooth_world(
                self.arrays['vertices'], 1.0)
            np.savez_compressed(directory / 'sequence.npz', **arrays)
        if status == 'completed':
            report.setdefault('sha256', {})['sequence.npz'] = self.digest(
                directory / 'sequence.npz')
        (directory / 'report.json').write_text(json.dumps(report))
        return directory

    def fixture(self, *, b_star_alias='world_gaussian'):
        source = self.root / 'source'; source.mkdir()
        np.savez_compressed(source / 'sequence.npz', **self.arrays)
        self.source_sequence_sha = self.digest(source / 'sequence.npz')
        source_report = {
            'status': 'completed', 'uid': self.uid, 'seed': self.seed,
            'baseline_arm': 'native', 'sigma_frames': 1.0,
            'implementation_sha256': self.digest(
                Path(comparison.controls_module.__file__)),
            'sha256': {'sequence.npz': self.source_sequence_sha}}
        (source / 'report.json').write_text(json.dumps(source_report))
        self.source_report_sha = self.digest(source / 'report.json')
        def move(frame, value):
            def mutate(arrays):
                arrays['vertices'][frame, 0, 0] = value
            return mutate
        arms = {
            'b0': source,
            'world_gaussian': self.write_arm('world_gaussian', role_name='world_gaussian',
                                       mutate=move(1, 1.)),
            'body_gaussian': self.write_arm(
                'body_gaussian', mutate=move(2, 2.)),
            'corotational_residual': self.write_arm('corotational_residual',
                                                 mutate=move(3, 3.)),
        }
        b0_implementation = self.root/'b0_impl.py'
        b0_implementation.write_bytes(
            Path(comparison.controls_module.__file__).read_bytes())
        if b_star_alias is None:
            arms['b_star'] = self.write_arm('b_star', mutate=move(1, 1.))
            (arms['b_star']/'sequence.npz').write_bytes(
                (arms['world_gaussian']/'sequence.npz').read_bytes())
            report = comparison.read_json(arms['b_star']/'report.json')
            report['sha256']['sequence.npz'] = self.digest(
                arms['b_star']/'sequence.npz')
            (arms['b_star']/'report.json').write_text(json.dumps(report))
        role_rows = []
        for role in comparison.ROLES:
            if role == 'b_star' and b_star_alias:
                role_rows.append({'role': role, 'method_id': b_star_alias,
                                  'alias_of': b_star_alias})
                continue
            directory = arms[role]
            role_rows.append({
                'role': role, 'method_id': role,
                'report_ref': comparison.file_ref(self.root, directory / 'report.json'),
                'sequence_ref': comparison.file_ref(self.root, directory / 'sequence.npz'),
                **({'implementation_ref': comparison.file_ref(
                    self.root, b0_implementation)}
                   if role == 'b0' else {}),
                **({'poses_ref': comparison.file_ref(
                    self.root, directory / 'poses.npz')}
                   if role == 'body_gaussian' else {}),
                **({'certificate_ref': comparison.file_ref(
                    self.root, directory / 'certificate.npz')}
                   if role == 'corotational_residual' else {}),
            })
        decision = self.root / 'b-star-decision.json'
        basis = self.root / 'b-star-basis.json'
        basis.write_text(json.dumps({'scope': 'prospective engineering fixture'}))
        decision_value = {
            'kind': 'c14-b-star-decision', 'version': 1,
            'candidate_id': comparison.CANDIDATE_ID,
            'uid': self.uid, 'inference_seed': self.seed,
            'decided_at': '2026-10-08T00:00:00+00:00',
            'selected_role': b_star_alias or 'b_star',
            'selected_method_id': b_star_alias or 'b_star',
            'selected_without_c14_native_outcomes': True,
            'selection_basis_refs': [comparison.file_ref(self.root, basis)],
        }
        decision_value['decision_digest'] = comparison.canonical_digest(decision_value)
        decision.write_text(json.dumps(decision_value))
        freeze = {
            'kind': 'c14-native-comparison-freeze', 'version': 1,
            'candidate_id': comparison.CANDIDATE_ID,
            'frozen_at': '2026-10-08T00:00:00+00:00',
            'uid': self.uid, 'inference_seed': self.seed, 'scoring_seed': 44,
            'primary_metric': 'cd_motion',
            'guardrail_metrics': ['cd_3d', 'cd_4d'],
            'roles': role_rows,
            'b_star_decision_ref': comparison.file_ref(self.root, decision),
            'source_sequence_ref': comparison.file_ref(self.root, source / 'sequence.npz'),
            'source_report_ref': comparison.file_ref(self.root, source / 'report.json'),
        }
        freeze['freeze_digest'] = comparison.canonical_digest(freeze)
        freeze_path = self.root / 'freeze.json'
        freeze_path.write_text(json.dumps(freeze))
        return freeze_path, freeze

    def test_freeze_keeps_five_roles_but_deduplicates_explicit_b_star_alias(self):
        freeze_path, freeze = self.fixture()
        request = comparison.make_request(self.root, freeze_path=freeze_path)
        self.assertEqual([row['role'] for row in request['roles']], list(comparison.ROLES))
        self.assertLessEqual(len(request['scoring_cases']), 4)
        self.assertEqual(request['role_to_case']['b_star'], request['role_to_case']['world_gaussian'])
        self.assertEqual(request['freeze_ref'], comparison.file_ref(self.root, freeze_path))
        self.assertEqual(request['b_star_decision_ref'], freeze['b_star_decision_ref'])

    def test_b_star_must_be_explicitly_frozen_before_scores_exist(self):
        freeze_path, freeze = self.fixture()
        freeze['roles'] = [row for row in freeze['roles'] if row['role'] != 'b_star']
        freeze['freeze_digest'] = comparison.canonical_digest(
            {key: value for key, value in freeze.items() if key != 'freeze_digest'})
        freeze_path.write_text(json.dumps(freeze))
        with self.assertRaisesRegex(ValueError, 'five ordered roles'):
            comparison.make_request(self.root, freeze_path=freeze_path)

    def test_alias_cannot_point_to_itself_or_unknown_role(self):
        freeze_path, freeze = self.fixture()
        freeze['roles'][1]['alias_of'] = 'b_star'
        freeze['freeze_digest'] = comparison.canonical_digest(
            {key: value for key, value in freeze.items() if key != 'freeze_digest'})
        freeze_path.write_text(json.dumps(freeze))
        with self.assertRaisesRegex(ValueError, 'simple control role'):
            comparison.make_request(self.root, freeze_path=freeze_path)

    def test_all_byte_identical_physical_roles_share_one_measurement(self):
        freeze_path, _ = self.fixture(b_star_alias=None)
        request = comparison.make_request(self.root, freeze_path=freeze_path)
        self.assertEqual(request['role_to_case']['b_star'],
                         request['role_to_case']['world_gaussian'])

    def test_changed_timeline_or_topology_is_rejected(self):
        freeze_path, freeze = self.fixture()
        body = next(row for row in freeze['roles']
                    if row['role'] == 'body_gaussian')
        sequence = comparison.resolve_ref(self.root, body['sequence_ref'])
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        arrays['frame_indices'] = np.arange(16, dtype=np.int64)[::-1]
        np.savez_compressed(sequence, **arrays)
        body['sequence_ref'] = comparison.file_ref(self.root, sequence)
        report_path = comparison.resolve_ref(self.root, body['report_ref'])
        report = comparison.read_json(report_path)
        report['sha256']['sequence.npz'] = body['sequence_ref']['sha256']
        report_path.write_text(json.dumps(report))
        body['report_ref'] = comparison.file_ref(self.root, report_path)
        freeze['freeze_digest'] = comparison.canonical_digest(
            {key: value for key, value in freeze.items() if key != 'freeze_digest'})
        freeze_path.write_text(json.dumps(freeze))
        with self.assertRaises(ValueError):
            comparison.make_request(self.root, freeze_path=freeze_path)

    def test_changed_topology_is_rejected_even_when_report_rehashes_it(self):
        freeze_path, freeze = self.fixture()
        world_gaussian = next(row for row in freeze['roles'] if row['role'] == 'world_gaussian')
        sequence = comparison.resolve_ref(self.root, world_gaussian['sequence_ref'])
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        arrays['faces'] = arrays['faces'][:, ::-1]
        np.savez_compressed(sequence, **arrays)
        world_gaussian['sequence_ref'] = comparison.file_ref(self.root, sequence)
        report_path = comparison.resolve_ref(self.root, world_gaussian['report_ref'])
        report = comparison.read_json(report_path)
        report['sha256']['sequence.npz'] = world_gaussian['sequence_ref']['sha256']
        report_path.write_text(json.dumps(report))
        world_gaussian['report_ref'] = comparison.file_ref(self.root, report_path)
        freeze['freeze_digest'] = comparison.canonical_digest(
            {key: value for key, value in freeze.items() if key != 'freeze_digest'})
        freeze_path.write_text(json.dumps(freeze))
        with self.assertRaisesRegex(ValueError, 'topology'):
            comparison.make_request(self.root, freeze_path=freeze_path)

    def test_failed_physical_arm_is_retained_without_a_scoring_case(self):
        freeze_path, freeze = self.fixture(b_star_alias='world_gaussian')
        row = next(item for item in freeze['roles'] if item['role'] == 'body_gaussian')
        report_path = comparison.resolve_ref(self.root, row['report_ref'])
        report = comparison.read_json(report_path)
        report['status'] = 'error'; report.pop('sha256', None)
        report['exception_type'] = 'RuntimeError'
        report['error'] = 'fixture preparation failure'
        report_path.write_text(json.dumps(report))
        comparison.resolve_ref(self.root, row['sequence_ref']).unlink()
        row['report_ref'] = comparison.file_ref(self.root, report_path)
        row['sequence_ref'] = None
        row['poses_ref'] = None
        freeze['freeze_digest'] = comparison.canonical_digest(
            {key: value for key, value in freeze.items() if key != 'freeze_digest'})
        freeze_path.write_text(json.dumps(freeze))
        request = comparison.make_request(self.root, freeze_path=freeze_path)
        body = next(item for item in request['roles']
                    if item['role'] == 'body_gaussian')
        self.assertEqual(body['preparation_status'], 'error')
        self.assertIsNone(body['case_id'])
        self.assertEqual(request['logical_denominator']['n_roles'], 5)
        denominator = {item['role']: item for item in
                       request['logical_denominator']['roles']}
        self.assertEqual(denominator['body_gaussian']['preparation_status'],
                         'error')
        self.assertNotIn('body_gaussian',
                         [item['role'] for item in request['scoring_cases']])

    def test_candidate_certificate_is_part_of_the_scoring_input_closure(self):
        freeze_path, _ = self.fixture()
        request = comparison.make_request(self.root, freeze_path=freeze_path)
        candidate = next(item for item in request['scoring_cases']
                         if item['role'] == 'corotational_residual')
        self.assertIn('certificate_ref', candidate)
        self.assertIn(candidate['certificate_ref'], request['input_refs'])

if __name__ == '__main__':
    unittest.main()
