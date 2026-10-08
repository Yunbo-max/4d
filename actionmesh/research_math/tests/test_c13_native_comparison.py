"""Engineering contracts for C13's frozen five-role native comparison.

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
    comparison = importlib.import_module('research_math.c13_native_comparison')
except ModuleNotFoundError:
    comparison = None


class C13NativeComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(comparison, 'C13 native comparison implementation is missing')
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.uid = 'fixture'
        self.seed = 42
        self.arrays = {
            'vertices': np.zeros((16, 4, 3), dtype=np.float32),
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
        }
        if name == 'group_acceleration':
            report.update(candidate_id=comparison.CANDIDATE_ID,
                          candidate_arm='group_acceleration')
            np.savez_compressed(directory / 'certificate.npz', dual=np.zeros((1, 1, 3)))
            report.setdefault('sha256', {})['certificate.npz'] = self.digest(
                directory / 'certificate.npz')
        if name == 'b_star':
            report['method_id'] = 'b_star'
        if status == 'completed':
            report.setdefault('sha256', {})['sequence.npz'] = self.digest(
                directory / 'sequence.npz')
        (directory / 'report.json').write_text(json.dumps(report))
        return directory

    def fixture(self, *, b_star_alias='gaussian'):
        source = self.root / 'source'; source.mkdir()
        np.savez_compressed(source / 'sequence.npz', **self.arrays)
        self.source_sequence_sha = self.digest(source / 'sequence.npz')
        source_report = {'status': 'completed', 'uid': self.uid, 'seed': self.seed,
                         'sha256': {'sequence.npz': self.source_sequence_sha}}
        (source / 'report.json').write_text(json.dumps(source_report))
        self.source_report_sha = self.digest(source / 'report.json')
        def move(frame, value):
            def mutate(arrays):
                arrays['vertices'][frame, 0, 0] = value
            return mutate
        arms = {
            'b0': source,
            'gaussian': self.write_arm('gaussian', role_name='world_gaussian',
                                       mutate=move(1, 1.)),
            'quadratic_acceleration': self.write_arm(
                'quadratic_acceleration', mutate=move(2, 2.)),
            'group_acceleration': self.write_arm('group_acceleration',
                                                 mutate=move(3, 3.)),
        }
        if b_star_alias is None:
            arms['b_star'] = self.write_arm('b_star', mutate=move(1, 1.))
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
            })
        decision = self.root / 'b-star-decision.json'
        basis = self.root / 'b-star-basis.json'
        basis.write_text(json.dumps({'scope': 'prospective engineering fixture'}))
        decision_value = {
            'kind': 'c13-b-star-decision', 'version': 1,
            'candidate_id': comparison.CANDIDATE_ID,
            'uid': self.uid, 'inference_seed': self.seed,
            'decided_at': '2026-10-08T00:00:00+00:00',
            'selected_role': b_star_alias or 'b_star',
            'selected_method_id': b_star_alias or 'b_star',
            'selected_without_c13_native_outcomes': True,
            'selection_basis_refs': [comparison.file_ref(self.root, basis)],
        }
        decision_value['decision_digest'] = comparison.canonical_digest(decision_value)
        decision.write_text(json.dumps(decision_value))
        freeze = {
            'kind': 'c13-native-comparison-freeze', 'version': 1,
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
        self.assertEqual(len(request['scoring_cases']), 4)
        self.assertEqual(request['role_to_case']['b_star'], request['role_to_case']['gaussian'])
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

    def test_physical_b_star_duplicate_of_later_control_requires_alias(self):
        freeze_path, _ = self.fixture(b_star_alias=None)
        with self.assertRaisesRegex(ValueError, 'must use explicit alias_of'):
            comparison.make_request(self.root, freeze_path=freeze_path)

    def test_changed_timeline_or_topology_is_rejected(self):
        freeze_path, freeze = self.fixture()
        quadratic = next(row for row in freeze['roles']
                         if row['role'] == 'quadratic_acceleration')
        sequence = comparison.resolve_ref(self.root, quadratic['sequence_ref'])
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        arrays['frame_indices'] = np.arange(16, dtype=np.int64)[::-1]
        np.savez_compressed(sequence, **arrays)
        quadratic['sequence_ref'] = comparison.file_ref(self.root, sequence)
        report_path = comparison.resolve_ref(self.root, quadratic['report_ref'])
        report = comparison.read_json(report_path)
        report['sha256']['sequence.npz'] = quadratic['sequence_ref']['sha256']
        report_path.write_text(json.dumps(report))
        quadratic['report_ref'] = comparison.file_ref(self.root, report_path)
        freeze['freeze_digest'] = comparison.canonical_digest(
            {key: value for key, value in freeze.items() if key != 'freeze_digest'})
        freeze_path.write_text(json.dumps(freeze))
        with self.assertRaises(ValueError):
            comparison.make_request(self.root, freeze_path=freeze_path)

    def test_changed_topology_is_rejected_even_when_report_rehashes_it(self):
        freeze_path, freeze = self.fixture()
        gaussian = next(row for row in freeze['roles'] if row['role'] == 'gaussian')
        sequence = comparison.resolve_ref(self.root, gaussian['sequence_ref'])
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        arrays['faces'] = arrays['faces'][:, ::-1]
        np.savez_compressed(sequence, **arrays)
        gaussian['sequence_ref'] = comparison.file_ref(self.root, sequence)
        report_path = comparison.resolve_ref(self.root, gaussian['report_ref'])
        report = comparison.read_json(report_path)
        report['sha256']['sequence.npz'] = gaussian['sequence_ref']['sha256']
        report_path.write_text(json.dumps(report))
        gaussian['report_ref'] = comparison.file_ref(self.root, report_path)
        freeze['freeze_digest'] = comparison.canonical_digest(
            {key: value for key, value in freeze.items() if key != 'freeze_digest'})
        freeze_path.write_text(json.dumps(freeze))
        with self.assertRaisesRegex(ValueError, 'topology'):
            comparison.make_request(self.root, freeze_path=freeze_path)

    def test_failed_physical_arm_is_retained_without_a_scoring_case(self):
        freeze_path, freeze = self.fixture(b_star_alias='gaussian')
        row = next(item for item in freeze['roles'] if item['role'] == 'quadratic_acceleration')
        report_path = comparison.resolve_ref(self.root, row['report_ref'])
        report = comparison.read_json(report_path)
        report['status'] = 'error'; report.pop('sha256', None)
        report_path.write_text(json.dumps(report))
        comparison.resolve_ref(self.root, row['sequence_ref']).unlink()
        row['report_ref'] = comparison.file_ref(self.root, report_path)
        row['sequence_ref'] = None
        freeze['freeze_digest'] = comparison.canonical_digest(
            {key: value for key, value in freeze.items() if key != 'freeze_digest'})
        freeze_path.write_text(json.dumps(freeze))
        request = comparison.make_request(self.root, freeze_path=freeze_path)
        quadratic = next(item for item in request['roles']
                         if item['role'] == 'quadratic_acceleration')
        self.assertEqual(quadratic['preparation_status'], 'error')
        self.assertIsNone(quadratic['case_id'])
        self.assertEqual(request['logical_denominator']['n_roles'], 5)
        denominator = {item['role']: item for item in
                       request['logical_denominator']['roles']}
        self.assertEqual(denominator['quadratic_acceleration']['preparation_status'],
                         'error')
        self.assertNotIn('quadratic_acceleration',
                         [item['role'] for item in request['scoring_cases']])

if __name__ == '__main__':
    unittest.main()
