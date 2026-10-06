"""Planned Local engineering checks; fixture values are never benchmark evidence."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math import simple_mesh_controls as controls

try:
    scoring = importlib.import_module('research_math.control_scoring')
except ModuleNotFoundError:
    scoring = None


class ControlScoringContracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(scoring, 'Generated scoring adapter is absent')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_ref_rejects_path_escape(self):
        with self.assertRaises(ValueError):
            scoring.resolve_ref(self.root, {'path': '../outside', 'sha256': '0'*64})

    def test_ref_rejects_changed_bytes(self):
        path = self.root/'input'; path.write_bytes(b'one')
        ref = scoring.file_ref(self.root, path); path.write_bytes(b'two')
        with self.assertRaises(ValueError): scoring.resolve_ref(self.root, ref)

    def test_ref_rejects_absolute_path(self):
        with self.assertRaises(ValueError):
            scoring.resolve_ref(self.root, {'path': str(self.root/'input'), 'sha256': '0'*64})

    def test_arm_ids_cannot_be_counted_as_assets(self):
        rows = {arm: {'status': 'success', 'cd_3d': 1., 'cd_4d': 2., 'cd_motion': 3.}
                for arm in scoring.ARMS}
        result = scoring.paired_readout(rows)
        self.assertEqual(result['n_assets'], 1)
        self.assertEqual(result['n_arms'], 3)
        self.assertIsNone(result['confidence_interval'])
        self.assertFalse(result['candidate_methods_tested'])

    def test_missing_arm_retains_declared_denominator(self):
        result = scoring.paired_readout({'native': {'status': 'error'}})
        self.assertEqual(result['n_arms'], 3)
        self.assertEqual(result['n_complete_assets'], 0)
        self.assertEqual(result['complete_pair_deltas'], {})
        self.assertEqual(set(result['failed_or_missing_arms']), set(scoring.ARMS))

    def test_deltas_are_control_minus_native(self):
        rows = {arm: {'status': 'success', 'cd_3d': float(i+1),
                     'cd_4d': float(i+1), 'cd_motion': float(i+1)}
                for i, arm in enumerate(scoring.ARMS)}
        result = scoring.paired_readout(rows)
        self.assertEqual(result['complete_pair_deltas']['world_gaussian']['cd_motion'], 1.)
        self.assertEqual(result['complete_pair_deltas']['body_gaussian']['cd_motion'], 2.)

    def test_nonfinite_success_is_not_a_complete_pair(self):
        rows = {arm: {'status': 'success', 'cd_3d': 1., 'cd_4d': 1., 'cd_motion': 1.}
                for arm in scoring.ARMS}
        rows['body_gaussian']['cd_motion'] = float('nan')
        self.assertEqual(scoring.paired_readout(rows)['n_complete_assets'], 0)

    def test_replay_disagreement_is_not_widened_into_a_pass(self):
        a = {'status': 'success', 'cd_3d': 1., 'cd_4d': 2., 'cd_motion': 3.}
        b = {**a, 'cd_motion': 3.0000001}
        self.assertEqual(scoring.replay_comparison(a, b)['status'], 'mismatch')

    def test_successful_identical_replay_has_no_scientific_gate_effect(self):
        a = {'status': 'success', 'cd_3d': 1., 'cd_4d': 2., 'cd_motion': 3.}
        result = scoring.replay_comparison(a, a)
        self.assertEqual(result['status'], 'identical')
        self.assertFalse(result['native_contract_qualified'])

    def test_error_twice_is_not_successful_replay(self):
        self.assertEqual(scoring.replay_comparison({'status': 'error'}, {'status': 'error'})['status'], 'unscored')

    def test_negative_score_cannot_be_a_successful_pair(self):
        self.assertFalse(scoring.valid_score({'status': 'success', 'cd_3d': -1., 'cd_4d': 1., 'cd_motion': 1.}))

    def test_boolean_metric_is_rejected(self):
        self.assertFalse(scoring.valid_score({'status': 'success', 'cd_3d': True, 'cd_4d': 1., 'cd_motion': 1.}))

    def test_telemetry_requires_physical_device_uuid(self):
        with self.assertRaises(ValueError): scoring.DeviceSamples('0', self.root/'telemetry.jsonl')

    def test_initial_record_keeps_all_six_unattempted_passes(self):
        records = scoring.initial_arm_records()
        self.assertEqual(set(records), set(scoring.ARMS))
        self.assertTrue(all(len(r['passes']) == 2 for r in records.values()))
        self.assertEqual(scoring.paired_readout({a: r['passes'][0] for a, r in records.items()})['n_complete_assets'], 0)

    def test_pass_records_do_not_alias_other_arms(self):
        records = scoring.initial_arm_records()
        records['native']['passes'][0]['status'] = 'error'
        self.assertEqual(records['body_gaussian']['passes'][0]['status'], 'pending')

    def dependency_request(self):
        fields = ('source_sequence_ref', 'source_report_ref', 'control_summary_ref',
                  'manifest_ref', 'ground_truth_ref', 'population_ref', 'body_pose_ref')
        request = {key: {'path': key, 'sha256': '0'*64} for key in fields}
        request['arms'] = [{'arm': arm, 'preparation_status': 'completed',
            'report_ref': {'path': arm+'-report', 'sha256': '0'*64},
            'sequence_ref': {'path': arm+'-sequence', 'sha256': '0'*64}} for arm in scoring.ARMS]
        request['input_refs'] = [request[key] for key in fields]
        request['input_refs'] += [ref for arm in request['arms'] for ref in (arm['report_ref'], arm['sequence_ref'])]
        return request

    def test_missing_source_report_dependency_is_rejected(self):
        request = self.dependency_request()
        request['input_refs'].remove(request['source_report_ref'])
        with self.assertRaises(ValueError): scoring.require_dependency_closure(request)

    def test_missing_body_pose_dependency_is_rejected(self):
        request = self.dependency_request(); del request['body_pose_ref']
        with self.assertRaises(ValueError): scoring.require_dependency_closure(request)

    def test_missing_summary_dependency_is_rejected(self):
        request = self.dependency_request(); del request['control_summary_ref']
        with self.assertRaises(ValueError): scoring.require_dependency_closure(request)

    def test_complete_dependency_inventory_is_accepted_as_inventory_only(self):
        scoring.require_dependency_closure(self.dependency_request())

    def test_different_contract_sampling_is_rejected_before_dispatch(self):
        with self.assertRaises(ValueError):
            scoring.check_contract_binding(self.root, {'native_protocol': {}}, {'sampling': {'policy': 'changed', 'parameters': {}}})

    def test_sparse_contract_budget_is_rejected(self):
        parameters = {'n_pts_chamfer': 100000, 'n_pts_icp': 10000, 'icp_initial_rotations': 24, 'icp_iterations': 200}
        contract = {'sampling': {'policy': 'official-actionbench-full-sequence', 'parameters': parameters},
                    'budget': {**parameters, 'n_pts_chamfer': 100}}
        with self.assertRaises(ValueError): scoring.check_contract_binding(self.root, {'native_protocol': parameters}, contract)

    def arrays(self):
        v = np.zeros((16, 4, 3), dtype=np.float32)
        v[:, 1, 0] = 1.; v[:, 2, 1] = 1.; v[:, 3, 2] = 1.
        return {'vertices': v, 'faces': np.array([[0, 1, 2], [0, 1, 3]]),
                'frame_indices': np.arange(16), 'timesteps': np.arange(16),
                'query_vertex_ids': np.arange(4)}

    def test_exact_full_timeline_and_topology_accepted(self):
        a = self.arrays(); b = copy.deepcopy(a); b['vertices'][8] += .1
        scoring.check_arm_arrays(a, b)

    def test_anchor_change_rejected(self):
        a = self.arrays(); b = copy.deepcopy(a); b['vertices'][0, 0, 0] = .1
        with self.assertRaises(ValueError): scoring.check_arm_arrays(a, b)

    def test_topology_change_rejected(self):
        a = self.arrays(); b = copy.deepcopy(a); b['faces'] = b['faces'][:, ::-1]
        with self.assertRaises(ValueError): scoring.check_arm_arrays(a, b)

    def test_metadata_drop_rejected(self):
        a = self.arrays(); b = copy.deepcopy(a); del b['query_vertex_ids']
        with self.assertRaises(ValueError): scoring.check_arm_arrays(a, b)

    def test_extra_native_array_change_rejected(self):
        a = self.arrays(); a['feature'] = np.array([1.]); b = copy.deepcopy(a); b['feature'][0] = 2.
        with self.assertRaises(ValueError): scoring.check_arm_arrays(a, b)

    def test_fractional_frame_indices_rejected(self):
        a = self.arrays(); a['frame_indices'] = np.arange(16, dtype=float)
        with self.assertRaises(ValueError): scoring.check_arm_arrays(a, a)

    def test_three_arm_manifest_requires_exact_case_mapping(self):
        valid = {'cases': [{'case_id': 'fixture-'+a, 'uid': 'fixture', 'case_dir': a} for a in scoring.ARMS]}
        scoring.check_manifest(valid, 'fixture')
        invalid = copy.deepcopy(valid); invalid['cases'][1]['case_dir'] = '../native'
        with self.assertRaises(ValueError): scoring.check_manifest(invalid, 'fixture')

    def test_bundle_excludes_ground_truth_but_keeps_all_raw_arms(self):
        files = []
        for name in ('native/sequence.npz', 'world_gaussian/sequence.npz', 'body_gaussian/report.json', 'gt/surfaces.npy'):
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'fixture')
            files.append(scoring.file_ref(self.root, path))
        request = {'input_refs': files, 'code_refs': [], 'ground_truth_ref': files[-1]}
        result = scoring.bundle_refs(request)
        self.assertEqual(len(result), 3)
        self.assertNotIn(files[-1], result)

    def test_input_bundle_is_available_before_scoring(self):
        source = self.root/'native/sequence.npz'; source.parent.mkdir(); source.write_bytes(b'engineering')
        gt = self.root/'gt/surfaces.npy'; gt.parent.mkdir(); gt.write_bytes(b'labels')
        ref = scoring.file_ref(self.root, source)
        request = {'input_refs': [ref, scoring.file_ref(self.root, gt)], 'code_refs': [],
                   'ground_truth_ref': scoring.file_ref(self.root, gt)}
        output = self.root/'output'; output.mkdir()
        self.assertEqual(scoring.copy_evidence_bundle(self.root, request, output), [])
        self.assertEqual((output/'bundle/native/sequence.npz').read_bytes(), b'engineering')
        self.assertFalse((output/'bundle/gt/surfaces.npy').exists())

    def test_missing_input_is_retained_as_transport_failure(self):
        source = self.root/'input'; source.write_bytes(b'engineering')
        ref = scoring.file_ref(self.root, source); source.unlink()
        request = {'input_refs': [ref], 'code_refs': [], 'ground_truth_ref': {'path': 'gt', 'sha256': '0'*64}}
        output = self.root/'output'; output.mkdir()
        missing = scoring.copy_evidence_bundle(self.root, request, output)
        self.assertEqual(missing[0]['ref'], ref)
        self.assertEqual(json.loads((output/'bundle-manifest.json').read_text())['status'], 'incomplete_transport')

    def test_manifest_extra_arm_is_rejected(self):
        cases = [{'case_id': 'fixture-'+a, 'uid': 'fixture', 'case_dir': a} for a in scoring.ARMS]
        cases.append({'case_id': 'fixture-extra', 'uid': 'fixture', 'case_dir': 'extra'})
        with self.assertRaises(ValueError): scoring.check_manifest({'cases': cases}, 'fixture')

    def test_command_runs_existing_scorer_with_full_native_options(self):
        manifest = self.root/'controls/manifest.json'; manifest.parent.mkdir(); manifest.write_text('{}')
        gt = self.root/'gt/fixture/surfaces.npy'; gt.parent.mkdir(parents=True); gt.write_bytes(b'engineering')
        request = {'manifest_ref': scoring.file_ref(self.root, manifest),
                   'ground_truth_ref': scoring.file_ref(self.root, gt), 'repo_root': 'actionmesh/repo'}
        command = scoring.scorer_command(self.root, request, 'native', self.root/'stage', 'cuda:0')
        self.assertTrue(command[1].endswith('research_census_eval.py'))
        self.assertEqual(command[command.index('--seed')+1], '44')
        self.assertNotIn('--validate-only', command)
        self.assertNotIn('--disable-cpu-rng-fix', command)
        self.assertEqual(command[command.index('--repo-root')+1], str(self.root/'actionmesh/repo'))


if __name__ == '__main__': unittest.main()
