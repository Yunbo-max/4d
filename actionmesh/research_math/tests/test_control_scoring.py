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

    def native_output_fixture(self):
        """Parser contract data only; never invoke any metric or benchmark."""
        arm = 'native'; uid = 'fixture'
        paths = {'manifest': 'controls/manifest.json', 'sequence': 'controls/native/sequence.npz',
                 'generation_report': 'controls/native/report.json', 'ground_truth': 'gt/fixture/surfaces.npy'}
        refs = {}
        for name, relative in paths.items():
            path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'parser-only'); refs[name] = scoring.file_ref(self.root, path)
        code = []
        for name in scoring.census.OFFICIAL_FILES:
            path = self.root/'actionmesh/repo/actionbench'/name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'official-source-identity-fixture')
            code.append(scoring.file_ref(self.root, path))
        stage = self.root/'stage'; stage.mkdir()
        scoring.census.write_json(stage/'manifest.json', {'cases': [{'case_id': uid+'-'+arm, 'uid': uid, 'case_dir': arm}]})
        request = {'uid': uid, 'manifest_ref': refs['manifest'], 'ground_truth_ref': refs['ground_truth'],
                   'repo_root': 'actionmesh/repo', 'code_refs': code,
                   'native_protocol': {**scoring.census.PROTOCOL, 'sampling_seed': 44},
                   'arms': [{'arm': arm, 'preparation_status': 'completed',
                             'sequence_ref': refs['sequence'], 'report_ref': refs['generation_report']}]}
        raw = {'schema_version': 1, 'protocol': copy.deepcopy(request['native_protocol']), 'device': 'cuda:0',
               'evaluator_sha256': scoring.census.digest(Path(scoring.census.__file__)),
               'official_source': {'source_root': str(self.root/'actionmesh/repo'),
                    'sha256': {Path(r['path']).name: r['sha256'] for r in code}},
               'denominator': {'frozen': True, 'manifest': str(stage/'manifest.json'),
                    'manifest_sha256': scoring.census.digest(stage/'manifest.json'), 'n_declared': 1},
               'cases': [{'case_id': uid+'-'+arm, 'uid': uid, 'case_dir': str(self.root/'controls/native'),
                    'status': 'success', 'cd_3d': 1., 'cd_4d': 2., 'cd_motion': 3.,
                    'inputs': {k: {'path': str(self.root/r['path']), 'sha256': r['sha256']}
                               for k, r in refs.items() if k != 'manifest'}}]}
        return request, stage, raw

    def validate_fixture_output(self, request, stage, raw, exit_code=0):
        return scoring.validate_native_output(self.root, request, 'native', stage, 'cuda:0', raw, exit_code)

    def test_bound_native_output_is_only_a_parser_acceptance(self):
        request, stage, raw = self.native_output_fixture()
        row = self.validate_fixture_output(request, stage, raw)
        self.assertEqual(row['uid'], 'fixture')

    def test_raw_output_wrong_uid_is_rejected_even_with_right_case_id(self):
        request, stage, raw = self.native_output_fixture(); raw['cases'][0]['uid'] = 'other'
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_changed_sampling_is_rejected(self):
        request, stage, raw = self.native_output_fixture(); raw['protocol']['n_pts_chamfer'] = 100
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_wrong_sampling_seed_is_rejected(self):
        request, stage, raw = self.native_output_fixture(); raw['protocol']['sampling_seed'] = 42
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_changed_evaluator_hash_is_rejected(self):
        request, stage, raw = self.native_output_fixture(); raw['evaluator_sha256'] = '0'*64
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_changed_official_source_hash_is_rejected(self):
        request, stage, raw = self.native_output_fixture(); raw['official_source']['sha256']['benchmark.py'] = '0'*64
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_missing_official_source_is_rejected_for_success(self):
        request, stage, raw = self.native_output_fixture(); raw['official_source'] = None
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_changed_sequence_hash_is_rejected(self):
        request, stage, raw = self.native_output_fixture(); raw['cases'][0]['inputs']['sequence']['sha256'] = '0'*64
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_changed_gt_hash_is_rejected(self):
        request, stage, raw = self.native_output_fixture(); raw['cases'][0]['inputs']['ground_truth']['sha256'] = '0'*64
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_missing_generator_report_is_rejected_for_success(self):
        request, stage, raw = self.native_output_fixture(); del raw['cases'][0]['inputs']['generation_report']
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_changed_manifest_hash_is_rejected(self):
        request, stage, raw = self.native_output_fixture(); raw['denominator']['manifest_sha256'] = '0'*64
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_wrong_device_is_rejected(self):
        request, stage, raw = self.native_output_fixture(); raw['device'] = 'cpu'
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_nonzero_success_exit_is_rejected(self):
        request, stage, raw = self.native_output_fixture()
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw, exit_code=1)

    def test_raw_output_invalid_metric_cannot_be_success(self):
        request, stage, raw = self.native_output_fixture(); raw['cases'][0]['cd_motion'] = float('nan')
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_raw_output_failed_preparation_cannot_score_as_success(self):
        request, stage, raw = self.native_output_fixture()
        request['arms'][0].update(preparation_status='error', sequence_ref=None)
        with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_native_backend_error_remains_an_error_with_declared_asset(self):
        request, stage, raw = self.native_output_fixture()
        raw['official_source'] = None
        raw['cases'][0] = {k: v for k, v in raw['cases'][0].items() if k in ('case_id', 'uid', 'case_dir')}
        raw['cases'][0].update(status='error', error='backend unavailable')
        row = self.validate_fixture_output(request, stage, raw, exit_code=1)
        self.assertEqual(row['status'], 'error')
        self.assertEqual(row['error'], 'backend unavailable')

    def test_raw_output_structural_and_path_mismatches_are_rejected(self):
        request, stage, original = self.native_output_fixture()
        mutations = [
            (('schema_version',), True), (('schema_version',), 2),
            (('denominator', 'frozen'), False), (('denominator', 'n_declared'), True),
            (('denominator', 'manifest'), str(stage/'other-manifest.json')),
            (('cases',), []), (('cases',), original['cases']*2),
            (('cases', 0, 'case_dir'), str(self.root/'controls/body_gaussian')),
            (('cases', 0, 'case_id'), 'fixture-body_gaussian'),
            (('official_source', 'source_root'), str(self.root/'other-repo')),
            (('cases', 0, 'inputs', 'sequence', 'path'), str(self.root/'other-sequence.npz')),
        ]
        for path, value in mutations:
            with self.subTest(path=path, value=value):
                raw = copy.deepcopy(original); target = raw
                for key in path[:-1]: target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError): self.validate_fixture_output(request, stage, raw)

    def test_nonescaping_symlink_refs_bind_canonical_scorer_paths(self):
        request, stage, raw = self.native_output_fixture()
        (self.root/'alias-controls').symlink_to(self.root/'controls', target_is_directory=True)
        request['manifest_ref']['path'] = 'alias-controls/manifest.json'
        request['arms'][0]['sequence_ref']['path'] = 'alias-controls/native/sequence.npz'
        request['arms'][0]['report_ref']['path'] = 'alias-controls/native/report.json'
        row = self.validate_fixture_output(request, stage, raw)
        self.assertEqual(row['status'], 'success')


if __name__ == '__main__': unittest.main()
