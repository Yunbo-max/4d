"""Engineering contracts for C13 official scoring and raw collection.

The tests use local byte/JSON fixtures and never invoke ActionBench, a model,
the scientific scorer, or a GPU.  Web authors these tests but does not run
project tests; Local must execute them through the common harness.
"""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


try:
    scoring = importlib.import_module('research_math.c13_native_scoring')
except ModuleNotFoundError:
    scoring = None


class C13NativeScoringTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(scoring, 'C13 native scoring implementation is missing')
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(json.dumps(value))
        return path

    def gpu_record(self):
        stdout = 'GPU-fixture, Fixture GPU, 11264\n'
        stderr = ''
        return {
            'kind': 'c13-gpu-identity-observation', 'version': 1,
            'gpu_uuid': 'GPU-fixture', 'name': 'Fixture GPU',
            'memory_total_mib': 11264.0,
            'cuda_visible_devices': 'GPU-fixture',
            'command': [
                'nvidia-smi', '--id=GPU-fixture',
                '--query-gpu=uuid,name,memory.total',
                '--format=csv,noheader,nounits'],
            'exit_code': 0, 'stdout': stdout, 'stderr': stderr,
            'stdout_sha256': hashlib.sha256(stdout.encode()).hexdigest(),
            'stderr_sha256': hashlib.sha256(stderr.encode()).hexdigest(),
        }

    def comparison(self):
        roles = [
            {'role': 'b0', 'method_id': 'b0', 'case_id': 'u-b0',
             'preparation_status': 'completed'},
            {'role': 'b_star', 'method_id': 'gaussian', 'alias_of': 'gaussian',
             'case_id': 'u-gaussian', 'preparation_status': 'completed'},
            {'role': 'gaussian', 'method_id': 'gaussian', 'case_id': 'u-gaussian',
             'preparation_status': 'completed'},
            {'role': 'quadratic_acceleration', 'method_id': 'quadratic',
             'case_id': 'u-quadratic', 'preparation_status': 'completed'},
            {'role': 'group_acceleration', 'method_id': 'group',
             'case_id': 'u-group', 'preparation_status': 'completed'},
        ]
        return {
            'kind': 'c13-native-comparison-request', 'version': 1,
            'candidate_id': '4d-math-20261006-c13', 'uid': 'u',
            'inference_seed': 42, 'scoring_seed': 44,
            'primary_metric': 'cd_motion',
            'guardrail_metrics': ['cd_3d', 'cd_4d'],
            'roles': roles,
            'role_to_case': {row['role']: row['case_id'] for row in roles},
            'logical_denominator': {'n_roles': 5},
            'scoring_cases': [], 'request_digest': 'b' * 64,
        }

    def official(self, failed=()):
        values = {
            'u-b0': (5.0, 6.0, 7.0),
            'u-gaussian': (4.0, 5.0, 6.0),
            'u-quadratic': (3.0, 4.0, 5.0),
            'u-group': (2.0, 3.0, 4.0),
        }
        rows = []
        for case_id, metrics in values.items():
            if case_id in failed:
                rows.append({'case_id': case_id, 'uid': 'u', 'status': 'error',
                             'error': 'fixture failure'})
            else:
                rows.append({'case_id': case_id, 'uid': 'u', 'status': 'success',
                             'n_frames': 16, 'cd_3d': metrics[0],
                             'cd_4d': metrics[1], 'cd_motion': metrics[2]})
        return {'cases': rows}

    def test_logical_readout_expands_alias_without_duplicate_measurement(self):
        readout = scoring.build_logical_readout(self.comparison(), self.official())
        self.assertEqual(readout['logical_denominator']['n_roles'], 5)
        self.assertEqual(len(readout['roles']), 5)
        b_star = next(row for row in readout['roles'] if row['role'] == 'b_star')
        gaussian = next(row for row in readout['roles'] if row['role'] == 'gaussian')
        self.assertEqual(b_star['metrics'], gaussian['metrics'])
        self.assertEqual(b_star['shared_measurement_with'], 'gaussian')
        self.assertEqual(readout['unique_physical_measurements'], 4)
        primary = next(row for row in readout['contrasts']
                       if row['control_role'] == 'b_star')
        self.assertEqual(primary['deltas']['cd_motion'], -2.0)
        self.assertEqual(primary['direction'], 'group minus control; lower is better')
        self.assertIsNone(readout['confidence_intervals'])
        self.assertEqual(readout['scientific_verdict'], 'not_computed')

    def test_failures_remain_in_denominator_and_are_never_zero_imputed(self):
        comparison = self.comparison()
        quadratic = next(row for row in comparison['roles']
                         if row['role'] == 'quadratic_acceleration')
        quadratic.update(preparation_status='error', case_id=None,
                         preparation_error='solver stopped')
        comparison['role_to_case']['quadratic_acceleration'] = None
        readout = scoring.build_logical_readout(
            comparison, self.official(failed={'u-group'}))
        rows = {row['role']: row for row in readout['roles']}
        self.assertEqual(rows['quadratic_acceleration']['status'], 'preparation_error')
        self.assertNotIn('metrics', rows['quadratic_acceleration'])
        self.assertEqual(rows['group_acceleration']['status'], 'scoring_error')
        self.assertNotIn('metrics', rows['group_acceleration'])
        self.assertEqual(readout['n_successful_roles'], 3)
        self.assertEqual(readout['contrasts'], [])

    def test_stage_cases_copies_exact_unique_physical_inputs(self):
        comparison = self.comparison()
        cases = []
        for role in ('b0', 'gaussian', 'quadratic_acceleration', 'group_acceleration'):
            report = self.write(f'inputs/{role}/report.json',
                                {'uid': 'u', 'status': 'completed'})
            sequence = self.write(f'inputs/{role}/sequence.npz', role.encode())
            cases.append({'role': role, 'method_id': role, 'case_id': f'u-{role}'.replace(
                'quadratic_acceleration', 'quadratic').replace('group_acceleration', 'group'),
                'preparation_status': 'completed',
                'report_ref': scoring.file_ref(self.root, report),
                'sequence_ref': scoring.file_ref(self.root, sequence)})
        comparison['scoring_cases'] = cases
        raw = self.root / 'raw'; raw.mkdir()
        manifest = scoring.stage_cases(self.root, comparison, raw)
        self.assertEqual(len(manifest['cases']), 4)
        self.assertEqual([row['case_id'] for row in manifest['cases']],
                         ['u-b0', 'u-gaussian', 'u-quadratic', 'u-group'])
        self.assertEqual((raw/'cases/u-gaussian/sequence.npz').read_bytes(), b'gaussian')
        self.assertFalse((raw/'cases/u-b_star').exists())

    def test_validate_official_report_rejects_changed_input_hash(self):
        comparison = self.comparison()
        report = self.write('inputs/group/report.json',
                            {'uid': 'u', 'status': 'completed'})
        sequence = self.write('inputs/group/sequence.npz', b'group')
        comparison['scoring_cases'] = [{
            'role': 'group_acceleration', 'method_id': 'group', 'case_id': 'u-group',
            'preparation_status': 'completed',
            'report_ref': scoring.file_ref(self.root, report),
            'sequence_ref': scoring.file_ref(self.root, sequence),
        }]
        raw = self.root/'raw'; raw.mkdir()
        manifest = scoring.stage_cases(self.root, comparison, raw)
        manifest_path = raw/'manifest.json'
        scoring.write_json(manifest_path, manifest)
        ground_truth = self.write('gt/u/surfaces.npy', b'ground truth')
        source_names = ('benchmark.py', 'chamfer.py', 'icp.py', 'sample_mesh.py',
                        'sample_point_cloud.py', 'evaluate_dataset.py')
        source_paths = {}
        for name in source_names:
            content = (b'devices=[verts.device], enabled=True' if name == 'sample_mesh.py'
                       else name.encode())
            source_paths[name] = self.write('repo/actionbench/' + name, content)
        source_hashes = {name: self.sha(path) for name, path in source_paths.items()}
        patched_hashes = dict(source_hashes)
        patched_hashes['sample_mesh.py'] = hashlib.sha256(
            b'devices=([verts.device] if verts.is_cuda else []), enabled=True').hexdigest()
        for name in source_names:
            content = (b'devices=([verts.device] if verts.is_cuda else []), enabled=True'
                       if name == 'sample_mesh.py' else source_paths[name].read_bytes())
            self.write('raw/official-scores.json.official/official-source/' + name,
                       content)
        self.write('raw/gpu-identity.json', self.gpu_record())
        official = {
            'schema_version': 1, 'device': 'cuda:0', 'seed': 44,
            'adapter_sha256': 'b' * 64,
            'denominator': {'frozen': True, 'manifest': str(manifest_path.resolve()),
                            'manifest_sha256': self.sha(manifest_path), 'n_declared': 1},
            'official_source': {
                'original_sha256': source_hashes,
                'patched_source_sha256': patched_hashes,
                'compatibility_patch': {
                    'file': 'sample_mesh.py',
                    'function': 'get_baryc_sampling_mesh',
                    'old': 'devices=[verts.device], enabled=True',
                    'new': 'devices=([verts.device] if verts.is_cuda else []), enabled=True',
                    'reason': ('CPU surface sampler must not query CUDA RNG state '
                               'for a CPU device'),
                    'metric_or_draw_change': False}},
            'cases': [{'case_id': 'u-group', 'uid': 'u', 'status': 'success',
                       'case_dir': str((raw/'cases/u-group').resolve()), 'n_frames': 16,
                       'cd_3d': 1.0, 'cd_4d': 2.0, 'cd_motion': 3.0,
                       'inputs': {
                           'sequence': {'path': str((raw/'cases/u-group/sequence.npz').resolve()),
                                        'sha256': '0' * 64},
                           'ground_truth': {'path': str(ground_truth.resolve()),
                                            'sha256': self.sha(ground_truth)}},
                       'official_outputs': {}}],
            'summary': {'n_total': 1, 'n_success': 1, 'n_failed': 0,
                        'success_rate': 1.0},
        }
        request = {'uid': 'u', 'scoring_seed': 44, 'device': 'cuda:0',
                   'comparison_request_digest': comparison['request_digest'],
                   'comparison_ref': {'path': 'comparison.json', 'sha256': 'a' * 64},
                   'ground_truth_ref': scoring.file_ref(self.root, ground_truth),
                   'adapter_ref': {'path': 'adapter.py', 'sha256': 'b' * 64},
                   'official_source_refs': [scoring.file_ref(self.root, source_paths[name])
                                            for name in source_names],
                   'runtime_policy': 'strict-cuda-forward-upstream-cpu-knn-backward-v1'}
        with self.assertRaisesRegex(ValueError, 'sequence'):
            scoring.validate_official_report(
                self.root, request, comparison, raw, official,
                adapter_exit_code=0, gpu_uuid='GPU-fixture')

    def test_raw_bundle_is_deterministic_and_tamper_evident(self):
        archives = []
        for index in range(2):
            output = self.root/f'out{index}'; output.mkdir()
            raw = output/'raw'; raw.mkdir()
            self.write(f'out{index}/raw/a.json', {'value': 1})
            self.write(f'out{index}/raw/nested/b.log', b'evidence\n')
            self.write(f'out{index}/raw/comparison-request.json', self.comparison())
            self.write(f'out{index}/raw/scoring-request.json', {
                'request_digest': 'a' * 64,
                'comparison_request_digest': 'b' * 64,
                'ground_truth_ref': {
                    'path': 'gt/u/surfaces.npy', 'sha256': 'c' * 64},
            })
            self.write(f'out{index}/raw/gpu-identity.json', self.gpu_record())
            self.write(f'out{index}/raw/adapter-execution.json', {
                'exit_code': 1, 'error': 'fixture adapter failure',
                'gpu_uuid': 'GPU-fixture',
            })
            bundle = scoring.finalize_raw_bundle(
                raw, output, request_digest='a' * 64,
                comparison_request_digest='b' * 64,
                excluded_ground_truth_ref={'path': 'gt/u/surfaces.npy', 'sha256': 'c' * 64})
            archives.append(output/'raw-evidence.tar')
            self.assertEqual(bundle['raw_file_count'], 6)
        self.assertEqual(self.sha(archives[0]), self.sha(archives[1]))
        validation_error = 'fixture validation failure'
        readout = scoring.build_logical_readout(self.comparison(), {'cases': []})
        readout['evidence_validation_error'] = validation_error
        result = {
            'kind': 'c13-native-scoring-result', 'version': 1,
            'candidate_id': '4d-math-20261006-c13', 'uid': 'u',
            'status': 'failed_validation',
            'gpu_uuid': 'GPU-fixture',
            'request_digest': 'a' * 64,
            'comparison_request_digest': 'b' * 64,
            'ground_truth_ref': {'path': 'gt/u/surfaces.npy', 'sha256': 'c' * 64},
            'adapter_exit_code': 1, 'adapter_error': 'fixture adapter failure',
            'readout': readout, 'validation_error': validation_error,
            'native_qualified': False,
            'scientific_effect_qualification': False,
            'scientific_verdict': 'not_computed', 'dispatch_ready': False,
            'raw_bundle': {
                'archive': json.loads((self.root/'out0/raw-manifest.json').read_text())[
                    'bundle_ref']['archive'],
                'manifest': {
                    'path': 'raw-manifest.json',
                    'sha256': self.sha(self.root/'out0/raw-manifest.json')},
                'raw_file_count': 6,
            },
            'scope': ('One frozen C13 physical scoring pass and receipt-bound raw '
                      'collection; no confidence interval, gate, qualification or verdict'),
        }
        result_path = self.write('out0/result.json', result)
        validated = scoring.validate_delivery(
            result_path, self.root/'out0/raw-manifest.json', archives[0],
            expected_result_sha256=self.sha(result_path),
            expected_manifest_sha256=self.sha(self.root/'out0/raw-manifest.json'),
            expected_archive_sha256=self.sha(archives[0]),
            expected_request_digest='a' * 64)
        self.assertEqual(validated['status'], 'validated_unqualified')
        archives[0].write_bytes(archives[0].read_bytes() + b'tamper')
        with self.assertRaises(ValueError):
            scoring.validate_delivery(
                result_path, self.root/'out0/raw-manifest.json', archives[0],
                expected_result_sha256=self.sha(result_path),
                expected_manifest_sha256=self.sha(self.root/'out0/raw-manifest.json'),
                expected_archive_sha256=('0' * 64),
                expected_request_digest='a' * 64)

    def test_raw_bundle_rejects_links(self):
        output = self.root/'linked'; output.mkdir()
        raw = output/'raw'; raw.mkdir()
        target = self.write('target.bin', b'x')
        (raw/'linked.bin').symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'regular'):
            scoring.finalize_raw_bundle(
                raw, output, request_digest='a' * 64,
                comparison_request_digest='b' * 64,
                excluded_ground_truth_ref={'path': 'gt/u/surfaces.npy', 'sha256': 'c' * 64})


if __name__ == '__main__':
    unittest.main()
