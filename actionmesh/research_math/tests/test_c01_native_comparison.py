"""Local C01 comparison contracts and a real retained-context acceptance path.

Native acceptance is a separate, explicitly selected test. Set
C01_NATIVE_ARTIFACT to the current-version candidate artifact directory retained
by a completed common-harness CPU operation. A skipped native test is not
acceptance. The native test reads real producer tensors, no decoder mock or GT.
"""
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math import c01_native_comparison as comparison
from research_math import self_map_candidate as candidate


class C01ComparisonContracts(unittest.TestCase):
    def test_exact_native_anchor_and_topology_are_required(self):
        arrays = dict(vertices=np.zeros((16, 3, 3), dtype=np.float32),
                      faces=np.array([[0, 1, 2]], dtype=np.int64),
                      timesteps=np.arange(16, dtype=np.float32),
                      frame_indices=np.arange(16, dtype=np.int64),
                      query_vertex_ids=np.arange(3, dtype=np.int64))
        changed = {key: value.copy() for key, value in arrays.items()}
        changed['vertices'][0, 0, 0] = .125
        with self.assertRaisesRegex(ValueError, 'anchor'):
            comparison._same_native(arrays, changed)
        changed = {key: value.copy() for key, value in arrays.items()}
        changed['faces'] = np.array([[0, 2, 1]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, 'identity'):
            comparison._same_native(arrays, changed)

    def test_physical_content_digest_is_array_exact(self):
        arrays = {'vertices': np.arange(18, dtype=np.float32).reshape(2, 3, 3),
                  'faces': np.array([[0, 1, 2]], dtype=np.int64)}
        reordered = dict(reversed(list(arrays.items())))
        self.assertEqual(comparison._sequence_content_digest(arrays),
                         comparison._sequence_content_digest(reordered))
        changed = {key: value.copy() for key, value in arrays.items()}
        changed['vertices'][1, 0, 0] += 1.
        self.assertNotEqual(comparison._sequence_content_digest(arrays),
                            comparison._sequence_content_digest(changed))

    def test_b_star_cannot_be_selected_after_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            basis = root / 'basis.json'; basis.write_text('{}\n')
            decision = {
                'kind': 'c01-b-star-decision', 'version': 1,
                'candidate_id': candidate.CANDIDATE_ID, 'uid': 'fixture',
                'inference_seed': 42, 'decided_at': '2026-10-09T00:00:00Z',
                'selected_role': 'b0', 'selected_method_id': 'native-b0',
                'selected_without_c01_native_outcomes': True,
                'selection_basis_refs': [comparison.file_ref(root, basis)],
            }
            decision['decision_digest'] = comparison.canonical_digest(decision)
            path = root / 'decision.json'; path.write_text(json.dumps(decision))
            freeze = {'uid': 'fixture', 'inference_seed': 42,
                      'frozen_at': '2026-10-08T00:00:00Z',
                      'b_star_decision_ref': comparison.file_ref(root, path)}
            with self.assertRaisesRegex(ValueError, 'precede freeze'):
                comparison._decision(root, freeze)


@unittest.skipUnless(os.environ.get('C01_NATIVE_ARTIFACT'),
                     'requires retained real current-version C01 native artifact; not acceptance')
class C01RetainedNativeComparison(unittest.TestCase):
    def test_full_native_artifact_freeze_recompute_and_request_tamper(self):
        root = Path(__file__).resolve().parents[3]
        output = Path(os.environ['C01_NATIVE_ARTIFACT']).resolve()
        output.relative_to(root)
        verified = candidate.verify_candidate_artifacts(output)
        source = verified['source_sequence']
        source_report = comparison.read_json(verified['source_report'])
        with tempfile.TemporaryDirectory(dir=root) as directory:
            workspace = Path(directory)
            def save(name, value):
                path = workspace / name
                path.write_text(json.dumps(value, allow_nan=False) + '\n')
                return comparison.file_ref(root, path)
            decision = {
                'kind': 'c01-b-star-decision', 'version': 1,
                'candidate_id': candidate.CANDIDATE_ID,
                'uid': source_report['uid'], 'inference_seed': source_report['seed'],
                'decided_at': datetime.now(timezone.utc).isoformat(),
                'selected_role': 'b0', 'selected_method_id': 'native-b0',
                'selected_without_c01_native_outcomes': True,
                'selection_basis_refs': [save('software-test-basis.json', {
                    'scope': 'engineering comparison acceptance only; no scientific B* selection'})],
            }
            decision['decision_digest'] = comparison.canonical_digest(decision)
            source_ref = comparison.file_ref(root, source)
            report_ref = comparison.file_ref(root, verified['source_report'])
            rows = [{
                'role': 'b0', 'method_id': 'native-b0', 'sequence_ref': source_ref,
                'report_ref': report_ref,
                'implementation_ref': comparison.file_ref(
                    root, root / 'actionmesh/research_math/native_context_runner.py'),
                'generation_identity_ref': comparison.file_ref(
                    root, source.parent.parent / 'generation-identity.json'),
            }, {'role': 'b_star', 'method_id': 'native-b0', 'alias_of': 'b0'}]
            for role in candidate.ROLES:
                report = comparison.read_json(output / role / 'report.json')
                rows.append({
                    'role': role, 'method_id': report['method_id'],
                    'report_ref': comparison.file_ref(root, output / role / 'report.json'),
                    'sequence_ref': (comparison.file_ref(root, verified['arms'][role])
                                     if report['status'] == 'completed' else None),
                    'implementation_ref': comparison.file_ref(root, Path(candidate.__file__)),
                    'certificate_ref': comparison.file_ref(root, verified['certificate']),
                })
            freeze = {
                'kind': 'c01-native-comparison-freeze', 'version': 1,
                'candidate_id': candidate.CANDIDATE_ID,
                'uid': source_report['uid'], 'inference_seed': source_report['seed'],
                'scoring_seed': 44, 'primary_metric': 'cd_motion',
                'guardrail_metrics': ['cd_3d', 'cd_4d'],
                'frozen_at': datetime.now(timezone.utc).isoformat(),
                'b_star_decision_ref': save('decision.json', decision),
                'source_sequence_ref': source_ref, 'source_report_ref': report_ref,
                'roles': rows,
            }
            freeze['freeze_digest'] = comparison.canonical_digest(freeze)
            ref = save('freeze.json', freeze)
            request = comparison.make_request(root, freeze_path=root / ref['path'])
            self.assertEqual(request['logical_denominator']['n_roles'], 5)
            self.assertFalse(request['dispatch_ready'])
            self.assertLessEqual(len(request['scoring_cases']), 4)
            pinned = {item['path'] for item in request['input_refs']}
            for path in verified['context_files']:
                self.assertIn(path.relative_to(root).as_posix(), pinned)
            for row in rows:
                if row.get('sequence_ref', True) is None:
                    logical = next(item for item in request['roles'] if item['role'] == row['role'])
                    self.assertEqual(logical['preparation_status'], 'error')
                    self.assertIsNone(logical['case_id'])
            # Exercise the actual installed staging boundary across distinct
            # roots with every real context file declared, never a fake mapper.
            import run_experiments
            request_ref = save('request.json', request)
            attempt = workspace / 'attempt'; attempt.mkdir()
            code_paths = [root / 'actionmesh/research_math' / name for name in (
                '__init__.py', 'c01_native_comparison.py', 'self_map_candidate.py',
                'native_context_delivery.py', 'native_context_runner.py',
                'pipeline_decoder_observer.py', 'decoder_observer.py',
                'complete_unit_export.py')]
            job = {
                'trial_id': 'c01-real-comparison-staging', 'cwd': 'actionmesh',
                'command': [sys.executable, '-m', 'research_math.c01_native_comparison',
                            'request', '--root', '..', '--freeze', str(root / ref['path']),
                            '--output', 'staged-comparison.json'],
                'input_refs': [*request['input_refs'], request_ref],
                'code_refs': [comparison.file_ref(root, path) for path in code_paths],
                'output_paths': ['actionmesh/staged-comparison.json'], 'seed': 42,
            }
            command, _, staged = run_experiments._stage(root, job, attempt)
            staged_freeze = Path(command[command.index('--freeze') + 1])
            staged_freeze.relative_to(staged)
            reconstructed = comparison.make_request(staged, freeze_path=staged_freeze)
            self.assertEqual(reconstructed, request)
            comparison.verify_request(staged, reconstructed)
            changed = dict(request, primary_metric='cd_3d')
            changed['request_digest'] = comparison.canonical_digest({
                key: value for key, value in changed.items() if key != 'request_digest'})
            with self.assertRaisesRegex(ValueError, 'differs'):
                comparison.verify_request(root, changed)


if __name__ == '__main__':
    unittest.main()
