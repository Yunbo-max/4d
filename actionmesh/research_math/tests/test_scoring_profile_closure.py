"""Authored engineering regressions; run only through Local's CPU harness.

These tests import isolated source snapshots and validate admission records.
They never build/launch a scientific plan, score meshes, or use a GPU.  Web has
not executed them.  The protocol digest comes from the installed harness.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


class ScoringProfileClosureTests(unittest.TestCase):
    def test_scorer_and_launcher_import_from_only_prescribed_sources(self):
        repository = Path(__file__).resolve().parents[3]
        for profile in ('c01', 'c02', 'c06', 'c11', 'c14'):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as directory:
                planner = importlib.import_module('prepare_' + profile + '_native_scoring')
                staged = Path(directory)
                # These are the shared builder's fixed code refs plus the
                # profile's real declarations. C14's frozen arm implementation
                # refs supply its two method modules; no other profile needs them.
                sources = {
                    'actionmesh/research_math/__init__.py',
                    'actionmesh/official_actionbench_adapter.py',
                    'actionmesh/deterministic_actionbench_entry.py',
                    'actionmesh/research_math/deterministic_knn.py',
                    planner.COMPARISON_SOURCE, planner.SCORING_SOURCE,
                    planner.LAUNCHER_SOURCE, *planner.EXTRA_CODE_SOURCES,
                }
                if profile == 'c14':
                    sources.update({
                        'actionmesh/research_math/corotational_residual_candidate.py',
                        'actionmesh/research_math/simple_mesh_controls.py',
                    })
                for relative in sources:
                    target = staged / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(repository / relative, target)
                script = r'''
import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
profile = sys.argv[2]
sys.path.insert(0, str(root / 'actionmesh'))
scorer = importlib.import_module('research_math.' + profile + '_native_scoring')
launcher = importlib.import_module('launch_' + profile + '_native_scoring')
entry = importlib.import_module('deterministic_actionbench_entry')
backend = importlib.import_module('research_math.deterministic_knn')
assert scorer.REQUEST_KIND == profile + '-native-scoring-request'
assert launcher.CANDIDATE_ID == '4d-math-20261006-' + profile
assert callable(scorer.validate_delivery) and callable(launcher.launch)
assert callable(backend.cpu_knn_backward)
if profile not in ('c11', 'c14'):
    assert 'research_math.c14_native_comparison' not in sys.modules
    assert 'research_math.corotational_residual_candidate' not in sys.modules
for name, module in tuple(sys.modules.items()):
    if name.startswith(('research_math', 'research_ten', 'prepare_', 'launch_',
                        'official_actionbench', 'deterministic_actionbench')):
        filename = getattr(module, '__file__', None)
        if filename:
            pathlib.Path(filename).resolve().relative_to(root)
print('isolated-import-ok')
'''
                environment = dict(os.environ)
                environment.pop('PYTHONPATH', None)
                completed = subprocess.run(
                    [sys.executable, '-I', '-c', script, str(staged), profile],
                    cwd=staged, env=environment, text=True, capture_output=True,
                    timeout=30, check=False)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout.strip(), 'isolated-import-ok')

    def _admission_fixture(self, root, planner, installed_core, *, mutate_core=False):
        """Real hash graph; unrelated Gate 0/IPCG schemas are fixture-scoped.

        The proxy below rehashes those two records. Their complete schema
        acceptance belongs to the installed harness's separate acceptance suite;
        this test exercises the actual full C01 admission binding implementation.
        """
        scoring = planner.scoring
        timestamp = '2000-01-01T00:00:00+00:00'
        candidate = planner.CANDIDATE_ID

        def pinned(name, value, kind=None, digest_key=None):
            value = deepcopy(value)
            if kind:
                value.update(kind=kind, version=1)
            if digest_key:
                value[digest_key] = planner.canonical_record_digest(value, digest_key)
            path = root / ('inputs/' + name + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
            return scoring.file_ref(root, path)

        method_ref = pinned('method', {'fixture': 'method'})
        environment_ref = pinned('environment', {'fixture': 'environment'})
        source_ref = pinned('source', {'families': {'dev': 'a', 'uid': 'b'}})
        gate_ref = pinned('gate', {
            'schema_id': 'natural-gate-0', 'schema_version': '1.0.0',
            'outcome': 'PASS', 'parent_problem_ref': source_ref})
        importance_ref = pinned('importance', {
            'schema_id': 'importance-decision', 'schema_version': '1.0.0',
            'outcome': 'CLEAR', 'natural_gate_0_ref': gate_ref,
            'parent_problem_ref': source_ref})
        family_map = {'dev': 'a', 'uid': 'b'}
        derivation_ref = pinned('derivation', {
            'candidate_id': candidate, 'benchmark_revision': 'fixture-v1',
            'family_by_uid': family_map, 'source_refs': [source_ref],
            'assignment_rule': 'retained fixture family identifiers',
            'reviewed': True, 'derived_at': timestamp},
            planner.FAMILY_DERIVATION_KIND, 'derivation_digest')
        review_ref = pinned('review', {
            'candidate_id': candidate, 'derivation_ref': derivation_ref,
            'outcome': 'verified', 'checks': [
                'source_identity', 'mapping_reproduction', 'split_independence'],
            'reviewer': 'fixture-independent-reviewer', 'reviewed_at': timestamp},
            planner.FAMILY_REVIEW_KIND, 'review_digest')
        assignments_ref = pinned('assignments', {
            'candidate_id': candidate, 'benchmark_revision': 'fixture-v1',
            'family_by_uid': family_map, 'source_refs': [source_ref],
            'derivation_ref': derivation_ref,
            'assignment_rule': 'retained fixture family identifiers',
            'review_ref': review_ref, 'assigned_at': timestamp},
            planner.FAMILY_ASSIGNMENTS_KIND, 'assignment_digest')
        family_ref = pinned('family', {
            'candidate_id': candidate, 'benchmark_revision': 'fixture-v1',
            'development_ids': ['dev'], 'confirmation_ids': ['uid'],
            'family_by_uid': family_map, 'family_assignments_ref': assignments_ref,
            'frozen_at': timestamp}, planner.FAMILY_SPLIT_KIND, 'split_digest')
        paired = {'unit': 'asset_family', 'pairing': 'same_uid_same_source',
                  'method': 'paired-bootstrap', 'confidence_level': .95}
        multiplicity = {'method': 'holm', 'family': ['cd_motion', 'cd_3d', 'cd_4d']}
        criteria = [
            {'metric': 'cd_motion', 'direction': 'minimize', 'min_effect': .1,
             'paired_statistics': paired, 'multiplicity': multiplicity},
            *({'metric': metric, 'direction': 'minimize',
               'noninferiority_margin': .01, 'paired_statistics': paired,
               'multiplicity': multiplicity} for metric in ('cd_3d', 'cd_4d')),
        ]
        protocol = {
            'protocol_id': 'fixture-c01', 'criteria': criteria,
            'method_discovery': {'candidate_id': candidate},
            'seed_policy': {'seeds': [42], 'allow_extra': False},
            'required_groups': ['confirmation'],
            'evidence_mode': 'prospective_confirmatory',
        }
        analysis_ref = pinned('analysis', {
            'candidate_id': candidate,
            'protocol_core_digest': planner.analysis_protocol_core_digest(protocol),
            'primary_metric': 'cd_motion', 'direction': 'minimize',
            'min_effect': .1, 'guardrail_margins': {'cd_3d': .01, 'cd_4d': .01},
            'independent_unit': 'asset_family',
            'failure_policy': 'all_frozen_roles_and_assets_in_denominator_no_zero_imputation',
            'paired_statistics': paired, 'multiplicity': multiplicity,
            'sensitivity': [{'name': 'source-query-roundoff', 'values': [0., 1e-6]}],
            'development_confirmation': {
                'development_ids': ['dev'], 'confirmation_ids': ['uid'],
                'confirmation_locked': True},
            'protocol_criteria': criteria, 'frozen_at': timestamp},
            planner.ANALYSIS_KIND, 'plan_digest')
        # This order was impossible when the analysis required the final digest.
        protocol['analysis_plan_ref'] = analysis_ref
        if mutate_core:
            protocol['seed_policy']['seeds'] = [43]
        protocol['frozen_at'] = timestamp
        protocol['protocol_digest'] = installed_core.protocol_hash(protocol)
        protocol_ref = pinned('protocol', protocol)
        request = {'uid': 'uid', 'benchmark_revision': 'fixture-v1',
                   'comparison_ref': source_ref}
        request_ref = pinned('request', request)
        criteria_ref = pinned('criteria', {
            'candidate_id': candidate, 'comparison_digest': source_ref['sha256'],
            'protocol_digest': protocol['protocol_digest'],
            'analysis_plan_ref': analysis_ref, 'primary_metric': 'cd_motion',
            'guardrail_metrics': ['cd_3d', 'cd_4d'], 'min_effect': .1,
            'noninferiority_margins': {'cd_3d': .01, 'cd_4d': .01},
            'independent_unit': 'asset_family', 'frozen_at': timestamp},
            planner.CRITERIA_KIND, 'criteria_digest')
        now = datetime.now(timezone.utc)
        authorization_ref = pinned('authorization', {
            'candidate_id': candidate, 'uid': 'uid', 'gpu_uuid': 'GPU-fixture',
            'protocol_digest': protocol['protocol_digest'],
            'comparison_digest': source_ref['sha256'], 'request_ref': request_ref,
            'method_batch_ref': method_ref, 'environment_ref': environment_ref,
            'run_id': 'fixture-run', 'task_id': planner.TASK_ID,
            'group': 'confirmation', 'contract_digest': 'a' * 64,
            'scope': planner.AUTHORIZATION_SCOPE, 'max_gpu_task_seconds': 600,
            'status': 'authorized', 'authorization_nonce': 'b' * 64,
            'stop_acknowledgement': 'explicit_resume_for_exact_attempt',
            'authorized_at': (now - timedelta(minutes=1)).isoformat(),
            'expires_at': (now + timedelta(hours=1)).isoformat()},
            planner.AUTHORIZATION_KIND, 'authorization_digest')
        admission_ref = pinned('admission', {
            'candidate_id': candidate, 'uid': 'uid',
            'benchmark_revision': 'fixture-v1', 'comparison_digest': source_ref['sha256'],
            'protocol_ref': protocol_ref, 'method_batch_ref': method_ref,
            'natural_gate_0_ref': gate_ref, 'importance_decision_ref': importance_ref,
            'family_split_ref': family_ref, 'outcome_criteria_ref': criteria_ref,
            'gpu_resume_authorization_ref': authorization_ref,
            'status': 'admitted', 'admitted_at': now.isoformat()},
            planner.ADMISSION_KIND, 'admission_digest')
        return dict(
            root=root, admission_path=root / admission_ref['path'], request=request,
            protocol_ref=protocol_ref, protocol=protocol, request_ref=request_ref,
            method_batch_ref=method_ref, environment_ref=environment_ref,
            run_id='fixture-run', group='confirmation', contract_digest='a' * 64,
            gpu_uuid='GPU-fixture', wall_seconds=600,
            core=SimpleNamespace(verify_ref=lambda project, ref, schema:
                                 scoring.resolve_ref(project, ref)))

    def test_acyclic_installed_protocol_freeze_and_rehashed_core_tamper(self):
        import _autoresearch as installed_core
        import prepare_c01_native_scoring as planner

        with tempfile.TemporaryDirectory() as directory:
            arguments = self._admission_fixture(Path(directory), planner, installed_core)
            self.assertEqual(arguments['protocol']['protocol_digest'],
                             installed_core.protocol_hash(arguments['protocol']))
            result = planner.require_dispatch_admission(**arguments)
            self.assertEqual(result[3]['status'], 'authorized')
        with tempfile.TemporaryDirectory() as directory:
            arguments = self._admission_fixture(
                Path(directory), planner, installed_core, mutate_core=True)
            # Every outer hash, criteria and authorization has been regenerated.
            # Only the analysis's old core binding exposes the changed seed.
            with self.assertRaisesRegex(ValueError, 'G01 analysis plan'):
                planner.require_dispatch_admission(**arguments)


if __name__ == '__main__':
    unittest.main()
