"""Local-only real staging coverage for dedicated native acceptance plans.

The small engineering capture exercises transport and staging, never native
scientific qualification. The dedicated acceptance test uses real retained
ActionMesh producer artifacts; this builder test never substitutes its fixture.
"""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from research_math import self_map_candidate as candidate
from research_math import c01_native_comparison as comparison
from research_math.tests.test_self_map_plan import engineering_context
import prepare_c01_native_acceptance as planner


class C01NativeAcceptancePlanTests(unittest.TestCase):
    def test_explicit_candidate_path_and_entire_closure_survive_real_staging(self):
        import run_experiments
        project = Path(__file__).resolve().parents[3]
        skill = Path(os.environ.get('RESEARCH_AUTOPILOT_SKILL_DIR',
                                     str(Path(run_experiments.__file__).resolve().parent.parent)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in planner.SOURCE_NAMES:
                destination = root / 'actionmesh' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(project / 'actionmesh' / relative, destination)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture',
                            '-c', 'user.email=fixture@example.invalid',
                            'commit', '--allow-empty', '-qm', 'acceptance fixture'], check=True)
            context = engineering_context(root)
            manifest = candidate.load(context / 'bundle-manifest.json')
            output = root / 'retained-artifacts'
            result = candidate.export_self_map_candidate(
                context / 'bundle-consumption.json', context / 'bundle-manifest.json',
                context / 'bundle-result.json',
                {row['path']: context / 'raw' / row['path'] for row in manifest['files']},
                output, expected_consumption_sha256=candidate.digest(context / 'bundle-consumption.json'),
                coordinate_bounds=[-1., 1.], bounds_policy='preserve_and_report')
            self.assertEqual(result['status'], 'completed', result)
            native, outer = planner.build_plans(
                root, skill_dir=skill, artifact_candidate=output / 'candidate.json',
                plan_dir=root / 'plans/native-acceptance', run_id='c01-acceptance-fixture',
                wall_seconds=300, ram_mib=4096)
            self.assertEqual(native['limits']['max_attempts'], 1)
            self.assertEqual(native['limits']['max_retries_per_trial'], 0)
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'], 0)
            self.assertEqual(outer['limits']['max_gpu_task_seconds'], 0)
            job = native['jobs'][0]
            self.assertEqual(job['command'][-1], str(output / 'candidate.json'))
            attempt = root / 'attempt'; attempt.mkdir()
            command, cwd, workspace = run_experiments._stage(root, job, attempt)
            staged_candidate = Path(command[-1])
            self.assertTrue(staged_candidate.is_relative_to(workspace))
            self.assertNotEqual(staged_candidate, output / 'candidate.json')
            # A real prospective freeze is an explicit staged input, separate
            # from the acceptance test's engineering-only comparison fixture.
            basis = root / 'basis.json'; basis.write_text('{"scope":"software fixture"}\n')
            source = output / 'context/raw/observed/sequence.npz'
            source_report = candidate.load(source.with_name('report.json'))
            decision = {
                'kind': 'c01-b-star-decision', 'version': 1,
                'candidate_id': candidate.CANDIDATE_ID, 'uid': source_report['uid'],
                'inference_seed': 42, 'decided_at': datetime.now(timezone.utc).isoformat(),
                'selected_role': 'b0', 'selected_method_id': 'native-b0',
                'selected_without_c01_native_outcomes': True,
                'selection_basis_refs': [planner.file_ref(root, basis)]}
            decision['decision_digest'] = comparison.canonical_digest(decision)
            decision_path = root / 'decision.json'; decision_path.write_text(json.dumps(decision))
            roles = [{
                'role': 'b0', 'method_id': 'native-b0',
                'sequence_ref': planner.file_ref(root, source),
                'report_ref': planner.file_ref(root, source.with_name('report.json')),
                'generation_identity_ref': planner.file_ref(
                    root, source.parent.parent / 'generation-identity.json'),
                'implementation_ref': planner.file_ref(root,
                    root / 'actionmesh/research_math/native_context_runner.py')},
                {'role': 'b_star', 'method_id': 'native-b0', 'alias_of': 'b0'}]
            for role in candidate.ROLES:
                report = candidate.load(output / role / 'report.json')
                roles.append({
                    'role': role, 'method_id': report['method_id'],
                    'report_ref': planner.file_ref(root, output / role / 'report.json'),
                    'sequence_ref': planner.file_ref(root, output / role / 'sequence.npz'),
                    'certificate_ref': planner.file_ref(root, output / 'certificate.npz'),
                    'implementation_ref': planner.file_ref(root,
                        root / 'actionmesh/research_math/self_map_candidate.py')})
            freeze = {
                'kind': 'c01-native-comparison-freeze', 'version': 1,
                'candidate_id': candidate.CANDIDATE_ID,
                'uid': source_report['uid'], 'inference_seed': 42, 'scoring_seed': 44,
                'primary_metric': 'cd_motion', 'guardrail_metrics': ['cd_3d', 'cd_4d'],
                'frozen_at': datetime.now(timezone.utc).isoformat(),
                'source_sequence_ref': roles[0]['sequence_ref'],
                'source_report_ref': roles[0]['report_ref'],
                'b_star_decision_ref': planner.file_ref(root, decision_path), 'roles': roles}
            freeze['freeze_digest'] = comparison.canonical_digest(freeze)
            freeze_path = root / 'prospective-freeze.json'; freeze_path.write_text(json.dumps(freeze))
            frozen_native, _ = planner.build_plans(
                root, skill_dir=skill, artifact_candidate=output / 'candidate.json',
                plan_dir=root / 'plans/prospective-acceptance', run_id='c01-freeze-fixture',
                wall_seconds=300, ram_mib=4096, freeze=freeze_path)
            frozen_job = frozen_native['jobs'][0]
            self.assertEqual(frozen_job['output_paths'], ['actionmesh/c01-comparison-request.json'])
            self.assertEqual(frozen_job['command'][-2:], [str(output / 'candidate.json'), str(freeze_path)])
            self.assertNotIn(str(output), frozen_job['command'][2])
            self.assertIn('Path(sys.argv[2])', frozen_job['command'][2])
            self.assertIn('if not result.wasSuccessful() or result.skipped', frozen_job['command'][2])
            ast.parse(frozen_job['command'][2])
            frozen_attempt = root / 'frozen-attempt'; frozen_attempt.mkdir()
            frozen_command, _, frozen_workspace = run_experiments._stage(
                root, frozen_job, frozen_attempt)
            for path in frozen_command[-2:]:
                self.assertTrue(Path(path).is_relative_to(frozen_workspace))
            prospective_refs = planner.prospective_freeze_refs(
                frozen_workspace, Path(frozen_command[-2]), Path(frozen_command[-1]))
            self.assertIn(planner.file_ref(frozen_workspace, frozen_workspace / 'basis.json'), prospective_refs)

            # Removing live producer/candidate data cannot alter staged inputs.
            shutil.rmtree(output)
            shutil.rmtree(context)
            staged_refs = planner.artifact_refs(workspace, staged_candidate)
            self.assertEqual({(ref['path'], ref['sha256']) for ref in staged_refs},
                             {(ref['path'], ref['sha256']) for ref in job['input_refs']})
            staged_report = staged_candidate.parent / 'mean_bias/report.json'
            changed = json.loads(staged_report.read_text())
            changed['sha256']['certificate.npz'] = '0' * 64
            staged_report.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'certificate metadata'):
                planner.artifact_refs(workspace, staged_candidate)


if __name__ == '__main__':
    unittest.main()
