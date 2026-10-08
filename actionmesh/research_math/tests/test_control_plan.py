"""Plan admission tests only; fixtures are not scientific evaluations."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    planner = importlib.import_module('prepare_mesh_controls')
    quadratic_planner = importlib.import_module('prepare_quadratic_acceleration_control')
    group_planner = importlib.import_module('prepare_group_acceleration_candidate')
    acceptance = importlib.import_module('prepare_control_scoring_checks')
except ModuleNotFoundError:
    planner = None
    quadratic_planner = None
    group_planner = None
    acceptance = None


class ControlPlanTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(planner, 'Baseline preparation plan builder missing')
        self.assertIsNotNone(quadratic_planner, 'Quadratic control plan builder missing')
        self.assertIsNotNone(group_planner, 'C13 group candidate plan builder missing')
        self.assertIsNotNone(acceptance, 'Acceptance plan builder missing')

    def test_acceptance_source_closure_includes_parity_finalizer(self):
        project_root = Path(__file__).resolve().parents[3]
        paths = {path.relative_to(project_root).as_posix()
                 for path in acceptance.acceptance_sources(project_root)}
        self.assertIn('actionmesh/finalize_actionbench_parity.py', paths)
        self.assertIn(
            'actionmesh/prepare_actionbench_parity_finalization.py', paths)
        self.assertIn(
            'actionmesh/research_math/tests/test_actionbench_parity.py', paths)
        self.assertIn(
            'actionmesh/prepare_actionbench_dataset_semantics.py', paths)
        self.assertIn(
            'actionmesh/research_math/tests/test_actionbench_dataset_semantics.py', paths)

    def test_acceptance_source_closure_includes_full128_root_imports(self):
        """Catch isolated acceptance omitting modules imported by window tests."""
        project_root = Path(__file__).resolve().parents[3]
        paths = {path.relative_to(project_root).as_posix()
                 for path in acceptance.acceptance_sources(project_root)}
        for path in (
                'actionmesh/prepare_actionbench_full128_window.py',
                'actionmesh/prepare_actionbench_active_batch_snapshot.py',
                'actionmesh/prepare_actionbench_active_batch_reconciliation.py'):
            self.assertIn(path, paths)

    def test_acceptance_stages_native_context_and_external_supervisor(self):
        """An isolated acceptance workspace must include the out-of-package CLI."""
        project_root = Path(__file__).resolve().parents[3]
        paths = {path.relative_to(project_root).as_posix()
                 for path in acceptance.acceptance_sources(project_root)}
        for path in (
                'scripts/research_supervisor.py',
                'actionmesh/prepare_native_context.py',
                'actionmesh/research_math/native_context_delivery.py',
                'actionmesh/research_math/native_context_runner.py',
                'actionmesh/research_math/pipeline_decoder_observer.py',
                'actionmesh/research_math/tests/test_research_supervisor.py'):
            self.assertIn(path, paths)

    def test_acceptance_stages_quadratic_control_and_plan(self):
        project_root = Path(__file__).resolve().parents[3]
        paths = {path.relative_to(project_root).as_posix()
                 for path in acceptance.acceptance_sources(project_root)}
        self.assertIn('actionmesh/prepare_quadratic_acceleration_control.py', paths)
        self.assertIn('actionmesh/research_math/quadratic_acceleration_control.py', paths)
        self.assertIn(
            'docs/research-math-20261006/longgoal-20261007/CANDIDATE_INPUT_AUDIT.json', paths)
        self.assertIn('actionmesh/prepare_native_context_consumption.py', paths)

    def test_acceptance_stages_real_g01_design_and_every_bound_source(self):
        project_root = Path(__file__).resolve().parents[3]
        paths = {path.relative_to(project_root).as_posix()
                 for path in acceptance.acceptance_sources(project_root)}
        required = {
            'docs/research-math-20261006/longgoal-20261007/G01_DESIGN.json',
            'docs/research-math-20261006/revisions/20261006-mechanism-boundaries/selection.json',
            'docs/research-math-20261006/evidence/exposure-and-contract.json',
            'actionmesh/research_overnight/assets/actionbench_population.json',
            'actionmesh/repo/actionbench/README.md',
            'actionmesh/repo/actionbench/benchmark.py',
            'actionmesh/repo/actionbench/chamfer.py',
            'actionmesh/repo/actionbench/evaluate_dataset.py',
            'actionmesh/repo/actionbench/icp.py',
            'actionmesh/repo/actionbench/sample_mesh.py',
            'actionmesh/repo/actionbench/sample_point_cloud.py',
            'docs/research-math-20261006/actionbench-full128-dataset-semantics-contract.json',
            'docs/research-math-20261006/actionbench-full128-reproduction-contract.json',
        }
        required.update(
            'docs/research-math-20261006/revisions/20261006-mechanism-boundaries/specs/'
            + name for name in (
                '4d-math-20261006-c01.json', '4d-math-20261006-c02.json',
                '4d-math-20261006-c03.json', '4d-math-20261006-c04.json',
                '4d-math-20261006-c05.json', '4d-math-20261006-c06.json',
                '4d-math-20261006-c07.json', '4d-math-20261006-c08.json',
                '4d-math-20261006-c10.json', '4d-math-20261006-c11.json',
                '4d-math-20261006-c12.json', '4d-math-20261006-c13.json',
                '4d-math-20261006-c14.json', '4d-math-20261006-c15.json',
                '4d-math-20261006-c20.json'))
        self.assertTrue(required.issubset(paths), sorted(required - paths))

    def test_acceptance_stages_c13_candidate_and_plan(self):
        project_root = Path(__file__).resolve().parents[3]
        paths = {path.relative_to(project_root).as_posix()
                 for path in acceptance.acceptance_sources(project_root)}
        self.assertIn('actionmesh/prepare_group_acceleration_candidate.py', paths)
        self.assertIn('actionmesh/research_math/group_acceleration_candidate.py', paths)
        self.assertIn(
            'actionmesh/research_math/tests/test_group_acceleration_candidate.py', paths)

    def test_delivery_inventory_attaches_quadratic_control_only_to_c13(self):
        project_root = Path(__file__).resolve().parents[3]
        audit = json.loads((project_root/'docs/research-math-20261006/longgoal-20261007/CANDIDATE_INPUT_AUDIT.json').read_text())
        candidates = {row['id'].rsplit('-', 1)[-1].upper(): row for row in audit['candidates']}
        self.assertEqual(candidates['C13']['strong_control_code_entry'],
            'actionmesh/research_math/quadratic_acceleration_control.py:export_quadratic_control')
        self.assertNotIn('strong_control_code_entry', candidates['C10'])
        self.assertFalse(candidates['C13']['full_method_source_complete'])

    def project(self, root):
        code = root/'actionmesh/research_math'; code.mkdir(parents=True)
        (code/'__init__.py').write_text('')
        (code/'simple_mesh_controls.py').write_text('fixture not executable\n')
        data = root/'inputs/case'; data.mkdir(parents=True)
        seq = data/'sequence.npz'; seq.write_bytes(b'engineering opaque input, no scorer')
        (data/'report.json').write_text(json.dumps({'status': 'completed', 'uid': 'fixture', 'seed': 42,
            'sha256': {'sequence.npz': hashlib.sha256(seq.read_bytes()).hexdigest()}}))
        return seq

    def test_plan_pins_both_source_files_and_cpu_only_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seq = self.project(root)
            native, outer = planner.build_plans(root, source_sequence=seq, run_id='fixture', sigma=1.,
                                                plan_dir=root/'plans', wall_seconds=120)
            paths = {x['path'] for x in native['jobs'][0]['input_refs']}
            self.assertEqual(paths, {'inputs/case/sequence.npz', 'inputs/case/report.json'})
            self.assertEqual(native['purpose'], 'engineering')
            self.assertIsNone(native['protocol_ref'])
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'], 0)
            self.assertIn(str(seq), native['jobs'][0]['command'])
            self.assertEqual(len(native['jobs'][0]['output_paths']), 9)

    def test_changed_source_report_blocks_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seq = self.project(root)
            seq.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                planner.build_plans(root, source_sequence=seq, run_id='fixture', sigma=1., plan_dir=root/'plans', wall_seconds=120)
            self.assertFalse((root/'plans').exists())

    def test_source_outside_project_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seq = self.project(root)
            other = root/'other'; other.mkdir()
            with self.assertRaises(ValueError):
                planner.build_plans(other, source_sequence=seq, run_id='fixture', sigma=1., plan_dir=other/'plans', wall_seconds=120)

    def test_plan_output_cannot_overwrite_existing_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seq = self.project(root)
            planner.build_plans(root, source_sequence=seq, run_id='fixture', sigma=1., plan_dir=root/'plans', wall_seconds=120)
            with self.assertRaises(FileExistsError):
                planner.build_plans(root, source_sequence=seq, run_id='fixture', sigma=1., plan_dir=root/'plans', wall_seconds=120)

    def test_quadratic_plan_is_single_cpu_arm_and_pins_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seq = self.project(root)
            (root/'actionmesh/research_math/quadratic_acceleration_control.py').write_text('fixture not executable\n')
            native, outer = quadratic_planner.build_plans(
                root, source_sequence=seq, run_id='quadratic-fixture', weight=2.,
                plan_dir=root/'quadratic-plans', wall_seconds=120)
            job = native['jobs'][0]
            self.assertEqual(native['purpose'], 'engineering')
            self.assertEqual(job['arm_role'], 'strong-simple-control-no-scorer')
            self.assertEqual({x['path'] for x in job['input_refs']},
                             {'inputs/case/sequence.npz', 'inputs/case/report.json'})
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'], 0)
            self.assertEqual(job['output_paths'], [
                'actionmesh/quadratic-control-output/controls.json',
                'actionmesh/quadratic-control-output/manifest.json',
                'actionmesh/quadratic-control-output/quadratic_acceleration/sequence.npz',
                'actionmesh/quadratic-control-output/quadratic_acceleration/report.json'])

    def test_group_candidate_plan_uses_identity_metric_and_zero_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seq = self.project(root)
            (root/'actionmesh/research_math/group_acceleration_candidate.py').write_text(
                'fixture not executable\n')
            (root/'actionmesh/research_math/quadratic_acceleration_control.py').write_text(
                'fixture not executable\n')
            native, outer = group_planner.build_plans(
                root, source_sequence=seq, observation_metric=None,
                run_id='c13-fixture', group_weight=.25, rho=1.,
                absolute_tolerance=1e-8, relative_tolerance=1e-7,
                gap_tolerance=1e-7, max_iterations=1000,
                plan_dir=root/'c13-plans', wall_seconds=600)
            job = native['jobs'][0]
            self.assertEqual(native['purpose'], 'engineering')
            self.assertEqual(job['arm_role'], 'candidate-artifact-no-scorer-no-admission')
            self.assertEqual({x['path'] for x in job['input_refs']},
                             {'inputs/case/sequence.npz', 'inputs/case/report.json'})
            self.assertIn('--identity-observation-metric', job['command'])
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'], 0)
            self.assertEqual(job['output_paths'], [
                'actionmesh/c13-group-output/candidate.json',
                'actionmesh/c13-group-output/manifest.json',
                'actionmesh/c13-group-output/group_acceleration/sequence.npz',
                'actionmesh/c13-group-output/group_acceleration/certificate.npz',
                'actionmesh/c13-group-output/group_acceleration/report.json'])

    def test_group_candidate_plan_hash_pins_explicit_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seq = self.project(root)
            (root/'actionmesh/research_math/group_acceleration_candidate.py').write_text(
                'fixture not executable\n')
            (root/'actionmesh/research_math/quadratic_acceleration_control.py').write_text(
                'fixture not executable\n')
            metric = root/'inputs/metric.npz'
            np.savez(metric, observation_metric=np.eye(16))
            native, _ = group_planner.build_plans(
                root, source_sequence=seq, observation_metric=metric,
                run_id='c13-metric-fixture', group_weight=.25, rho=1.,
                absolute_tolerance=1e-8, relative_tolerance=1e-7,
                gap_tolerance=1e-7, max_iterations=1000,
                plan_dir=root/'c13-metric-plans', wall_seconds=600)
            job = native['jobs'][0]
            refs = {x['path']: x['sha256'] for x in job['input_refs']}
            self.assertEqual(set(refs), {
                'inputs/case/sequence.npz', 'inputs/case/report.json',
                'inputs/metric.npz'})
            self.assertIn('--observation-metric', job['command'])
            self.assertIn('--expected-metric-sha256', job['command'])
            self.assertEqual(
                job['command'][job['command'].index('--expected-metric-sha256') + 1],
                refs['inputs/metric.npz'])

    def test_group_candidate_plan_keeps_outer_cpu_budget_within_work_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); seq = self.project(root)
            (root/'actionmesh/research_math/group_acceleration_candidate.py').write_text(
                'fixture not executable\n')
            (root/'actionmesh/research_math/quadratic_acceleration_control.py').write_text(
                'fixture not executable\n')
            with self.assertRaises(ValueError):
                group_planner.build_plans(
                    root, source_sequence=seq, observation_metric=None,
                    run_id='c13-over-budget', group_weight=.25, rho=1.,
                    absolute_tolerance=1e-8, relative_tolerance=1e-7,
                    gap_tolerance=1e-7, max_iterations=1000,
                    plan_dir=root/'c13-over-budget-plans', wall_seconds=27000)
            self.assertFalse((root/'c13-over-budget-plans').exists())


if __name__ == '__main__': unittest.main()
