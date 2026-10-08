"""Plan admission tests only; fixtures are not scientific evaluations."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    planner = importlib.import_module('prepare_mesh_controls')
    acceptance = importlib.import_module('prepare_control_scoring_checks')
except ModuleNotFoundError:
    planner = None
    acceptance = None


class ControlPlanTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(planner, 'Baseline preparation plan builder missing')
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


if __name__ == '__main__': unittest.main()
