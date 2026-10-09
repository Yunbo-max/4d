"""Unexecuted Local plan/staging integration using the actual installed harness.

No model, scoring, or GPU execution. Fixture JSON tests plan plumbing only.
The actual native staging function is deliberately not mocked.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


class NativeContextPlanTests(unittest.TestCase):
    def fixture(self, root):
        from research_math.tests.test_native_context_runner import NativeContextRunnerTests
        project = Path(__file__).resolve().parents[3]
        package = root / 'actionmesh/research_math'
        package.mkdir(parents=True)
        for path in (project / 'actionmesh/research_math').glob('*.py'):
            shutil.copy2(path, package / path.name)
        for name in ('official_actionbench_adapter.py', 'research_census_eval.py',
                     'deterministic_actionbench_entry.py', 'prepare_native_context.py'):
            shutil.copy2(project / 'actionmesh' / name, root / 'actionmesh' / name)
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit',
                        '--allow-empty', '-qm', 'engineering fixture'], check=True)
        inputs = root / 'inputs'; inputs.mkdir()
        paths = {}
        for name in ('contract', 'population', 'snapshot_contract', 'snapshot_admission',
                     'dataset_semantics', 'unit_manifest'):
            paths[name] = inputs / (name + '.json')
            paths[name].write_text('{}')
        for name, status in (('snapshot_admission', 'admitted_engineering_snapshot'),
                             ('dataset_semantics', 'admitted_engineering_dataset_semantics')):
            paths[name].write_text(json.dumps({'status': status}))
        paths['unit_manifest'].write_text(json.dumps({
            'status': 'frozen_engineering_current_release_unit',
            'generation': NativeContextRunnerTests().generation(),
            'calibration_unit': {'uid': 'engineering-fixture'},
            'population': {'revision': 'd' * 40}}))
        runtime = inputs / 'native-runtime'; runtime.mkdir()
        packages = {name: 'fixture' for name in ('numpy', 'torch', 'trimesh', 'scipy', 'pytorch3d')}
        common = {'python_executable': sys.executable, 'python_version': '3.12',
                  'python_prefix': '/env', 'conda_prefix': '/conda', 'packages': packages}
        dependency = dict(common, kind='installed-native-dependency-inventory', version='1.0.0',
                          scope='fixture', conda_packages=[{'name': 'python'}])
        lock = runtime / 'dependencies.json'; lock.write_text(json.dumps(dependency))
        environment = dict(common, execution_mode='native_host', gpu_uuid='GPU-fixture',
            dependency_lock_refs=[{'path': 'inputs/native-runtime/dependencies.json',
                                  'sha256': hashlib.sha256(lock.read_bytes()).hexdigest()}],
            captured_at='2026-10-08T00:00:00Z', gpu_identity_source='fixture',
            gpu_identity_verified=False, native_contract_qualified=False, scope='fixture')
        paths['environment'] = runtime / 'environment.json'
        paths['environment'].write_text(json.dumps(environment))
        for name in ('source_root', 'dataset_root', 'weights_root'):
            paths[name] = root / name; paths[name].mkdir()
        skill = os.environ.get('RESEARCH_AUTOPILOT_SKILL_DIR')
        self.assertIsNotNone(skill, 'Use the acceptance builder-selected installed skill')
        return SimpleNamespace(root=root, plan_dir=root / 'plans/paired', skill_dir=Path(skill),
            run_id='paired-fixture', gpu_uuid='GPU-fixture', generation_seed=314,
            instrument_wall_seconds=5000,
            atol=0., rtol=0., max_capture_bytes=67108864, source_time_query=True,
            cpu_cores=2, ram_mib=4096, **paths)

    def test_real_plans_and_staging_preserve_identity_paths_budget_and_outputs(self):
        from prepare_native_context import build_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = self.fixture(root)
            receipt = build_plan(args)
            native = json.loads((args.plan_dir / 'native.json').read_text())
            outer = json.loads((args.plan_dir / 'harness.json').read_text())
            self.assertFalse(receipt['execution_started'])
            self.assertFalse(receipt['dispatch_ready'])
            self.assertEqual(native['purpose'], 'engineering')
            self.assertEqual(native['limits']['max_retries_per_trial'], 0)
            self.assertEqual(outer['limits']['total_wall_seconds'], 5000)
            self.assertEqual(outer['limits']['window_seconds'], 6800)
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'], 1)
            job = native['jobs'][0]
            self.assertEqual(job['seed'], 314)
            self.assertEqual(job['command'][job['command'].index('--generation-seed') + 1], '314')
            self.assertIn('actionmesh/context-output/raw-evidence.tar', job['output_paths'])
            import run_experiments
            attempt = root / 'attempt'; attempt.mkdir()
            command, cwd, workspace = run_experiments._stage(root, job, attempt)
            for name in ('contract', 'population', 'snapshot_contract', 'snapshot_admission',
                         'dataset_semantics', 'unit_manifest', 'environment'):
                value = command[command.index('--' + name.replace('_', '-')) + 1]
                self.assertEqual(Path(value), workspace / getattr(args, name).relative_to(root))
                self.assertTrue(Path(value).is_file())
            self.assertEqual(cwd, workspace / 'actionmesh')
            self.assertEqual(command[command.index('--source-root') + 1], str(args.source_root))
            self.assertTrue((workspace / 'inputs/native-runtime/dependencies.json').is_file())
            for name in ('native_context_runner.py', 'native_context_delivery.py',
                         'pipeline_decoder_observer.py', 'decoder_observer.py'):
                self.assertTrue((workspace / 'actionmesh/research_math' / name).is_file())
            with self.assertRaises(FileExistsError):
                build_plan(args)

    def test_bad_budget_or_tolerance_writes_no_plan(self):
        from prepare_native_context import build_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = self.fixture(root)
            for key, value in (('instrument_wall_seconds', 27001), ('atol', float('nan')),
                               ('max_capture_bytes', 0), ('gpu_uuid', 'GPU-a,GPU-b')):
                old = getattr(args, key); setattr(args, key, value)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    build_plan(args)
                setattr(args, key, old)
                self.assertFalse(args.plan_dir.exists())

    def test_full128_manifest_cannot_enter_calibration_instrument(self):
        from prepare_native_context import build_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = self.fixture(root)
            record = json.loads(args.unit_manifest.read_text())
            record['status'] = 'frozen_engineering_full128_unit'
            args.unit_manifest.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                build_plan(args)
            self.assertFalse(args.plan_dir.exists())

    def test_changed_dependency_lock_is_rejected_before_plan_creation(self):
        from prepare_native_context import build_plan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = self.fixture(root)
            (root / 'inputs/native-runtime/dependencies.json').write_text('{}')
            with self.assertRaises(ValueError):
                build_plan(args)
            self.assertFalse(args.plan_dir.exists())


if __name__ == '__main__':
    unittest.main()
