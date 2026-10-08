"""Plan contracts for consuming a real paired-context bundle; unexecuted."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class NativeContextConsumptionPlanTests(unittest.TestCase):
    def bundle(self, root):
        from research_math.tests.test_native_context_delivery import NativeContextDeliveryTests
        source = root/'inputs/native-context'; source.mkdir(parents=True)
        return source, NativeContextDeliveryTests().completed_bundle(source)

    def project(self, root):
        project = Path(__file__).resolve().parents[3]
        package = root/'actionmesh/research_math'; package.mkdir(parents=True)
        for name in ('__init__.py', 'native_context_delivery.py'):
            shutil.copy2(project/'actionmesh/research_math'/name, package/name)
        shutil.copy2(project/'actionmesh/prepare_native_context_consumption.py',
                     root/'actionmesh/prepare_native_context_consumption.py')
        subprocess.run(['git', 'init', '-q', str(root)], check=True)
        subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture',
                        '-c', 'user.email=fixture@example.invalid', 'commit',
                        '--allow-empty', '-qm', 'fixture'], check=True)

    def build_args(self, expected):
        return {
            'expected_result_sha256': expected['result.json'],
            'expected_manifest_sha256': expected['raw-manifest.json'],
            'expected_archive_sha256': expected['raw-evidence.tar'],
            'expected_uid': expected['uid'], 'expected_gpu_uuid': expected['gpu_uuid'],
            'expected_generation_identity_sha256': expected['generation_identity_sha256'],
            'expected_source_time_query': expected['source_time_query'],
            'max_files': 256, 'max_unpacked_bytes': 16 * 1024 * 1024,
            'max_member_bytes': 4 * 1024 * 1024,
            'max_archive_bytes': 16 * 1024 * 1024,
            'max_metadata_bytes': 1024 * 1024,
        }

    def test_plan_pins_three_bundle_files_and_declares_every_extracted_member(self):
        from prepare_native_context_consumption import build_plans
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.project(root); bundle, expected = self.bundle(root)
            native, outer = build_plans(root, bundle_root=bundle, run_id='consume-fixture',
                plan_dir=root/'plans/consume', wall_seconds=300, **self.build_args(expected))
            job = native['jobs'][0]
            self.assertEqual(native['purpose'], 'engineering')
            self.assertEqual(job['arm_role'], 'paired-context-consumer-no-candidate')
            self.assertEqual({row['path'] for row in job['input_refs']}, {
                'inputs/native-context/result.json',
                'inputs/native-context/raw-manifest.json',
                'inputs/native-context/raw-evidence.tar'})
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'], 0)
            self.assertEqual(native['limits']['max_attempts'], 1)
            self.assertEqual(native['limits']['max_retries_per_trial'], 0)
            manifest = json.loads((bundle/'raw-manifest.json').read_text())
            outputs = set(job['output_paths'])
            for row in manifest['files']:
                self.assertIn('actionmesh/context-consumed/raw/'+row['path'], outputs)
            for name in ('bundle-result.json', 'bundle-manifest.json', 'bundle-consumption.json'):
                self.assertIn('actionmesh/context-consumed/'+name, outputs)
            import run_experiments
            attempt = root/'attempt'; attempt.mkdir()
            command, cwd, workspace = run_experiments._stage(root, job, attempt)
            originals = {}
            for option, name in (('--result', 'result.json'), ('--manifest', 'raw-manifest.json'),
                                 ('--archive', 'raw-evidence.tar')):
                staged = Path(command[command.index(option)+1])
                self.assertEqual(staged, workspace/(bundle/name).relative_to(root))
                originals[name] = staged.read_bytes()
            for name in originals:
                (bundle/name).write_bytes(b'changed after staging')
                self.assertEqual((workspace/(bundle/name).relative_to(root)).read_bytes(),
                                 originals[name])
            self.assertEqual(cwd, workspace/'actionmesh')

    def test_stale_pin_or_existing_plan_directory_fails_closed(self):
        from prepare_native_context_consumption import build_plans
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.project(root); bundle, expected = self.bundle(root)
            with self.assertRaises(ValueError):
                build_plans(root, bundle_root=bundle, run_id='consume-fixture',
                    plan_dir=root/'plans/consume', wall_seconds=300,
                    **dict(self.build_args(expected), expected_result_sha256='0'*64))
            self.assertFalse((root/'plans/consume').exists())
            build_plans(root, bundle_root=bundle, run_id='consume-fixture',
                plan_dir=root/'plans/consume', wall_seconds=300, **self.build_args(expected))
            with self.assertRaises(FileExistsError):
                build_plans(root, bundle_root=bundle, run_id='consume-fixture-2',
                    plan_dir=root/'plans/consume', wall_seconds=300,
                    **self.build_args(expected))


if __name__ == '__main__': unittest.main()
