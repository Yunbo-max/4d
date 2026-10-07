"""Environment inventory checks; no scorer, GPU workload or scientific evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_math import native_runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prefix = self.root/'conda'
        (self.prefix/'conda-meta').mkdir(parents=True)
        (self.prefix/'conda-meta/python.json').write_text(json.dumps(
            {'name': 'python', 'version': '3.11.9', 'build': 'fixture',
             'subdir': 'linux-64', 'url': 'https://secret.invalid/token'}))
        (self.prefix/'conda-meta/numpy.json').write_text(json.dumps(
            {'name': 'numpy', 'version': '2.4.4', 'build': 'base', 'subdir': 'linux-64'}))
        self.venv_prefix = self.root/'venv'
        self.venv_site = self.venv_prefix/'lib/python3.11/site-packages'
        self.base_site = self.prefix/'lib/python3.11/site-packages'

    def distributions(self):
        return [SimpleNamespace(metadata={'Name': name}, version='1.0')
                for name in native_runtime.REQUIRED_PACKAGES]

    def capture(self, output='inputs/native-runtime', uuid='GPU-fixture'):
        with patch.object(native_runtime.metadata, 'distributions', return_value=self.distributions()), \
             patch.object(native_runtime.sys, 'prefix', str(self.venv_prefix)), \
             patch.object(native_runtime.sys, 'base_prefix', str(self.prefix)):
            return native_runtime.capture(self.root, self.root/output, uuid)

    def test_capture_binds_lock_bytes_at_the_final_project_relative_path(self):
        env = self.capture()
        ref = env['dependency_lock_refs'][0]
        self.assertEqual(ref['path'], 'inputs/native-runtime/dependencies.json')
        self.assertEqual(ref['sha256'], hashlib.sha256((self.root/ref['path']).read_bytes()).hexdigest())
        self.assertEqual(env['packages'], {name: '1.0' for name in native_runtime.REQUIRED_PACKAGES})
        self.assertFalse(env['native_contract_qualified'])
        self.assertFalse(env['gpu_identity_verified'])

    def test_existing_capture_is_preserved(self):
        self.capture()
        before = (self.root/'inputs/native-runtime/environment.json').read_bytes()
        with self.assertRaises(FileExistsError): self.capture()
        self.assertEqual((self.root/'inputs/native-runtime/environment.json').read_bytes(), before)

    def test_missing_required_package_cannot_emit_ready_runtime(self):
        with patch.object(native_runtime.metadata, 'distributions', return_value=self.distributions()[:-1]):
            with self.assertRaises(ValueError): native_runtime.package_inventory()
        self.assertFalse((self.root/'inputs/native-runtime').exists())

    def test_conflicting_distribution_versions_are_rejected(self):
        values = self.distributions()+[SimpleNamespace(metadata={'Name': 'NumPy'}, version='2.0')]
        with patch.object(native_runtime.metadata, 'distributions', return_value=values):
            with self.assertRaises(ValueError): native_runtime.package_inventory()

    def test_system_site_overlay_uses_active_venv_package_and_base_conda_records(self):
        def distribution(name, version, location):
            return SimpleNamespace(metadata={'Name': name}, version=version,
                                   locate_file=lambda _: location)

        values = [distribution(name, '1.0', self.venv_site)
                  for name in native_runtime.REQUIRED_PACKAGES if name != 'torch']
        values += [distribution('numpy', '2.4.4', self.base_site),
                   distribution('torch', '2.12.1', self.base_site)]
        # Metadata enumeration order must not override Python's sys.path precedence.
        values.reverse()
        with patch.object(native_runtime.metadata, 'distributions', return_value=values), \
             patch.object(native_runtime.sys, 'path', [str(self.venv_site), str(self.base_site)]), \
             patch.object(native_runtime.sys, 'prefix', str(self.venv_prefix)), \
             patch.object(native_runtime.sys, 'base_prefix', str(self.prefix)):
            packages = native_runtime.package_inventory()
            conda_packages = native_runtime.conda_inventory()

        self.assertEqual(packages['numpy'], '1.0')
        self.assertEqual(packages['torch'], '2.12.1')
        self.assertEqual([p['version'] for p in conda_packages if p['name'] == 'numpy'], ['2.4.4'])

    def test_capture_records_venv_and_base_conda_prefixes(self):
        env = self.capture()
        self.assertEqual(env['python_prefix'], str(self.venv_prefix))
        self.assertEqual(env['conda_prefix'], str(self.prefix.resolve()))

    def test_conda_capture_excludes_urls_channels_and_credentials(self):
        self.capture()
        lock = json.loads((self.root/'inputs/native-runtime/dependencies.json').read_text())
        self.assertEqual(lock['conda_packages'], [
            {'name': 'numpy', 'version': '2.4.4', 'build': 'base', 'subdir': 'linux-64'},
            {'name': 'python', 'version': '3.11.9', 'build': 'fixture', 'subdir': 'linux-64'}])
        self.assertNotIn('secret.invalid', json.dumps(lock))
        self.assertNotIn('url', lock['conda_packages'][0])

    def test_nonconda_interpreter_is_rejected(self):
        with patch.object(native_runtime.sys, 'prefix', str(self.root/'not-conda')), \
             patch.object(native_runtime.sys, 'base_prefix', str(self.root/'not-conda')):
            with self.assertRaises(ValueError): native_runtime.conda_inventory()

    def test_escaping_output_directory_is_rejected_before_write(self):
        outside = self.root.parent/(self.root.name+'-outside')
        with self.assertRaises(ValueError): native_runtime.capture(self.root, outside, 'GPU-fixture')
        self.assertFalse(outside.exists())

    def test_multiple_or_missing_gpu_identity_is_rejected_before_write(self):
        for uuid in ('', '0', 'GPU-a,GPU-b', 'GPU-a\nGPU-b'):
            with self.subTest(uuid=uuid):
                with self.assertRaises(ValueError): self.capture(uuid=uuid)
        self.assertFalse((self.root/'inputs/native-runtime').exists())


if __name__ == '__main__': unittest.main()
