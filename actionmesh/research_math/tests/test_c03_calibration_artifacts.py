"""Software-only fixtures; no native correspondence or scientific evidence.

These tests are authored but not executed by the Web supervisor. The staging
case uses the actual installed native runner, with no _stage/input-path mock.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from research_math import c03_calibration_artifacts as artifacts


def fixture(root):
    data = root / 'retained/data.npz'; data.parent.mkdir(parents=True)
    np.savez_compressed(data, reference_residual=np.zeros((4, 3)),
        labeled_error=np.zeros((15, 4, 3)), weights=np.ones(4),
        uids=np.array(['software-dev'] * 4))
    evidence = root / 'retained/correspondence.json'
    evidence.write_text('{"scope":"software fixture, not qualified native correspondence"}\n')
    policy = {'kind': 'c03-development-calibration-policy', 'version': 1,
        'data_sha256': artifacts.digest(data), 'development_uids': ['software-dev'],
        'confirmation_uids': ['software-confirm'],
        'uid_to_family': {'software-dev': 'family-d', 'software-confirm': 'family-c'},
        'coordinate_policy': 'plain_pointwise_no_asset_specific_icp',
        'correspondence_evidence': {'mapping-review.json': artifacts.digest(evidence)},
        'generation_seed': 42, 'frame_indices': list(range(16)),
        'parameters': {'ridge': 1., 'tau': .1, 'max_iterations': 100,
                       'gradient_tolerance': 1e-8, 'objective_tolerance': 1e-12}}
    policy_path = root / 'retained/policy.json'
    artifacts.write(policy_path, policy)
    return data, policy_path, {'mapping-review.json': evidence}, policy


class CalibrationArtifactTests(unittest.TestCase):
    def test_partition_and_parameter_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            data, path, evidence, policy = fixture(Path(directory))
            bad = copy.deepcopy(policy)
            bad['uid_to_family']['software-confirm'] = 'family-d'
            with self.assertRaises(ValueError): artifacts.validate_policy(bad)
            bank = artifacts.load_development_bank(data, policy)
            self.assertEqual(bank['labeled_error'].shape, (15, 4, 3))
            result = artifacts.fit_bundle(data, path, evidence, Path(directory) / 'fit')
            self.assertEqual(result['status'], 'completed', result)
            bad = copy.deepcopy(result)
            bad['policy']['parameters']['max_iterations'] += 1
            bad['bundle_digest'] = artifacts.canonical_digest(
                {k: v for k, v in bad.items() if k != 'bundle_digest'})
            with self.assertRaises(ValueError): artifacts.validate_bundle(bad)
            raw = np.zeros((16, 4, 3)); anchor = np.ones((4, 3))
            arrays, failures = artifacts.apply_bundle_arrays(raw, anchor * 0, anchor, result,
                uid='software-confirm', family='family-c')
            self.assertEqual(set(arrays), set(artifacts.FIT_ROLES) | {'unit_C01'})
            self.assertEqual(failures, {})
            for value in arrays.values(): np.testing.assert_array_equal(value[0], anchor)
            with self.assertRaises(ValueError):
                artifacts.apply_bundle_arrays(raw, anchor * 0, anchor, result,
                    uid='software-dev', family='family-d')

    def test_confirmation_rows_and_corrupt_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, path, evidence, policy = fixture(root)
            with np.load(data, allow_pickle=False) as bank:
                values = {k: bank[k] for k in bank.files}
            values['uids'] = np.array(['software-confirm'] * 4)
            np.savez_compressed(data, **values)
            policy['data_sha256'] = artifacts.digest(data)
            with self.assertRaises(ValueError): artifacts.load_development_bank(data, policy)
            evidence['mapping-review.json'].write_text('changed bytes')
            result = artifacts.fit_bundle(data, path, evidence, root / 'fit')
            self.assertEqual(result['status'], 'error')
            self.assertNotIn('role_results', result)
            self.assertTrue((root / 'fit/fit-bundle.json').is_file())

    def test_actual_cross_root_staging_has_no_live_input_dependency(self):
        from prepare_c03_calibration import build_plans
        import run_experiments
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = Path(__file__).resolve().parents[3]
            package = root / 'actionmesh/research_math'; package.mkdir(parents=True)
            for name in ('__init__.py', 'correlated_calibration.py',
                         'c03_calibration_artifacts.py', 'self_map_candidate.py'):
                shutil.copy2(source_root / 'actionmesh/research_math' / name, package / name)
            shutil.copy2(source_root / 'actionmesh/prepare_c03_calibration.py',
                         root / 'actionmesh/prepare_c03_calibration.py')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c',
                'user.email=fixture@example.invalid', 'commit', '--allow-empty', '-qm', 'fixture'], check=True)
            data, policy, evidence, _ = fixture(root)
            native, outer = build_plans(root, data=data, policy=policy, evidence_files=evidence,
                plan_dir=root / 'plans/c03', run_id='c03-software-fixture', wall_seconds=60, ram_mib=1024)
            self.assertEqual(outer['tasks'][0]['resources']['gpu_count'], 0)
            self.assertEqual(native['limits']['max_retries_per_trial'], 0)
            attempt = root / 'attempt'; attempt.mkdir()
            command, cwd, workspace = run_experiments._stage(root, native['jobs'][0], attempt)
            for option in ('--data', '--policy'):
                self.assertTrue(Path(command[command.index(option) + 1]).is_relative_to(workspace))
            index = command.index('--evidence-file')
            name, path = command[index + 1:index + 3]
            self.assertTrue(Path(path).is_relative_to(workspace))
            shutil.rmtree(root / 'retained')
            # Invoke the staged module in a fresh process, not live imported code.
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            bundle = artifacts.load(cwd / 'c03-calibration-output/fit-bundle.json')
            self.assertEqual(artifacts.validate_bundle(bundle)['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
