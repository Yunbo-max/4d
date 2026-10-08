"""Local-only actual-file contracts for native B0 implementation provenance.

Fixtures intentionally match collect_native_sequence's missing candidate-only
implementation field; they do not execute generation, scoring or a harness.
"""
import json
from pathlib import Path
import tempfile
import unittest

import prepare_c01_native_scoring as c01
import prepare_c02_native_scoring as c02
import prepare_c06_native_scoring as c06
import prepare_c11_native_scoring as c11
import prepare_c14_native_scoring as c14


class NativeB0ContractTests(unittest.TestCase):
    def fixture(self, root, profile):
        def write(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
            return profile.scoring.file_ref(root, path)
        rows, pinned, arms = [], [], {}
        for key, role in profile.ROLE_FOR_CONTRACT_ARM.items():
            if role == 'b_star':
                continue
            implementation = write('code/' + role + '.json', {'source': role})
            report = write('cases/' + role + '/report.json', {
                'status': 'completed', 'uid': 'fixture', 'seed': 42,
                **({} if role == 'b0' else {'implementation_sha256': implementation['sha256']})})
            row = {'role': role, 'method_id': role + '-method',
                   'report_ref': report, 'implementation_ref': implementation,
                   'implementation_sha256': implementation['sha256']}
            pinned.extend([implementation, report])
            if role == 'b0' and profile.PROFILE in ('c01', 'c06', 'c11'):
                ref = write('native/generation-identity.json', {
                    'kind': 'native-context-generation-identity', 'uid': 'fixture',
                    'instrument_code_sha256': {
                        'research_math/native_context_runner.py': implementation['sha256']}})
                row['generation_identity_ref'] = ref
                pinned.append(ref)
            if role == 'b0' and profile.PROFILE == 'c02':
                ref = write('cases/b0/command.json', {
                    'uid': 'fixture', 'seed': 42, 'offline': True,
                    'script_sha256': implementation['sha256']})
                row['command_ref'] = ref
                pinned.append(ref)
            rows.append(row)
            arms[key] = {'name': role, 'revision': row['method_id'],
                         'implementation_refs': [implementation]}
        rows.append({'role': 'b_star', 'method_id': 'b0-method', 'alias_of': 'b0'})
        arms['baseline'] = {**arms['b0'], 'name': 'b_star'}
        return ({'arm_requirements': arms},
                {'uid': 'fixture', 'inference_seed': 42, 'roles': rows, 'input_refs': pinned})

    def test_each_real_profile_accepts_native_report_without_candidate_hash(self):
        for profile in (c01, c02, c06, c11, c14):
            with self.subTest(profile=profile.PROFILE), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                contract, comparison = self.fixture(root, profile)
                profile.bind_contract_arms(contract, comparison, root)

    def test_present_mismatch_and_changed_bytes_are_rejected(self):
        for profile in (c01, c02, c06, c11, c14):
            for corruption in ('reported_hash', 'code_bytes', 'unpinned_ref'):
                with self.subTest(profile=profile.PROFILE, corruption=corruption), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    contract, comparison = self.fixture(root, profile)
                    row = next(row for row in comparison['roles'] if row['role'] == 'b0')
                    if corruption == 'reported_hash':
                        path = root / row['report_ref']['path']
                        report = json.loads(path.read_text())
                        report['implementation_sha256'] = '0' * 64
                        path.write_text(json.dumps(report))
                        row['report_ref'] = profile.scoring.file_ref(root, path)
                    elif corruption == 'code_bytes':
                        (root / row['implementation_ref']['path']).write_text('changed')
                    else:
                        comparison['input_refs'].remove(row['implementation_ref'])
                    with self.assertRaises(ValueError):
                        profile.bind_contract_arms(contract, comparison, root)

    def test_wrong_native_producer_or_command_is_rejected(self):
        for profile, ref_key, wrong_field in (
                (c01, 'generation_identity_ref', 'instrument_code_sha256'),
                (c06, 'generation_identity_ref', 'instrument_code_sha256'),
                (c11, 'generation_identity_ref', 'instrument_code_sha256'),
                (c02, 'command_ref', 'script_sha256')):
            with self.subTest(profile=profile.PROFILE), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                contract, comparison = self.fixture(root, profile)
                row = next(row for row in comparison['roles'] if row['role'] == 'b0')
                path = root / row[ref_key]['path']
                record = json.loads(path.read_text())
                record[wrong_field] = ({'research_math/native_context_runner.py': '0' * 64}
                                       if profile.PROFILE in ('c01', 'c06', 'c11') else '0' * 64)
                path.write_text(json.dumps(record))
                old = row[ref_key]
                row[ref_key] = profile.scoring.file_ref(root, path)
                comparison['input_refs'].remove(old)
                comparison['input_refs'].append(row[ref_key])
                with self.assertRaisesRegex(ValueError, 'implementation mismatch'):
                    profile.bind_contract_arms(contract, comparison, root)

    def test_non_b0_report_stays_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, comparison = self.fixture(root, c14)
            row = next(row for row in comparison['roles'] if row['role'] == 'corotational_residual')
            path = root / row['report_ref']['path']
            path.write_text('{}')
            row['report_ref'] = c14.scoring.file_ref(root, path)
            with self.assertRaisesRegex(ValueError, 'arm/report implementation'):
                c14.bind_contract_arms(contract, comparison, root)


if __name__ == '__main__':
    unittest.main()
