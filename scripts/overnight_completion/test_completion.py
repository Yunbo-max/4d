import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from research_overnight.engine import Unit, atomic_json, fingerprint, process_identity, write_receipt
from completion import finish_units, verify_authorization, wait_for_runner


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = os.environ.copy()
        self.signature = 'fixture-signature'

    def unit(self, name, fail_first=False, always_fail=False):
        folder = self.root / 'units' / name
        folder.mkdir(parents=True)
        worker = folder / 'worker.py'
        worker.write_text('''import sys
from pathlib import Path
from research_overnight.engine import write_receipt
p=Path(__file__).parent
counter=p/'launches.txt'
n=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(n))
if sys.argv[1]=='always' or (sys.argv[1]=='first' and n==1):
    raise SystemExit(7)
artifact=p/'result.txt'
artifact.write_text('verified output')
write_receipt(p/'receipt.json', 'fixture-signature', [artifact])
''')
        mode = 'always' if always_fail else 'first' if fail_first else 'never'
        return Unit(name, 'generation', [sys.executable, str(worker), mode], 2400,
                    folder / 'receipt.json')

    def run_units(self, units):
        return finish_units(units, self.root, self.signature, env=self.env, poll_seconds=.01)

    def test_skips_verified_completed_outputs_but_runs_pending(self):
        done, pending = self.unit('done'), self.unit('pending')
        artifact = done.receipt.parent / 'original.txt'
        artifact.write_text('original generation preserved')
        write_receipt(done.receipt, self.signature, [artifact])
        before = done.receipt.read_bytes()
        state = self.run_units([done, pending])
        self.assertEqual(state['counts'], {'declared': 2, 'complete': 2})
        self.assertEqual(state['status'], 'inventory_complete')
        self.assertFalse((done.receipt.parent / 'launches.txt').exists())
        self.assertEqual(done.receipt.read_bytes(), before)
        self.assertEqual((pending.receipt.parent / 'launches.txt').read_text(), '1')

    def test_finishes_after_expired_window_and_preserves_it_and_failed_attempt(self):
        window = self.root / 'window.json'
        window.write_text('{"deadline_utc":"2000-01-01T00:00:00+00:00"}')
        before = window.read_bytes()
        unit = self.unit('retry', fail_first=True)
        state = self.run_units([unit])
        self.assertEqual(state['status'], 'inventory_complete')
        attempts = state['units']['retry']['attempts']
        self.assertEqual([a['returncode'] for a in attempts], [7, 0])
        self.assertTrue(all(Path(a['log']).is_file() for a in attempts))
        self.assertEqual(window.read_bytes(), before)

    def test_failure_is_not_completion_and_other_assets_still_finish(self):
        state = self.run_units([self.unit('bad', always_fail=True), self.unit('good')])
        self.assertEqual(state['status'], 'incomplete_requires_debug')
        self.assertEqual(state['counts'], {'declared': 2, 'complete': 1})
        self.assertEqual(len(state['units']['bad']['attempts']), 2)
        self.assertEqual(state['units']['good']['status'], 'complete')

    def test_corrupt_receipt_cannot_be_overwritten_or_counted(self):
        unit = self.unit('corrupt')
        artifact = unit.receipt.parent / 'result.txt'
        artifact.write_text('old')
        write_receipt(unit.receipt, self.signature, [artifact])
        artifact.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'receipt'):
            self.run_units([unit])
        self.assertFalse((unit.receipt.parent / 'launches.txt').exists())

    def test_refuses_overlapping_live_previous_worker(self):
        unit = self.unit('pending')
        atomic_json(self.root / 'queue.json', {'unit_ids': ['pending'], 'units': {},
                    'active_process': process_identity(os.getpid())})
        with self.assertRaisesRegex(RuntimeError, 'already running'):
            self.run_units([unit])
        self.assertFalse((unit.receipt.parent / 'launches.txt').exists())

    def test_resource_failure_is_saved_and_does_not_launch_next_asset(self):
        first, second = self.unit('oom'), self.unit('later')
        (first.receipt.parent / 'worker.py').write_text(
            "import sys\nprint('CUDA out of memory', flush=True)\nsys.exit(1)\n")
        state = self.run_units([first, second])
        self.assertEqual(state['status'], 'resource_stop')
        self.assertEqual(state['counts'], {'declared': 2, 'complete': 0})
        self.assertEqual(state['units']['oom']['attempts'][0]['returncode'], 1)
        self.assertFalse((second.receipt.parent / 'launches.txt').exists())

    def test_interruption_kills_worker_and_records_incomplete_attempt(self):
        unit = self.unit('interrupt')
        marker = unit.receipt.parent / 'alive'
        worker = unit.receipt.parent / 'worker.py'
        worker.write_text("import time\nfrom pathlib import Path\np=Path(__file__).parent\n"
                          "(p/'alive').write_text('started')\ntime.sleep(10)\n"
                          "(p/'should_not_finish').write_text('bad')\n")
        def interrupt():
            if marker.exists():
                raise KeyboardInterrupt('test stop')
        with self.assertRaises(KeyboardInterrupt):
            finish_units([unit], self.root, self.signature, env=self.env,
                         monitor=interrupt, poll_seconds=.01)
        state = json.loads((self.root / 'queue.json').read_text())
        self.assertEqual(state['status'], 'interrupted')
        self.assertIsNone(state['active_process'])
        attempt = state['units']['interrupt']['attempts'][0]
        self.assertEqual(attempt['status'], 'interrupted')
        self.assertIsNotNone(attempt['returncode'])
        self.assertFalse((unit.receipt.parent / 'should_not_finish').exists())

    def test_waits_for_predecessor_lock_without_changing_its_records(self):
        marker = self.root / 'locked'
        code = '''import sys,time
from pathlib import Path
from research_overnight.engine import Lease
with Lease(Path(sys.argv[1])):
 Path(sys.argv[2]).write_text('locked')
 time.sleep(.6)
 Path(sys.argv[2]).write_text('releasing')
'''
        child = subprocess.Popen([sys.executable, '-c', code, str(self.root / 'runner.lock'), str(marker)], env=self.env)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        limit = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < limit:
            time.sleep(.01)
        self.assertTrue(marker.exists())
        queue = self.root / 'queue.json'
        queue.write_text('{"original":"untouched"}')
        notices = []
        with wait_for_runner(self.root / 'runner.lock', lambda **v: notices.append(v), poll_seconds=.02):
            self.assertEqual(marker.read_text(), 'releasing')
            self.assertEqual(queue.read_text(), '{"original":"untouched"}')
        child.wait(timeout=2)
        self.assertTrue(any(n['status'] == 'waiting_for_original_runner' for n in notices))

    def test_authorization_is_bound_to_original_protocol_and_inventory(self):
        protocol = {'generation': {'cohort': {'uids': ['a','b']}}, 'suite': 'generation'}
        signature = fingerprint(protocol)
        atomic_json(self.root / 'protocol.json', protocol)
        atomic_json(self.root / 'window.json', {'fingerprint': signature})
        authorization = {'mode':'until_inventory_complete', 'fingerprint':signature,
                         'unit_ids':['gen-a','gen-b']}
        verify_authorization(self.root, authorization)
        authorization['unit_ids'] = ['gen-a']
        with self.assertRaisesRegex(ValueError, 'inventory'):
            verify_authorization(self.root, authorization)
        authorization['unit_ids'] = ['gen-a','gen-b']
        protocol['suite'] = 'both'
        atomic_json(self.root / 'protocol.json', protocol)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            verify_authorization(self.root, authorization)


if __name__ == '__main__':
    unittest.main()
