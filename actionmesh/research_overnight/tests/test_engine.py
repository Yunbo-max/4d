"""Supervisor engineering tests; no scientific benchmark scores are generated."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone

class EngineTests(unittest.TestCase):
    def setUp(self):
        try:
            self.engine = importlib.import_module('research_overnight.engine')
        except ModuleNotFoundError:
            self.fail('The bounded overnight engine has not been implemented')

    def test_restart_keeps_original_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'window.json'
            start = datetime(2026, 10, 5, 22, tzinfo=timezone.utc)
            first = self.engine.Window.open(p, 8, 'frozen', now=start)
            resumed = self.engine.Window.open(p, 8, 'frozen', now=start + timedelta(hours=3))
            self.assertEqual(first.deadline, resumed.deadline)
            self.assertAlmostEqual(resumed.remaining(now=start + timedelta(hours=3)), 5 * 3600, delta=0.05)

    def test_changed_protocol_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'window.json'
            self.engine.Window.open(p, 8, 'old')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                self.engine.Window.open(p, 8, 'new')

    def test_changed_budget_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'window.json'
            self.engine.Window.open(p, 8, 'same')
            with self.assertRaisesRegex(ValueError, 'hours'):
                self.engine.Window.open(p, 4, 'same')

    def test_wall_clock_rollback_does_not_add_budget(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'window.json'
            now = datetime.now(timezone.utc)
            self.engine.Window.open(p, 8, 'same', now=now).checkpoint(now=now + timedelta(hours=2))
            resumed = self.engine.Window.open(p, 8, 'same', now=now + timedelta(hours=1))
            self.assertLessEqual(resumed.remaining(now=now + timedelta(hours=1)), 6 * 3600)

    def test_timeout_kills_process_group_and_retains_log(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            heartbeat = root / 'heartbeat'
            child = "import time,pathlib;p=pathlib.Path(" + repr(str(heartbeat)) + ");\nfor i in range(400):p.write_text(str(i));time.sleep(.05)"
            code = "import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-u','-c'," + repr(child) + "]);print('started',flush=True);time.sleep(20)"
            result = self.engine.execute([sys.executable, '-u', '-c', code], root / 'log.txt', 0.4, poll_seconds=0.02)
            self.assertEqual(result['status'], 'timeout')
            self.assertIn('started', (root / 'log.txt').read_text())
            before = heartbeat.read_text()
            time.sleep(.15)
            self.assertEqual(heartbeat.read_text(), before)

    def test_nonzero_exit_is_retained_as_failure(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.engine.execute([sys.executable, '-c', "print('failure evidence');raise SystemExit(3)"], Path(d) / 'log.txt', 5, poll_seconds=0.02)
            self.assertEqual(r['status'], 'failed')
            self.assertEqual(r['returncode'], 3)
            self.assertIn('failure evidence', (Path(d) / 'log.txt').read_text())

    def test_exited_leader_does_not_leave_running_child(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            heartbeat = root / 'heartbeat'
            child = "import time,pathlib;p=pathlib.Path(" + repr(str(heartbeat)) + ");\nfor i in range(400):p.write_text(str(i));time.sleep(.05)"
            code = "import subprocess,sys,time,pathlib;p=subprocess.Popen([sys.executable,'-u','-c'," + repr(child) + "]);\nwhile not pathlib.Path(" + repr(str(heartbeat)) + ").exists():time.sleep(.01)"
            result = self.engine.execute([sys.executable, '-c', code], root / 'log.txt', 5, poll_seconds=0.02)
            self.assertEqual(result['status'], 'completed')
            before = heartbeat.read_text()
            time.sleep(.15)
            self.assertEqual(heartbeat.read_text(), before)

    def test_receipt_corruption_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            artifact = root / 'artifact.txt'
            artifact.write_text('real bytes')
            self.engine.write_receipt(root / 'receipt.json', 'frozen', [artifact])
            self.assertTrue(self.engine.valid_receipt(root / 'receipt.json', 'frozen'))
            artifact.write_text('changed')
            self.assertFalse(self.engine.valid_receipt(root / 'receipt.json', 'frozen'))

    def test_lock_prevents_duplicate_runner(self):
        with tempfile.TemporaryDirectory() as d:
            lock = Path(d) / 'lock'
            with self.engine.Lease(lock):
                with self.assertRaisesRegex(RuntimeError, 'already'):
                    with self.engine.Lease(lock):
                        pass

    def test_budget_stop_keeps_all_units_in_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            window = self.engine.Window.open(root / 'window.json', .0002, 'same')
            units = [self.engine.Unit('one', 'job', [sys.executable, '-c', 'raise SystemExit(99)'], 999, root / 'one/receipt.json'),
                     self.engine.Unit('two', 'job', [sys.executable, '-c', 'raise SystemExit(99)'], 999, root / 'two/receipt.json')]
            state = self.engine.run_queue(units, root, window, env=os.environ.copy(), reserve_seconds=.1)
            self.assertEqual(set(state['units']), {'one', 'two'})
            self.assertEqual(state['counts']['declared'], 2)
            self.assertEqual(state['counts']['complete'], 0)
            self.assertEqual(state['units']['two']['status'], 'queued')

    def test_preflight_failure_preserves_existing_inventory_and_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            units = [self.engine.Unit('one', 'job', [], 1, root / 'one/receipt.json')]
            self.engine.atomic_json(root / 'queue.json', {'unit_ids': ['one'], 'status': 'interrupted',
                'units': {'one': {'family': 'job', 'status': 'interrupted', 'attempts': [{'attempt': 1}]}}})
            state = self.engine.initialize_queue(units, root)
            self.engine.record_preflight(root, state, {'status': 'failed', 'returncode': 2})
            saved = json.loads((root / 'queue.json').read_text())
            self.assertEqual(saved['unit_ids'], ['one'])
            self.assertEqual(saved['units']['one']['attempts'], [{'attempt': 1}])
            self.assertEqual(saved['status'], 'preflight_blocked')

    def test_successful_job_is_not_repeated_after_resume(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            artifact = root / 'one/count.txt'
            receipt = root / 'one/receipt.json'
            code = ('from pathlib import Path;from research_overnight.engine import write_receipt;'
                    'p=Path(' + repr(str(artifact)) + ');p.parent.mkdir(parents=True,exist_ok=True);'
                    'p.write_text(str(int(p.read_text())+1) if p.exists() else "1");'
                    'write_receipt(Path(' + repr(str(receipt)) + '),"same",[p])')
            units = [self.engine.Unit('one', 'job', [sys.executable, '-c', code], 1, receipt)]
            window = self.engine.Window.open(root / 'window.json', .01, 'same')
            first = self.engine.run_queue(units, root, window, env=os.environ.copy(), reserve_seconds=.1)
            second = self.engine.run_queue(units, root, window, env=os.environ.copy(), reserve_seconds=.1)
            self.assertEqual(first['counts']['complete'], 1)
            self.assertEqual(second['counts']['complete'], 1)
            self.assertEqual(artifact.read_text(), '1')
            self.assertEqual(len(second['units']['one']['attempts']), 1)

    def test_completed_receipt_is_recovered_after_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            artifact = root / 'one/artifact.txt'
            artifact.parent.mkdir()
            artifact.write_text('finished before parent checkpoint')
            receipt = artifact.parent / 'receipt.json'
            self.engine.write_receipt(receipt, 'same', [artifact])
            window = self.engine.Window.open(root / 'window.json', .0001, 'same')
            units = [self.engine.Unit('one', 'job', [sys.executable, '-c', 'raise SystemExit(99)'], 999, receipt)]
            state = self.engine.run_queue(units, root, window, env=os.environ.copy(), reserve_seconds=120)
            self.assertEqual(state['units']['one']['status'], 'complete')
            self.assertEqual(state['counts']['complete'], 1)

    def test_interrupt_updates_attempt_history_after_child_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            window = self.engine.Window.open(root / 'window.json', .01, 'same')
            units = [self.engine.Unit('one', 'job', [sys.executable, '-c', 'import time;time.sleep(20)'], 1, root / 'one/receipt.json')]
            def interrupt():
                raise KeyboardInterrupt('operator stop')
            with self.assertRaises(KeyboardInterrupt):
                self.engine.run_queue(units, root, window, env=os.environ.copy(), reserve_seconds=.1, monitor=interrupt)
            state = json.loads((root / 'queue.json').read_text())
            self.assertEqual(state['status'], 'interrupted')
            self.assertEqual(state['units']['one']['attempts'][0]['status'], 'interrupted')
            self.assertIsNone(state['active_process'])

if __name__ == '__main__':
    unittest.main()
