"""CPU-only failure/drain tests with mocked GPU telemetry and tiny owned children."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import research_census_queue_v2 as queue


class QueueDrainTests(unittest.TestCase):
    def job(self, root, identifier, code, **kwargs):
        return dict(id=identifier, output=str(root / (identifier + '.out')),
                    estimated_peak_mib=100, timeout_seconds=2,
                    argv=[sys.executable, '-c', code], **kwargs)

    def execute(self, root, jobs, snapshot=None, concurrency=2):
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({'jobs': jobs}))
        args = argparse.Namespace(manifest=manifest, output=root / 'queue', project_root=root,
            max_concurrency=concurrency, reserve_mib=100, max_seconds=3, gpu_index=0)
        telemetry = snapshot or (lambda _: dict(uuid='mock', total_mib=1000, free_mib=1000, used_mib=0,
                                                utilization_percent=0, processes=[]))
        code = queue.run(args, snapshot_fn=telemetry, sample_interval=.01)
        return code, json.loads((args.output / 'queue.json').read_text())

    def test_failure_latches_no_new_jobs_and_healthy_sibling_drains(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            healthy_code = f'import time,pathlib; time.sleep(.18); pathlib.Path({str(root / "healthy.out")!r}).write_text("done")'
            fail_code = 'import time,sys; time.sleep(.05); sys.exit(7)'
            pending_code = f'import pathlib; pathlib.Path({str(root / "unexpected-launch")!r}).touch()'
            jobs = [self.job(root, 'failing', fail_code), self.job(root, 'healthy', healthy_code),
                    self.job(root, 'unrelated', pending_code),
                    self.job(root, 'dependent', pending_code, depends_on=['failing']),
                    self.job(root, 'transitive', pending_code, depends_on=['dependent'])]
            code, report = self.execute(root, jobs)
            by_id = {job['id']: job for job in report['jobs']}
            self.assertEqual(code, 1)
            self.assertEqual(report['status'], 'stopped_after_admission_failure')
            self.assertEqual(report['admission_stop']['reason'], 'child_failed')
            self.assertEqual(by_id['failing']['exit_code'], 7)
            self.assertEqual(by_id['healthy']['status'], 'completed')
            self.assertEqual((root / 'healthy.out').read_text(), 'done')
            self.assertEqual(by_id['unrelated']['status'], 'aborted_not_started')
            self.assertEqual(by_id['dependent']['status'], 'dependency_failed')
            self.assertEqual(by_id['transitive']['status'], 'dependency_failed')
            self.assertNotIn('pid', by_id['unrelated'])
            self.assertFalse((root / 'unexpected-launch').exists())

    def test_timeout_does_not_admit_next_task(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self.job(root, 'timeout', 'import time; time.sleep(30)')
            first['timeout_seconds'] = .06
            pending = self.job(root, 'pending', 'raise RuntimeError("must not run")')
            code, report = self.execute(root, [first, pending], concurrency=1)
            self.assertEqual(code, 1)
            self.assertEqual(report['jobs'][0]['status'], 'timeout')
            self.assertIsNotNone(report['jobs'][0]['exit_code'])
            self.assertEqual(report['jobs'][1]['status'], 'aborted_not_started')

    def test_reserve_violation_stays_latched_after_memory_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            active = self.job(root, 'active', f'import time,pathlib; time.sleep(.13); pathlib.Path({str(root / "active.out")!r}).touch()')
            pending = self.job(root, 'pending', 'raise RuntimeError("must not run")')
            calls = []
            def telemetry(_):
                calls.append(1)
                return dict(uuid='mock', total_mib=1000, free_mib=50 if len(calls) == 3 else 1000,
                            used_mib=0, utilization_percent=0, processes=[])
            code, report = self.execute(root, [active, pending], telemetry, concurrency=1)
            self.assertEqual(code, 1)
            self.assertGreater(len(calls), 3)
            self.assertEqual(report['admission_stop']['reason'], 'gpu_reserve_breached')
            self.assertEqual(report['jobs'][0]['status'], 'completed')
            self.assertEqual(report['jobs'][1]['status'], 'aborted_not_started')

    def test_success_and_dependency_order_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self.job(root, 'first', f'import pathlib; pathlib.Path({str(root / "first.out")!r}).touch()')
            second = self.job(root, 'second',
                f'import pathlib; assert pathlib.Path({str(root / "first.out")!r}).exists(); pathlib.Path({str(root / "second.out")!r}).touch()',
                depends_on=['first'])
            code, report = self.execute(root, [first, second])
            self.assertEqual(code, 0)
            self.assertEqual(report['status'], 'completed')
            self.assertNotIn('admission_stop', report)
            self.assertTrue(all(job['status'] == 'completed' for job in report['jobs']))

    def test_unknown_dependency_refused_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bad = self.job(root, 'bad', 'raise RuntimeError("must not run")', depends_on=['missing'])
            with self.assertRaisesRegex(ValueError, 'Unknown dependency'):
                self.execute(root, [bad])
            self.assertFalse((root / 'queue').exists())

    def test_missing_output_is_a_failure_not_a_successful_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            empty = self.job(root, 'empty', 'pass')
            pending = self.job(root, 'pending', 'raise RuntimeError("must not run")')
            code, report = self.execute(root, [empty, pending], concurrency=1)
            self.assertEqual(code, 1)
            self.assertEqual(report['jobs'][0]['status'], 'missing_output')
            self.assertEqual(report['jobs'][1]['status'], 'aborted_not_started')


if __name__ == '__main__':
    unittest.main()
