"""CPU-only scheduler checks; no GPU, downloads, SSH, or model inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import research_census_queue as queue


class QueueTests(unittest.TestCase):
    def test_dependencies_wait_propagate_failure_and_reject_cycles(self):
        jobs = [dict(id='generation', status='running', depends_on=[]),
                dict(id='eval', status='pending', depends_on=['generation']),
                dict(id='summary', status='pending', depends_on=['eval'])]
        self.assertEqual(queue.eligible_jobs(jobs), [])
        jobs[0]['status'] = 'completed'
        self.assertEqual([j['id'] for j in queue.eligible_jobs(jobs)], ['eval'])
        jobs[1]['status'] = 'failed'
        self.assertEqual(queue.eligible_jobs(jobs), [])
        self.assertEqual(jobs[2]['status'], 'dependency_failed')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'manifest.json'
            jobs = [dict(id='a', output=str(root / 'a'), argv=[sys.executable], depends_on=['b']),
                    dict(id='b', output=str(root / 'b'), argv=[sys.executable], depends_on=['a'])]
            manifest.write_text(json.dumps({'jobs': jobs}))
            with self.assertRaisesRegex(ValueError, 'cycle'):
                queue.load_jobs(manifest, root)

    def test_unrealized_peak_prevents_double_admission(self):
        # A newly spawned child has allocated nothing: physical free is misleading.
        active = [dict(estimated_peak_mib=10209, gpu_memory_mib=0)]
        self.assertTrue(queue.admitted(22400, [], 10209, 2048, 2))
        self.assertFalse(queue.admitted(22400, active, 10209, 2048, 2))
        # Once usage materializes, the answer must stay invariant, not double-count it.
        active[0]['gpu_memory_mib'] = 8000
        self.assertFalse(queue.admitted(14400, active, 10209, 2048, 2))
        # Device/process queries may straddle an allocation. A stale high-free
        # reading must not override the known initial capacity for both peaks.
        self.assertFalse(queue.admitted(22400, active, 10209, 2048, 2, initial_free_mib=22400))
        self.assertTrue(queue.admitted(22528, [dict(estimated_peak_mib=8000, gpu_memory_mib=0)], 8000, 2048, 2))

    def test_timeout_reaped_and_rerun_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root / 'marker'
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'jobs': [dict(id='timeout', output=str(root / 'job'),
                estimated_peak_mib=100, timeout_seconds=.12,
                argv=[sys.executable, '-c',
                      'import pathlib,time; pathlib.Path(' + repr(str(marker)) + ').write_text("once"); time.sleep(30)'])]}))
            args = argparse.Namespace(manifest=manifest, output=root / 'run', project_root=root,
                                      max_concurrency=2, reserve_mib=100, max_seconds=2, gpu_index=0)
            gpu = lambda _: dict(utc='mock', uuid='mock', total_mib=1000, used_mib=0,
                                 free_mib=1000, utilization_percent=0, processes=[])
            self.assertEqual(queue.run(args, snapshot_fn=gpu, sample_interval=.03), 1)
            report = json.loads((args.output / 'queue.json').read_text())
            child = report['jobs'][0]
            self.assertEqual(child['status'], 'timeout')
            self.assertIsNotNone(child['exit_code'])
            self.assertEqual(marker.read_text(), 'once')
            # poll through kill -0: the timed-out direct child was reaped, not orphaned.
            result = subprocess.run(['kill', '-0', str(child['pid'])], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            with self.assertRaises(FileExistsError):
                queue.run(args, snapshot_fn=gpu, sample_interval=.03)

    def test_foreign_gpu_user_prevents_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root / 'marker'
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'jobs': [dict(id='blocked', output=str(root / 'job'),
                estimated_peak_mib=100,
                argv=[sys.executable, '-c', 'open(' + repr(str(marker)) + ', "w").close()'])]}))
            args = argparse.Namespace(manifest=manifest, output=root / 'run', project_root=root,
                                      max_concurrency=2, reserve_mib=100, max_seconds=2, gpu_index=0)
            gpu = lambda _: dict(utc='mock', uuid='mock', total_mib=1000, used_mib=10,
                                 free_mib=990, utilization_percent=1,
                                 processes=[dict(pid=123, memory_mib=10)])
            self.assertEqual(queue.run(args, snapshot_fn=gpu, sample_interval=.03), 1)
            report = json.loads((args.output / 'queue.json').read_text())
            self.assertEqual(report['status'], 'external_gpu_busy_initially')
            self.assertEqual(report['jobs'][0]['status'], 'not_started')
            self.assertFalse(marker.exists())

    def test_budget_terminates_only_our_child_and_preserves_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'jobs': [dict(id='job' + str(i), output=str(root / str(i)),
                estimated_peak_mib=700, argv=[sys.executable, '-c', 'import time; time.sleep(30)'])
                for i in range(2)]}))
            args = argparse.Namespace(manifest=manifest, output=root / 'run', project_root=root,
                                      max_concurrency=2, reserve_mib=100, max_seconds=.12, gpu_index=0)
            gpu = lambda _: dict(utc='mock', uuid='mock', total_mib=1000, used_mib=0,
                                 free_mib=1000, utilization_percent=0, processes=[])
            bystander = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            try:
                self.assertEqual(queue.run(args, snapshot_fn=gpu, sample_interval=.02), 1)
                report = json.loads((args.output / 'queue.json').read_text())
                self.assertEqual(report['status'], 'budget_exhausted')
                self.assertEqual(report['jobs'][0]['status'], 'cancelled_budget_exhausted')
                self.assertEqual(report['jobs'][1]['status'], 'not_started')
                self.assertIsNone(bystander.poll())
            finally:
                bystander.terminate()
                bystander.wait()


if __name__ == '__main__':
    unittest.main()
