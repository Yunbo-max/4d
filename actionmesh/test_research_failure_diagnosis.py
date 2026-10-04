"""Engineering checks against retained native reports, not new evaluation cases."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/failure-diagnosis/manifest-20261004.json'


class DiagnosisTests(unittest.TestCase):
    def runner(self):
        spec = importlib.util.find_spec('research_failure_diagnosis')
        self.assertIsNotNone(spec, 'Missing executable natural-failure diagnostic')
        import research_failure_diagnosis
        return research_failure_diagnosis

    def manifest(self):
        return json.loads(MANIFEST.read_text())

    def test_manifest_keeps_ten_runs_eight_assets_and_exact_source_hashes(self):
        runner = self.runner()
        manifest = runner.load_manifest(MANIFEST)
        self.assertEqual(len(manifest['cases']), 10)
        self.assertEqual(len({row['uid'] for row in manifest['cases']}), 8)
        for row in manifest['cases']:
            report = json.loads((ROOT / row['reference_evaluation']).read_text())
            self.assertEqual(row['sequence_sha256'], report['cases'][0]['inputs']['sequence']['sha256'])
            self.assertEqual(row['gt_sha256'], report['cases'][0]['inputs']['ground_truth']['sha256'])

    def test_missing_outputs_remain_in_full_denominator(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as folder:
            report = runner.collect(Path(folder), self.manifest(), ROOT)
        self.assertEqual(report['n_planned_runs'], 10)
        self.assertEqual(report['n_unique_assets'], 8)
        self.assertEqual(report['n_planned_arm_scores'], 30)
        self.assertEqual(report['run_status_counts'], {'pending': 10})
        self.assertEqual(report['scientific_status'], 'INCONCLUSIVE')

    def test_retained_counterexamples_prevent_universal_failure_story(self):
        report = self.runner().historical_stage_summary(self.manifest(), ROOT)
        self.assertEqual(report['n_assets_with_stage_report'], 8)
        self.assertEqual(report['stageI_future_mean_above_anchor'], 7)
        self.assertEqual(report['stageII_future_mean_above_anchor'], 8)
        self.assertEqual(report['stageII_sampled_monotonic_increase'], 3)
        self.assertEqual(report['stageII_future_mean_above_stageI'], 5)
        self.assertEqual(report['measured_frames'], [0, 8, 15])
        self.assertFalse(report['meaningful_failure_threshold_established'])

    def test_replay_accepts_actual_scores_and_rejects_changed_scores(self):
        runner = self.runner()
        row = self.manifest()['cases'][1]
        metrics = row['reference_metrics']
        self.assertTrue(runner.replay_check(metrics, metrics, self.manifest()['replay_tolerance'])['pass'])
        changed = dict(metrics, cd_motion=metrics['cd_motion'] + .01)
        self.assertFalse(runner.replay_check(metrics, changed, self.manifest()['replay_tolerance'])['pass'])

    def test_duplicate_cases_and_changed_native_budget_are_rejected(self):
        runner = self.runner()
        for change in ('duplicate', 'sample_budget', 'replay_tolerance', 'case_budget'):
            value = copy.deepcopy(self.manifest())
            if change == 'duplicate':
                value['cases'][1] = value['cases'][0]
            elif change == 'sample_budget':
                value['protocol']['n_pts_chamfer'] = 5000
            elif change == 'replay_tolerance':
                value['replay_tolerance']['atol'] = .1
            else:
                value['budgets']['case_seconds'] = 0
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'manifest.json'
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    runner.load_manifest(path)

    def test_prior_failed_attempt_is_retained_and_not_an_extra_asset(self):
        runner = self.runner()
        with tempfile.TemporaryDirectory() as folder:
            report = runner.collect(Path(folder), self.manifest(), ROOT)
        self.assertEqual(len(report['historical_failed_attempts']), 1)
        self.assertEqual(report['historical_failed_attempts'][0]['error_type'], 'TypeError')
        self.assertEqual(report['n_unique_assets'], 8)

    def test_wrong_round_output_cannot_be_silently_collected(self):
        runner = self.runner()
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            case = root / manifest['cases'][0]['case_id']
            case.mkdir()
            # Reuse a real report of another protocol as a mismatched engineering input.
            real = ROOT / manifest['cases'][0]['reference_evaluation']
            (case / 'report.json').write_bytes(real.read_bytes())
            report = runner.collect(root, manifest, ROOT)
        self.assertEqual(report['run_status_counts'], {'invalid_report': 1, 'pending': 9})

    def test_child_failure_before_report_is_not_hidden_as_pending(self):
        runner = self.runner()
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            queue = {'manifest_id': runner.manifest_id(manifest), 'jobs': [
                {'case_id': manifest['cases'][0]['case_id'], 'status': 'failed',
                 'error': 'child exited before writing its receipt'}]}
            (root / 'queue.json').write_text(json.dumps(queue))
            report = runner.collect(root, manifest, ROOT)
        self.assertEqual(report['run_status_counts'], {'failed': 1, 'pending': 9})
        self.assertEqual(report['arm_status_counts'], {'blocked': 3, 'pending': 27})

    def test_later_case_resource_monitor_failure_stops_admissions(self):
        runner = self.runner()
        policy = getattr(runner, 'should_halt_after_case', None)
        self.assertIsNotNone(policy, 'Missing resource failure admission rule')
        self.assertTrue(policy(3, {'status': 'failed'}, {'resource_qualification': 'failed'}))
        self.assertTrue(policy(3, {'status': 'failed'}, {'resource_collection_error': 'monitor failed'}))
        self.assertFalse(policy(3, {'status': 'failed'}, {'error': 'FileNotFoundError: absent natural cache'}))

    def test_owned_child_cleanup_is_idempotent_and_leaves_other_process_alone(self):
        runner = self.runner()
        cleanup = getattr(runner, 'stop_owned_child', None)
        self.assertIsNotNone(cleanup, 'Missing owned process group cleanup')
        command = [sys.executable, '-c', 'import time; time.sleep(120)']
        owned = subprocess.Popen(command, start_new_session=True)
        other = subprocess.Popen(command, start_new_session=True)
        try:
            cleanup(owned)
            self.assertIsNotNone(owned.poll())
            cleanup(owned)
            self.assertIsNone(other.poll())
        finally:
            if owned.poll() is None:
                owned.kill()
            other.terminate()
            owned.wait(timeout=5)
            other.wait(timeout=5)

    def test_launch_oserror_is_terminal_failure_with_full_denominator(self):
        from argparse import Namespace
        runner = self.runner()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'round'
            args = Namespace(manifest=MANIFEST, root=Path(folder), output=output, device='cuda:0')
            # Fault injection tests process admission, never native scorer behavior.
            with patch.object(runner, 'gpu_free', return_value=8192), patch.object(
                    runner.subprocess, 'Popen', side_effect=OSError('intentional process launch failure')):
                result = runner.run_round(args)
            report = json.loads((output / 'summary.json').read_text())
        self.assertEqual(result, 1)
        self.assertEqual(report['run_status_counts'], {'failed': 1, 'pending': 9})
        self.assertEqual(report['queue']['jobs'][0]['status'], 'failed')

    def test_retained_gpu_log_is_required_and_changes_are_rejected(self):
        runner = self.runner()
        verify = getattr(runner, 'verify_resource_receipt', None)
        self.assertIsNotNone(verify, 'Missing GPU evidence integrity check')
        source = ROOT / 'results/census-20261002/stage-gt/bat-seed42-v1'
        natural = json.loads((source / 'report.json').read_text())
        receipt = {'resources': natural['resources'], 'resource_qualification': 'passed',
                   'case_artifact_sha256': {'gpu-memory.csv': runner.digest(source / 'gpu-memory.csv')}}
        verify(receipt, source)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            with self.assertRaises(FileNotFoundError):
                verify(receipt, target)
            (target / 'gpu-memory.csv').write_bytes((source / 'gpu-memory.csv').read_bytes() + b'\n')
            with self.assertRaises(ValueError):
                verify(receipt, target)


if __name__ == '__main__':
    unittest.main()
