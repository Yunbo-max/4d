"""CLI/report engineering checks; no model is loaded by these tests."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class CLITests(unittest.TestCase):
    def test_plan_runs_without_cuda_or_models(self):
        result = subprocess.run([sys.executable, '-m', 'research_overnight', 'plan'],
                                capture_output=True, text=True, env=os.environ.copy())
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['resources']['gpu_workers'], 1)
        self.assertEqual(len(plan['generation']['uids']), 16)
        self.assertFalse(plan['generation']['new_candidate_methods_enabled'])

    def test_report_preserves_unscored_inventory(self):
        try:
            report = importlib.import_module('research_overnight.reporting')
        except ModuleNotFoundError:
            self.fail('reporting is not implemented')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'protocol.json').write_text(json.dumps({'generation': {'cohort': {'uids': ['existing-unit', 'not-run-unit']}}, 'perception': None}))
            (root / 'queue.json').write_text(json.dumps({'unit_ids': ['gen-existing-unit', 'gen-not-run-unit'], 'units': {'gen-existing-unit': {'status': 'failed'}}, 'status': 'budget_boundary'}))
            summary = report.collect(root)
            self.assertEqual(summary['generation']['declared_assets'], 2)
            self.assertEqual(summary['generation']['complete_pairs'], 0)
            self.assertEqual(len(summary['generation']['pending_or_failed']), 2)
            self.assertFalse(summary['scientific_gate_pass'])

    def test_prepared_records_reject_changed_input(self):
        from research_overnight.protocol import verify_records
        from research_overnight.engine import digest
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input.bin'
            p.write_bytes(b'serialized input')
            record = {str(p): {'sha256': digest(p)}}
            p.write_bytes(b'changed input')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_records(record)

    def test_launcher_honors_explicit_inference_python(self):
        with tempfile.TemporaryDirectory() as d:
            chosen = Path(d) / 'python'
            chosen.write_text('#!/bin/sh\nprintf "selected-environment\\n"\n')
            chosen.chmod(0o755)
            env = {**os.environ, 'OVERNIGHT_PYTHON': str(chosen)}
            from research_overnight.protocol import ACTIONMESH
            result = subprocess.run(['bash', str(ACTIONMESH.parent / 'scripts/run_8h_2080ti.sh'), 'plan'],
                                    capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'selected-environment')

    def test_cached_weights_are_frozen_even_without_expected_hashes(self):
        from research_overnight import resources
        from research_overnight.protocol import verify_records
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'manifests').mkdir()
            (root / 'weight.bin').write_bytes(b'old bytes')
            manifest = root / 'manifests/weights.json'
            manifest.write_text(json.dumps([{'path': 'weight.bin', 'size': 9}]))
            frozen = resources.freeze_weight_records(root)
            self.assertIn(str(manifest), frozen)
            verify_records(frozen)
            (root / 'weight.bin').write_bytes(b'new bytes')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_records(frozen)

    def test_partial_qa_object_is_not_counted_as_complete(self):
        from research_overnight.reporting import collect
        from research_overnight.engine import atomic_json, write_receipt
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            qids = ['objectA_1', 'objectA_2']
            atomic_json(root / 'protocol.json', {'generation': None, 'perception': {'question_ids': qids, 'categories': ['motion']}})
            atomic_json(root / 'window.json', {'fingerprint': 'same'})
            unit_id = 'qa-' + hashlib.sha256(qids[0].encode()).hexdigest()[:16]
            unit = root / 'units' / unit_id
            pair = {'question_id': qids[0], 'uid': 'objectA', 'category': 'motion',
                    'arms': {arm: {'correctness': 1} for arm in ('official_concat', 'separate_views')}}
            atomic_json(unit / 'pair.json', pair)
            write_receipt(unit / 'receipt.json', 'same', [unit / 'pair.json'])
            atomic_json(root / 'queue.json', {'status': 'budget_boundary', 'units': {unit_id: {'status': 'complete'}}})
            summary = collect(root)['perception']
            self.assertEqual(summary['complete_pairs'], 1)
            self.assertEqual(summary['complete_objects'], 0)
            self.assertEqual(summary['partially_scored_objects'], 1)

    def test_stock_11gib_card_is_admitted_when_free_memory_clears_observed_margin(self):
        from research_overnight.resources import require_available
        require_available({'applications': [], 'total_mib': 11264, 'free_mib': 11000}, 'generation')

if __name__ == '__main__':
    unittest.main()
