"""Serialization/protocol engineering checks, not synthetic scientific evaluation."""
import ast
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

class AdapterTests(unittest.TestCase):
    def module(self, name):
        try:
            return importlib.import_module('research_overnight.' + name)
        except ModuleNotFoundError:
            self.fail(name + ' is not implemented')

    def test_cohort_excludes_inspected_assets_and_is_stable(self):
        p = self.module('protocol')
        assets = Path(__file__).resolve().parents[1] / 'assets/actionbench_population.json'
        population = json.loads(assets.read_text())['uids']
        cohort = p.generation_cohort(population, 16)
        self.assertEqual(len(set(cohort)), 16)
        self.assertFalse(set(cohort) & set(p.INSPECTED_UIDS))
        self.assertTrue(set(cohort) <= set(population))
        self.assertEqual(cohort, p.generation_cohort(list(reversed(population)), 16))

    def test_native_stationary_control_keeps_anchor_and_faces(self):
        g = self.module('generation')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            vertices = np.arange(16 * 3 * 3, dtype=np.float32).reshape(16, 3, 3)
            faces = np.array([[0, 1, 2]], dtype=np.int64)
            np.savez(root / 'source.npz', vertices=vertices, faces=faces, frame_indices=np.arange(16))
            g.stationary_control(root / 'source.npz', root / 'control.npz')
            with np.load(root / 'control.npz') as saved:
                self.assertTrue(np.array_equal(saved['faces'], faces))
                self.assertTrue(np.array_equal(saved['vertices'][0], vertices[0]))
                self.assertTrue(np.array_equal(saved['vertices'][15], vertices[0]))
                self.assertEqual(saved['vertices'].shape, (16, 3, 3))
                self.assertTrue(np.array_equal(saved['frame_indices'], np.arange(16)))

    def test_stationary_control_rejects_truncated_native_sequence(self):
        g = self.module('generation')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.savez(root / 'source.npz', vertices=np.zeros((2, 3, 3)), faces=np.array([[0, 1, 2]]), frame_indices=[0, 1])
            with self.assertRaisesRegex(ValueError, '16'):
                g.stationary_control(root / 'source.npz', root / 'control.npz')

    def test_qa_native_interval_is_preserved(self):
        q = self.module('perception')
        self.assertEqual(q.native_indices(29), [0, 4, 8, 12, 16, 20])
        with self.assertRaises(ValueError):
            q.native_indices(5)

    def test_qa_input_never_contains_labels(self):
        q = self.module('perception')
        row = {'Question': 'schema fixture', 'Options': ['one', 'two', 'three', 'four'], 'Answer index': 2, 'Category': 'Counting'}
        public, label, category = q.public_question(row)
        self.assertEqual(label, 2)
        self.assertEqual(category, 'Counting')
        self.assertNotIn('Answer index', public)
        self.assertNotIn('Category', public)
        self.assertIn('Answer index', row)

    def test_qa_invalid_format_stays_in_denominator(self):
        q = self.module('perception')
        result = q.native_score('No option supplied', 1)
        self.assertEqual(result, (-1, -1))
        self.assertEqual(q.native_score('(A)', 1), (1, 1))
        self.assertEqual(q.native_score('(B)', 1), (0, 2))

    def test_qa_extraction_matches_pinned_native_functions(self):
        q = self.module('perception')
        path = Path(__file__).resolve().parents[1] / 'assets/4dbench_qwen2_reference.py'
        module = ast.parse(path.read_text())
        selected = ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in ('extract_answer_option', 'handle_vqa_result')], type_ignores=[])
        namespace = {'re': __import__('re')}
        exec(compile(selected, str(path), 'exec'), namespace)
        for text in ['(A)', 'A (B)', 'Answer: C.', 'No option supplied', '(D) then (A)', '', None]:
            self.assertEqual(q.native_score(text, 2), namespace['handle_vqa_result'](text, 2))

    def test_native_asset_validation_failure_is_not_scorer_unavailability(self):
        from research_overnight.generation import evaluation_failure
        from research_overnight.engine import atomic_json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'metrics.json'
            atomic_json(p, {'cases': [{'status': 'error', 'error': 'ValueError: no finite positive surface area'}]})
            error = evaluation_failure(p)
            self.assertIn('Native asset evaluation failed', error)
            self.assertNotIn('Native scorer unavailable', error)

    def test_missing_native_backend_is_a_fatal_resource_failure(self):
        from research_overnight.generation import evaluation_failure
        from research_overnight.engine import atomic_json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'metrics.json'
            atomic_json(p, {'backend_error': 'ModuleNotFoundError: pytorch3d', 'cases': []})
            self.assertIn('Native scorer unavailable', evaluation_failure(p))

if __name__ == '__main__':
    unittest.main()
