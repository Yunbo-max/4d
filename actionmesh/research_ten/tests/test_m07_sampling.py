"""Behavior tests: strict paid queries, exact samples and sparse motion reconstruction."""
import importlib
import unittest
import numpy as np

try:
    method = importlib.import_module("research_ten.m07_sampling")
except ModuleNotFoundError:
    method = None


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(method, "The budgeted sparse query allocator is not implemented")

    @staticmethod
    def fixture():
        points = np.column_stack((np.linspace(0, 1, 33), np.zeros((33, 2))))
        truth = np.broadcast_to(points, (5, 33, 3)).copy()
        truth[:, :, 1] += np.sin(np.linspace(0, np.pi, 5))[:, None] * np.exp(-((points[:, 0] - .5) / .10)**2)
        return points, truth

    def test_all_strategies_charge_unique_probes_and_use_exact_budget(self):
        points, truth = self.fixture()
        for strategy in ["fps", "curvature", "motion", "residual"]:
            calls = []
            def query(indices):
                calls.extend(np.asarray(indices).tolist())
                return truth[:, indices, :]
            r = method.allocate_controls(points, query, 9, strategy=strategy,
                                         curvature=np.linspace(0, 1, len(points)))
            self.assertEqual(len(calls), 9)
            self.assertEqual(len(set(calls)), 9)
            self.assertEqual(r["query_count"], 9)
            self.assertEqual(r["probe_count"] + r["adaptive_count"], 9)
            np.testing.assert_array_equal(r["sample_indices"], calls)
            np.testing.assert_allclose(r["trajectories"][:, calls], truth[:, calls], atol=1e-12)

    def test_global_translation_reproduced_from_one_paid_query(self):
        points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        delta = np.array([[0, 0, 0], [2, 3, 4]])
        r = method.allocate_controls(points, lambda ids: points[ids][None] + delta[:, None], 1)
        np.testing.assert_allclose(r["trajectories"], points[None] + delta[:, None])

    def test_no_full_trajectory_argument_and_callback_receives_only_budgeted_indices(self):
        points, truth = self.fixture()
        def query(ids):
            self.assertLessEqual(len(ids), 5)
            return truth[:, ids, :]
        r = method.allocate_controls(points, query, 5, strategy="residual")
        self.assertEqual(r["query_count"], 5)

    def test_residual_score_changes_allocation_after_observed_nonuniform_motion(self):
        points, truth = self.fixture()
        fps = method.allocate_controls(points, lambda ids: truth[:, ids], 9, strategy="fps")
        adaptive = method.allocate_controls(points, lambda ids: truth[:, ids], 9, strategy="residual")
        self.assertFalse(np.array_equal(fps["sample_indices"], adaptive["sample_indices"]))
        self.assertTrue(any(row["observed_residual_max"] > .01 for row in adaptive["history"]))

    def test_invalid_budget_geometry_and_decoder_shape_are_rejected(self):
        points, truth = self.fixture()
        for budget in [0, 34, 2.5, True]:
            with self.assertRaises(ValueError):
                method.allocate_controls(points, lambda ids: truth[:, ids], budget)
        with self.assertRaises(ValueError):
            method.allocate_controls(np.zeros((3, 3)), lambda ids: truth[:, ids], 2)
        with self.assertRaises(ValueError):
            method.allocate_controls(points, lambda ids: np.full((5, len(ids), 3), np.nan), 2)
        with self.assertRaises(ValueError):
            method.allocate_controls(points, lambda ids: np.zeros((len(ids), 3)), 2)

    def test_distance_chunk_partition_preserves_queries_and_interpolation(self):
        points, truth = self.fixture()
        for strategy in ["fps", "curvature", "motion", "residual"]:
            results = [method.allocate_controls(points, lambda ids: truth[:, ids], 9,
                        strategy=strategy, curvature=np.linspace(0., 1., len(points)),
                        distance_chunk_size=chunk) for chunk in [1, 7, 256]]
            for r in results[1:]:
                np.testing.assert_array_equal(r["sample_indices"], results[0]["sample_indices"])
                np.testing.assert_array_equal(r["trajectories"], results[0]["trajectories"])
        for chunk in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError):
                method.allocate_controls(points, lambda ids: truth[:, ids], 5,
                                         distance_chunk_size=chunk)


if __name__ == "__main__":
    unittest.main()
