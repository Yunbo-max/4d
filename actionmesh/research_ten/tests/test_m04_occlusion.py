"""Behavior tests: two-sided evidence, exact anchors and neighbor-supported repair."""
import importlib
import unittest
import numpy as np

try:
    method = importlib.import_module("research_ten.m04_occlusion")
except ModuleNotFoundError:
    method = None


class OcclusionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(method, "The constrained occlusion repair is not implemented")

    def test_visible_samples_exact_and_neighbor_recovers_nonlinear_gap(self):
        tracks = np.zeros((5, 2, 3))
        tracks[:, 0, 0] = [0, 9, 9, 9, 4]
        tracks[:, 1, 0] = [0, 1, 2, 3, 4]
        tracks[:, 1, 1] = [1, 2, 3, 2, 1]
        visible = np.ones((5, 2), bool)
        visible[1:4, 0] = False
        r = method.repair_occlusions(tracks, visible, surface_edges=[[0, 1]],
                                     temporal_weight=.01, surface_weight=10.)
        np.testing.assert_array_equal(r["trajectories"][visible], tracks[visible])
        truth = np.column_stack(([0, 1, 2, 3, 4], [0, 1, 2, 1, 0], np.zeros(5)))
        linear = method.linear_interpolation(tracks, visible)["trajectories"]
        self.assertLess(np.square(r["trajectories"][:, 0] - truth).sum(),
                        np.square(linear[:, 0] - truth).sum() * .05)
        self.assertEqual(r["repaired_count"], 3)

    def test_one_sided_gap_abstains(self):
        tracks = np.arange(18.).reshape(6, 1, 3)
        visible = np.array([[0], [0], [1], [1], [0], [0]], bool)
        r = method.repair_occlusions(tracks, visible)
        np.testing.assert_array_equal(r["trajectories"], tracks)
        self.assertEqual(r["repaired_count"], 0)
        self.assertEqual(r["abstained_count"], 4)

    def test_plain_temporal_baseline_matches_linear_for_affine_motion(self):
        tracks = np.zeros((5, 1, 3))
        tracks[-1, 0, 0] = 4
        visible = np.array([[1], [0], [0], [0], [1]], bool)
        r = method.repair_occlusions(tracks, visible, surface_weight=0.)
        np.testing.assert_allclose(r["trajectories"][:, 0, 0], [0, 1, 2, 3, 4], atol=1e-10)

    def test_irregular_timestamps_and_multiple_gaps(self):
        tracks = np.zeros((6, 1, 3))
        tracks[:, 0, 0] = [0, 99, 2, 4, 99, 10]
        visible = np.array([[1], [0], [1], [1], [0], [1]], bool)
        r = method.linear_interpolation(tracks, visible, timestamps=[0, 1, 2, 4, 7, 10])
        np.testing.assert_allclose(r["trajectories"][:, 0, 0], [0, 1, 2, 4, 7, 10])

    def test_bad_visibility_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            method.repair_occlusions(np.zeros((4, 1, 3)), np.ones((4, 1)) * .3)
        with self.assertRaises(ValueError):
            method.repair_occlusions(np.full((4, 1, 3), np.nan), np.ones((4, 1), bool))

    def test_long_closed_gap_abstains_before_dense_solve(self):
        tracks = np.arange(900., dtype=float).reshape(300, 1, 3)
        visible = np.zeros((300, 1), bool)
        visible[[0, -1]] = True
        r = method.repair_occlusions(tracks, visible)
        np.testing.assert_array_equal(r["trajectories"], tracks)
        self.assertEqual(r["repaired_count"], 0)
        self.assertEqual(r["abstained_count"], 298)
        self.assertEqual(r["intervals"][0]["status"], "gap_budget_exceeded")
        short = tracks[:6].copy()
        short_visible = np.zeros((6, 1), bool)
        short_visible[[0, -1]] = True
        limited = method.repair_occlusions(short, short_visible, max_gap_frames=3)
        self.assertEqual(limited["repaired_count"], 0)
        for limit in [0, -1, 2.5, True]:
            with self.assertRaises(ValueError):
                method.repair_occlusions(short, short_visible, max_gap_frames=limit)


if __name__ == "__main__":
    unittest.main()
