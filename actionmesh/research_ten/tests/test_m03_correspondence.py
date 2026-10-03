"""Behavior tests: temporal identity, candidate validity and surface coupling."""
import importlib
import itertools
import unittest

import numpy as np

try:
    method = importlib.import_module("research_ten.m03_correspondence")
except ModuleNotFoundError:
    method = None


class CorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(method, "The explicit-candidate correspondence solver is not implemented")

    def test_acceleration_preserves_identity_through_ambiguous_crossing(self):
        # Independent constant-velocity physical tracks; framewise scores swap at crossing.
        candidates = np.zeros((5, 1, 2, 3))
        candidates[:, 0, 0, 0] = [-2, -1, 0, 1, 2]
        candidates[:, 0, 1, 0] = [2, 1, 0, -1, -2]
        costs = np.zeros((5, 1, 2))
        costs[3:, 0, 0] = 0.1
        result = method.select_correspondences(
            candidates, costs, np.ones_like(costs, bool), np.zeros((1, 3)),
            point_ids=[19], anchor_indices=[0], temporal_weight=2., surface_weight=0.)
        np.testing.assert_allclose(result["trajectories"][:, 0, 0], [-2, -1, 0, 1, 2])
        np.testing.assert_array_equal(result["point_ids"], [19])
        self.assertGreater(np.mean((method.top1(candidates, costs, np.ones_like(costs, bool))[3:, 0, 0] - [1, 2]) ** 2), 1.)

    def test_surface_evidence_keeps_adjacent_points_together(self):
        candidates = np.zeros((3, 2, 2, 3))
        candidates[:, 0, 1, 0] = 8
        candidates[:, 1, 0, 0] = 1
        candidates[:, 1, 1, 0] = 9
        costs = np.zeros((3, 2, 2))
        costs[:, 0, 1] = 2
        costs[:, 1, 0] = .1
        result = method.select_correspondences(
            candidates, costs, np.ones_like(costs, bool), [[0, 0, 0], [1, 0, 0]],
            surface_edges=[[0, 1]], temporal_weight=0., surface_weight=1.)
        np.testing.assert_allclose(result["trajectories"][:, :, 0], [[0, 1]] * 3)

    def test_candidate_slot_reordering_does_not_change_selected_physical_track(self):
        c = np.zeros((4, 1, 2, 3))
        c[:, 0, 0, 0] = [0, 1, 2, 3]
        c[:, 0, 1, 0] = [8, 7, 6, 5]
        scores = np.zeros((4, 1, 2))
        ids = np.broadcast_to([41, 82], (4, 1, 2)).copy()
        c[2] = c[2, :, ::-1]
        ids[2] = ids[2, :, ::-1]
        r = method.select_correspondences(c, scores, np.ones_like(scores, bool), [[0, 0, 0]],
                                         anchor_indices=[0], candidate_ids=ids)
        np.testing.assert_array_equal(r["selected_candidate_ids"], [[41]] * 4)

    def test_mask_is_respected_and_all_invalid_is_explicit(self):
        c = np.zeros((3, 1, 2, 3))
        c[:, :, 1, :] = 1
        cost = np.zeros((3, 1, 2))
        valid = np.ones_like(cost, bool)
        valid[:, :, 0] = False
        r = method.select_correspondences(c, cost, valid, [[0, 0, 0]])
        np.testing.assert_array_equal(r["candidate_indices"], np.ones((3, 1), int))
        valid[1] = False
        with self.assertRaises(ValueError):
            method.select_correspondences(c, cost, valid, [[0, 0, 0]])

    def test_dp_matches_exhaustive_temporal_objective(self):
        c = np.zeros((4, 1, 2, 3))
        c[:, 0, :, 0] = [[0, 3], [1, 2], [2, 0], [3, -1]]
        cost = np.array([[[0, .3]], [[.2, 0]], [[.4, 0]], [[.2, 0]]])
        r = method.select_correspondences(c, cost, np.ones_like(cost, bool), [[0, 0, 0]],
                                         temporal_weight=.7, surface_weight=0.)
        objectives = []
        for path in itertools.product(range(2), repeat=4):
            x = np.array([c[t, 0, k] for t, k in enumerate(path)])
            objectives.append(sum(cost[t, 0, k] for t, k in enumerate(path)) +
                              .7 * np.square(np.diff(x, n=2, axis=0)).sum())
        self.assertAlmostEqual(r["objective"], min(objectives), places=10)

    def test_unidentifiable_and_oversized_components_are_not_silent(self):
        c = np.zeros((3, 1, 2, 3))
        c[:, 0, 1, 0] = 1
        cost = np.zeros((3, 1, 2))
        r = method.select_correspondences(c, cost, np.ones_like(cost, bool), [[0, 0, 0]])
        self.assertTrue(r["ambiguous"])
        c = np.zeros((3, 3, 3, 3))
        with self.assertRaises(ValueError):
            method.select_correspondences(c, np.zeros((3, 3, 3)), np.ones((3, 3, 3), bool),
                                          np.zeros((3, 3)), surface_edges=[[0, 1], [1, 2]], max_joint_states=8)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            method.select_correspondences(np.full((3, 1, 1, 3), np.nan),
                                          np.zeros((3, 1, 1)), np.ones((3, 1, 1), bool), [[0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
