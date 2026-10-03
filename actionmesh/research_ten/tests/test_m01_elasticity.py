"""Independent fixtures for observation-supported ARAP."""
import importlib
import unittest

import numpy as np


def api():
    try:
        return importlib.import_module("research_ten.m01_elasticity")
    except ModuleNotFoundError:
        raise AssertionError("observation-supported ARAP implementation is missing")


def fixture():
    tetra = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rest = np.concatenate([tetra, tetra + [3, 0, 0]])
    edges = np.array([(a + k, b + k) for k in (0, 4)
                      for a in range(4) for b in range(a + 1, 4)])
    target = rest.copy()
    for k in (0, 4):
        center = rest[k:k + 4].mean(0)
        target[k:k + 4] = center + 1.4 * (rest[k:k + 4] - center)
    return rest, np.stack([rest, target]), edges


class ElasticityTests(unittest.TestCase):
    def test_observed_evidence_selects_strain_not_motion_amplitude(self):
        m = api()
        rest, track, edges = fixture()
        uv = track[..., :2].copy()
        uv[1, 4:] = rest[4:, :2]
        weights = m.observation_supported_weights(
            rest, track, edges, uv, lambda x, t: x[:, :2], np.ones((2, 8)),
            stiffness=30, min_stiffness=.05, evidence_floor=.001)
        self.assertGreater(weights.support[1, :4].mean(), .5)
        self.assertEqual(float(weights.support[1, 4:].max()), 0.)
        self.assertLess(weights.edge_weights[1, :6].mean(),
                        weights.edge_weights[1, 6:].mean())
        hidden = m.observation_supported_weights(
            rest, track, edges, np.full((2, 8, 2), np.nan),
            lambda x, t: x[:, :2], np.zeros((2, 8)), stiffness=30)
        np.testing.assert_array_equal(hidden.edge_weights, 30.)

    def test_arap_preserves_rigid_motion_nonuniform_time_and_exact_anchor(self):
        m = api()
        rest, _, edges = fixture()
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        track = np.stack([rest, rest @ rotation.T + [2, -3, .5], rest + [5, 1, 2]])
        result = m.arap_fit(rest, track, edges, 20., temporal_weight=3.,
                            times=np.array([0., .2, 1.]), iterations=10)
        np.testing.assert_allclose(result.trajectories, track, atol=1e-8)
        np.testing.assert_array_equal(result.trajectories[0], track[0])

    def test_local_elasticity_retains_supported_action_and_regularizes_noise(self):
        m = api()
        rest, track, edges = fixture()
        weights = np.full((2, len(edges)), 40.)
        weights[1, :6] = .001
        fitted = m.arap_fit(rest, track, edges, weights, iterations=20).trajectories
        uniform = m.arap_fit(rest, track, edges, 40., iterations=20).trajectories
        self.assertLess(np.linalg.norm(fitted[1, :4] - track[1, :4]), .01)
        self.assertGreater(np.linalg.norm(uniform[1, :4] - track[1, :4]), .3)
        self.assertLess(np.linalg.norm(fitted[1, 4:] - rest[4:]), .02)
        np.testing.assert_array_equal(fitted[0], rest)

    def test_shuffled_control_preserves_weight_distribution(self):
        m = api()
        rest, track, edges = fixture()
        weights = np.arange(24.).reshape(2, 12) + 1
        controls = m.stiffness_controls(rest, track, edges, weights, uniform_grid=(1., 5.))
        np.testing.assert_array_equal(np.sort(controls["shuffled"], axis=1), weights)
        self.assertFalse(np.array_equal(controls["shuffled"], weights))
        self.assertIn("motion", controls)
        self.assertIn("uniform_5", controls)

    def test_invalid_time_and_edges_are_rejected(self):
        m = api()
        rest, track, edges = fixture()
        with self.assertRaises(ValueError):
            m.arap_fit(rest, track, edges, 1., times=[0, 0])
        with self.assertRaises(ValueError):
            m.arap_fit(rest, track, [[0, 100]], 1.)

    def test_face_edges_and_fixed_vertices_are_respected(self):
        m = api()
        rest, track, _ = fixture()
        edges = m.mesh_edges(np.array([[0, 1, 2], [2, 1, 3]]), len(rest))
        self.assertEqual(len(edges), 5)
        fitted = m.arap_fit(rest, track, edges, 50., fixed_vertices=[0]).trajectories
        np.testing.assert_array_equal(fitted[:, 0], track[:, 0])
        with self.assertRaises(ValueError):
            m.mesh_edges(np.array([[0, 0, 2]]), len(rest))
        with self.assertRaises(ValueError):
            m.arap_fit(rest, track, edges, 1., cg_maxiter=1.5)


if __name__ == "__main__":
    unittest.main()
