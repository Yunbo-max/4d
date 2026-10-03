"""Sim(3) controls with known rigid motion and independent outliers."""
import importlib
import unittest

import numpy as np


def api():
    try:
        return importlib.import_module("research_ten.m05_scale")
    except ModuleNotFoundError:
        raise AssertionError("stable-region scale decomposition is missing")


class ScaleTests(unittest.TestCase):
    def test_similarity_recovers_known_transform_despite_stable_region_outlier(self):
        m = api()
        points = np.random.default_rng(8).normal(size=(30, 3))
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        target = 1.7 * points @ rotation.T + [3, 4, -2]
        target[-1] += [40, -60, 90]
        fit = m.fit_similarity(points, target, robust=True)
        self.assertAlmostEqual(fit.scale, 1.7, places=6)
        np.testing.assert_allclose(fit.rotation, rotation, atol=1e-6)
        np.testing.assert_allclose(fit.translation, [3, 4, -2], atol=1e-6)
        self.assertLess(fit.weights[-1], .01)

    def test_scale_fix_keeps_translation_rotation_and_articulation(self):
        m = api()
        rest = np.array([[-1., -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [3, 0, 0]])
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        deformed = rest.copy()
        deformed[-1] += [0, 2, 1]
        truth = deformed @ rotation.T + [5, -2, 3]
        distorted = 1.4 * deformed @ rotation.T + [5, -2, 3]
        result = m.stabilize_scale(rest, np.stack([rest, distorted]),
                                   stable_mask=[True, True, True, True, False])
        np.testing.assert_allclose(result.corrected[1], truth, atol=1e-8)
        np.testing.assert_array_equal(result.corrected[0], rest)
        np.testing.assert_allclose(result.centers[1], [5, -2, 3], atol=1e-8)
        self.assertAlmostEqual(result.scales[1], 1.4, places=8)

    def test_rigid_sequence_is_identity_and_missing_stability_abstains(self):
        m = api()
        rest = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        track = np.stack([rest, rest + [5, -2, 3]])
        result = m.stabilize_scale(rest, track, stable_mask=np.ones(4, bool))
        np.testing.assert_allclose(result.corrected, track, atol=1e-10)
        with self.assertRaises(ValueError):
            m.stabilize_scale(rest, track, stable_mask=None)
        with self.assertRaises(ValueError):
            m.stabilize_scale(rest, track, stable_mask=[True, False, False, False])
        with self.assertRaises(ValueError):
            m.stabilize_scale(rest, track, stable_mask=np.ones(4, bool), camera_convention="unknown")

    def test_bbox_and_global_controls_are_executable(self):
        m = api()
        rest = np.array([[-1., -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [3, 0, 0]])
        target = rest.copy()
        target[-1] += [5, 0, 0]
        track = np.stack([rest, target])
        controls = m.scale_controls(rest, track)
        np.testing.assert_array_equal(controls["original"], track)
        np.testing.assert_array_equal(controls["bbox"][0], rest)
        self.assertLess(np.linalg.norm(controls["bbox"][1, 0] - controls["bbox"][1, 1]), 2.)
        self.assertIn("global_sim3", controls)

    def test_fractional_masks_and_degenerate_region_are_rejected(self):
        m = api()
        rest = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            m.stabilize_scale(rest, rest[None], [.5, 1, 1, 0])
        line = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaises(ValueError):
            m.fit_similarity(line, 2 * line)
        with self.assertRaises(ValueError):
            m.fit_similarity(rest, rest, weights=np.zeros(4))


if __name__ == "__main__":
    unittest.main()
