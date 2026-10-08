"""Local-only engineering acceptance for the complete C02 construction."""
from __future__ import annotations

import unittest

import numpy as np

from research_math import protected_geometry_candidate as c02


class ProtectedGeometryCandidateTests(unittest.TestCase):
    def setUp(self):
        self.anchor = np.array([
            [0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
            [0., 1., 0.], [1., 1., 0.], [2., 1., 0.],
        ])
        self.faces = np.array([[0, 1, 4], [0, 4, 3],
                               [1, 2, 5], [1, 5, 4]], dtype=np.int64)

    def test_patch_velocity_projection_and_strength_match_certificate(self):
        labels, seeds = c02.deterministic_patches(self.anchor, self.faces, 2)
        self.assertEqual(set(labels.tolist()), {0, 1})
        self.assertEqual(len(seeds), 2)
        area = c02.vertex_area_weights(self.anchor, self.faces)
        rng = np.random.default_rng(19)
        desired = rng.normal(size=(16, len(self.anchor), 3))
        desired[0] = 0.
        projected, diagnostics = c02.protected_projection(desired, area, labels)
        for patch in range(2):
            mask = labels == patch
            centroids = np.einsum(
                "v,tvc->tc", area[mask], projected[:, mask]) / area[mask].sum()
            np.testing.assert_allclose(centroids, 0., rtol=0., atol=diagnostics["numerical_tolerance"])
        self.assertTrue(np.array_equal(projected[0], np.zeros_like(projected[0])))
        self.assertGreaterEqual(diagnostics["rho"], 0.)
        self.assertLessEqual(diagnostics["rho"], 1.)
        self.assertAlmostEqual(
            diagnostics["projected_w_norm_squared"],
            diagnostics["strength_matched_w_norm_squared"], places=10)
        self.assertGreaterEqual(
            diagnostics["strength_matched_local_gain"] + 1e-12,
            diagnostics["protected_local_gain"])

    def test_projection_is_rotation_equivariant_and_trivial_step_is_stable(self):
        labels, _ = c02.deterministic_patches(self.anchor, self.faces, 3)
        area = c02.vertex_area_weights(self.anchor, self.faces)
        desired = np.zeros((16, len(self.anchor), 3), dtype=np.float64)
        projected, diagnostics = c02.protected_projection(desired, area, labels)
        self.assertTrue(np.array_equal(projected, desired))
        self.assertEqual(diagnostics["rho"], 0.)
        desired[1:, :, 0] = np.arange(1, 16)[:, None]
        theta = .71
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                             [np.sin(theta), np.cos(theta), 0.],
                             [0., 0., 1.]])
        left, _ = c02.protected_projection(
            np.einsum("tvi,ij->tvj", desired, rotation), area, labels)
        right, _ = c02.protected_projection(desired, area, labels)
        np.testing.assert_allclose(
            left, np.einsum("tvi,ij->tvj", right, rotation), rtol=1e-12, atol=1e-12)

    def test_patch_builder_rejects_missing_component_seed_and_bad_area(self):
        disconnected_anchor = np.concatenate([self.anchor[:3], self.anchor[:3] + [10., 0., 0.]])
        disconnected_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "seed every connected component"):
            c02.deterministic_patches(disconnected_anchor, disconnected_faces, 1)
        bad = self.anchor.copy(); bad[4] = bad[1]
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            c02.vertex_area_weights(bad, self.faces)

    def test_native_identity_rejects_wrong_frame_or_vertex_mapping(self):
        vertices = np.repeat(self.anchor[None].astype(np.float32), 16, axis=0)
        arrays = {
            "vertices": vertices, "faces": self.faces,
            "timesteps": np.arange(16, dtype=np.float32),
            "frame_indices": np.arange(16, dtype=np.int64),
            "query_vertex_ids": np.arange(len(self.anchor), dtype=np.int64),
        }
        c02.validate_native_arrays(arrays)
        changed = dict(arrays); changed["frame_indices"] = np.arange(16)[::-1]
        with self.assertRaisesRegex(ValueError, "frame order"):
            c02.validate_native_arrays(changed)
        changed = dict(arrays); changed["query_vertex_ids"] = np.arange(len(self.anchor))[::-1]
        with self.assertRaisesRegex(ValueError, "identity"):
            c02.validate_native_arrays(changed)


if __name__ == "__main__":
    unittest.main()
