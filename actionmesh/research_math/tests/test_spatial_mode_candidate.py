"""Authored, generated-unexecuted tests for the C05 mathematical core."""
import unittest

import numpy as np

from research_math.spatial_mode_candidate import (
    METHOD_IDS,
    cluster_empirical_trajectories,
    independent_top1,
    localized_mean,
    normalized_mode_separations,
    project_to_sampled_surface_union,
    select_label_trajectories,
    solve_joint_labels_icm,
    spatial_mode_energy,
    temperature_mean,
    union_surface_projected_mean,
    validate_native_arrays,
)


def _bank():
    # K=3, T=2, L=2.  Draws 10/30 form the large near-zero cluster and
    # draw 20 is the observed separated alternative at both landmarks.
    draws = np.asarray([
        [[[0.00, 0.00, 0.00], [1.00, 0.00, 0.00]],
         [[0.00, 0.00, 0.00], [1.00, 0.00, 0.00]]],
        [[[0.02, 0.00, 0.00], [1.02, 0.00, 0.00]],
         [[0.02, 0.00, 0.00], [1.02, 0.00, 0.00]]],
        [[[2.00, 0.00, 0.00], [3.00, 0.00, 0.00]],
         [[2.00, 0.00, 0.00], [3.00, 0.00, 0.00]]],
    ])
    return cluster_empirical_trajectories(
        draws, draw_seeds=(30, 10, 20), landmark_scales=(1.0, 1.0), cluster_radius=0.05)


class EmpiricalModesTest(unittest.TestCase):
    def test_declared_method_roles_and_native_shape_boundary(self):
        self.assertEqual(tuple(METHOD_IDS), (
            "localized_mean", "temperature_matched_mean", "surface_projected_mean",
            "independent_top1", "joint_spatial_labels"))
        self.assertEqual(METHOD_IDS["joint_spatial_labels"],
                         "4d-math-20261006-c05")
        surfaces = np.zeros((2, 2, 3, 3), dtype=np.float32)
        native = validate_native_arrays(
            surfaces,
            anchor_vertices=np.zeros((3, 3), dtype=np.float32),
            faces=np.asarray([[0, 1, 2]]),
            target_frame_indices=np.asarray([1, 2]),
            target_timesteps=np.asarray([0.5, 1.0]),
            query_vertex_ids=np.asarray([10, 20, 30]),
            draw_seeds=(101, 202),
        )
        self.assertFalse(native["sampled_surfaces"].flags.writeable)
        with self.assertRaisesRegex(ValueError, "distinct integer"):
            validate_native_arrays(
                surfaces, anchor_vertices=np.zeros((3, 3)), faces=np.asarray([[0, 1, 2]]),
                target_frame_indices=np.asarray([1, 2]), target_timesteps=np.asarray([0.5, 1.0]),
                query_vertex_ids=np.asarray([10, 20, 30]), draw_seeds=(101, 101))

    def test_cluster_masses_are_empirical_and_atoms_are_observed_medoids(self):
        bank = _bank()
        self.assertEqual(bank.draw_seeds, (10, 20, 30))
        for mode in bank.modes:
            np.testing.assert_allclose(mode.masses, (2.0 / 3.0, 1.0 / 3.0))
            self.assertEqual(mode.member_draw_indices, ((0, 2), (1,)))
            # Equal medoid distance within the first cluster uses lower seed 10.
            self.assertEqual(mode.medoid_draw_indices, (0, 1))
        np.testing.assert_allclose(bank.modes[0].atoms[0, :, 0], (0.02, 0.02))
        np.testing.assert_allclose(bank.modes[0].atoms[1, :, 0], (2.0, 2.0))

    def test_draw_order_does_not_change_bank(self):
        bank = _bank()
        # Reconstruct canonical draws in seed order, then permute again.
        canonical = np.asarray([
            [[[0.02, 0, 0], [1.02, 0, 0]], [[0.02, 0, 0], [1.02, 0, 0]]],
            [[[2.00, 0, 0], [3.00, 0, 0]], [[2.00, 0, 0], [3.00, 0, 0]]],
            [[[0.00, 0, 0], [1.00, 0, 0]], [[0.00, 0, 0], [1.00, 0, 0]]],
        ])
        other = cluster_empirical_trajectories(
            canonical, draw_seeds=(10, 20, 30), landmark_scales=(1, 1), cluster_radius=0.05)
        for left, right in zip(bank.modes, other.modes):
            np.testing.assert_array_equal(left.atoms, right.atoms)
            np.testing.assert_array_equal(left.masses, right.masses)

    def test_reductions_use_the_same_bank(self):
        bank = _bank()
        top1, labels = independent_top1(bank)
        np.testing.assert_array_equal(top1, select_label_trajectories(bank, labels))
        local, support = localized_mean(bank, radius=0.1)
        self.assertEqual(support, ((0,), (0,)))
        np.testing.assert_array_equal(local, top1)
        ordinary = temperature_mean(bank, temperature=1.0)
        hotter = temperature_mean(bank, temperature=2.0)
        self.assertGreater(float(hotter[0, 0, 0]), float(ordinary[0, 0, 0]))

    def test_mode_separation_is_measured_between_observed_medoids(self):
        bank = _bank()
        separation = normalized_mode_separations(bank)
        self.assertEqual(separation.shape, (2,))
        # The first observed medoids are x=0.02 and x=2.0 for all frames.
        np.testing.assert_allclose(separation, (1.98, 1.98))
        self.assertFalse(separation.flags.writeable)


class JointEnergyTest(unittest.TestCase):
    def test_icm_is_monotone_and_returns_a_one_flip_certificate(self):
        bank = _bank()
        edges = np.asarray([[0, 1]], dtype=np.int64)
        anchor = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        rotations = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
        result = solve_joint_labels_icm(
            bank,
            edges=edges,
            anchor_vertices=anchor,
            rotations=rotations,
            time_weights=(1, 1),
            unary_weight=0.01,
            spatial_weight=1.0,
            max_sweeps=10,
            additional_starts=(("mixed", (0, 1)),),
        )
        self.assertEqual(result.labels[0], result.labels[1])
        self.assertLessEqual(result.one_flip_residual, 1e-12)
        for run in result.runs:
            self.assertTrue(all(right <= left + 1e-12
                                for left, right in zip(run.objective_trace, run.objective_trace[1:])))
        direct = spatial_mode_energy(
            bank, result.labels, edges=edges, anchor_vertices=anchor,
            rotations=rotations, time_weights=(1, 1), unary_weight=0.01,
            spatial_weight=1.0)
        self.assertAlmostEqual(result.energy["total"], direct["total"])

    def test_improper_rotation_hints_are_rejected(self):
        bank = _bank()
        reflection = np.broadcast_to(np.diag((1.0, 1.0, -1.0)), (1, 2, 3, 3)).copy()
        with self.assertRaisesRegex(ValueError, "Proper orthogonal"):
            spatial_mode_energy(
                bank, (0, 0), edges=np.asarray([[0, 1]]),
                anchor_vertices=np.asarray([[0, 0, 0], [1, 0, 0]]),
                rotations=reflection, time_weights=(1, 1))


class SurfaceProjectionTest(unittest.TestCase):
    def test_projection_uses_union_and_has_deterministic_ties(self):
        points = np.asarray([[[0.25, 0.25, 1.0]]])
        base = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        surfaces = np.stack((base[None], (base + [0, 0, 2])[None]))
        projected, evidence = project_to_sampled_surface_union(
            points, surfaces, np.asarray([[0, 1, 2]]), face_chunk_size=1)
        # Point is equidistant; exact tie resolves to draw zero.
        np.testing.assert_allclose(projected, [[[0.25, 0.25, 0.0]]])
        self.assertEqual(int(evidence["draw_indices"][0, 0]), 0)
        self.assertEqual(int(evidence["face_indices"][0, 0]), 0)
        np.testing.assert_allclose(evidence["barycentrics"][0, 0], (0.5, 0.25, 0.25))

    def test_projected_mean_is_not_projection_to_the_mean_mesh(self):
        draws = np.asarray([
            [[[0.25, 0.25, 0.0]]],
            [[[0.25, 0.25, 2.0]]],
        ])
        bank = cluster_empirical_trajectories(
            draws, draw_seeds=(1, 2), landmark_scales=(1,), cluster_radius=0.1)
        base = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        surfaces = np.stack((base[None], (base + [0, 0, 2])[None]))
        projected, evidence = union_surface_projected_mean(
            bank, localized_radius=3.0, sampled_surfaces=surfaces,
            faces=np.asarray([[0, 1, 2]]), face_chunk_size=1)
        np.testing.assert_allclose(projected, [[[0.25, 0.25, 0.0]]])
        self.assertAlmostEqual(float(evidence["squared_distances"][0, 0]), 1.0)


if __name__ == "__main__":  # pragma: no cover - authored Local entry only
    unittest.main()
