"""Authored acceptance checks for C14; Web does not execute this file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math.corotational_residual_candidate import (
    export_corotational_candidate,
    fit_frozen_rigid_factors,
    reconstruct_with_fixed_pose,
    repair_body_residual,
)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class CorotationalResidualCandidateTest(unittest.TestCase):
    def setUp(self):
        self.anchor = np.array([
            [-1.0, -0.5, 0.0], [1.0, -0.5, 0.0],
            [0.7, 0.9, 0.2], [-0.4, 0.8, -0.3],
        ])

    def test_exact_time_varying_rigid_motion_is_fixed_point(self):
        rotations = np.stack([_rotation_z(value) for value in (0.0, 0.4, 1.1, 1.8)])
        translations = np.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.3],
                                 [0.5, 0.4, -0.2], [1.0, 0.1, 0.5]])
        sequence = np.einsum("vi,tij->tvj", self.anchor, rotations) + translations[:, None, :]
        factors = fit_frozen_rigid_factors(sequence)
        self.assertLess(np.max(np.abs(factors["body_residual"])), 1e-12)
        repaired, certificate = repair_body_residual(
            factors["body_residual"], np.arange(4.0),
            weight=0.7, rho=1.0, absolute_tolerance=1e-10,
            relative_tolerance=1e-10, max_iterations=100,
        )
        rebuilt = reconstruct_with_fixed_pose(
            factors["anchor"], repaired, factors["rotation_rows"], factors["centroids"])
        np.testing.assert_allclose(rebuilt, sequence, atol=1e-11, rtol=1e-11)
        self.assertTrue(certificate["converged"])
        self.assertEqual(certificate["termination_reason"], "exact_zero_fixed_point")

    def test_declared_group_tv_is_not_body_gaussian(self):
        residual = np.zeros((6, 2, 3), dtype=np.float64)
        residual[2:4, :, 0] = 1.0
        residual[2:4, :, 1] = 0.5
        repaired, certificate = repair_body_residual(
            residual, np.arange(6.0), weight=0.25, rho=1.0,
            absolute_tolerance=1e-8, relative_tolerance=1e-8,
            max_iterations=2000,
        )
        self.assertTrue(certificate["converged"])
        self.assertEqual(repaired.shape, residual.shape)
        np.testing.assert_array_equal(repaired[0], residual[0])
        # Vector shrinkage keeps XYZ groups coupled and yields a sparse innovation path.
        self.assertGreater(certificate["zero_innovation_groups"], 0)
        self.assertEqual(certificate["operator"], "anchored_nonuniform_group_tv_first_difference")
        np.testing.assert_allclose(repaired[:, :, 0], 2.0 * repaired[:, :, 1],
                                   atol=1e-10, rtol=1e-10)

    def test_nonuniform_timestamps_change_the_declared_derivative_scaling(self):
        residual = np.zeros((5, 3, 3), dtype=np.float64)
        residual[2:, :, 1] = 0.5
        uniform, uniform_certificate = repair_body_residual(
            residual, np.arange(5.0), weight=0.1, rho=1.0,
            absolute_tolerance=1e-8, relative_tolerance=1e-8,
            max_iterations=2000,
        )
        nonuniform, nonuniform_certificate = repair_body_residual(
            residual, np.array([0.0, 0.5, 2.0, 4.5, 8.0]),
            weight=0.1, rho=1.0, absolute_tolerance=1e-8,
            relative_tolerance=1e-8, max_iterations=2000,
        )
        self.assertTrue(uniform_certificate["converged"])
        self.assertTrue(nonuniform_certificate["converged"])
        self.assertFalse(np.allclose(uniform, nonuniform))

    def test_pose_fit_refuses_collinear_anchor(self):
        line = np.zeros((3, 4, 3), dtype=np.float64)
        line[:, :, 0] = np.arange(4.0)
        with self.assertRaisesRegex(ValueError, "rank-deficient"):
            fit_frozen_rigid_factors(line)

    def test_reconstruction_rejects_non_so3_factor(self):
        rotations = np.repeat(np.eye(3)[None], 2, axis=0)
        rotations[1, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, r"SO\(3\)"):
            reconstruct_with_fixed_pose(
                self.anchor, np.zeros((2, 4, 3)), rotations, np.zeros((2, 3)))

    def test_export_preserves_full_native_identity_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "source"
            case.mkdir()
            rotations = np.stack([_rotation_z(index / 20.0) for index in range(16)])
            vertices = (np.einsum("vi,tij->tvj", self.anchor, rotations)
                        + np.arange(16.0)[:, None, None] * np.array([0.01, 0.02, 0.0]))
            arrays = {
                "vertices": vertices.astype(np.float32),
                "faces": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
                "timesteps": np.arange(16, dtype=np.float64),
                "frame_indices": np.arange(16, dtype=np.int64),
                "query_vertex_ids": np.arange(4, dtype=np.int64),
            }
            np.savez_compressed(case / "sequence.npz", **arrays)
            sequence_sha = hashlib.sha256((case / "sequence.npz").read_bytes()).hexdigest()
            (case / "report.json").write_text(json.dumps({
                "status": "completed", "uid": "unit-001", "seed": 42,
                "sha256": {"sequence.npz": sequence_sha},
            }))
            output = root / "output"
            summary = export_corotational_candidate(
                case / "sequence.npz", output, uid="unit-001",
                expected_sequence_sha256=sequence_sha, weight=0.2, rho=1.0,
                absolute_tolerance=1e-7, relative_tolerance=1e-7,
                max_iterations=500,
            )
            self.assertEqual(summary["status"], "completed")
            self.assertFalse(summary["native_qualified"])
            with np.load(output / "declared_body_residual_repair" / "sequence.npz") as saved:
                np.testing.assert_array_equal(saved["faces"], arrays["faces"])
                np.testing.assert_array_equal(saved["frame_indices"], arrays["frame_indices"])
                np.testing.assert_array_equal(saved["query_vertex_ids"], arrays["query_vertex_ids"])
                self.assertEqual(saved["vertices"].dtype, np.float32)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["cases"], [{
                "case_id": "unit-001-declared_body_residual_repair",
                "uid": "unit-001", "case_dir": "declared_body_residual_repair",
            }])
            report = json.loads((output / "declared_body_residual_repair" / "report.json").read_text())
            self.assertEqual(report["method_id"], "4d-math-20261006-c14")
            self.assertEqual(report["pose_source"], "predicted_sequence_geometry_only")
            self.assertEqual(report["scientific_verdict"], "not_computed")

    def test_nonconvergence_is_retained_as_failed_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "source"
            case.mkdir()
            sequence = np.repeat(self.anchor[None], 16, axis=0).astype(np.float32)
            sequence[8:, 2, 2] += 0.5
            np.savez_compressed(
                case / "sequence.npz", vertices=sequence,
                faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
                timesteps=np.arange(16, dtype=np.float64),
                frame_indices=np.arange(16, dtype=np.int64),
                query_vertex_ids=np.arange(4, dtype=np.int64),
            )
            sequence_sha = hashlib.sha256((case / "sequence.npz").read_bytes()).hexdigest()
            (case / "report.json").write_text(json.dumps({
                "status": "completed", "uid": "unit-002", "seed": 42,
                "sha256": {"sequence.npz": sequence_sha},
            }))
            summary = export_corotational_candidate(
                case / "sequence.npz", root / "output", uid="unit-002",
                expected_sequence_sha256=sequence_sha, weight=0.2, rho=1.0,
                absolute_tolerance=1e-30, relative_tolerance=1e-30,
                max_iterations=1,
            )
            self.assertEqual(summary["status"], "incomplete")
            report = json.loads((root / "output" / "declared_body_residual_repair" / "report.json").read_text())
            self.assertEqual(report["status"], "error")
            self.assertIn("did not converge", report["error"])
            self.assertEqual(report["solver"]["termination_reason"], "iteration_limit")
            self.assertEqual(report["solver"]["iterations"], 1)
            for name in ("objective", "primal_residual", "dual_residual",
                         "primal_tolerance", "dual_tolerance",
                         "linear_system_condition"):
                self.assertIn(name, report["solver"])
            self.assertFalse((root / "output" / "declared_body_residual_repair" / "sequence.npz").exists())


if __name__ == "__main__":
    unittest.main()
