"""Local-only numerical, identity and retained-artifact checks for C08."""
from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

from research_math import trajectory_bridge_candidate as c08


def tetrahedral_sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      dtype=np.float32)
    frames = []
    for step in range(16):
        frame = base.copy()
        frame[1, 0] += np.float32(0.01 * step)
        frame[2, 1] += np.float32(0.005 * step)
        frames.append(frame)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
                       dtype=np.int64)
    return np.asarray(frames, dtype=np.float32), faces, np.arange(16, dtype=np.float64)


class C08NumericalTests(unittest.TestCase):
    def test_whole_path_conditional_mean_has_known_two_state_value(self):
        vertices = np.zeros((16, 2, 3), dtype=np.float32)
        vertices[:, 0, 0] = 0.0; vertices[:, 1, 0] = 2.0
        rows = np.asarray([0, 0, 1, 1], dtype=np.int64)
        columns = np.asarray([0, 1, 0, 1], dtype=np.int64)
        kernels = [{"rows": rows, "columns": columns} for _ in range(15)]
        policies = [np.full(4, 0.5, dtype=np.float64) for _ in range(15)]
        output = c08.whole_path_conditional_mean_lift(vertices, kernels, policies)
        np.testing.assert_array_equal(output[0], vertices[0])
        np.testing.assert_allclose(output[1:, :, 0], 1.0, atol=1e-7, rtol=0)

    def test_sparse_witness_and_bridge_close_every_required_marginal(self):
        vertices, faces, _ = tetrahedral_sequence()
        kernels, masses = c08.build_geometry_chain(
            vertices, faces, epsilon=0.5, neighbors=2)
        self.assertEqual(len(kernels), 15)
        for step, kernel in enumerate(kernels):
            row = np.zeros(4); column = np.zeros(4)
            np.add.at(row, kernel["rows"], kernel["witness"])
            np.add.at(column, kernel["columns"], kernel["witness"])
            np.testing.assert_allclose(row, masses[step], atol=1e-13, rtol=1e-12)
            np.testing.assert_allclose(column, masses[step + 1], atol=1e-13, rtol=1e-12)
        policies, diagnostic = c08.solve_endpoint_bridge(
            kernels, masses[0], masses[-1], tolerance=1e-9,
            max_iterations=2000)
        self.assertLessEqual(diagnostic["maximum_absolute_endpoint_residual"], 1e-9)
        marginal = masses[0].copy()
        for kernel, policy in zip(kernels, policies):
            next_marginal = np.zeros(4)
            np.add.at(next_marginal, kernel["columns"], marginal[kernel["rows"]] * policy)
            marginal = next_marginal
        np.testing.assert_allclose(marginal, masses[-1], atol=2e-9, rtol=2e-9)

    def test_uninformative_terminal_collapses_to_reference_chain(self):
        vertices, faces, _ = tetrahedral_sequence()
        kernels, masses = c08.build_geometry_chain(
            vertices, faces, epsilon=0.5, neighbors=2)
        reference = [np.exp(kernel["log_probability"]) for kernel in kernels]
        terminal = masses[0].copy()
        for kernel, policy in zip(kernels, reference):
            observed = np.zeros_like(terminal)
            np.add.at(observed, kernel["columns"], terminal[kernel["rows"]] * policy)
            terminal = observed
        bridge, _ = c08.solve_endpoint_bridge(
            kernels, masses[0], terminal, tolerance=1e-10, max_iterations=2000)
        for observed, expected in zip(bridge, reference):
            np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=1e-9)

    def test_whole_path_outputs_preserve_anchor_shape_and_finiteness(self):
        vertices, faces, times = tetrahedral_sequence()
        outputs, certificates, diagnostics, shared = c08.build_role_sequences(
            vertices, faces, times, epsilon=0.5, neighbors=2,
            tolerance=1e-9, max_iterations=2000, smoothing_strength=0.1)
        self.assertEqual(set(outputs), set(c08.ROLES))
        for role, output in outputs.items():
            self.assertEqual(output.shape, vertices.shape)
            self.assertEqual(output.dtype, np.float32)
            self.assertTrue(np.isfinite(output).all())
            self.assertTrue(np.array_equal(output[0], vertices[0]))
            self.assertEqual(certificates[role]["path_state_ids"].shape, (16, 4))
            if role != "coordinate_smoother":
                self.assertEqual(certificates[role]["witness_sha256"].shape, (15,))
                self.assertLessEqual(
                    float(certificates[role]["witness_row_residual"].max()), 1e-13)
                self.assertLessEqual(
                    float(certificates[role]["witness_column_residual"].max()), 1e-13)
        self.assertTrue(diagnostics["endpoint_bridge"]["probabilistic_lift"])
        self.assertFalse(shared["ground_truth_attention_or_scorer_input"])

    def test_invalid_geometry_and_nonconvergence_fail_closed(self):
        vertices, faces, _ = tetrahedral_sequence()
        broken = vertices.copy(); broken[1, 3] = broken[1, 0]
        with self.assertRaises((ValueError, c08.TrajectoryBridgeError)):
            c08.build_geometry_chain(broken, faces, epsilon=0.5, neighbors=2)
        kernels, masses = c08.build_geometry_chain(
            vertices, faces, epsilon=0.5, neighbors=2)
        with self.assertRaises(c08.TrajectoryBridgeError):
            c08.solve_endpoint_bridge(kernels, masses[0], masses[-1],
                                      tolerance=1e-30, max_iterations=1)


class C08RetainedNativeAcceptance(unittest.TestCase):
    """Run only through the zero-GPU Local acceptance plan."""

    def test_retained_real_artifact_replays_exactly(self):
        root = os.environ.get("C08_NATIVE_ROOT")
        artifact = os.environ.get("C08_NATIVE_ARTIFACT")
        if not root or not artifact:
            self.skipTest("C08 retained-artifact environment is Local-only")
        result = c08.validate_candidate_artifact(Path(root), Path(artifact))
        self.assertEqual(result["completed_roles"], list(c08.ROLES))
        self.assertFalse(result["native_qualified"])
        self.assertFalse(result["scientific_admission"])


if __name__ == "__main__":
    unittest.main()
