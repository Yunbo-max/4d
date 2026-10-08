"""Engineering contracts for C10; these fixtures are not native evidence."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


try:
    candidate = importlib.import_module("research_math.integrable_gradient_candidate")
except ModuleNotFoundError:
    candidate = None


class IntegrableGradientCandidateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(candidate, "C10 integrable-gradient candidate is missing")
        anchor = np.array([
            [0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.],
            [5., 0., 0.], [6., 0., 0.], [5., 2., 0.],
        ], dtype=np.float32)
        self.vertices = np.repeat(anchor[None], 16, axis=0)
        self.vertices[:, 2, 2] = np.linspace(0., .5, 16, dtype=np.float32)
        self.faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]], dtype=np.int64)
        self.times = np.arange(16, dtype=np.float32)

    def common(self):
        return candidate.generate_common_differential_target(
            self.vertices, self.faces, target_strength=.5,
            max_relative_change=.25)

    def source(self, root):
        case = root / "source"
        case.mkdir()
        np.savez_compressed(
            case / "sequence.npz", vertices=self.vertices, faces=self.faces,
            frame_indices=np.arange(16), timesteps=self.times,
            query_vertex_ids=np.arange(len(self.vertices[0])))
        digest = hashlib.sha256((case / "sequence.npz").read_bytes()).hexdigest()
        (case / "report.json").write_text(json.dumps({
            "status": "completed", "uid": "fixture", "seed": 42,
            "sha256": {"sequence.npz": digest}}))
        return case, digest

    def test_geometry_only_target_is_deterministic_and_component_pinned(self):
        first = self.common()
        second = self.common()
        for key in ("edges", "target", "weights", "component_labels", "pins",
                    "source_edge_lengths", "desired_edge_lengths"):
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first["pins"], np.array([0, 4]))
        self.assertTrue(np.all(first["weights"] > 0.))
        self.assertEqual(first["weights"].shape, (len(first["edges"]),))
        np.testing.assert_array_equal(first["target"][0],
                                      np.zeros_like(first["target"][0]))
        self.assertGreater(np.linalg.norm(first["target"]), 0.)
        # A local edge-length rule is generally cycle-inconsistent on this mesh.
        triangle = {(int(a), int(b)): index for index, (a, b) in enumerate(first["edges"])}
        cycle = (first["target"][:, triangle[(0, 1)]]
                 + first["target"][:, triangle[(1, 2)]]
                 - first["target"][:, triangle[(0, 2)]])
        self.assertGreater(np.linalg.norm(cycle), 0.)

    def test_three_arms_share_target_but_have_distinct_constructions(self):
        common = self.common()
        arms = candidate.construct_all_arms(
            self.vertices, self.faces, common, absolute_tolerance=1e-10,
            relative_tolerance=1e-9, max_iterations=10000)
        self.assertEqual(tuple(arms), candidate.ROLES)
        for role, (displacement, diagnostics) in arms.items():
            self.assertEqual(displacement.shape, self.vertices.shape)
            np.testing.assert_array_equal(displacement[:, common["pins"]], 0.)
            self.assertEqual(diagnostics["pin_residual_linf"], 0.)
            self.assertIn("construction", diagnostics)
            self.assertTrue(np.isfinite(displacement).all(), role)
        self.assertFalse(np.array_equal(
            arms["direct_common_lift"][0], arms["independent_local_repair"][0]))
        self.assertLessEqual(
            arms["pinned_integrable_solve"][1]["weighted_residual_l2"],
            arms["direct_common_lift"][1]["weighted_residual_l2"] + 1e-9)

    def test_integrable_input_is_recovered_up_to_numerical_tolerance(self):
        common = self.common()
        displacement = np.zeros_like(self.vertices, dtype=np.float64)
        displacement[:, 1, 0] = .2
        displacement[:, 2, 1] = -.1
        displacement[:, 3, 2] = .3
        displacement[:, 5, 0] = -.4
        displacement[:, 6, 1] = .15
        common = {**common, "target": candidate._realized(displacement, common["edges"]),
                  "weights": np.ones_like(common["weights"])}
        solved, diagnostics = candidate.pinned_integrable_solve(
            common, self.vertices.shape[1], absolute_tolerance=1e-12,
            relative_tolerance=1e-11, max_iterations=10000)
        np.testing.assert_allclose(solved, displacement, atol=1e-9, rtol=1e-9)
        self.assertLess(diagnostics["weighted_residual_l2"], 1e-8)
        self.assertLess(diagnostics["normal_equation_residual_l2"], 1e-8)

    def test_full_export_preserves_16_frames_topology_ids_and_recomputes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, digest = self.source(root)
            output = root / "artifact"
            result = candidate.export_integrable_candidate(
                case, output, uid="fixture", expected_sequence_sha256=digest,
                source_sequence_ref="source/sequence.npz",
                source_report_ref="source/report.json",
                target_strength=.5, max_relative_change=.25,
                absolute_tolerance=1e-10, relative_tolerance=1e-9,
                max_iterations=10000, coordinate_lower=-10., coordinate_upper=10.,
                bounds_policy="preserve_and_report", max_artifact_bytes=16 * 1024 * 1024)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["roles"], list(candidate.ROLES))
            record = candidate.validate_candidate_artifact(root, output / "candidate.json")
            self.assertEqual(record["status"], "completed")
            self.assertEqual([arm["status"] for arm in record["arms"]],
                             ["completed"] * 3)
            self.assertIn("artifact_archive", record)
            materialized = root / "materialized"
            rows = candidate.materialize_candidate_archive(
                output / "artifact.tar", materialized,
                max_artifact_bytes=16 * 1024 * 1024,
                max_member_bytes=8 * 1024 * 1024)
            self.assertTrue(rows)
            rematerialized = candidate.validate_candidate_artifact(
                root, materialized / "candidate.json")
            self.assertEqual(rematerialized["status"], "completed")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual([case["arm_role"] for case in manifest["cases"]],
                             list(candidate.ROLES))
            for role in candidate.ROLES:
                with np.load(output / role / "sequence.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["vertices"].shape, self.vertices.shape)
                    np.testing.assert_array_equal(saved["faces"], self.faces)
                    np.testing.assert_array_equal(saved["frame_indices"], np.arange(16))
                    np.testing.assert_array_equal(
                        saved["query_vertex_ids"], np.arange(self.vertices.shape[1]))
                    np.testing.assert_array_equal(
                        saved["vertices"][:, [0, 4]], self.vertices[:, [0, 4]])

    def test_failed_bounds_roles_remain_terminal_without_scoreable_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, digest = self.source(root)
            output = root / "artifact"
            result = candidate.export_integrable_candidate(
                case, output, uid="fixture", expected_sequence_sha256=digest,
                source_sequence_ref="source/sequence.npz",
                source_report_ref="source/report.json",
                target_strength=.5, max_relative_change=.25,
                absolute_tolerance=1e-10, relative_tolerance=1e-9,
                max_iterations=10000, coordinate_lower=-.01, coordinate_upper=.01,
                bounds_policy="reject", max_artifact_bytes=16 * 1024 * 1024)
            self.assertEqual(result["status"], "incomplete")
            validated = candidate.validate_candidate_artifact(root, output / "candidate.json")
            self.assertEqual(validated["status"], "incomplete")
            self.assertEqual([arm["status"] for arm in validated["arms"]], ["error"] * 3)
            for role in candidate.ROLES:
                report = json.loads((output / role / "report.json").read_text())
                self.assertEqual(report["status"], "error")
                self.assertFalse((output / role / "sequence.npz").exists())
                self.assertFalse((output / role / "certificate.npz").exists())

    def test_tampered_target_and_source_contract_fail_closed(self):
        with self.assertRaises(ValueError):
            candidate.generate_common_differential_target(
                self.vertices.astype(np.float64), self.faces,
                target_strength=.5, max_relative_change=.25)
        collapsed = self.vertices.copy()
        collapsed[:, 1] = collapsed[:, 0]
        with self.assertRaises(ValueError):
            candidate.generate_common_differential_target(
                collapsed, self.faces, target_strength=.5, max_relative_change=.25)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, digest = self.source(root)
            with self.assertRaises(ValueError):
                candidate.export_integrable_candidate(
                    case, root / "artifact", uid="fixture",
                    expected_sequence_sha256="0" * 64,
                    source_sequence_ref="source/sequence.npz",
                    source_report_ref="source/report.json",
                    target_strength=.5, max_relative_change=.25,
                    absolute_tolerance=1e-10, relative_tolerance=1e-9,
                    max_iterations=10000, coordinate_lower=-10., coordinate_upper=10.,
                    bounds_policy="preserve_and_report",
                    max_artifact_bytes=16 * 1024 * 1024)
            self.assertFalse((root / "artifact").exists())


if __name__ == "__main__":
    unittest.main()
