"""Generated Local acceptance contracts for C12; not executed by Web."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math import area_admission_candidate as candidate


class AreaAdmissionCandidateTests(unittest.TestCase):
    def sequence(self):
        anchor = np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
        ], dtype=np.float32)
        vertices = np.repeat(anchor[None], 16, axis=0)
        vertices[:, :, 2] = np.linspace(0., .15, 16, dtype=np.float32)[:, None]
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        return vertices, faces

    def source(self, root: Path):
        case = root / "source"; case.mkdir()
        vertices, faces = self.sequence()
        np.savez_compressed(
            case / "sequence.npz", vertices=vertices, faces=faces,
            timesteps=np.arange(16, dtype=np.float32),
            frame_indices=np.arange(16, dtype=np.int64),
            query_vertex_ids=np.arange(len(vertices[0]), dtype=np.int64))
        digest = hashlib.sha256((case / "sequence.npz").read_bytes()).hexdigest()
        (case / "report.json").write_text(json.dumps({
            "status": "completed", "uid": "fixture-c12", "seed": 42,
            "sha256": {"sequence.npz": digest}}) + "\n")
        return case, digest

    def test_exact_quadratic_finds_crossing_and_ignores_tangency(self):
        vertices = np.repeat(np.array([[
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
        ]]), 16, axis=0)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        desired = np.zeros_like(vertices)
        desired[:, 2, 1] = -2.0
        coefficients = candidate.form_exact_face_quadratics(
            vertices, desired, faces, beta=.2, degeneracy_epsilon=1e-12)
        crossing = candidate.find_first_violation_interval(
            coefficients, root_tolerance=1e-12)
        self.assertAlmostEqual(crossing["boundary_alpha"], .4, places=10)
        self.assertEqual(crossing["boundary_kind"], "crossing")

        tangent = {
            "q0": np.array([[.25]]), "q1": np.array([[-1.]]),
            "q2": np.array([[1.]]), "scale": np.ones((1, 1)),
        }
        admitted = candidate.find_first_violation_interval(
            tangent, root_tolerance=1e-12)
        self.assertEqual(admitted["boundary_alpha"], 1.0)
        self.assertEqual(admitted["crossing_count"], 0)
        self.assertEqual(admitted["tangency_count"], 1)

        endpoint = {
            "q0": np.array([[1.]]), "q1": np.array([[-1.]]),
            "q2": np.array([[0.]]), "scale": np.ones((1, 1)),
        }
        endpoint_crossing = candidate.find_first_violation_interval(
            endpoint, root_tolerance=1e-12)
        self.assertEqual(endpoint_crossing["boundary_alpha"], 1.0)
        self.assertEqual(endpoint_crossing["boundary_kind"], "crossing")

        delta = 1e-14
        near_tangent = {
            "q0": np.array([[.25 - delta]]), "q1": np.array([[-1.]]),
            "q2": np.array([[1.]]), "scale": np.ones((1, 1)),
        }
        narrow_negative_interval = candidate.find_first_violation_interval(
            near_tangent, root_tolerance=1e-12)
        self.assertEqual(narrow_negative_interval["boundary_kind"], "crossing")
        self.assertLess(narrow_negative_interval["boundary_alpha"], .5)

        tiny_scale = {key: value * 1e-24 for key, value in endpoint.items()}
        tiny_scale["scale"] = np.array([[1e-24]])
        tiny_crossing = candidate.find_first_violation_interval(
            tiny_scale, root_tolerance=1e-12)
        self.assertEqual(tiny_crossing["boundary_kind"], "crossing")

        near_linear = {
            "q0": np.array([[.5]]), "q1": np.array([[-1.]]),
            "q2": np.array([[1e-30]]), "scale": np.ones((1, 1)),
        }
        near_linear_crossing = candidate.find_first_violation_interval(
            near_linear, root_tolerance=1e-12)
        self.assertEqual(near_linear_crossing["boundary_kind"], "crossing")
        self.assertAlmostEqual(near_linear_crossing["boundary_alpha"], .5)

    def test_three_roles_share_update_and_candidate_uses_inward_margin(self):
        vertices = np.repeat(np.array([[
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.],
        ]]), 16, axis=0)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        desired = np.zeros_like(vertices)
        desired[:, 2, 1] = -2.0
        alphas = {}
        for role in candidate.ROLES:
            _, certificate, diagnostics = candidate.admit_role(
                vertices, desired, faces, role, beta=.2, fixed_alpha=.25,
                backtracking_factor=.5, backtracking_max_steps=8,
                degeneracy_epsilon=1e-12, root_tolerance=1e-12,
                absolute_margin=1e-6, relative_margin=1e-6)
            self.assertTrue(candidate.check_all_face_area_constraints(
                certificate, diagnostics["admitted_alpha"], tolerance=1e-10)[0])
            self.assertEqual(
                diagnostics["export_constraint_check"]["violating_constraints"], 0)
            alphas[role] = diagnostics["admitted_alpha"]
        self.assertEqual(alphas["fixed_damping"], .25)
        self.assertEqual(alphas["generic_backtracking"], .25)
        self.assertLess(alphas["exact_quadratic_admission"], .4)
        self.assertGreater(alphas["exact_quadratic_admission"], .399)

    def test_full_export_revalidates_all_16_frames_and_exact_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, digest = self.source(root)
            output = root / "candidate"
            record = candidate.export_area_admission_candidate(
                source, output, uid="fixture-c12",
                expected_sequence_sha256=digest,
                source_sequence_ref="source/sequence.npz",
                source_report_ref="source/report.json",
                beta=.2, fixed_alpha=.5, backtracking_factor=.5,
                backtracking_max_steps=8, degeneracy_epsilon=1e-12,
                root_tolerance=1e-10, absolute_margin=1e-8,
                relative_margin=1e-8, arap_weight=1., temporal_weight=.1,
                arap_iterations=2, cg_tolerance=1e-8,
                cg_max_iterations=1000, coordinate_lower=-10.,
                coordinate_upper=10., bounds_policy="preserve_and_report",
                max_artifact_bytes=32 * 1024 * 1024)
            self.assertEqual(record["status"], "completed")
            checked = candidate.validate_candidate_artifact(
                root, output / "candidate.json")
            self.assertEqual(checked["completed_roles"], list(candidate.ROLES))
            original, faces = self.sequence()
            for role in candidate.ROLES:
                with np.load(output / role / "sequence.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["vertices"].shape, original.shape)
                    self.assertEqual(saved["vertices"].dtype, np.float32)
                    np.testing.assert_array_equal(saved["vertices"][0], original[0])
                    np.testing.assert_array_equal(saved["faces"], faces)

    def test_degenerate_source_and_artifact_tampering_fail_closed(self):
        vertices, faces = self.sequence()
        vertices[:, 2] = vertices[:, 1]
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            candidate.form_exact_face_quadratics(
                vertices, np.zeros_like(vertices), faces,
                beta=.2, degeneracy_epsilon=1e-12)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, digest = self.source(root)
            output = root / "candidate"
            candidate.export_area_admission_candidate(
                source, output, uid="fixture-c12",
                expected_sequence_sha256=digest,
                source_sequence_ref="source/sequence.npz",
                source_report_ref="source/report.json",
                beta=.2, fixed_alpha=.5, backtracking_factor=.5,
                backtracking_max_steps=8, degeneracy_epsilon=1e-12,
                root_tolerance=1e-10, absolute_margin=1e-8,
                relative_margin=1e-8, arap_weight=1., temporal_weight=.1,
                arap_iterations=2, cg_tolerance=1e-8,
                cg_max_iterations=1000, coordinate_lower=-10.,
                coordinate_upper=10., bounds_policy="preserve_and_report",
                max_artifact_bytes=32 * 1024 * 1024)
            report_path = output / "exact_quadratic_admission" / "report.json"
            report = json.loads(report_path.read_text())
            report["native_qualified"] = True
            report_path.write_text(json.dumps(report) + "\n")
            with self.assertRaisesRegex(ValueError, "report|evidence"):
                candidate.validate_candidate_artifact(root, output / "candidate.json")

    def test_validation_replays_failed_arms_and_materializer_prevalidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, digest = self.source(root)
            output = root / "candidate"
            candidate.export_area_admission_candidate(
                source, output, uid="fixture-c12", expected_sequence_sha256=digest,
                source_sequence_ref="source/sequence.npz",
                source_report_ref="source/report.json", beta=.2,
                fixed_alpha=.5, backtracking_factor=.5,
                backtracking_max_steps=8, degeneracy_epsilon=1e-12,
                root_tolerance=1e-10, absolute_margin=1e-8,
                relative_margin=1e-8, arap_weight=1., temporal_weight=.1,
                arap_iterations=2, cg_tolerance=1e-8,
                cg_max_iterations=1000, coordinate_lower=-10.,
                coordinate_upper=10., bounds_policy="preserve_and_report",
                max_artifact_bytes=32 * 1024 * 1024)
            record_path = output / "artifact-archive.json"
            record = json.loads(record_path.read_text())
            record["members"][0]["sha256"] = "0" * 64
            record_path.write_text(json.dumps(record) + "\n")
            destination = root / "materialized"
            with self.assertRaisesRegex(ValueError, "inventory"):
                candidate.materialize_candidate_archive(
                    output / "artifact.tar", destination,
                    max_artifact_bytes=32 * 1024 * 1024,
                    max_member_bytes=16 * 1024 * 1024)
            self.assertFalse(destination.exists())

            # Restore the terminal archive, then forge a successful role as an
            # error while keeping all surrounding records internally coherent.
            reports = [json.loads((output / role / "report.json").read_text())
                       for role in candidate.ROLES]
            candidate._write_archive(output, reports, 32 * 1024 * 1024)
            forged = reports[0]
            for name in ("diagnostics", "bounds", "sha256"):
                forged.pop(name, None)
            forged.update(status="error", exception_type="RuntimeError", error="forged")
            (output / candidate.ROLES[0] / "sequence.npz").unlink()
            (output / candidate.ROLES[0] / "certificate.npz").unlink()
            candidate.write_json(output / candidate.ROLES[0] / "report.json", forged)
            candidate_record = json.loads((output / "candidate.json").read_text())
            candidate_record["status"] = "incomplete"
            candidate_record["arms"][0] = forged
            candidate.write_json(output / "candidate.json", candidate_record)
            manifest = json.loads((output / "manifest.json").read_text())
            manifest["cases"] = manifest["cases"][1:]
            candidate.write_json(output / "manifest.json", manifest)
            candidate._write_archive(
                output, [forged, *reports[1:]], 32 * 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "C12 failed role lacks bounded terminal evidence"):
                candidate.validate_candidate_artifact(root, output / "candidate.json")


if __name__ == "__main__":
    unittest.main()
