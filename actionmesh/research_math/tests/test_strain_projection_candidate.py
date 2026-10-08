"""Engineering contracts for C11; Local must execute through the CPU harness."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np

from research_math import strain_projection_candidate as candidate


class StrainProjectionCandidateTests(unittest.TestCase):
    def sequence(self):
        anchor = np.array([
            [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
        ], dtype=np.float32)
        vertices = np.repeat(anchor[None], 16, axis=0)
        scale = np.linspace(1., 1.8, 16, dtype=np.float32)
        vertices[:, :, 0] *= scale[:, None]
        vertices[:, 2:, 1] *= np.linspace(1., .7, 16, dtype=np.float32)[:, None]
        return vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

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
            "status": "completed", "uid": "fixture-c11", "seed": 42,
            "sha256": {"sequence.npz": digest}}) + "\n")
        return case, digest

    def test_polar_projection_retains_rotation_and_has_distinct_controls(self):
        vertices, faces = self.sequence()
        maps = candidate.qualify_square_maps(
            vertices, faces, degeneracy_epsilon=1e-10)
        outputs = {}
        for role in candidate.ROLES:
            transforms, diagnostics = candidate.project_stretch_spectrum(
                maps, role, lower=.8, upper=1.2, elastic_weight=2.)
            outputs[role] = transforms
            self.assertLessEqual(diagnostics["rotation_preservation_linf"], 1e-9)
        self.assertFalse(np.array_equal(outputs["arap_repair"],
                                        outputs["elastic_repair"]))
        self.assertFalse(np.array_equal(outputs["elastic_repair"],
                                        outputs["rotation_preserving_stretch_projection"]))

    def test_full_export_and_revalidation_preserve_native_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, digest = self.source(root)
            output = root / "candidate"
            record = candidate.export_strain_projection_candidate(
                source, output, uid="fixture-c11",
                expected_sequence_sha256=digest,
                source_sequence_ref="source/sequence.npz",
                source_report_ref="source/report.json",
                lower=.8, upper=1.2, elastic_weight=2.,
                degeneracy_epsilon=1e-10,
                absolute_tolerance=1e-10, relative_tolerance=1e-8,
                max_iterations=1000, coordinate_lower=-10.,
                coordinate_upper=10., bounds_policy="preserve_and_report",
                max_artifact_bytes=32 * 1024 * 1024)
            self.assertEqual(record["status"], "completed")
            checked = candidate.validate_candidate_artifact(
                root, output / "candidate.json")
            self.assertEqual(checked["completed_roles"], list(candidate.ROLES))
            materialized = root / "materialized"
            candidate.materialize_candidate_archive(
                output / "artifact.tar", materialized,
                max_artifact_bytes=32 * 1024 * 1024,
                max_member_bytes=16 * 1024 * 1024)
            self.assertEqual(candidate.validate_candidate_artifact(
                root, materialized / "candidate.json")["status"], "completed")
            for role in candidate.ROLES:
                with np.load(output / role / "sequence.npz", allow_pickle=False) as saved:
                    self.assertEqual(saved["vertices"].dtype, np.float32)
                    np.testing.assert_array_equal(saved["vertices"][0],
                                                  self.sequence()[0][0])
                    with np.load(output / role / "certificate.npz",
                                 allow_pickle=False) as certificate:
                        np.testing.assert_array_equal(
                            saved["vertices"][:, certificate["pins"]],
                            self.sequence()[0][:, certificate["pins"]])
                    np.testing.assert_array_equal(saved["faces"], self.sequence()[1])
                    np.testing.assert_array_equal(saved["frame_indices"], np.arange(16))

    def test_degenerate_or_reflected_maps_fail_without_fallback(self):
        vertices, faces = self.sequence()
        collapsed = vertices.copy()
        collapsed[3, 2] = collapsed[3, 1]
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            candidate.qualify_square_maps(
                collapsed, faces, degeneracy_epsilon=1e-10)
        reflected = vertices.copy()
        reflected[5, [1, 2]] = reflected[5, [2, 1]]
        with self.assertRaisesRegex(ValueError, "Reflected|oriented"):
            candidate.qualify_square_maps(
                reflected, faces, degeneracy_epsilon=1e-10)

    def test_bounds_failure_retains_reports_and_removes_scoreable_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, digest = self.source(root)
            output = root / "candidate"
            record = candidate.export_strain_projection_candidate(
                source, output, uid="fixture-c11", expected_sequence_sha256=digest,
                source_sequence_ref="source/sequence.npz",
                source_report_ref="source/report.json",
                lower=.8, upper=1.2, elastic_weight=2.,
                degeneracy_epsilon=1e-10, absolute_tolerance=1e-10,
                relative_tolerance=1e-8, max_iterations=1000,
                coordinate_lower=-.01, coordinate_upper=.01,
                bounds_policy="reject", max_artifact_bytes=32 * 1024 * 1024)
            self.assertEqual(record["status"], "incomplete")
            for role in candidate.ROLES:
                report = json.loads((output / role / "report.json").read_text())
                self.assertEqual(report["status"], "error")
                self.assertFalse((output / role / "sequence.npz").exists())
                self.assertFalse((output / role / "certificate.npz").exists())

    def test_archive_member_and_evidence_overclaim_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, digest = self.source(root)
            output = root / "candidate"
            candidate.export_strain_projection_candidate(
                source, output, uid="fixture-c11", expected_sequence_sha256=digest,
                source_sequence_ref="source/sequence.npz",
                source_report_ref="source/report.json",
                lower=.8, upper=1.2, elastic_weight=2.,
                degeneracy_epsilon=1e-10, absolute_tolerance=1e-10,
                relative_tolerance=1e-8, max_iterations=1000,
                coordinate_lower=-10., coordinate_upper=10.,
                bounds_policy="reject", max_artifact_bytes=32 * 1024 * 1024)
            record_path = output / "artifact-archive.json"
            record = json.loads(record_path.read_text())
            rows = []
            with tarfile.open(output / "artifact.tar", "r:") as bundle:
                for member in bundle.getmembers():
                    data = bundle.extractfile(member).read()
                    if member.name == "candidate.json":
                        data += b" "
                    rows.append((member.name, data))
            with tarfile.open(output / "artifact-tampered.tar", "w",
                              format=tarfile.USTAR_FORMAT) as bundle:
                for name, data in rows:
                    info = tarfile.TarInfo(name)
                    info.size = len(data); info.mode = 0o644
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    bundle.addfile(info, io.BytesIO(data))
            (output / "artifact-tampered.tar").replace(output / "artifact.tar")
            record["archive"]["sha256"] = hashlib.sha256(
                (output / "artifact.tar").read_bytes()).hexdigest()
            record["archive"]["size_bytes"] = (output / "artifact.tar").stat().st_size
            record_path.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "archive"):
                candidate.validate_candidate_artifact(root, output / "candidate.json")

            candidate_json = json.loads((output / "candidate.json").read_text())
            candidate_json["native_qualified"] = True
            (output / "candidate.json").write_text(json.dumps(candidate_json) + "\n")
            with self.assertRaisesRegex(ValueError, "evidence flags"):
                candidate.validate_candidate_artifact(root, output / "candidate.json")


if __name__ == "__main__":
    unittest.main()
