"""Authored software checks for C05 artifact materialization.

These fixtures are not native method qualification and are not executed by the
Web supervisor.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_math import c05_candidate_artifacts as artifacts
from research_math import c05_mode_bank as mode_bank_module


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _producer_ref(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": _digest(path),
            "bytes": path.stat().st_size}


def _content_sha(arrays: dict[str, np.ndarray], names) -> str:
    return mode_bank_module._sha256_bytes(mode_bank_module._canonical_json({
        name: mode_bank_module._array_sha256(arrays[name], name=name)
        for name in names
    }))


def _parity_ref(root: Path, path: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": _digest(path)}


def fixture(root: Path, *, workspace: Path | None = None,
            multimodal: bool = True):
    workspace = root if workspace is None else workspace
    mode_root = workspace / "retained/mode-bank"
    (mode_root / "sequences").mkdir(parents=True)
    anchor = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    faces = np.asarray([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    frames = np.arange(16, dtype=np.int64)
    times = np.linspace(0.0, 1.5, 16, dtype=np.float32)
    ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    seeds = np.asarray([42, 1001, 1002, 1003], dtype=np.int64)
    draws = []
    branch_arrays = []
    for branch in range(4):
        sequence = np.broadcast_to(anchor, (16, 4, 3)).copy()
        sequence[1:, :, 1] += np.linspace(0.0, 0.2, 15, dtype=np.float32)[:, None]
        if multimodal and branch >= 2:
            # Same anchor, then a coherent alternative complete trajectory.
            sequence[1:, :, 0] += np.linspace(0.1, 0.6, 15, dtype=np.float32)[:, None]
        elif not multimodal:
            sequence[1:, :, 0] += branch * 1e-5
        draws.append(sequence)
        np.savez_compressed(
            mode_root / "sequences" / f"branch-{branch:04d}.npz",
            vertices=sequence, faces=faces, frame_indices=frames,
            timesteps=times, query_vertex_ids=ids)
        branch_arrays.append({
            "vertices": sequence, "faces": faces, "frame_indices": frames,
            "timesteps": times, "query_vertex_ids": ids,
        })
    draws = np.stack(draws)
    bank_arrays = {
        "vertices": draws, "faces": faces, "frame_indices": frames,
        "timesteps": times, "query_vertex_ids": ids,
        "equal_mass_scores": np.full(4, 0.25, dtype=np.float64),
        "inner_stage1_seeds": seeds,
        "outer_generation_seed": np.asarray(42, dtype=np.int64),
    }
    np.savez_compressed(
        mode_root / "mode-bank.npz", **bank_arrays)
    branches = []
    for branch, seed in enumerate(seeds):
        path = mode_root / "sequences" / f"branch-{branch:04d}.npz"
        ref = _producer_ref(mode_root, path)
        ref["content_sha256"] = _content_sha(
            branch_arrays[branch],
            ("vertices", "faces", "frame_indices", "timesteps", "query_vertex_ids"))
        branches.append({
            "branch_index": branch,
            "role": "b0_algorithmic_path_unverified" if branch == 0 else "empirical_mode",
            "stage1_seed": int(seed), "sequence_ref": ref,
        })
    scope = {name: False for name in (
        "native_scientific_qualification", "scientific_effect_qualification",
        "candidate_methods_tested", "dispatch_ready")}
    request = workspace / "plans/c05-mode-fixture/request.json"
    request.parent.mkdir(parents=True, exist_ok=True)
    _json(request, {
        "kind": "c05-mode-bank-request", "version": 1,
        "output_relative": mode_root.relative_to(root).as_posix(),
    })
    producer_inputs = [
        _parity_ref(root, request),
        _parity_ref(root, Path(__file__).resolve()),
    ]
    producer_code = [_parity_ref(root, Path(inspect.getsourcefile(
        artifacts.validate_retained_mode_bank)).resolve())]
    bank_ref = _producer_ref(mode_root, mode_root / "mode-bank.npz")
    bank_ref["content_sha256"] = _content_sha(bank_arrays, (
        "vertices", "faces", "frame_indices", "timesteps", "query_vertex_ids",
        "equal_mass_scores", "inner_stage1_seeds"))
    manifest = {
        "kind": "c05-same-anchor-stage1-mode-bank-manifest", "version": 1,
        "status": "completed_unqualified", "candidate_id": "C05",
        "outer_generation_seed": 42, "inner_stage1_seeds": seeds.tolist(),
        "branch_count": 4, "frames": 16, "vertices_per_frame": 4,
        "provenance": {"generation_uid": "software-uid", "gpu_uuid": "GPU-fixture"},
        "producer_code_sha256": _digest(Path(inspect.getsourcefile(
            artifacts.validate_retained_mode_bank))),
        "producer_input_refs": producer_inputs,
        "producer_code_refs": producer_code,
        "mode_semantics": "equal-mass empirical model-output modes",
        "posterior_correspondence_claim": False,
        "calibrated_probability_claim": False,
        "mode_bank_ref": bank_ref,
        "branches": branches, **scope,
    }
    manifest_path = mode_root / "raw-manifest.json"
    _json(manifest_path, manifest)
    result = {
        "kind": "c05-same-anchor-stage1-mode-bank-result", "version": 1,
        "status": "completed_unqualified", "candidate_id": "C05",
        "manifest_ref": _producer_ref(mode_root, manifest_path),
        "mode_bank_ref": manifest["mode_bank_ref"],
        "producer_input_refs": producer_inputs,
        "producer_code_refs": producer_code,
        "outer_generation_seed": 42, "inner_stage1_seeds": seeds.tolist(),
        **scope,
    }
    result_path = mode_root / "result.json"
    _json(result_path, result)

    b0_dir = workspace / "retained/b0"
    b0_dir.mkdir(parents=True)
    b0_sequence = b0_dir / "sequence.npz"
    np.savez_compressed(
        b0_sequence, vertices=draws[0], faces=faces,
        frame_indices=frames, timesteps=times, query_vertex_ids=ids)
    b0_report = b0_dir / "report.json"
    _json(b0_report, {"status": "completed", "uid": "software-uid", "seed": 42,
                      "sha256": {"sequence.npz": _digest(b0_sequence)}})
    parity = workspace / "retained/c05-b0-parity.json"
    _json(parity, {
        "kind": "c05-b0-parity-receipt", "version": 1,
        "candidate_id": artifacts.CANDIDATE_ID,
        "uid": "software-uid", "outer_seed": 42, "matches": True,
        "comparison": "exact array equality for all native sequence fields",
        "mode_bank_result_ref": _parity_ref(root, result_path),
        "mode_bank_manifest_ref": _parity_ref(root, manifest_path),
        "mode_bank_array_ref": _parity_ref(root, mode_root / "mode-bank.npz"),
        "branch_zero_ref": _parity_ref(root, mode_root / "sequences/branch-0000.npz"),
        "retained_b0_sequence_ref": _parity_ref(root, b0_sequence),
        "retained_b0_report_ref": _parity_ref(root, b0_report),
        "candidate_methods_tested": False,
        "scientific_effect_qualification": False, "native_qualified": False,
    })
    return mode_root, parity, b0_sequence, b0_report


def materialize(root: Path, *, workspace: Path | None = None,
                multimodal: bool = True, output_name: str = "candidate"):
    workspace = root if workspace is None else workspace
    mode_root, parity, b0_sequence, b0_report = fixture(
        root, workspace=workspace, multimodal=multimodal)
    output = workspace / output_name
    result = artifacts.materialize_candidate(
        root, mode_root, parity, b0_sequence, b0_report, output,
        landmark_count=4, cluster_radius=0.05,
        natural_gate_min_fraction=0.75, localized_radius=1.0,
        temperature=2.0, unary_weight=0.1, spatial_weight=1.0,
        max_sweeps=20, displacement_clip_multiplier=4.0,
        max_artifact_bytes=8 * 1024 * 1024, face_chunk_size=2)
    return output, result


class C05CandidateArtifactTests(unittest.TestCase):
    def test_materializes_all_distinct_roles_with_exact_native_identity(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            output, result = materialize(root, workspace=workspace)
            self.assertEqual(result["status"], "completed_unqualified")
            self.assertEqual(result["role_denominator"], len(artifacts.ROLE_ORDER))
            self.assertEqual(result["roles_failed"], [])
            candidate = artifacts.validate_candidate_artifact(root, output)
            self.assertEqual(tuple(candidate["role_refs"]), artifacts.ROLE_ORDER)
            self.assertEqual(candidate["method_ids"], artifacts.METHOD_IDS)
            self.assertEqual(set(candidate["input_refs"]), {
                "mode_bank_result", "mode_bank_manifest", "mode_bank",
                "branch0_sequence", "b0_sequence", "b0_report", "parity_receipt",
                "artifact_implementation", "math_implementation",
                "mode_bank_implementation",
            })
            with np.load(root / candidate["input_refs"]["b0_sequence"]["path"],
                         allow_pickle=False) as archive:
                b0 = {name: archive[name].copy() for name in archive.files}
            policies = set()
            for role in artifacts.ROLE_ORDER:
                with np.load(output / f"roles/{role}/sequence.npz",
                             allow_pickle=False) as archive:
                    self.assertEqual(set(archive.files), artifacts._SEQUENCE_FIELDS)
                    np.testing.assert_array_equal(archive["vertices"][0], b0["vertices"][0])
                    for name in artifacts._SEQUENCE_FIELDS - {"vertices"}:
                        np.testing.assert_array_equal(archive[name], b0[name])
                report = json.loads((output / f"roles/{role}/report.json").read_text())
                policies.add(report["lift_policy"])
                self.assertEqual(report["method_id"], artifacts.METHOD_IDS[role])
                self.assertEqual(report["implementation_ref"],
                                 candidate["input_refs"]["math_implementation"])
                self.assertFalse(report["native_scientific_qualification"])
            self.assertEqual(len(policies), 1)

    def test_natural_gate_rejection_has_no_scoreable_role_or_fallback(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            output, result = materialize(root, workspace=workspace, multimodal=False)
            self.assertEqual(result["status"], "incomplete_natural_gate")
            candidate = artifacts.validate_candidate_artifact(root, output)
            self.assertEqual(tuple(candidate["role_refs"]), artifacts.ROLE_ORDER)
            self.assertEqual(result["roles_completed"], [])
            self.assertEqual(result["roles_failed"], list(artifacts.ROLE_ORDER))
            self.assertEqual(result["role_denominator"], len(artifacts.ROLE_ORDER))
            for role in artifacts.ROLE_ORDER:
                row = candidate["role_refs"][role]
                self.assertEqual(row["preparation_status"], "error")
                self.assertIsNone(row["sequence_ref"])
                self.assertFalse((output / f"roles/{role}/sequence.npz").exists())
                report = json.loads(
                    (output / f"roles/{role}/report.json").read_text())
                self.assertEqual(report["status"], "error")
                self.assertEqual(report["exception_type"], "C05ArtifactError")
                self.assertIsNone(report["sequence_ref"])
            certificate = json.loads((output / "certificate.json").read_text())
            self.assertEqual(certificate["status"], "rejected")
            self.assertLess(certificate["multimodal_landmark_count"],
                            certificate["required_multimodal_count"])
            self.assertEqual(len(certificate["max_medoid_separations"]), 4)
            self.assertEqual(certificate["separated_landmark_count"], 0)

    def test_current_release_identity_requires_float32_frames_zero_through_fifteen(self):
        arrays = {
            "vertices": np.zeros((2, 16, 4, 3), dtype=np.float32),
            "faces": np.asarray([[0, 1, 2]], dtype=np.int64),
            "frame_indices": np.arange(1, 17, dtype=np.int64),
            "timesteps": np.arange(16, dtype=np.float32),
            "query_vertex_ids": np.arange(4, dtype=np.int64),
            "equal_mass_scores": np.full(2, 0.5, dtype=np.float64),
            "inner_stage1_seeds": np.asarray([42, 314], dtype=np.int64),
            "outer_generation_seed": np.asarray(42, dtype=np.int64),
        }
        b0 = {name: arrays[name].copy() for name in artifacts._SEQUENCE_FIELDS}
        with self.assertRaisesRegex(artifacts.C05ArtifactError, "frame order"):
            artifacts.validate_native_arrays(arrays, b0)
        arrays["frame_indices"] = np.arange(16, dtype=np.int64)
        b0["frame_indices"] = arrays["frame_indices"].copy()
        arrays["vertices"] = arrays["vertices"].astype(np.float64)
        b0["vertices"] = arrays["vertices"][0].copy()
        with self.assertRaisesRegex(artifacts.C05ArtifactError, "float32"):
            artifacts.validate_native_arrays(arrays, b0)

    def test_parity_must_bind_every_exact_input_and_b0_report(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            mode_root, parity, b0_sequence, b0_report = fixture(
                root, workspace=workspace)
            value = json.loads(parity.read_text())
            value["branch_zero_ref"]["sha256"] = "0" * 64
            _json(parity, value)
            with self.assertRaisesRegex(artifacts.C05ArtifactError, "branch_zero_ref"):
                artifacts.materialize_candidate(
                    root, mode_root, parity, b0_sequence, b0_report, workspace / "bad",
                    landmark_count=4, cluster_radius=.05,
                    natural_gate_min_fraction=.5, localized_radius=1,
                    temperature=1, unary_weight=1, spatial_weight=1,
                    max_sweeps=5, displacement_clip_multiplier=4,
                    max_artifact_bytes=8 * 1024 * 1024)

    def test_aggregate_bank_must_equal_every_ordered_retained_branch(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            mode_root, _, _, _ = fixture(root, workspace=workspace)
            manifest = json.loads((mode_root / "raw-manifest.json").read_text())
            mode_arrays = artifacts._load_npz(
                mode_root / "mode-bank.npz", artifacts._BANK_FIELDS)
            branch_path = mode_root / "sequences/branch-0001.npz"
            branch = artifacts._load_npz(branch_path, artifacts._SEQUENCE_FIELDS)
            branch["vertices"][1, 0, 0] += np.float32(0.25)
            np.savez_compressed(branch_path, **branch)
            with self.assertRaisesRegex(
                    artifacts.C05ArtifactError, "aggregate mode-bank branch"):
                artifacts._validate_branch_closure(mode_root, manifest, mode_arrays)

    def test_kabsch_failure_blocks_only_joint_and_keeps_fixed_denominator(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            with patch.object(
                    artifacts, "_proper_kabsch_hints",
                    side_effect=artifacts.C05ArtifactError("rank-deficient fixture")):
                output, result = materialize(root, workspace=workspace)
                candidate = artifacts.validate_candidate_artifact(root, output)
            self.assertEqual(result["role_denominator"], 5)
            self.assertEqual(result["roles_failed"], ["joint_spatial_labels"])
            self.assertEqual(candidate["roles_completed"], list(artifacts.ROLE_ORDER[:-1]))
            self.assertFalse(
                (output / "roles/joint_spatial_labels/sequence.npz").exists())
            report = json.loads(
                (output / "roles/joint_spatial_labels/report.json").read_text())
            self.assertEqual(report["status"], "error")
            self.assertEqual(report["exception_type"], "C05ArtifactError")
            self.assertEqual(report["error"], "rank-deficient fixture")
            for role in artifacts.ROLE_ORDER[:-1]:
                self.assertTrue((output / f"roles/{role}/sequence.npz").is_file())

    def test_union_projection_failure_blocks_only_surface_and_is_bounded(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            message = "projection fixture " + "x" * 5000
            with patch.object(
                    artifacts, "union_surface_projected_mean",
                    side_effect=RuntimeError(message)):
                output, result = materialize(root, workspace=workspace)
                candidate = artifacts.validate_candidate_artifact(root, output)
            failed = "surface_projected_mean"
            self.assertEqual(result["roles_failed"], [failed])
            self.assertEqual(candidate["role_refs"][failed]["preparation_status"],
                             "error")
            self.assertIsNone(candidate["role_refs"][failed]["sequence_ref"])
            report = json.loads((output / f"roles/{failed}/report.json").read_text())
            self.assertEqual(report["exception_type"], "RuntimeError")
            self.assertEqual(len(report["error"]), 4096)
            self.assertFalse((output / f"roles/{failed}/sequence.npz").exists())
            for role in artifacts.ROLE_ORDER:
                if role != failed:
                    self.assertTrue((output / f"roles/{role}/sequence.npz").is_file())

    def test_icm_failure_blocks_only_joint_after_kabsch_evidence(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            with patch.object(
                    artifacts, "solve_joint_labels_icm",
                    side_effect=RuntimeError("ICM fixture")):
                output, result = materialize(root, workspace=workspace)
                artifacts.validate_candidate_artifact(root, output)
            self.assertEqual(result["roles_failed"], ["joint_spatial_labels"])
            solver = json.loads((output / "solver-certificate.json").read_text())
            self.assertIsInstance(solver["rotation_hints"], dict)
            self.assertEqual(len(solver["time_weights"]), 16)
            self.assertIsNone(solver["joint"])
            self.assertEqual(
                solver["role_errors"]["joint_spatial_labels"]["error"],
                "ICM fixture")

    def test_archive_and_generated_files_are_exact_and_tamper_evident(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            output, _ = materialize(root, workspace=workspace)
            archive = output / "raw-evidence.tar"
            original = archive.read_bytes()
            self.assertEqual(original, artifacts._tar_bytes(
                output, [row["path"] for row in
                         json.loads((output / "raw-manifest.json").read_text())["members"]]))
            report = output / "roles/joint_spatial_labels/report.json"
            report.write_bytes(report.read_bytes() + b"changed")
            with self.assertRaises(artifacts.C05ArtifactError):
                artifacts.validate_candidate_artifact(root, output)

    def test_byte_ceiling_and_append_only_output_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            mode_root, parity, b0_sequence, b0_report = fixture(
                root, workspace=workspace)
            kwargs = dict(
                landmark_count=4, cluster_radius=.05,
                natural_gate_min_fraction=.5, localized_radius=1,
                temperature=1, unary_weight=1, spatial_weight=1,
                max_sweeps=5, displacement_clip_multiplier=4,
                max_artifact_bytes=1)
            with self.assertRaisesRegex(artifacts.C05ArtifactError, "byte ceiling"):
                artifacts.materialize_candidate(
                    root, mode_root, parity, b0_sequence, b0_report,
                    workspace / "bounded", **kwargs)
            with self.assertRaises(FileExistsError):
                artifacts.materialize_candidate(
                    root, mode_root, parity, b0_sequence, b0_report,
                    workspace / "bounded", **{**kwargs, "max_artifact_bytes": 2**20})

    def test_interface_has_no_forbidden_coordinate_or_scoring_input(self):
        parameters = set(inspect.signature(artifacts.materialize_candidate).parameters)
        self.assertEqual(parameters, {
            "root", "mode_bank_root", "parity_receipt", "b0_sequence", "b0_report",
            "output_dir", "landmark_count", "cluster_radius",
            "natural_gate_min_fraction", "localized_radius", "temperature",
            "unary_weight", "spatial_weight", "max_sweeps",
            "displacement_clip_multiplier", "max_artifact_bytes", "face_chunk_size",
        })
        args = artifacts.parse_args([
            "validate", "--root", "/project", "--output", "/project/c05"])
        self.assertEqual(args.operation, "validate")
        source = Path(inspect.getsourcefile(artifacts.materialize_candidate)).read_text()
        self.assertNotIn("official_actionbench_adapter", source)
        self.assertNotIn("attention_map", source)

    def test_changed_output_and_extra_file_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            root, workspace = PROJECT_ROOT, Path(directory)
            output, _ = materialize(root, workspace=workspace)
            (output / "unlisted.bin").write_bytes(b"not retained")
            with self.assertRaisesRegex(artifacts.C05ArtifactError, "extra"):
                artifacts.validate_candidate_artifact(root, output)


if __name__ == "__main__":  # pragma: no cover - Local execution only
    unittest.main()
