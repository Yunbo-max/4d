"""Authored engineering checks for C15; Web does not execute this file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_math.protected_lowrank_candidate import (
    ARMS,
    CANDIDATE_ARM,
    RANK_MATCHED_ARM,
    RANK_POLICY,
    UNPROTECTED_ARM,
    build_anchored_dct_basis,
    export_protected_lowrank_candidate,
    solve_rank_matched_truncated_svd,
    solve_residual_svt,
    solve_unprotected_svt,
    validate_protected_lowrank_artifact,
    validate_temporal_basis,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProtectedLowrankCandidateTest(unittest.TestCase):
    def test_analytic_basis_is_exactly_anchored_and_nontrivial(self):
        basis = build_anchored_dct_basis(16, 4)
        checked, diagnostics = validate_temporal_basis(basis, 16, 1e-10)
        np.testing.assert_array_equal(checked[:, 0], np.eye(16)[:, 0])
        np.testing.assert_array_equal(checked[0, 1:], np.zeros(3))
        self.assertEqual(diagnostics["basis_rank"], 4)
        self.assertEqual(diagnostics["complement_dimension"], 12)
        with self.assertRaisesRegex(ValueError, "2<=K<T"):
            validate_temporal_basis(np.eye(16), 16, 1e-10)

    def test_candidate_and_anchor_only_controls_share_exact_anchor(self):
        rng = np.random.default_rng(7)
        trajectory = rng.normal(size=(16, 21)).astype(np.float64)
        basis = build_anchored_dct_basis(16, 3)
        candidate, diagnostics, _ = solve_residual_svt(
            trajectory, basis, lambda_value=0.2,
            orthogonality_tolerance=1e-10)
        unprotected, control_diagnostics, _ = solve_unprotected_svt(
            trajectory, lambda_value=0.2)
        np.testing.assert_array_equal(candidate[0], trajectory[0])
        np.testing.assert_array_equal(unprotected[0], trajectory[0])
        self.assertLess(diagnostics["protected_coefficient_drift_frobenius"], 1e-10)
        self.assertFalse(control_diagnostics["additional_action_subspace_protected"])
        self.assertFalse(np.allclose(candidate, unprotected))

    def test_rank_control_matches_frozen_candidate_export_rank(self):
        rng = np.random.default_rng(11)
        trajectory = rng.normal(size=(16, 18)).astype(np.float64)
        basis = build_anchored_dct_basis(16, 4)
        candidate, _, _ = solve_residual_svt(
            trajectory, basis, lambda_value=0.5,
            orthogonality_tolerance=1e-10)
        exported = candidate.astype(np.float32).astype(np.float64)
        tolerance = 1e-6
        singular = np.linalg.svd(exported, compute_uv=False)
        target_rank = int(np.count_nonzero(singular > tolerance * singular[0]))
        control, diagnostics, _ = solve_rank_matched_truncated_svd(
            trajectory, target_numeric_rank=target_rank,
            numeric_rank_tolerance=tolerance,
            residual_rank_policy=RANK_POLICY)
        self.assertEqual(diagnostics["output_numeric_rank"], target_rank)
        np.testing.assert_array_equal(control[0], trajectory[0])

    def _case(self, root: Path):
        case = root / "retained/case"
        case.mkdir(parents=True)
        rng = np.random.default_rng(13)
        anchor = rng.normal(size=(5, 3))
        vertices = np.stack([
            anchor + 0.03 * frame * rng.normal(size=(5, 3))
            for frame in range(16)
        ]).astype(np.float32)
        arrays = {
            "vertices": vertices,
            "faces": np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=np.int64),
            "timesteps": np.linspace(0.0, 1.5, 16, dtype=np.float64),
            "frame_indices": np.arange(16, dtype=np.int64),
            "query_vertex_ids": np.arange(5, dtype=np.int64),
        }
        np.savez_compressed(case / "sequence.npz", **arrays)
        sequence_sha = _sha(case / "sequence.npz")
        (case / "report.json").write_text(json.dumps({
            "status": "completed", "uid": "unit-c15", "seed": 42,
            "sha256": {"sequence.npz": sequence_sha},
        }))
        math_ref = root / "basis-policy.md"
        math_ref.write_text("Prospective analytic anchored DCT policy.\n")
        basis = build_anchored_dct_basis(16, 3)
        basis_path = root / "basis.npy"
        np.save(basis_path, basis, allow_pickle=False)
        evidence_path = root / "basis-evidence.json"
        evidence_path.write_text(json.dumps({
            "kind": "c15-protection-basis-evidence", "version": "1.0.0",
            "status": "completed", "basis_sha256": _sha(basis_path),
            "basis_source_policy": "analytic_anchored_dct",
            "confirmation_outcomes_used": False, "prospective_freeze": True,
            "frozen_at": "2026-10-08T12:00:00+00:00",
            "construction_rule": "exact_e0_plus_orthonormal_dct_ii_on_frames_1_to_T_minus_1",
            "frame_count": 16, "basis_rank": 3, "input_sequence_sha256": None,
            "source_refs": [{"path": "basis-policy.md", "sha256": _sha(math_ref)}],
        }))
        return case, arrays, basis_path, evidence_path

    def _export(self, root: Path, *, output_name="output"):
        case, arrays, basis_path, evidence_path = self._case(root)
        summary = export_protected_lowrank_candidate(
            case / "sequence.npz", basis_path, root / output_name,
            project_root=root, uid="unit-c15",
            expected_sequence_sha256=_sha(case / "sequence.npz"),
            expected_basis_sha256=_sha(basis_path),
            basis_source_policy="analytic_anchored_dct",
            lambda_value=0.1, residual_rank_policy=RANK_POLICY,
            orthogonality_tolerance=1e-10,
            protected_coefficient_tolerance=1e-5,
            numeric_rank_tolerance=1e-6,
            max_artifact_bytes=8 * 1024 * 1024,
            basis_evidence=evidence_path,
            expected_basis_evidence_sha256=_sha(evidence_path))
        return summary, case, arrays, basis_path, evidence_path

    def test_export_preserves_native_identity_and_validates_terminal_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, case, arrays, basis_path, evidence_path = self._export(root)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(tuple(summary["arms"]), ARMS)
            for arm in ARMS:
                with np.load(root / "output" / arm / "sequence.npz") as saved:
                    np.testing.assert_array_equal(saved["faces"], arrays["faces"])
                    np.testing.assert_array_equal(saved["frame_indices"], arrays["frame_indices"])
                    np.testing.assert_array_equal(saved["timesteps"], arrays["timesteps"])
                    np.testing.assert_array_equal(saved["query_vertex_ids"], arrays["query_vertex_ids"])
                    np.testing.assert_array_equal(saved["vertices"][0], arrays["vertices"][0])
            validation = validate_protected_lowrank_artifact(
                root / "output", case / "sequence.npz", basis_path,
                project_root=root, uid="unit-c15",
                expected_sequence_sha256=_sha(case / "sequence.npz"),
                expected_basis_sha256=_sha(basis_path),
                basis_source_policy="analytic_anchored_dct",
                lambda_value=0.1, residual_rank_policy=RANK_POLICY,
                orthogonality_tolerance=1e-10,
                protected_coefficient_tolerance=1e-5,
                numeric_rank_tolerance=1e-6,
                basis_evidence=evidence_path,
                expected_basis_evidence_sha256=_sha(evidence_path))
            self.assertEqual(validation["terminal_status"], "completed")
            self.assertEqual(set(validation["roles"]), set(ARMS))
            self.assertTrue((root / "output/artifact.tar").is_file())

    def test_one_arm_failure_is_retained_without_erasing_independent_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case, _, basis_path, evidence_path = self._case(root)
            with patch("research_math.protected_lowrank_candidate.solve_residual_svt",
                       side_effect=RuntimeError("fixture candidate failure")):
                summary = export_protected_lowrank_candidate(
                    case / "sequence.npz", basis_path, root / "failed-output",
                    project_root=root, uid="unit-c15",
                    expected_sequence_sha256=_sha(case / "sequence.npz"),
                    expected_basis_sha256=_sha(basis_path),
                    basis_source_policy="analytic_anchored_dct",
                    lambda_value=0.1, residual_rank_policy=RANK_POLICY,
                    orthogonality_tolerance=1e-10,
                    protected_coefficient_tolerance=1e-5,
                    numeric_rank_tolerance=1e-6,
                    max_artifact_bytes=8 * 1024 * 1024,
                    basis_evidence=evidence_path,
                    expected_basis_evidence_sha256=_sha(evidence_path))
                validation = validate_protected_lowrank_artifact(
                    root / "failed-output", case / "sequence.npz", basis_path,
                    project_root=root, uid="unit-c15",
                    expected_sequence_sha256=_sha(case / "sequence.npz"),
                    expected_basis_sha256=_sha(basis_path),
                    basis_source_policy="analytic_anchored_dct",
                    lambda_value=0.1, residual_rank_policy=RANK_POLICY,
                    orthogonality_tolerance=1e-10,
                    protected_coefficient_tolerance=1e-5,
                    numeric_rank_tolerance=1e-6,
                    basis_evidence=evidence_path,
                    expected_basis_evidence_sha256=_sha(evidence_path))
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["arms"][CANDIDATE_ARM]["status"], "error")
            self.assertEqual(summary["arms"][UNPROTECTED_ARM]["status"], "completed")
            self.assertEqual(summary["arms"][RANK_MATCHED_ARM]["status"], "error")
            self.assertTrue((root / "failed-output/artifact.tar").is_file())
            self.assertFalse((root / "failed-output/protected_residual_svt/sequence.npz").exists())
            self.assertEqual(validation["terminal_status"], "incomplete")
            self.assertEqual(validation["roles"][CANDIDATE_ARM]["status"], "error")
            self.assertEqual(validation["roles"][UNPROTECTED_ARM]["status"], "completed")

    def test_stale_basis_source_ref_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, case, _, basis_path, evidence_path = self._export(root)
            (root / "basis-policy.md").write_text("mutated after prospective freeze\n")
            with self.assertRaisesRegex(ValueError, "missing or stale"):
                validate_protected_lowrank_artifact(
                    root / "output", case / "sequence.npz", basis_path,
                    project_root=root, uid="unit-c15",
                    expected_sequence_sha256=_sha(case / "sequence.npz"),
                    expected_basis_sha256=_sha(basis_path),
                    basis_source_policy="analytic_anchored_dct",
                    lambda_value=0.1, residual_rank_policy=RANK_POLICY,
                    orthogonality_tolerance=1e-10,
                    protected_coefficient_tolerance=1e-5,
                    numeric_rank_tolerance=1e-6,
                    basis_evidence=evidence_path,
                    expected_basis_evidence_sha256=_sha(evidence_path))

    def test_fabricated_terminal_error_is_not_denominator_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, case, _, basis_path, evidence_path = self._export(root)
            report_path = root / "output/unprotected_svt/report.json"
            report = json.loads(report_path.read_text())
            report.update({
                "status": "error", "failure_phase": "unprotected_svt_compute",
                "exception_type": "RuntimeError", "error": "fabricated",
                "traceback": "fabricated traceback",
            })
            report.pop("sha256", None)
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "bounded terminal error"):
                validate_protected_lowrank_artifact(
                    root / "output", case / "sequence.npz", basis_path,
                    project_root=root, uid="unit-c15",
                    expected_sequence_sha256=_sha(case / "sequence.npz"),
                    expected_basis_sha256=_sha(basis_path),
                    basis_source_policy="analytic_anchored_dct",
                    lambda_value=0.1, residual_rank_policy=RANK_POLICY,
                    orthogonality_tolerance=1e-10,
                    protected_coefficient_tolerance=1e-5,
                    numeric_rank_tolerance=1e-6,
                    basis_evidence=evidence_path,
                    expected_basis_evidence_sha256=_sha(evidence_path))


if __name__ == "__main__":
    unittest.main()
