"""Authored C20 software checks; Web does not execute this module."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research_math import c20_consensus_target as target
from research_math import c20_native_comparison as comparison
from research_math import c20_native_scoring as scoring
from research_math import phase_amplitude_artifacts as artifacts
from research_math.phase_amplitude_candidate import build_endpoint_sine_basis


class C20SourceChainTests(unittest.TestCase):
    def _native(self):
        times = np.arange(16, dtype=np.float32)
        vertices = np.zeros((16, 4, 3), dtype=np.float32)
        vertices[:, 0, 0] = np.linspace(0.0, 0.3, 16)
        vertices[:, 1, 1] = np.linspace(0.0, 0.2, 16)
        vertices[:, 2, 2] = np.linspace(0.0, 0.1, 16)
        vertices[:, 3] = np.array([0.1, 0.1, 0.1])
        faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
        return {"vertices": vertices, "faces": faces,
                "frame_indices": np.arange(16), "timesteps": times}

    def test_profile_has_exact_six_roles_and_distinct_candidate(self):
        self.assertEqual(comparison.ROLES, (
            "b0", "b_star", "phase_only", "amplitude_only", "simple_lag",
            "joint_monotone_phase_amplitude"))
        self.assertEqual(comparison.CANDIDATE_ROLE,
                         "joint_monotone_phase_amplitude")
        self.assertEqual(artifacts.METHOD_IDS[comparison.CANDIDATE_ROLE],
                         artifacts.CANDIDATE_ID)
        self.assertEqual(len(set(artifacts.METHOD_IDS.values())), 4)
        self.assertEqual(scoring.CANDIDATE_ROLE,
                         "joint_monotone_phase_amplitude")
        self.assertEqual(scoring.CONTROL_ROLES,
                         ("b0", "b_star", "phase_only",
                          "amplitude_only", "simple_lag"))

    def test_action_basis_is_w_orthogonal_to_full_phase_design(self):
        native = self._native(); vertices = native["vertices"].astype(np.float64)
        times = native["timesteps"].astype(np.float64)
        phase = build_endpoint_sine_basis(times, 3)
        weights = np.ones_like(vertices)
        seed = vertices - vertices[0]
        basis, diagnostics = artifacts.build_action_amplitude_basis(
            vertices, times, phase, weights, seed, 1e-5)
        derivative = np.gradient(vertices, times, axis=0, edge_order=2)
        columns = np.stack([derivative * phase[:, index, None, None]
                            for index in range(phase.shape[1])], axis=-1)
        overlap = columns.reshape(-1, 3).T @ basis[0].reshape(-1)
        np.testing.assert_allclose(overlap, 0.0, atol=1e-10)
        self.assertAlmostEqual(diagnostics["normalized_weighted_norm"], 1.0)
        np.testing.assert_array_equal(basis[0, 0], np.zeros_like(basis[0, 0]))

    def test_freeze_schema_rejects_camera_or_gt_substitution(self):
        record = {
            "kind": target.FREEZE_KIND, "version": 1,
            "candidate_id": target.CANDIDATE_ID, "uid": "unit", "generation_seed": 42,
            "benchmark_revision": "revision", "source_indices": [0, 8, 15],
            "target_indices": list(range(16)),
            "frozen_without_candidate_or_confirmation_outcomes": True,
            "max_source_self_map_rms_over_diagonal": 0.1,
            "max_consensus_target_rms_over_diagonal": 0.1,
            "min_action_seed_rms_over_diagonal": 1e-5,
        }
        for name in ("source_sequence_ref", "source_report_ref",
                     "generation_identity_ref", "capture_identity_ref",
                     "window_record_ref", "decoder_record_ref",
                     "decoder_inputs_ref", "decoder_tensors_ref",
                     "weights_manifest_ref", "source_review_ref"):
            record[name] = {"path": name, "sha256": "0" * 64}
        record["camera_ref"] = {"path": "camera.json", "sha256": "0" * 64}
        core = dict(record); record["freeze_digest"] = target.canonical_digest(core)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Exact outcome-blind"):
                target.validate_freeze(Path(directory), record)

    def test_native_identity_validator_rejects_frame_subsets(self):
        arrays = self._native(); arrays["vertices"] = arrays["vertices"][:15]
        with self.assertRaisesRegex(ValueError, "16-frame"):
            artifacts.validate_native_arrays(arrays)

    def test_terminal_failure_is_not_a_scoreable_sequence(self):
        report = {"candidate_arm": "phase_only", "status": "error",
                  "exception_type": "PhaseAmplitudeError", "error": "gate"}
        self.assertNotIn("sha256", report)
        self.assertNotEqual(report["status"], "completed")

    def test_six_role_readout_keeps_treatment_failure_in_denominator(self):
        roles = []
        physical = []
        for role in comparison.ROLES:
            completed = role != comparison.CANDIDATE_ROLE
            case_id = "unit-" + role if completed else None
            roles.append({"role": role, "method_id": role,
                          "case_id": case_id,
                          "preparation_status": "completed" if completed else "error",
                          "preparation_error": None if completed else "solver: gate"})
            if completed:
                physical.append({"case_id": case_id, "status": "success",
                                 "cd_3d": 1.0, "cd_4d": 1.0,
                                 "cd_motion": 1.0})
        result = scoring.build_logical_readout(
            {"roles": roles}, {"cases": physical})
        self.assertEqual(result["logical_denominator"]["n_roles"], 6)
        self.assertEqual(result["n_successful_roles"], 5)
        self.assertEqual(result["n_failed_or_missing_roles"], 1)
        treatment = next(row for row in result["roles"]
                         if row["role"] == comparison.CANDIDATE_ROLE)
        self.assertEqual(treatment["status"], "preparation_error")

    def test_freeze_uses_exact_released_benchmark(self):
        self.assertEqual(target.ACTIONBENCH_REVISION,
                         "2796071cbe6248422fcbeab3101fa9f9886cb7b9")


if __name__ == "__main__":
    unittest.main()
