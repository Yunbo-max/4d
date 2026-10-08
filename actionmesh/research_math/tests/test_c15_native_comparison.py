"""Local source checks for C15's prospective six-role comparison boundary."""
from __future__ import annotations

from datetime import datetime, timezone
import unittest

import numpy as np

from research_math import c15_native_comparison as comparison


class C15NativeComparisonTests(unittest.TestCase):
    def arrays(self):
        vertices = np.zeros((16, 4, 3), dtype=np.float32)
        return {
            "vertices": vertices,
            "faces": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
            "timesteps": np.arange(16, dtype=np.float64),
            "frame_indices": np.arange(16, dtype=np.int64),
            "query_vertex_ids": np.arange(4, dtype=np.int64),
        }

    def freeze(self):
        rows = [{"role": role} for role in comparison.ROLES]
        value = {
            "kind": "c15-native-comparison-freeze", "version": 1,
            "candidate_id": comparison.CANDIDATE_ID,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "uid": "native-uid", "inference_seed": 42, "scoring_seed": 44,
            "primary_metric": "cd_motion",
            "guardrail_metrics": ["cd_3d", "cd_4d"], "roles": rows,
            "basis_source_policy": "analytic_anchored_dct",
            "residual_rank_policy": "candidate_export_numeric_rank",
            "lambda_value": 0.1,
            "basis_orthogonality_tolerance": 1e-10,
            "protected_coefficient_tolerance": 1e-6,
            "numeric_rank_tolerance": 1e-8,
            "max_artifact_bytes": 1024 * 1024,
        }
        for index, name in enumerate((
                "source_sequence_ref", "source_report_ref",
                "candidate_artifact_ref", "basis_ref", "basis_evidence_ref",
                "b_star_decision_ref")):
            value[name] = {"path": name + ".json",
                           "sha256": format(index + 1, "064x")}
        value["freeze_digest"] = comparison.canonical_digest(value)
        return value

    def test_freeze_requires_six_ordered_roles_and_motion_primary(self):
        comparison._freeze_core(self.freeze())
        changed = self.freeze()
        changed["roles"] = changed["roles"][:-1]
        changed["freeze_digest"] = comparison.canonical_digest(
            {key: value for key, value in changed.items()
             if key != "freeze_digest"})
        with self.assertRaisesRegex(ValueError, "six ordered"):
            comparison._freeze_core(changed)
        changed = self.freeze()
        changed["primary_metric"] = "cd_3d"
        changed["freeze_digest"] = comparison.canonical_digest(
            {key: value for key, value in changed.items()
             if key != "freeze_digest"})
        with self.assertRaisesRegex(ValueError, "frozen C15"):
            comparison._freeze_core(changed)

    def test_every_completed_role_requires_exact_anchor(self):
        source = self.arrays()
        drifted = {name: value.copy() for name, value in source.items()}
        drifted["vertices"][0, 0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "anchor"):
            comparison._same_native(source, drifted, require_anchor=True)

    def test_content_digest_deduplicates_equal_arrays_not_container_bytes(self):
        first = self.arrays()
        second = {name: value.copy() for name, value in first.items()}
        self.assertEqual(comparison._sequence_content_digest(first),
                         comparison._sequence_content_digest(second))
        second["vertices"][1, 0, 0] = 1.0
        self.assertNotEqual(comparison._sequence_content_digest(first),
                            comparison._sequence_content_digest(second))

    def test_named_gaussian_is_not_replaced_by_b0_or_b_star(self):
        self.assertEqual(comparison.ROLES, (
            "b0", "b_star", "gaussian", "unprotected_svt",
            "rank_matched_tsvd", "protected_residual_svt"))
        self.assertIn("gaussian", comparison.CONTROL_ROLES)
        self.assertEqual(comparison.ARTIFACT_ROLES["rank_matched_tsvd"],
                         "rank_matched_tsvd")


if __name__ == "__main__":
    unittest.main()
