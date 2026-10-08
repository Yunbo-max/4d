"""Source profile checks for C07; Local executes these checks."""
from __future__ import annotations

import unittest

import prepare_c07_native_scoring as plan
from research_math import c07_native_comparison as comparison
from research_math import c07_native_scoring as scoring


class C07NativeScoringProfileTests(unittest.TestCase):
    def test_profile_has_exact_five_roles_and_candidate_identity(self):
        self.assertEqual(comparison.CANDIDATE_ID, "4d-math-20261006-c07")
        self.assertEqual(comparison.ROLES, (
            "b0", "b_star", "confidence_threshold_fallback", "full_mass_transport",
            "partial_mass_native_fallback"))
        self.assertEqual(scoring.CANDIDATE_ROLE, "partial_mass_native_fallback")
        self.assertEqual(scoring.REQUEST_KIND, "c07-native-scoring-request")
        self.assertEqual(scoring.CONTROL_ROLES, (
            "b0", "b_star", "confidence_threshold_fallback", "full_mass_transport"))
        self.assertEqual(comparison.ALLOWED_INFERENCE_SEEDS, (42,))
        self.assertEqual(comparison.PRIMARY_METRIC, "cd_motion")
        self.assertEqual(comparison.GUARDRAIL_METRICS, ("cd_3d", "cd_4d"))

    def test_plan_profile_keeps_single_attempt_scope(self):
        self.assertEqual(plan.CANDIDATE_ID, "4d-math-20261006-c07")
        self.assertEqual(plan.PROFILE, "c07")
        self.assertEqual(plan.AUTHORIZATION_SCOPE, "single_c07_scoring_attempt")
        self.assertEqual(plan.ROLE_FOR_CONTRACT_ARM["treatment"],
                         "partial_mass_native_fallback")
        self.assertEqual(scoring.CONTROLLER_ENV_PREFIX, "C07")

    def test_three_preparation_failures_stay_in_c07_denominator(self):
        roles = [{
            "role": "b0", "method_id": "native-actionmesh-b0",
            "case_id": "u-b0", "preparation_status": "completed",
        }, {
            "role": "b_star", "method_id": "native-actionmesh-b0",
            "case_id": "u-b0", "alias_of": "b0",
            "preparation_status": "completed",
        }]
        for role in comparison.ROLES[2:]:
            roles.append({
                "role": role, "method_id": "method-" + role,
                "case_id": None, "preparation_status": "error",
                "preparation_error": "retained C07 preparation failure",
            })
        frozen = {"roles": roles}
        official = {"cases": [{
            "case_id": "u-b0", "status": "success", "n_frames": 16,
            "cd_3d": 1.0, "cd_4d": 2.0, "cd_motion": 3.0,
        }]}
        readout = scoring.build_logical_readout(frozen, official)
        rows = {row["role"]: row for row in readout["roles"]}
        for role in comparison.ROLES[2:]:
            self.assertEqual(rows[role]["status"], "preparation_error")
            self.assertNotIn("metrics", rows[role])
        self.assertEqual(readout["n_failed_or_missing_roles"], 3)
        self.assertEqual(readout["n_successful_roles"], 2)
        self.assertEqual(readout["contrasts"], [])


if __name__ == "__main__":
    unittest.main()
