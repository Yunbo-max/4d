"""Local-only non-regression checks for the isolated C15 scoring profile."""
from __future__ import annotations

import unittest

import launch_c15_native_scoring as launcher
import prepare_c15_native_scoring as plan
from research_math import c15_native_comparison as comparison
from research_math import c15_native_scoring as scoring
from research_math import c14_native_scoring as c14_scoring


class C15NativeScoringProfileTests(unittest.TestCase):
    def test_profile_isolated_from_c14_defaults(self):
        self.assertEqual(c14_scoring.REQUEST_KIND, "c14-native-scoring-request")
        self.assertEqual(c14_scoring.CANDIDATE_ROLE, "corotational_residual")
        self.assertEqual(scoring.REQUEST_KIND, "c15-native-scoring-request")
        self.assertEqual(scoring.CANDIDATE_ROLE, "protected_residual_svt")
        self.assertEqual(scoring.CONTROL_ROLES, (
            "b0", "b_star", "gaussian", "unprotected_svt",
            "rank_matched_tsvd"))
        self.assertEqual(len(comparison.ROLES), 6)

    def test_plan_and_launcher_have_only_c15_identity(self):
        self.assertEqual(plan.TASK_ID, "c15-six-role-official-scoring")
        self.assertEqual(plan.PRIMARY_METRIC, "cd_motion")
        self.assertEqual(plan.GUARDRAIL_METRICS, ("cd_3d", "cd_4d"))
        self.assertEqual(plan.AUTHORIZATION_SCOPE, "single_c15_scoring_attempt")
        self.assertEqual(plan.AUTHORIZATION_KIND, "c15-gpu-resume-authorization")
        self.assertEqual(launcher.CLAIM_KIND, "c15-launch-claim")
        self.assertEqual(launcher.ENV_PREFIX, "C15")
        self.assertIn("actionmesh/research_math/protected_lowrank_candidate.py",
                      plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/prepare_protected_lowrank_candidate.py",
                      plan.EXTRA_CODE_SOURCES)


if __name__ == "__main__":
    unittest.main()
