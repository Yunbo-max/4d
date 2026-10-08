"""Local-only non-regression checks for the isolated C10 scoring profile."""
from __future__ import annotations

import unittest

import launch_c10_native_scoring as launcher
import prepare_c10_native_scoring as plan
from research_math import c10_native_scoring as scoring
from research_math import c14_native_scoring as c14_scoring


class C10NativeScoringProfileTests(unittest.TestCase):
    def test_profile_isolated_from_c14_defaults(self):
        self.assertEqual(c14_scoring.REQUEST_KIND, "c14-native-scoring-request")
        self.assertEqual(c14_scoring.CANDIDATE_ROLE, "corotational_residual")
        self.assertEqual(scoring.REQUEST_KIND, "c10-native-scoring-request")
        self.assertEqual(scoring.CANDIDATE_ROLE, "pinned_integrable_solve")
        self.assertEqual(scoring.CONTROL_ROLES, (
            "b0", "b_star", "direct_common_lift", "independent_local_repair"))
        self.assertEqual(
            scoring.CONTRAST_DIRECTION,
            "pinned integrable solve minus control; lower is better")

    def test_plan_and_launcher_have_only_c10_identity(self):
        self.assertEqual(plan.TASK_ID, "c10-five-role-official-scoring")
        self.assertEqual(plan.PRIMARY_METRIC, "cd_3d")
        self.assertEqual(plan.GUARDRAIL_METRICS, ("cd_4d", "cd_motion"))
        self.assertEqual(plan.AUTHORIZATION_SCOPE, "single_c10_scoring_attempt")
        self.assertEqual(plan.AUTHORIZATION_KIND, "c10-gpu-resume-authorization")
        self.assertEqual(launcher.CLAIM_KIND, "c10-launch-claim")
        self.assertEqual(launcher.ENV_PREFIX, "C10")
        self.assertIn("actionmesh/research_math/integrable_gradient_candidate.py",
                      plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/prepare_c14_native_scoring.py",
                      plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/launch_c14_native_scoring.py",
                      plan.EXTRA_CODE_SOURCES)


if __name__ == "__main__":
    unittest.main()
