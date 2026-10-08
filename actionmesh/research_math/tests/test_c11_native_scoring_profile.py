"""Local-only non-regression checks for the isolated C11 scoring profile."""
from __future__ import annotations

import unittest

import launch_c11_native_scoring as launcher
import prepare_c11_native_scoring as plan
from research_math import c11_native_scoring as scoring
from research_math import c14_native_scoring as c14_scoring


class C11NativeScoringProfileTests(unittest.TestCase):
    def test_profile_isolated_from_c14_defaults(self):
        self.assertEqual(c14_scoring.CANDIDATE_ROLE, "corotational_residual")
        self.assertEqual(scoring.REQUEST_KIND, "c11-native-scoring-request")
        self.assertEqual(scoring.CANDIDATE_ROLE,
                         "rotation_preserving_stretch_projection")
        self.assertEqual(scoring.CONTROL_ROLES,
                         ("b0", "b_star", "arap_repair", "elastic_repair"))

    def test_plan_and_launcher_keep_c11_identity(self):
        self.assertEqual(plan.TASK_ID, "c11-five-role-official-scoring")
        self.assertEqual(plan.PRIMARY_METRIC, "cd_3d")
        self.assertEqual(plan.GUARDRAIL_METRICS, ("cd_4d", "cd_motion"))
        self.assertEqual(plan.AUTHORIZATION_SCOPE, "single_c11_scoring_attempt")
        self.assertEqual(launcher.CLAIM_KIND, "c11-launch-claim")
        self.assertEqual(launcher.ENV_PREFIX, "C11")
        self.assertIn("actionmesh/research_math/strain_projection_candidate.py",
                      plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/research_math/integrable_gradient_candidate.py",
                      plan.EXTRA_CODE_SOURCES)


if __name__ == "__main__":
    unittest.main()
