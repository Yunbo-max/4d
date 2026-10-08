"""Local-only non-regression checks for the isolated C02 scoring profile."""
from __future__ import annotations

import unittest

import launch_c02_native_scoring as c02_launcher
import prepare_c02_native_scoring as c02_plan
from research_math import c02_native_scoring as c02_scoring
from research_math import c14_native_scoring as c14_scoring


class C02NativeScoringProfileTests(unittest.TestCase):
    def test_profile_is_isolated_from_c14_defaults(self):
        self.assertEqual(c14_scoring.REQUEST_KIND, "c14-native-scoring-request")
        self.assertEqual(c14_scoring.CANDIDATE_ROLE, "corotational_residual")
        self.assertEqual(c02_scoring.REQUEST_KIND, "c02-native-scoring-request")
        self.assertEqual(c02_scoring.CANDIDATE_ROLE, "protected_step")
        self.assertEqual(
            c02_scoring.CONTROL_ROLES,
            ("b0", "b_star", "geometry_only", "strength_matched_blend"))
        self.assertEqual(
            c02_scoring.CONTRAST_DIRECTION,
            "protected step minus control; lower is better")

    def test_plan_and_launcher_use_c02_identity(self):
        self.assertEqual(c02_plan.TASK_ID, "c02-five-role-official-scoring")
        self.assertEqual(c02_plan.PRIMARY_METRIC, "cd_3d")
        self.assertEqual(c02_plan.GUARDRAIL_METRICS, ("cd_4d", "cd_motion"))
        self.assertEqual(c02_plan.AUTHORIZATION_KIND, "c02-gpu-resume-authorization")
        self.assertEqual(c02_launcher.CLAIM_KIND, "c02-launch-claim")
        self.assertEqual(c02_launcher.ENV_PREFIX, "C02")
        self.assertIn("actionmesh/prepare_c02_native_scoring.py",
                      c02_plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/prepare_c14_native_scoring.py",
                      c02_plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/launch_c14_native_scoring.py",
                      c02_plan.EXTRA_CODE_SOURCES)


if __name__ == "__main__":
    unittest.main()
