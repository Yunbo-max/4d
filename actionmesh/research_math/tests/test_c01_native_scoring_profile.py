"""Local-only non-regression checks for the isolated C01 scoring profile."""
from __future__ import annotations

import unittest

import launch_c01_native_scoring as c01_launcher
import prepare_c01_native_scoring as c01_plan
from research_math import c01_native_scoring as c01_scoring
from research_math import c14_native_scoring as c14_scoring


class C01NativeScoringProfileTests(unittest.TestCase):
    def test_profile_is_isolated_from_c14_defaults(self):
        self.assertEqual(c14_scoring.REQUEST_KIND, "c14-native-scoring-request")
        self.assertEqual(c14_scoring.CANDIDATE_ROLE, "corotational_residual")
        self.assertEqual(c01_scoring.REQUEST_KIND, "c01-native-scoring-request")
        self.assertEqual(c01_scoring.CANDIDATE_ROLE, "self_map_subtraction")
        self.assertEqual(
            c01_scoring.CONTROL_ROLES,
            ("b0", "b_star", "raw_uncorrected", "mean_bias"))
        self.assertEqual(
            c01_scoring.CONTRAST_DIRECTION,
            "self-map subtraction minus control; lower is better")

    def test_plan_and_launcher_use_c01_identity(self):
        self.assertEqual(c01_plan.TASK_ID, "c01-five-role-official-scoring")
        self.assertEqual(c01_plan.PRIMARY_METRIC, "cd_motion")
        self.assertEqual(c01_plan.GUARDRAIL_METRICS, ("cd_3d", "cd_4d"))
        self.assertEqual(c01_plan.AUTHORIZATION_KIND, "c01-gpu-resume-authorization")
        self.assertEqual(c01_launcher.CLAIM_KIND, "c01-launch-claim")
        self.assertEqual(c01_launcher.ENV_PREFIX, "C01")
        self.assertIn("actionmesh/prepare_c01_native_scoring.py",
                      c01_plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/prepare_c14_native_scoring.py",
                      c01_plan.EXTRA_CODE_SOURCES)
        self.assertIn("actionmesh/launch_c14_native_scoring.py",
                      c01_plan.EXTRA_CODE_SOURCES)


if __name__ == "__main__":
    unittest.main()
