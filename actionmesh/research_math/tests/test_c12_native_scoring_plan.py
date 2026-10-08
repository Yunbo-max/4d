"""Static plan/launcher contracts for C12; Local executes these checks."""
from __future__ import annotations

import unittest

import prepare_c12_native_scoring as plan
from research_math import c12_native_scoring as scoring


class C12NativeScoringPlanTests(unittest.TestCase):
    def test_profile_keeps_zero_retry_and_single_use_scope(self):
        self.assertEqual(plan.CANDIDATE_ID, "4d-math-20261006-c12")
        self.assertEqual(plan.AUTHORIZATION_SCOPE, "single_c12_scoring_attempt")
        self.assertEqual(plan.PROFILE, "c12")
        self.assertEqual(plan.ROLE_FOR_CONTRACT_ARM["treatment"],
                         "exact_quadratic_admission")
        self.assertEqual(scoring.CONTROLLER_ENV_PREFIX, "C12")


if __name__ == "__main__":
    unittest.main()
