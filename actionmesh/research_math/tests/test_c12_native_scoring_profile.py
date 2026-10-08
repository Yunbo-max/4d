"""Source profile checks for C12's shared scorer; Local executes them."""
from __future__ import annotations

import unittest

from research_math import c12_native_comparison as comparison
from research_math import c12_native_scoring as scoring


class C12NativeScoringProfileTests(unittest.TestCase):
    def test_profile_has_exact_five_roles_and_candidate_identity(self):
        self.assertEqual(comparison.CANDIDATE_ID, "4d-math-20261006-c12")
        self.assertEqual(comparison.ROLES, (
            "b0", "b_star", "fixed_damping", "generic_backtracking",
            "exact_quadratic_admission"))
        self.assertEqual(scoring.CANDIDATE_ROLE, "exact_quadratic_admission")
        self.assertEqual(scoring.REQUEST_KIND, "c12-native-scoring-request")
        self.assertEqual(scoring.CONTROL_ROLES, (
            "b0", "b_star", "fixed_damping", "generic_backtracking"))


if __name__ == "__main__":
    unittest.main()
