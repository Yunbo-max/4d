import unittest

import numpy as np

from research_math.protected_projection import project_protected_step


class ProtectedProjectionTests(unittest.TestCase):
    def test_removes_the_component_seen_by_an_action_constraint(self):
        result = project_protected_step(
            np.array([1.0, 2.0]), np.array([[1.0, 0.0]]), np.eye(2)
        )
        np.testing.assert_allclose(result, [0.0, 2.0], atol=1e-12)

    def test_weighted_projection_satisfies_constraint_and_minimizes_change(self):
        result = project_protected_step(
            np.array([1.0, 0.0]), np.array([[1.0, 1.0]]),
            np.diag([2.0, 1.0])
        )
        np.testing.assert_allclose(result, [2.0 / 3.0, -2.0 / 3.0], atol=1e-12)
        np.testing.assert_allclose(np.array([[1.0, 1.0]]) @ result, [0.0], atol=1e-12)

    def test_redundant_constraints_use_a_stable_pseudoinverse(self):
        result = project_protected_step(
            np.array([2.0, -1.0]), np.array([[1.0, 1.0], [2.0, 2.0]]), np.eye(2)
        )
        np.testing.assert_allclose(result, [1.5, -1.5], atol=1e-12)

    def test_pinned_coordinates_are_zero_and_never_reintroduced(self):
        result = project_protected_step(
            np.array([4.0, 1.0, 2.0]), np.array([[0.0, 1.0, 1.0]]),
            np.eye(3), pins=[0]
        )
        np.testing.assert_allclose(result, [0.0, -0.5, 0.5], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
