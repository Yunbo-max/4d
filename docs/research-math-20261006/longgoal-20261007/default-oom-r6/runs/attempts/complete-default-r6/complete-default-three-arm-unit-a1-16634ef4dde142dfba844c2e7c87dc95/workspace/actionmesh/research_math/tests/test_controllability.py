"""Numerical controls, not evidence of any 4D editing method's efficacy.

Every expectation below is analytic or an invariance/property of the problem;
no test computes its expected result through the implementation's helpers.
"""
import unittest

import numpy as np

try:
    from research_math import controllability as ctl
except ImportError:
    ctl = None


class ControllabilityTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ctl, "B controllability implementation is absent")

    def test_nullspace_rank_deficient_and_empty(self):
        # Losing full_matrices=True discards the nullspace of a wide matrix.
        matrix = np.array([[1., 2., 0.], [2., 4., 0.]])
        basis = ctl.nullspace(matrix)
        self.assertEqual(basis.shape, (3, 2))
        np.testing.assert_allclose(matrix @ basis, 0., atol=1e-12)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
        np.testing.assert_array_equal(ctl.nullspace(np.zeros((0, 3))), np.eye(3))
        self.assertEqual(ctl.nullspace(np.eye(3)).shape, (3, 0))

    def test_metric_projection_and_first_order_preservation(self):
        # An ordinary Euclidean projection gives (1,1), not the W projection.
        metric = np.diag([4., 1., 2.])
        contract = np.array([[1., -1., 0.]])
        step = ctl.project_model_step(np.eye(3), contract, [2., 0., 3.],
                                      output_metric=metric)
        np.testing.assert_allclose(step, [1.6, 1.6, 3.], atol=1e-12)
        np.testing.assert_allclose(contract @ step, 0., atol=1e-12)
        basis = ctl.reachable_basis(np.eye(3), contract, output_metric=metric)
        np.testing.assert_allclose(basis.T @ metric @ basis, np.eye(2), atol=1e-12)

    def test_invertible_latent_reparameterization(self):
        # A latent Euclidean trust ball would change after this rescaling/mixing.
        j = np.array([[1., 0.], [0., 1.], [1., 2.]])
        transform = np.array([[2., 1.], [.5, 3.]])
        args = (np.array([[1., -1., 0.]]), np.eye(3), np.array([1., 0., 2.]))
        kw = dict(output_metric=np.diag([2., 3., .5]), radius=.7)
        original = ctl.analyze_edit(j, *args, **kw)
        changed = ctl.analyze_edit(j @ transform, *args, **kw)
        np.testing.assert_allclose(original.step, changed.step, atol=1e-11)
        self.assertAlmostEqual(original.residual_squared, changed.residual_squared, places=11)
        np.testing.assert_allclose(args[0] @ original.step, 0., atol=1e-12)
        np.testing.assert_allclose(j @ np.linalg.lstsq(j, original.step, rcond=None)[0],
                                   original.step, atol=1e-12)

    def test_output_coordinates_and_metric_transform_together(self):
        # Raw-coordinate norms create a spurious cross-representation difference.
        j = np.array([[1.], [2.]])
        a = np.zeros((0, 2))
        b = np.array([[1., 1.]])
        w = np.array([[2., .3], [.3, 1.]])
        s = np.array([[2., .5], [0., 3.]])
        inverse = np.linalg.inv(s)
        first = ctl.analyze_edit(j, a, b, [2.], output_metric=w, radius=.4)
        second = ctl.analyze_edit(s @ j, a @ inverse, b @ inverse, [2.],
                                  output_metric=inverse.T @ w @ inverse, radius=.4)
        np.testing.assert_allclose(second.step, s @ first.step, atol=1e-11)
        self.assertAlmostEqual(first.residual_squared, second.residual_squared, places=11)

    def test_unreachable_rank_deficient_and_no_direction(self):
        j = np.array([[1., 2.], [0., 0.]])
        result = ctl.analyze_edit(j, np.zeros((0, 2)), np.eye(2), [0., 1.])
        self.assertEqual(result.preserving_rank, 1)
        self.assertAlmostEqual(result.residual_squared, 1.)
        np.testing.assert_allclose(result.step, [0., 0.], atol=1e-12)
        for no_j in (np.zeros((2, 0)), np.zeros((2, 2))):
            empty = ctl.analyze_edit(no_j, np.zeros((0, 2)), np.eye(2), [3., 4.])
            self.assertEqual(empty.preserving_rank, 0)
            self.assertAlmostEqual(empty.residual_squared, 25.)
            self.assertTrue(empty.converged)
        frozen = ctl.analyze_edit(np.eye(2), np.eye(2), np.eye(2), [3., 4.])
        self.assertEqual(frozen.preserving_rank, 0)

    def test_unbounded_reachability_does_not_imply_bounded_success(self):
        # Edit gain .001 reaches d=1 only with an output displacement of 1000.
        args = (np.ones((1, 1)), np.zeros((0, 1)), np.array([[.001]]), [1.])
        unlimited = ctl.analyze_edit(*args)
        limited = ctl.analyze_edit(*args, radius=1.)
        self.assertAlmostEqual(unlimited.residual_squared, 0., places=14)
        self.assertAlmostEqual(unlimited.step[0], 1000., places=8)
        self.assertAlmostEqual(limited.step[0], 1., places=10)
        self.assertAlmostEqual(limited.residual_squared, .998001, places=10)
        self.assertTrue(limited.converged)

    def test_trust_solve_is_not_clipped_unbounded_solution(self):
        # Hand KKT solution: C=diag(2,1), d=(2,1), lambda=1 gives a=(.8,.5).
        radius = np.sqrt(.89)
        result = ctl.analyze_edit(np.eye(2), np.zeros((0, 2)), np.diag([2., 1.]),
                                 [2., 1.], radius=radius)
        np.testing.assert_allclose(result.step, [.8, .5], atol=1e-10)
        self.assertAlmostEqual(result.lagrange_multiplier, 1., places=9)
        self.assertAlmostEqual(result.residual_squared, .41, places=10)
        self.assertLessEqual(result.step_norm, radius * (1 + 1e-12))
        self.assertLess(result.stationarity_residual, 1e-10)
        self.assertTrue(result.converged)

    def test_zero_radius_and_zero_target_are_not_nan_scores(self):
        zero_radius = ctl.analyze_edit(np.eye(2), np.zeros((0, 2)), np.eye(2),
                                       [1., 2.], radius=0.)
        np.testing.assert_array_equal(zero_radius.step, np.zeros(2))
        self.assertEqual(zero_radius.residual_squared, 5.)
        self.assertIsNone(zero_radius.stationarity_residual)
        zero_target = ctl.analyze_edit(np.eye(2), np.zeros((0, 2)), np.eye(2), [0., 0.])
        self.assertIsNone(zero_target.normalized_residual)
        np.testing.assert_array_equal(zero_target.step, np.zeros(2))

    def test_nested_gap_and_false_nesting_refusal(self):
        j = np.array([[1.], [0.]])
        args = (np.zeros((0, 2)), np.eye(2), [0., 1.])
        nested = ctl.nested_gap(j, np.eye(2), *args, radius=1.)
        self.assertAlmostEqual(nested.gap, 1., places=10)
        self.assertAlmostEqual(nested.reference.residual_squared, 0., places=10)
        with self.assertRaises(ValueError):
            ctl.nested_gap(j, np.array([[0.], [1.]]), *args, radius=1.)

    def test_contract_only_projector_can_leave_model_tangent(self):
        # Contract-only projection retains e2; the model only exposes e1.
        j = np.array([[1.], [0.], [0.]])
        a = np.array([[0., 0., 1.]])
        v = np.array([0., 1., 0.])
        bad_step = (np.eye(3) - a.T @ np.linalg.pinv(a @ a.T) @ a) @ v
        np.testing.assert_array_equal(bad_step, v)
        good_step = ctl.project_model_step(j, a, v)
        np.testing.assert_allclose(good_step, np.zeros(3), atol=1e-12)
        model = ctl.analyze_edit(j, a, np.eye(3), v)
        output = ctl.analyze_edit(np.eye(3), a, np.eye(3), v)
        self.assertAlmostEqual(model.residual_squared, 1.)
        self.assertAlmostEqual(output.residual_squared, 0.)

    def test_contract_orthogonality_roundoff_does_not_destroy_nullspace(self):
        # Rank based only on the tiny computed A@U can spuriously freeze a model.
        j = np.array([[1.], [1.], [1.]])
        a = np.array([[1., -1., 0.]])
        v = ctl.reachable_basis(j, a)
        self.assertEqual(v.shape, (3, 1))
        np.testing.assert_allclose(a @ v, 0., atol=1e-12)

    def test_full_spd_edit_metric_changes_optimum(self):
        # Min (a-0, -1)^T [[2,1],[1,2]] (a-0,-1) has a=.5, cost1.5.
        result = ctl.analyze_edit(np.array([[1.], [0.]]), np.zeros((0, 2)),
                                 np.eye(2), [0., 1.], edit_metric=[[2., 1.], [1., 2.]])
        np.testing.assert_allclose(result.step, [.5, 0.], atol=1e-12)
        self.assertAlmostEqual(result.residual_squared, 1.5)

    def test_nonlinear_acceptance_backtracks_and_does_not_hide_rejection(self):
        # Tangent to the unit circle drifts off it quadratically after retraction.
        x = np.array([1., 0.])
        proposed = np.array([0., .2])
        contract = lambda y: abs(float(y @ y) - 1.) <= .011
        accepted = ctl.backtrack_accept(x, proposed, contract, max_backtracks=4)
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.scale, .5)
        np.testing.assert_array_equal(accepted.point, [1., .1])
        rejected = ctl.backtrack_accept(x, proposed, lambda y: False, max_backtracks=2)
        self.assertFalse(rejected.accepted)
        np.testing.assert_array_equal(rejected.point, x)
        self.assertEqual(rejected.scale, 0.)

    def test_invalid_metrics_shapes_and_nonfinite_inputs_fail(self):
        args = (np.eye(2), np.zeros((0, 2)), np.eye(2), [1., 0.])
        for metric in (np.diag([1., 0.]), [[1., 2.], [0., 1.]], [[1., 0.], [0., -1.]]):
            with self.assertRaises(ValueError):
                ctl.analyze_edit(*args, output_metric=metric)
        with self.assertRaises(ValueError):
            ctl.analyze_edit(*args, radius=-1.)
        with self.assertRaises(ValueError):
            ctl.analyze_edit(np.eye(2), np.zeros((0, 3)), np.eye(2), [1., 0.])
        with self.assertRaises(ValueError):
            ctl.analyze_edit(np.eye(2), np.zeros((0, 2)), np.eye(2), [np.nan, 0.])

    def test_constructed_demo_explicitly_allows_output_baseline_to_win(self):
        demo = ctl.constructed_demo()
        self.assertEqual(demo["evidence_scope"], "constructed numerical controls; not 4D efficacy")
        self.assertGreater(demo["missing_direction"]["model_residual_squared"],
                           demo["missing_direction"]["output_residual_squared"])
        self.assertGreater(demo["trust_budget"]["bounded_residual_squared"],
                           demo["trust_budget"]["unbounded_residual_squared"])


if __name__ == "__main__":
    unittest.main()
