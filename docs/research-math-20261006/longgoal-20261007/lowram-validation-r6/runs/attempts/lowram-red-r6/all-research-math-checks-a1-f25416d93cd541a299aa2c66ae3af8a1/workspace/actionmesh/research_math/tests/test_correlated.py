"""Constructed algebra/constraint tests; no claim about natural 4D efficacy."""
import unittest
import numpy as np

from research_math.correlated import apply_covariance_inverse, solve_correction, constructed_benchmark


class CovarianceTests(unittest.TestCase):
    def test_woodbury_matches_dense_inverse_for_vectors_and_matrices(self):
        rng = np.random.default_rng(142)
        modes = rng.normal(size=(19, 4))
        variances = np.array([0., .2, 2., 7.])
        covariance = .7 * np.eye(19) + (modes * variances) @ modes.T
        for rhs in (rng.normal(size=19), rng.normal(size=(19, 3))):
            expected = np.linalg.solve(covariance, rhs)
            actual = apply_covariance_inverse(rhs, modes, variances, sigma2=.7)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    def test_repeated_shared_bias_precision_saturates(self):
        values = []
        for count in (1, 8, 128):
            ones = np.ones(count)
            precision = ones @ apply_covariance_inverse(ones, ones[:, None], np.array([2.]), sigma2=3.)
            self.assertAlmostEqual(precision, count / (3. + 2. * count), places=12)
            self.assertLess(precision, .5)
            values.append(precision)
        self.assertGreater(values[-1], values[0])
        self.assertGreater(128 / 3., 80 * values[-1])

    def test_bad_covariance_inputs_are_rejected(self):
        for rhs, modes, variances, sigma2 in (
            ([1., 2.], [[1.]], [1.], 1.),
            ([1.], [[1.]], [-1.], 1.),
            ([1.], [[1.]], [1.], 0.),
            ([np.nan], [[1.]], [1.], 1.),
        ):
            with self.assertRaises(ValueError):
                apply_covariance_inverse(rhs, modes, variances, sigma2=sigma2)


class CorrectionTests(unittest.TestCase):
    def solve(self, ro, jo, rs, js, **updates):
        p = np.asarray(jo).shape[1]
        kwargs = dict(bias_modes=np.zeros((p, 0)), bias_variances=np.zeros(0),
                      sigma2=1., synthetic_weight=1., ridge=.1, trust_radius=10.)
        kwargs.update(updates)
        return solve_correction(np.asarray(ro), np.asarray(jo), np.asarray(rs), np.asarray(js), **kwargs)

    def test_any_observed_group_worsening_rejects_step_and_preserves_zero(self):
        result = self.solve([-1., -1.], np.eye(2), [-4., 8.], 4*np.eye(2),
                            group_ids=[0, 1], group_caps=[0., 0.])
        self.assertTrue(result['feasible'])
        self.assertEqual(result['status'], 'rejected_constraints')
        np.testing.assert_array_equal(result['correction'], np.zeros(2))
        np.testing.assert_array_equal(result['group_errors_after'], result['group_errors_before'])

    def test_backtracking_honors_group_caps_and_trust_radius(self):
        result = self.solve([0., -1.], np.eye(2), [-5., -5.], np.eye(2),
                            group_ids=[0, 1], group_caps=[.04, 0.], trust_radius=.5)
        self.assertEqual(result['status'], 'accepted')
        self.assertGreater(result['backtracks'], 0)
        self.assertLessEqual(np.linalg.norm(result['correction']), .5 + 1e-14)
        self.assertLessEqual(result['group_errors_after'][0], .04)
        self.assertLessEqual(result['group_errors_after'][1], 1.)
        self.assertLess(result['objective_after'], result['objective_before'])

    def test_declared_budget_changes_precision_without_estimating_from_targets(self):
        common = dict(bias_modes=np.ones((1, 1)), ridge=1., group_caps=[10.])
        independent = self.solve([0.], [[1.]], -np.ones(4), np.ones((4, 1)),
                                 bias_variances=[0.], **common)
        correlated = self.solve([0.], [[1.]], -np.ones(4), np.ones((4, 1)),
                                bias_variances=[2.], **common)
        self.assertAlmostEqual(independent['correction'][0], 2/3, places=12)
        self.assertAlmostEqual(correlated['correction'][0], 2/11, places=12)
        self.assertEqual(correlated['bias_variances'], [2.])
        self.assertFalse(correlated['uncertainty_calibrated'])

    def test_zero_residual_control_remains_exactly_unchanged(self):
        result = self.solve(np.zeros(2), np.eye(2), np.zeros(4), np.tile(np.eye(2), (2, 1)),
                            bias_modes=np.eye(2), bias_variances=[3., 4.])
        np.testing.assert_array_equal(result['correction'], np.zeros(2))
        self.assertEqual(result['objective_after'], 0.)
        self.assertEqual(result['status'], 'unchanged_stationary')

    def test_bad_caps_nonfinite_jacobian_and_uncovered_groups_rejected(self):
        for updates in ({'group_caps': [-.1]}, {'group_ids': [1]}, {'trust_radius': -1.}):
            with self.assertRaises(ValueError):
                self.solve([-1.], [[1.]], [-1.], [[1.]], **updates)
        with self.assertRaises(ValueError):
            self.solve([-1.], [[np.inf]], [-1.], [[1.]])

    def test_seeded_benchmark_preserves_all_baselines_and_negative_controls(self):
        first = constructed_benchmark(42)
        second = constructed_benchmark(42)
        self.assertEqual(first, second)
        self.assertFalse(first['natural_efficacy_claim'])
        self.assertGreaterEqual(len(first['cases']), 4)
        for case in first['cases']:
            self.assertEqual(set(case['arms']), {'source_only', 'equal_weight', 'inverse_count', 'fixed_cap', 'covariance'})
            for arm in case['arms'].values():
                self.assertTrue(arm['feasible'])
                self.assertFalse(arm['constrained_optimum_certified'])
        positive = next(c for c in first['cases'] if c['name'] == 'shared_hidden_bias')
        negative = next(c for c in first['cases'] if c['name'] == 'helpful_hidden_prior')
        self.assertLess(positive['arms']['covariance']['parameter_error'], positive['arms']['equal_weight']['parameter_error'])
        self.assertGreater(negative['arms']['covariance']['parameter_error'], negative['arms']['equal_weight']['parameter_error'])


if __name__ == '__main__':
    unittest.main()
