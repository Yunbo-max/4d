"""Source-only software acceptance cases; algebra fixtures are not native data.

Run only by the authorized Local harness. No scientific qualification is implied
by these numerical or serialization checks, even after Local execution.
"""
import copy
import hashlib
import itertools
import json
import unittest

import numpy as np

from research_math.correlated_calibration import (
    CalibrationFailure, apply_frozen_fits, fit_affine, unit_c01,
    validate_fit, verify_fit,
)


class CorrelatedCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array(list(itertools.product((-1., 1.), repeat=3)))
        self.intercept = np.array([.4, -.2, .7])
        self.matrix = np.array([[2., -.3, .8], [.9, -1., .2], [-.5, .6, 1.2]])
        self.error = self.intercept + self.reference @ self.matrix.T
        self.weights = np.ones(8)
        self.config = dict(mode="full", loss="squared", ridge=.25, tau=.15,
                           max_iterations=3000, gradient_tolerance=1e-8,
                           objective_tolerance=1e-13)

    def fit(self, **changes):
        config = dict(self.config, **changes)
        return fit_affine(self.reference, self.error, self.weights, **config)

    @staticmethod
    def rehash(fit):
        data = {key: value for key, value in fit.items() if key != "fit_sha256"}
        fit["fit_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        return fit

    def test_full_squared_recovers_independent_closed_form_with_unpenalized_intercept(self):
        fit = self.fit()
        np.testing.assert_allclose(fit["intercept"], self.intercept, atol=1e-13)
        np.testing.assert_allclose(fit["matrix"], self.matrix * 8 / 8.25, atol=1e-13)
        self.assertEqual(fit["diagnostics"]["iterations"], 1)
        replay = verify_fit(fit, self.reference, self.error, self.weights)
        self.assertLessEqual(replay["gradient_inf"], 1e-8)

    def test_diagonal_and_intercept_are_real_restricted_optimizers(self):
        diagonal = self.fit(mode="diagonal")
        np.testing.assert_allclose(diagonal["matrix"], np.diag(np.diag(self.matrix)) * 8 / 8.25, atol=1e-13)
        intercept = self.fit(mode="intercept")
        np.testing.assert_array_equal(intercept["matrix"], np.zeros((3, 3)))
        np.testing.assert_allclose(intercept["intercept"], self.intercept, atol=1e-13)
        self.assertNotEqual(diagonal["fit_sha256"], intercept["fit_sha256"])

    def test_weight_scale_is_not_silently_normalized(self):
        doubled = fit_affine(self.reference, self.error, 2*self.weights, **self.config)
        np.testing.assert_allclose(doubled["matrix"], self.matrix * 16 / 16.25, atol=1e-13)
        self.assertEqual(doubled["diagnostics"]["weight_sum"], 16)

    def test_irls_all_restrictions_have_actual_vector_loss_stationarity(self):
        # Coupled output errors ensure shared Euclidean radial weights matter.
        error = self.error.copy()
        error[0] += [7., -2., 3.]
        weights = np.array([.5, 1., 2., 1., 1.5, .7, 1., 2.])
        for mode in ("full", "diagonal", "intercept"):
            with self.subTest(mode=mode):
                fit = fit_affine(self.reference, error, weights,
                                **dict(self.config, mode=mode, loss="smoothed_unsquared"))
                matrix, intercept = np.array(fit["matrix"]), np.array(fit["intercept"])
                residual = intercept + self.reference @ matrix.T - error
                radial = weights / np.sqrt(np.sum(residual**2, axis=1) + .15**2)
                g_b = (radial[:, None] * residual).sum(axis=0)
                g_m = (radial[:, None] * residual).T @ self.reference + .25 * matrix
                if mode == "diagonal":
                    g_m = np.diag(g_m)
                elif mode == "intercept":
                    g_m = np.zeros(1)
                self.assertLessEqual(max(np.max(np.abs(g_b)), np.max(np.abs(g_m))), 1e-8)
                history = fit["diagnostics"]["history"]
                for previous, current in zip(history, history[1:]):
                    self.assertLessEqual(current["objective"], previous["objective"] + 1e-13 * (1+abs(previous["objective"])))
                    self.assertLessEqual(current["surrogate_after"], current["surrogate_before"] + 1e-13 * (1+abs(current["surrogate_before"])))
                verify_fit(fit, self.reference, error, weights)

    def test_rank_deficient_reference_and_zero_weight_rows_are_legal(self):
        reference = np.zeros((4, 3))
        errors = np.array([[2., -1., 3.], [2., -1., 3.], [2., -1., 3.], [100., 0., -9.]])
        for loss in ("squared", "smoothed_unsquared"):
            fit = fit_affine(reference, errors, [1., 1., 1., 0.],
                             **dict(self.config, loss=loss))
            np.testing.assert_allclose(fit["intercept"], [2., -1., 3.], atol=1e-12)
            np.testing.assert_array_equal(fit["matrix"], np.zeros((3, 3)))
            self.assertEqual(fit["diagnostics"]["positive_weight_count"], 3)

    def test_nonconvergence_retains_diagnostics_and_never_returns_fallback(self):
        error = self.error.copy()
        error[0] += [7., -2., 3.]
        with self.assertRaises(CalibrationFailure) as caught:
            fit_affine(self.reference, error, self.weights,
                       **dict(self.config, loss="smoothed_unsquared", max_iterations=1,
                              gradient_tolerance=1e-14))
        diagnostics = caught.exception.diagnostics
        self.assertEqual(diagnostics["status"], "rejected_nonconvergence")
        self.assertFalse(diagnostics["converged"])
        self.assertEqual(len(diagnostics["history"]), 2)
        self.assertNotIn("matrix", diagnostics)
        json.dumps(diagnostics, allow_nan=False)

    def test_bad_controls_shapes_and_nonfinite_inputs_fail_closed(self):
        for key, value in (("ridge", 0), ("tau", 0), ("tau", float("nan")),
                           ("gradient_tolerance", 0), ("objective_tolerance", -1),
                           ("max_iterations", True), ("max_iterations", 1.5)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.fit(**{key: value})
        for weight in (np.zeros(8), -np.ones(8), np.ones(7)):
            with self.assertRaises(ValueError):
                fit_affine(self.reference, self.error, weight, **self.config)
        reference = self.reference.copy()
        reference[0, 0] = np.inf
        with self.assertRaises(ValueError):
            fit_affine(reference, self.error, self.weights, **self.config)

    def test_overflow_rejection_preserves_finite_history(self):
        with self.assertRaises(CalibrationFailure) as caught:
            fit_affine(self.reference, self.error*1e200, self.weights, **self.config)
        self.assertEqual(caught.exception.diagnostics["status"], "rejected_numerical_failure")
        json.dumps(caught.exception.diagnostics, allow_nan=False)

    def test_frozen_json_round_trip_and_tampering_checks(self):
        original = self.fit()
        restored = validate_fit(json.loads(json.dumps(original)))
        self.assertEqual(original, restored)
        changed = copy.deepcopy(original)
        changed["matrix"][0][0] += .1
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_fit(changed)
        # Even a rewritten local digest cannot substitute for development replay.
        self.rehash(changed)
        with self.assertRaisesRegex(ValueError, "replay"):
            verify_fit(changed, self.reference, self.error, self.weights)
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            verify_fit(original, self.reference, self.error + .01, self.weights)
        for mutate in (
            lambda f: f["diagnostics"].update(converged=False),
            lambda f: f["diagnostics"].update(iterations=100),
            lambda f: f.update(mode="intercept"),
            lambda f: f["diagnostics"]["history"][-1].update(gradient_inf=100.),
        ):
            bad = copy.deepcopy(original)
            mutate(bad)
            self.rehash(bad)
            with self.assertRaises(ValueError):
                validate_fit(bad)

    def test_native_full16_application_preserves_anchor_and_does_not_clip(self):
        fit = self.fit()
        raw = np.arange(16*8*3, dtype=np.float64).reshape(16, 8, 3) / 10
        raw_before = raw.copy()
        anchor = np.full((8, 3), .123456789, dtype=np.float32)
        output = apply_frozen_fits(raw, self.reference, anchor, [fit] * 15)
        expected = raw[1:] - np.array(fit["intercept"]) - self.reference @ np.array(fit["matrix"]).T
        np.testing.assert_array_equal(output[0], anchor)
        np.testing.assert_allclose(output[1:], expected, rtol=0, atol=0)
        np.testing.assert_array_equal(raw, raw_before)
        self.assertGreater(output.max(), 1.)
        control = unit_c01(raw, self.reference, anchor)
        np.testing.assert_array_equal(control[0], anchor)
        np.testing.assert_array_equal(control[1:], raw[1:]-self.reference)
        with self.assertRaises(ValueError):
            apply_frozen_fits(raw, self.reference, anchor, [fit] * 14)
        with self.assertRaises(ValueError):
            apply_frozen_fits(raw[:15], self.reference, anchor, [fit] * 15)
        with self.assertRaisesRegex(ValueError, "configuration"):
            apply_frozen_fits(raw, self.reference, anchor, [fit]*14 + [self.fit(mode="diagonal")])


if __name__ == "__main__":
    unittest.main()
