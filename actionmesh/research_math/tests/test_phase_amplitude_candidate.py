"""Authored engineering checks for the C20 solver; Web does not execute them."""
from __future__ import annotations

import unittest

import numpy as np

from research_math.phase_amplitude_candidate import (
    PhaseAmplitudeError,
    _interpolate_vertices,
    build_endpoint_sine_basis,
    solve_phase_amplitude,
)


class PhaseAmplitudeCandidateTest(unittest.TestCase):
    def _trajectory(self):
        times = np.linspace(0.0, 1.0, 16, dtype=np.float64)
        vertices = np.zeros((16, 3, 3), dtype=np.float64)
        vertices[:, 0, 0] = times + 0.2 * times ** 2
        vertices[:, 1, 1] = np.sin(np.pi * times)
        vertices[:, 2, 2] = times ** 2
        return times, vertices

    def test_inactive_problem_recovers_separated_phase_and_amplitude(self):
        times, vertices = self._trajectory()
        phase_basis = build_endpoint_sine_basis(times, rank=2)
        amplitude = np.zeros((2,) + vertices.shape, dtype=np.float64)
        amplitude[0, :, 0, 1] = np.sin(np.pi * times)
        amplitude[1, :, 2, 0] = times * (1.0 - times)
        xi = np.array([0.015, -0.008])
        gamma = np.array([0.06, -0.04])
        derivative = np.gradient(vertices, times, axis=0, edge_order=2)
        target = derivative * (phase_basis @ xi)[:, None, None]
        target += np.tensordot(gamma, amplitude, axes=(0, 0))

        result = solve_phase_amplitude(
            vertices, target, times, phase_basis, amplitude,
            lambda_value=1e-12, min_slope=0.25,
            identifiability_floor=1e-8, max_abs_warp=0.1,
            max_relative_linearization_remainder=0.2,
            max_gamma_noise_amplification=1e6,
            max_relative_stationarity_residual=1e-8,
        )

        np.testing.assert_allclose(result.phase_coefficients, xi, atol=2e-7)
        np.testing.assert_allclose(result.amplitude_coefficients, gamma, atol=2e-7)
        self.assertFalse(result.active_constraints)
        self.assertGreater(result.kappa, 1e-8)
        self.assertLess(result.inactive_gamma_noise_upper_bound, 1e6)
        self.assertEqual(result.conditioning_scope,
                         "unconstrained_zero_ridge_data_subspaces_only")
        self.assertGreaterEqual(result.minimum_warp_slope, 0.25)
        np.testing.assert_array_equal(result.warp_offsets[[0, -1]], np.zeros(2))
        self.assertEqual(result.output_vertices.shape, vertices.shape)

    def test_active_monotonicity_constraint_uses_full_block_solution(self):
        times, vertices = self._trajectory()
        phase_basis = build_endpoint_sine_basis(times, rank=1)
        amplitude = np.zeros((1,) + vertices.shape, dtype=np.float64)
        amplitude[0, :, :, 0] = 1.0
        derivative = np.gradient(vertices, times, axis=0, edge_order=2)
        # This target asks for a phase coefficient beyond the monotonicity bound.
        target = derivative * (phase_basis[:, 0] * 2.0)[:, None, None]

        result = solve_phase_amplitude(
            vertices, target, times, phase_basis, amplitude,
            lambda_value=1e-5, min_slope=0.6,
            identifiability_floor=1e-10, max_abs_warp=0.5,
            max_relative_linearization_remainder=2.0,
            max_gamma_noise_amplification=1e6,
            max_relative_stationarity_residual=1e-8,
        )

        self.assertTrue(result.active_constraints)
        self.assertGreaterEqual(result.minimum_warp_slope, 0.6 - 1e-10)
        self.assertEqual(result.solver_regime, "full_constrained_block")
        self.assertLessEqual(result.max_constraint_violation, 1e-10)
        differences = np.diff(phase_basis[:, 0])
        bounds = -(1.0 - 0.6) * np.diff(times)
        expected_upper = min(bounds[differences < 0.0] / differences[differences < 0.0])
        self.assertAlmostEqual(result.phase_coefficients[0], expected_upper, places=8)
        self.assertLessEqual(result.relative_stationarity_residual, 1e-8)
        self.assertTrue(all(value >= -1e-10 for value in result.active_multipliers))

    def test_overlapping_phase_and_amplitude_directions_fail_closed(self):
        times, vertices = self._trajectory()
        phase_basis = build_endpoint_sine_basis(times, rank=1)
        derivative = np.gradient(vertices, times, axis=0, edge_order=2)
        phase_direction = derivative * phase_basis[:, 0, None, None]
        amplitude = phase_direction[None, ...]
        target = phase_direction.copy()

        with self.assertRaisesRegex(PhaseAmplitudeError, "not identifiable"):
            solve_phase_amplitude(
                vertices, target, times, phase_basis, amplitude,
                lambda_value=1e-4, min_slope=0.2,
                identifiability_floor=1e-4, max_abs_warp=0.2,
                max_relative_linearization_remainder=0.2,
                max_gamma_noise_amplification=1e6,
                max_relative_stationarity_residual=1e-8,
            )

    def test_target_and_bases_must_match_original_identity(self):
        times, vertices = self._trajectory()
        phase_basis = build_endpoint_sine_basis(times, rank=1)
        amplitude = np.zeros((1,) + vertices.shape, dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "target"):
            solve_phase_amplitude(
                vertices, np.zeros((15, 3, 3)), times, phase_basis,
                amplitude, lambda_value=1e-4, min_slope=0.2,
                identifiability_floor=1e-8, max_abs_warp=0.2,
                max_relative_linearization_remainder=0.2,
                max_gamma_noise_amplification=1e6,
                max_relative_stationarity_residual=1e-8,
            )
        with self.assertRaisesRegex(ValueError, "endpoint"):
            bad = phase_basis.copy()
            bad[0, 0] = 1.0
            solve_phase_amplitude(
                vertices, np.zeros_like(vertices), times, bad, amplitude,
                lambda_value=1e-4, min_slope=0.2,
                identifiability_floor=1e-8, max_abs_warp=0.2,
                max_relative_linearization_remainder=0.2,
                max_gamma_noise_amplification=1e6,
                max_relative_stationarity_residual=1e-8,
            )

    def test_reduced_phase_only_and_amplitude_only_blocks_are_supported(self):
        times, vertices = self._trajectory()
        phase = build_endpoint_sine_basis(times, rank=1)
        derivative = np.gradient(vertices, times, axis=0, edge_order=2)
        phase_target = derivative * (0.01 * phase[:, 0])[:, None, None]
        phase_only = solve_phase_amplitude(
            vertices, phase_target, times, phase,
            np.zeros((0,) + vertices.shape), lambda_value=1e-4,
            min_slope=0.2, identifiability_floor=1e-8,
            max_abs_warp=0.05, max_relative_linearization_remainder=0.2,
            max_gamma_noise_amplification=1e6,
            max_relative_stationarity_residual=1e-8,
        )
        self.assertEqual(phase_only.amplitude_coefficients.size, 0)
        self.assertEqual(phase_only.kappa, 1.0)

        amplitude = np.zeros((1,) + vertices.shape)
        amplitude[0, :, 1, 0] = times * (1.0 - times)
        amplitude_target = 0.2 * amplitude[0]
        amplitude_only = solve_phase_amplitude(
            vertices, amplitude_target, times, np.zeros((16, 0)),
            amplitude, lambda_value=1e-12, min_slope=0.2,
            identifiability_floor=1e-8, max_abs_warp=0.05,
            max_relative_linearization_remainder=0.2,
            max_gamma_noise_amplification=1e6,
            max_relative_stationarity_residual=1e-8,
        )
        self.assertEqual(amplitude_only.phase_coefficients.size, 0)
        np.testing.assert_allclose(amplitude_only.amplitude_coefficients, [0.2], atol=1e-7)

    def test_nonlinear_renderer_uses_the_same_knot_derivative_and_remainder_gate(self):
        times, vertices = self._trajectory()
        phase = build_endpoint_sine_basis(times, rank=1)
        derivative = np.gradient(vertices, times, axis=0, edge_order=2)
        target = derivative * (0.01 * phase[:, 0])[:, None, None]
        result = solve_phase_amplitude(
            vertices, target, times, phase, np.zeros((0,) + vertices.shape),
            lambda_value=1e-4, min_slope=0.2,
            identifiability_floor=1e-8, max_abs_warp=0.05,
            max_relative_linearization_remainder=0.2,
            max_gamma_noise_amplification=1e6,
            max_relative_stationarity_residual=1e-8,
        )
        self.assertLess(result.relative_linearization_remainder, 0.2)
        self.assertGreater(result.output_vertices[5, 0, 0], vertices[5, 0, 0])

        derivative = np.gradient(vertices, times, axis=0, edge_order=2)
        offset = phase[:, 0]
        full = _interpolate_vertices(vertices, derivative, times, times + 0.01 * offset)
        half = _interpolate_vertices(vertices, derivative, times, times + 0.005 * offset)
        full_remainder = np.linalg.norm(full - vertices - derivative * (0.01 * offset)[:, None, None])
        half_remainder = np.linalg.norm(half - vertices - derivative * (0.005 * offset)[:, None, None])
        self.assertGreater(full_remainder / half_remainder, 3.0)
        left_full = _interpolate_vertices(vertices, derivative, times, times - 0.01 * offset)
        left_half = _interpolate_vertices(vertices, derivative, times, times - 0.005 * offset)
        left_full_remainder = np.linalg.norm(
            left_full - vertices + derivative * (0.01 * offset)[:, None, None])
        left_half_remainder = np.linalg.norm(
            left_half - vertices + derivative * (0.005 * offset)[:, None, None])
        self.assertGreater(left_full_remainder / left_half_remainder, 3.0)

        with self.assertRaisesRegex(PhaseAmplitudeError, "remainder"):
            solve_phase_amplitude(
                vertices, 20.0 * target, times, phase,
                np.zeros((0,) + vertices.shape), lambda_value=1e-4,
                min_slope=0.01, identifiability_floor=1e-8,
                max_abs_warp=0.9,
                max_relative_linearization_remainder=1e-8,
                max_gamma_noise_amplification=1e6,
                max_relative_stationarity_residual=1e-8,
            )
        with self.assertRaisesRegex(PhaseAmplitudeError, "small-warp"):
            solve_phase_amplitude(
                vertices, target, times, phase, np.zeros((0,) + vertices.shape),
                lambda_value=1e-4, min_slope=0.2,
                identifiability_floor=1e-8, max_abs_warp=1e-6,
                max_relative_linearization_remainder=0.2,
                max_gamma_noise_amplification=1e6,
                max_relative_stationarity_residual=1e-8,
            )

    def test_ill_scaled_amplitude_basis_is_rejected_by_noise_bound(self):
        times, vertices = self._trajectory()
        amplitude = np.zeros((1,) + vertices.shape)
        amplitude[0, :, 0, 0] = 1e-12 * times
        with self.assertRaisesRegex(PhaseAmplitudeError, "noise amplification"):
            solve_phase_amplitude(
                vertices, amplitude[0], times, np.zeros((16, 0)), amplitude,
                lambda_value=1e-4, min_slope=0.2,
                identifiability_floor=1e-8, max_abs_warp=0.1,
                max_relative_linearization_remainder=0.2,
                max_gamma_noise_amplification=1e5,
                max_relative_stationarity_residual=1e-8,
            )

    def test_nonuniform_times_and_weights_keep_the_same_weighted_solution(self):
        times, vertices = self._trajectory()
        times = times ** 1.3
        vertices[:, 0, 0] = times + 0.2 * times ** 2
        vertices[:, 1, 1] = np.sin(np.pi * times)
        vertices[:, 2, 2] = times ** 2
        phase = build_endpoint_sine_basis(times, rank=1)
        derivative = np.gradient(vertices, times, axis=0, edge_order=2)
        target = derivative * (0.005 * phase[:, 0])[:, None, None]
        weights = np.linspace(0.5, 2.0, times.size)[:, None, None]
        result = solve_phase_amplitude(
            vertices, target, times, phase, np.zeros((0,) + vertices.shape),
            weights=weights, lambda_value=1e-4, min_slope=0.2,
            identifiability_floor=1e-8, max_abs_warp=0.05,
            max_relative_linearization_remainder=0.2,
            max_gamma_noise_amplification=1e6,
            max_relative_stationarity_residual=1e-8,
        )
        np.testing.assert_allclose(result.phase_coefficients, [0.005], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
