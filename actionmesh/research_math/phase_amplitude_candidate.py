"""C20 phase/amplitude decomposition core for an explicitly qualified target.

The caller must supply the legal model/input-derived motion target and frozen
amplitude basis.  This module neither constructs that scientific input nor
loads ground truth, a scorer, model state, or confirmation outcomes.  It solves
the reviewed local problem, rejects phase/amplitude non-identifiability, enforces
endpoint-preserving monotone time, and renders at the original timestamps.

Web-authored source is ``generated_unexecuted`` until Local acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class PhaseAmplitudeError(RuntimeError):
    """A reviewed C20 precondition or solver certificate failed."""


@dataclass(frozen=True)
class PhaseAmplitudeResult:
    phase_coefficients: np.ndarray
    amplitude_coefficients: np.ndarray
    warp_offsets: np.ndarray
    amplitude_delta: np.ndarray
    linearized_repair: np.ndarray
    output_vertices: np.ndarray
    kappa: float
    cross_subspace_singular_value: float
    minimum_warp_slope: float
    max_constraint_violation: float
    active_constraints: tuple[int, ...]
    active_multipliers: tuple[float, ...]
    solver_regime: str
    weighted_linear_residual: float
    weighted_nonlinear_residual: float
    weighted_linearization_remainder: float
    relative_linearization_remainder: float
    inactive_eta_noise_upper_bound: float | None
    inactive_gamma_noise_upper_bound: float | None
    amplitude_minimum_singular_value: float | None
    conditioning_scope: str
    relative_stationarity_residual: float


def _positive(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _finite_float(name: str, value, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != ndim or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite floating rank-{ndim} array")
    return array.astype(np.float64, copy=False)


def build_endpoint_sine_basis(times, rank: int) -> np.ndarray:
    """Return a deterministic endpoint-zero phase basis on the supplied times."""
    t = _finite_float("times", times, 1)
    if t.size < 4 or not np.all(np.diff(t) > 0.0):
        raise ValueError("times must contain at least four strictly increasing values")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= t.size - 2:
        raise ValueError("rank must be an integer in [1,T-2]")
    normalized = (t - t[0]) / (t[-1] - t[0])
    basis = np.stack([
        np.sin(np.pi * (index + 1) * normalized) for index in range(rank)
    ], axis=1)
    basis[[0, -1], :] = 0.0
    return basis


def _orthogonal_basis(matrix: np.ndarray, tolerance: float) -> tuple[np.ndarray, int]:
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64), 0
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular.size == 0 or singular[0] == 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64), 0
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    return left[:, :rank], rank


def _identifiability(weighted_phase: np.ndarray, weighted_amplitude: np.ndarray,
                     tolerance: float) -> tuple[float, float, int, int, float]:
    phase_q, phase_rank = _orthogonal_basis(weighted_phase, tolerance)
    amplitude_q, amplitude_rank = _orthogonal_basis(weighted_amplitude, tolerance)
    if weighted_amplitude.shape[1] == 0:
        return 1.0, 0.0, phase_rank, 0, None
    if amplitude_rank != weighted_amplitude.shape[1]:
        raise PhaseAmplitudeError("amplitude basis is not full column rank")
    if phase_rank == 0:
        cross = 0.0
    else:
        cross_values = np.linalg.svd(phase_q.T @ amplitude_q, compute_uv=False)
        cross = float(cross_values[0]) if cross_values.size else 0.0
    kappa = max(0.0, min(1.0, 1.0 - cross * cross))
    singular = np.linalg.svd(weighted_amplitude, compute_uv=False)
    return kappa, cross, phase_rank, amplitude_rank, float(singular[-1])


def _solve_equality_ls(design: np.ndarray, target: np.ndarray,
                       constraints: np.ndarray, bounds: np.ndarray,
                       active: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    dimension = design.shape[1]
    if not active:
        solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        gradient = design.T @ (design @ solution - target)
        scale = max(1.0, float(np.linalg.norm(design) * np.linalg.norm(target)))
        residual = float(np.linalg.norm(gradient) / scale)
        return solution, np.zeros(0, dtype=np.float64), residual
    selected = constraints[np.asarray(active, dtype=np.int64)]
    selected_bounds = bounds[np.asarray(active, dtype=np.int64)]
    particular, _, _, _ = np.linalg.lstsq(selected, selected_bounds, rcond=None)
    if not np.allclose(selected @ particular, selected_bounds, rtol=0.0, atol=1e-10):
        raise PhaseAmplitudeError("active monotonicity face is inconsistent")
    _, singular, right_t = np.linalg.svd(selected, full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-12 * singular[0])) if singular.size else 0
    null_basis = right_t[rank:].T
    if null_basis.shape[1]:
        reduced, _, _, _ = np.linalg.lstsq(
            design @ null_basis, target - design @ particular, rcond=None)
        solution = particular + null_basis @ reduced
    else:
        solution = particular
    objective_gradient = design.T @ (design @ solution - target)
    multipliers, _, _, _ = np.linalg.lstsq(selected.T, objective_gradient, rcond=None)
    stationarity = objective_gradient - selected.T @ multipliers
    scale = max(1.0, float(np.linalg.norm(design) *
                           (np.linalg.norm(design @ solution) + np.linalg.norm(target))))
    residual = max(float(np.linalg.norm(stationarity) / scale),
                   float(np.linalg.norm(selected @ solution - selected_bounds)))
    return solution, multipliers, residual


def _solve_monotone_qp(design: np.ndarray, target: np.ndarray,
                       constraints: np.ndarray, bounds: np.ndarray,
                       tolerance: float, max_iterations: int) -> tuple[
                           np.ndarray, tuple[int, ...], tuple[float, ...], float]:
    """Primal active-set solve for C z >= b with retained KKT checks."""
    solution, multipliers, residual = _solve_equality_ls(
        design, target, constraints, bounds, [])
    active: list[int] = []
    for _ in range(max_iterations):
        slack = constraints @ solution - bounds
        inactive_violations = [
            index for index in range(bounds.size)
            if index not in active and slack[index] < -tolerance
        ]
        if inactive_violations:
            chosen = min(inactive_violations, key=lambda index: (slack[index], index))
            active.append(chosen)
            solution, multipliers, residual = _solve_equality_ls(
                design, target, constraints, bounds, active)
            continue
        negative = [index for index, value in enumerate(multipliers) if value < -tolerance]
        if negative:
            remove_at = min(negative, key=lambda index: (multipliers[index], index))
            active.pop(remove_at)
            solution, multipliers, residual = _solve_equality_ls(
                design, target, constraints, bounds, active)
            continue
        stationarity = design.T @ (design @ solution - target)
        if active:
            stationarity -= constraints[np.asarray(active)].T @ multipliers
        scale = max(1.0, float(np.linalg.norm(design) *
                               (np.linalg.norm(design @ solution) + np.linalg.norm(target))))
        certificate = max(
            float(np.max(np.maximum(bounds - constraints @ solution, 0.0))),
            float(np.linalg.norm(stationarity) / scale), residual,
        )
        if certificate > max(1e-8, 100.0 * tolerance):
            raise PhaseAmplitudeError("constrained block KKT certificate failed")
        ordered = sorted(zip(active, multipliers), key=lambda item: item[0])
        return (solution, tuple(item[0] for item in ordered),
                tuple(float(item[1]) for item in ordered), certificate)
    raise PhaseAmplitudeError("constrained block active-set iteration limit reached")


def _interpolate_vertices(vertices: np.ndarray, derivatives: np.ndarray,
                          times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Piecewise cubic Hermite rendering with the exact fitted knot derivative."""
    if (query_times[0] < times[0] - 1e-12
            or query_times[-1] > times[-1] + 1e-12
            or not np.all(np.diff(query_times) > 0.0)):
        raise PhaseAmplitudeError("warped times are outside the original monotone interval")
    output = np.empty_like(vertices)
    for frame, query in enumerate(query_times):
        right = int(np.searchsorted(times, query, side="right"))
        if right == 0:
            output[frame] = vertices[0]
        elif right >= times.size:
            output[frame] = vertices[-1]
        else:
            left = right - 1
            width = float(times[right] - times[left])
            alpha = float((query - times[left]) / width)
            alpha2 = alpha * alpha
            alpha3 = alpha2 * alpha
            output[frame] = (
                (2.0 * alpha3 - 3.0 * alpha2 + 1.0) * vertices[left]
                + (alpha3 - 2.0 * alpha2 + alpha) * width * derivatives[left]
                + (-2.0 * alpha3 + 3.0 * alpha2) * vertices[right]
                + (alpha3 - alpha2) * width * derivatives[right]
            )
    return output


def solve_phase_amplitude(vertices, target, times, phase_basis, amplitude_basis,
                          *, weights=None, lambda_value: float,
                          min_slope: float, identifiability_floor: float,
                          max_abs_warp: float,
                          max_relative_linearization_remainder: float,
                          max_gamma_noise_amplification: float,
                          max_relative_stationarity_residual: float,
                          rank_tolerance: float = 1e-10,
                          constraint_tolerance: float = 1e-10,
                          max_active_iterations: int = 512) -> PhaseAmplitudeResult:
    """Solve the reviewed C20 local problem and render at original time IDs.

    ``target`` is a desired additive motion correction with the same material
    vertex identity as ``vertices``.  This function intentionally does not infer
    or validate the target's scientific provenance; a receipt-bound native
    adapter must do that before this core can form a complete C20 method.
    """
    trajectory = _finite_float("vertices", vertices, 3)
    desired = _finite_float("target", target, 3)
    t = _finite_float("times", times, 1)
    phase = _finite_float("phase_basis", phase_basis, 2)
    amplitude = _finite_float("amplitude_basis", amplitude_basis, 4)
    if trajectory.shape[0] < 4 or trajectory.shape[2] != 3:
        raise ValueError("vertices must have shape [T,V,3] with T>=4")
    if desired.shape != trajectory.shape:
        raise ValueError("target must match the complete original vertex identity")
    if t.shape != (trajectory.shape[0],) or not np.all(np.diff(t) > 0.0):
        raise ValueError("times must match T and be strictly increasing")
    if (phase.shape[0] != trajectory.shape[0]
            or not np.array_equal(phase[[0, -1]], np.zeros((2, phase.shape[1])))):
        raise ValueError("phase basis must match T and be exactly endpoint preserving")
    if amplitude.shape[1:] != trajectory.shape or phase.shape[1] + amplitude.shape[0] < 1:
        raise ValueError("amplitude basis must have shape [Q,T,V,3]")
    ridge = _positive("lambda_value", lambda_value)
    slope_floor = _positive("min_slope", min_slope)
    if slope_floor >= 1.0:
        raise ValueError("min_slope must be less than one")
    kappa_floor = _positive("identifiability_floor", identifiability_floor)
    if kappa_floor > 1.0:
        raise ValueError("identifiability_floor must be <=1")
    warp_limit = _positive("max_abs_warp", max_abs_warp)
    remainder_limit = _positive(
        "max_relative_linearization_remainder", max_relative_linearization_remainder)
    gamma_noise_limit = _positive(
        "max_gamma_noise_amplification", max_gamma_noise_amplification)
    stationarity_limit = _positive(
        "max_relative_stationarity_residual",
        max_relative_stationarity_residual)
    rank_tolerance = _positive("rank_tolerance", rank_tolerance)
    constraint_tolerance = _positive("constraint_tolerance", constraint_tolerance)
    if isinstance(max_active_iterations, bool) or not isinstance(max_active_iterations, int) or max_active_iterations < 1:
        raise ValueError("max_active_iterations must be a positive integer")

    if weights is None:
        weight = np.ones_like(trajectory)
    else:
        raw_weight = np.asarray(weights)
        try:
            weight = np.broadcast_to(raw_weight, trajectory.shape).astype(np.float64, copy=False)
        except ValueError as error:
            raise ValueError("weights must broadcast to target shape") from error
        if not np.isfinite(weight).all() or np.any(weight <= 0.0):
            raise ValueError("weights must be finite and strictly positive")

    derivative = np.gradient(trajectory, t, axis=0, edge_order=2)
    if phase.shape[1]:
        phase_columns = np.stack([
            derivative * phase[:, index, None, None] for index in range(phase.shape[1])
        ], axis=-1).reshape(-1, phase.shape[1])
    else:
        phase_columns = np.zeros((trajectory.size, 0), dtype=np.float64)
    if amplitude.shape[0]:
        amplitude_columns = np.moveaxis(amplitude, 0, -1).reshape(
            trajectory.size, amplitude.shape[0])
    else:
        amplitude_columns = np.zeros((trajectory.size, 0), dtype=np.float64)
    target_vector = desired.reshape(-1)
    sqrt_weight = np.sqrt(weight.reshape(-1))
    weighted_phase = phase_columns * sqrt_weight[:, None]
    weighted_amplitude = amplitude_columns * sqrt_weight[:, None]
    weighted_target = target_vector * sqrt_weight

    kappa, cross, phase_rank, _, amplitude_minimum_singular = _identifiability(
        weighted_phase, weighted_amplitude, rank_tolerance)
    if phase_rank != phase.shape[1]:
        raise PhaseAmplitudeError("phase basis has no full-rank trajectory action")
    # Conservative fail-closed rule: active inequalities can sometimes remove an
    # alias, but this partial core does not claim cone-level identifiability.
    if weighted_amplitude.shape[1] and weighted_phase.shape[1] and kappa < kappa_floor:
        raise PhaseAmplitudeError(
            f"phase/amplitude decomposition is not identifiable: kappa={kappa:.17g}")

    inactive_eta_bound = 1.0 / math.sqrt(kappa) if weighted_amplitude.shape[1] else None
    inactive_gamma_bound = (
        inactive_eta_bound / amplitude_minimum_singular
        if weighted_amplitude.shape[1] else None)
    if inactive_gamma_bound is not None and inactive_gamma_bound > gamma_noise_limit:
        raise PhaseAmplitudeError("amplitude coefficient noise amplification exceeds frozen limit")

    design = np.concatenate((weighted_phase, weighted_amplitude), axis=1)
    ridge_rows = np.zeros((amplitude.shape[0], design.shape[1]), dtype=np.float64)
    if amplitude.shape[0]:
        ridge_rows[:, phase.shape[1]:] = math.sqrt(ridge) * np.eye(amplitude.shape[0])
    augmented_design = np.concatenate((design, ridge_rows), axis=0)
    augmented_target = np.concatenate((weighted_target, np.zeros(amplitude.shape[0])))
    dt = np.diff(t)
    difference = np.diff(phase, axis=0)
    constraints = np.zeros((dt.size, design.shape[1]), dtype=np.float64)
    constraints[:, :phase.shape[1]] = difference
    bounds = -(1.0 - slope_floor) * dt

    unconstrained, _, unconstrained_residual = _solve_equality_ls(
        augmented_design, augmented_target, constraints, bounds, [])
    unconstrained_slack = constraints @ unconstrained - bounds
    if np.all(unconstrained_slack >= -constraint_tolerance):
        coefficients = unconstrained
        active: tuple[int, ...] = ()
        active_multipliers: tuple[float, ...] = ()
        normal_residual = unconstrained_residual
        regime = "profiled_unconstrained_equivalent"
    else:
        coefficients, active, active_multipliers, normal_residual = _solve_monotone_qp(
            augmented_design, augmented_target, constraints, bounds,
            constraint_tolerance, max_active_iterations)
        regime = "full_constrained_block"
    if normal_residual > stationarity_limit:
        raise PhaseAmplitudeError(
            "relative stationarity residual exceeds frozen limit")

    xi = coefficients[:phase.shape[1]]
    gamma = coefficients[phase.shape[1]:]
    warp_offsets = phase @ xi
    if float(np.max(np.abs(warp_offsets), initial=0.0)) > warp_limit:
        raise PhaseAmplitudeError("phase warp exceeds frozen small-warp limit")
    warped_times = t + warp_offsets
    interval_slopes = np.diff(warped_times) / dt
    minimum_warp_slope = float(np.min(interval_slopes))
    violation = float(np.max(np.maximum(bounds - constraints @ coefficients, 0.0)))
    if (not np.array_equal(warp_offsets[[0, -1]], np.zeros(2))
            or minimum_warp_slope < slope_floor - 10.0 * constraint_tolerance
            or violation > 10.0 * constraint_tolerance):
        raise PhaseAmplitudeError("monotone phase certificate failed")

    phase_linear = (phase_columns @ xi).reshape(trajectory.shape)
    amplitude_delta = (amplitude_columns @ gamma).reshape(trajectory.shape)
    linearized_repair = phase_linear + amplitude_delta
    warped = _interpolate_vertices(trajectory, derivative, t, warped_times)
    output = warped + amplitude_delta
    nonlinear_repair = output - trajectory
    weighted_linear_residual = float(np.linalg.norm(
        sqrt_weight * (target_vector - linearized_repair.reshape(-1))))
    weighted_nonlinear_residual = float(np.linalg.norm(
        sqrt_weight * (target_vector - nonlinear_repair.reshape(-1))))
    weighted_remainder = float(np.linalg.norm(
        sqrt_weight * (nonlinear_repair - linearized_repair).reshape(-1)))
    weighted_phase_norm = float(np.linalg.norm(sqrt_weight * phase_linear.reshape(-1)))
    relative_remainder = weighted_remainder / max(weighted_phase_norm, np.finfo(float).tiny)
    if relative_remainder > remainder_limit:
        raise PhaseAmplitudeError("observed linearization remainder exceeds frozen limit")

    return PhaseAmplitudeResult(
        phase_coefficients=xi,
        amplitude_coefficients=gamma,
        warp_offsets=warp_offsets,
        amplitude_delta=amplitude_delta,
        linearized_repair=linearized_repair,
        output_vertices=output,
        kappa=kappa,
        cross_subspace_singular_value=cross,
        minimum_warp_slope=minimum_warp_slope,
        max_constraint_violation=violation,
        active_constraints=active,
        active_multipliers=active_multipliers,
        solver_regime=regime,
        weighted_linear_residual=weighted_linear_residual,
        weighted_nonlinear_residual=weighted_nonlinear_residual,
        weighted_linearization_remainder=weighted_remainder,
        relative_linearization_remainder=relative_remainder,
        inactive_eta_noise_upper_bound=inactive_eta_bound,
        inactive_gamma_noise_upper_bound=inactive_gamma_bound,
        amplitude_minimum_singular_value=amplitude_minimum_singular,
        conditioning_scope="unconstrained_zero_ridge_data_subspaces_only",
        relative_stationarity_residual=normal_residual,
    )
