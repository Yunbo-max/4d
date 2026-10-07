"""B: local action-preserving edit controllability, NumPy numerical prototype.

J maps allowed model controls to a COMMON output observable. A is the equality
contract Jacobian, B the edit-feature Jacobian, and d the desired feature change.
Metrics W and Q must be fixed across comparisons, in declared physical units.

V spans range(J) intersect ker(A), with V.T W V = I. We solve
    min_a ||B V a - d||_Q**2, optionally subject to ||a||_2 <= radius.
The radius is an OUTPUT displacement budget, never a latent-coordinate budget.
This is classical linear algebra/trust-region least squares, not a new method.
Unilateral inequalities are NOT implemented or silently treated as equalities.

All claims are local and subject to numerical rank. Invertible latent changes
preserve exact ranges, but ill-conditioned changes can cross an SVD threshold.
No finite edit, physical feasibility, decoder memory, or 4D efficacy is certified.
Dense Jacobians/metrics here are intended for small, declared control probes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EditResult:
    step: np.ndarray
    coefficients: np.ndarray
    basis: np.ndarray
    residual_squared: float
    normalized_residual: float | None
    singular_values: np.ndarray
    model_rank: int
    preserving_rank: int
    edit_rank: int
    step_norm: float
    radius: float | None
    lagrange_multiplier: float | None
    stationarity_residual: float | None
    complementarity_residual: float | None
    contract_residual: float
    range_residual: float
    iterations: int
    converged: bool
    status: str
    rtol: float
    atol: float
    edit_rank_threshold: float


@dataclass(frozen=True)
class GapResult:
    model: EditResult
    reference: EditResult
    gap: float
    raw_gap: float
    nesting_residual: float


@dataclass(frozen=True)
class AcceptanceResult:
    point: np.ndarray
    step: np.ndarray
    scale: float
    accepted: bool
    evaluations: int
    history: tuple[dict, ...]


def _matrix(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real matrix")
    return result


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return result


def _tolerances(rtol, atol):
    if not np.isfinite(rtol) or not 0 <= rtol < 1:
        raise ValueError("rtol must be finite and in [0,1)")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")


def _norm2(matrix):
    return float(np.linalg.norm(matrix, ord=2)) if matrix.size else 0.0


def _metric(value, size, name):
    metric = np.eye(size) if value is None else _matrix(value, name)
    if metric.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size},{size})")
    if not np.allclose(metric, metric.T, rtol=1e-12, atol=0.):
        raise ValueError(f"{name} must be symmetric positive definite")
    metric = (metric + metric.T) * .5
    try:
        # C.T @ C = metric; C is the whitening map for column vectors.
        whitening = np.linalg.cholesky(metric).T
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be symmetric positive definite") from error
    return metric, whitening


def nullspace(matrix, *, rtol=1e-10, atol=0.):
    """Euclidean orthonormal nullspace; wide/empty matrices are supported."""
    _tolerances(rtol, atol)
    matrix = _matrix(matrix, "matrix")
    if matrix.shape[0] == 0:
        return np.eye(matrix.shape[1])
    _, singular, right = np.linalg.svd(matrix, full_matrices=True)
    cutoff = max(atol, rtol * (float(singular[0]) if singular.size else 0.))
    rank = int(np.count_nonzero(singular > cutoff))
    return right[rank:].T.copy()


def _space(jacobian, contract, output_metric, rtol, atol):
    _tolerances(rtol, atol)
    j = _matrix(jacobian, "jacobian")
    a = _matrix(contract, "contract")
    size = j.shape[0]
    if size == 0 or a.shape[1] != size:
        raise ValueError("contract columns must match a nonempty output space")
    w, whitening = _metric(output_metric, size, "output_metric")
    left, singular, _ = np.linalg.svd(whitening @ j, full_matrices=False)
    cutoff = max(atol, rtol * (float(singular[0]) if singular.size else 0.))
    rank = int(np.count_nonzero(singular > cutoff))
    u_white = left[:, :rank]
    u = np.linalg.solve(whitening, u_white)
    # Use the ambient constraint operator's scale, not just A@U, so roundoff
    # in a truly orthogonal A@U is not misclassified as a nonzero constraint.
    a_white = np.linalg.solve(whitening.T, a.T).T
    constraint_cutoff = max(atol, rtol * _norm2(a_white))
    n = nullspace(a @ u, rtol=0., atol=constraint_cutoff)
    return j, a, w, whitening, u_white, u @ n, rank


def reachable_basis(jacobian, contract, *, output_metric=None, rtol=1e-10, atol=0.):
    """W-orthonormal basis of model-reachable, equality-preserving directions.

    Numerical nullspace tolerance uses the whitened ambient contract scale.
    This API does not infer active sets for contact inequalities.
    """
    return _space(jacobian, contract, output_metric, rtol, atol)[5]


def project_model_step(jacobian, contract, desired_step, *, output_metric=None,
                       rtol=1e-10, atol=0.):
    """Project an output velocity to range(J) intersect ker(A), in metric W."""
    j, _, w, _, _, basis, _ = _space(jacobian, contract, output_metric, rtol, atol)
    desired = _vector(desired_step, j.shape[0], "desired_step")
    return basis @ (basis.T @ w @ desired)


def analyze_edit(jacobian, contract, edit_jacobian, target, *, output_metric=None,
                 edit_metric=None, radius=None, rtol=1e-10, atol=0.,
                 solver_tol=1e-12, max_iterations=200):
    """Solve equality-preserving edit LS, with an optional output trust ball.

    SVD gives the minimum-output-norm unbounded solution. For an active positive
    radius, bisection solves the monotone secular equation for lambda >= 0:
    a(lambda) = R diag(s/(s**2+lambda)) L.T Qsqrt d.
    We retain the feasible bracket endpoint rather than clipping a solution.

    KKT diagnostics use the half-squared objective convention; the reported
    residual is the full squared norm. At radius zero the feasible set is a
    singleton, so a finite-multiplier KKT diagnostic is inapplicable (None).
    Discarded singular directions follow the declared numerical-rank threshold;
    the actual full residual and stationarity are still evaluated and reported.
    """
    if radius is not None and (not np.isfinite(radius) or radius < 0):
        raise ValueError("radius must be None or finite and nonnegative")
    if not np.isfinite(solver_tol) or solver_tol <= 0:
        raise ValueError("solver_tol must be positive and finite")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    j, a, w, whitening, u_white, basis, model_rank = _space(
        jacobian, contract, output_metric, rtol, atol)
    b = _matrix(edit_jacobian, "edit_jacobian")
    if b.shape[1] != j.shape[0] or b.shape[0] == 0:
        raise ValueError("edit_jacobian must map the output to nonempty edit features")
    d = _vector(target, b.shape[0], "target")
    _, q_white = _metric(edit_metric, b.shape[0], "edit_metric")
    c = q_white @ b @ basis
    desired = q_white @ d
    left, singular, right = np.linalg.svd(c, full_matrices=False)
    ambient_edit = np.linalg.solve(whitening.T, (q_white @ b).T).T
    cutoff = max(atol, rtol * _norm2(ambient_edit))
    rank = int(np.count_nonzero(singular > cutoff))
    s = singular[:rank]
    beta = left[:, :rank].T @ desired
    rotations = right[:rank].T
    coefficients = rotations @ (beta / s)
    iterations = 0
    multiplier = 0.
    secular_converged = True
    status = "unbounded" if radius is None else "trust_inactive"
    if radius == 0:
        coefficients = np.zeros(basis.shape[1])
        multiplier = None
        status = "zero_radius_singleton"
    elif radius is not None and np.linalg.norm(coefficients) > radius:
        status = "trust_active"
        numerator = s * beta
        low = 0.
        # This bound makes ||a(high)|| <= radius without a heuristic growth loop.
        high = float(np.linalg.norm(numerator) / radius)
        if not np.isfinite(high) or high <= 0:
            raise FloatingPointError("trust multiplier cannot be bracketed at float64 scale")
        secular_converged = False
        for iterations in range(1, max_iterations + 1):
            middle = low + (high - low) * .5
            trial = numerator / (s * s + middle)
            if np.linalg.norm(trial) > radius:
                low = middle
            else:
                high = middle
            feasible = numerator / (s * s + high)
            if abs(float(np.linalg.norm(feasible)) - radius) <= solver_tol * radius:
                secular_converged = True
                break
            if middle == low == high:
                break
        multiplier = high
        coefficients = rotations @ (numerator / (s * s + high))
    step = basis @ coefficients
    residual = c @ coefficients - desired
    residual_squared = float(residual @ residual)
    target_squared = float(desired @ desired)
    step_norm = float(np.linalg.norm(whitening @ step))
    if multiplier is None:
        stationarity = None
        complementarity = None
        kkt_ok = True
    else:
        gradient = c.T @ residual
        stationarity = float(np.linalg.norm(gradient + multiplier * coefficients))
        gradient_scale = max(1., float(np.linalg.norm(c.T @ desired)),
                             multiplier * float(np.linalg.norm(coefficients)))
        complementarity = 0. if radius is None else abs(
            multiplier * (float(coefficients @ coefficients) - radius * radius))
        kkt_ok = stationarity <= max(100 * solver_tol, 10 * rtol) * gradient_scale
    primal_ok = radius is None or step_norm <= radius + solver_tol * max(1., radius)
    white_step = whitening @ step
    range_residual = float(np.linalg.norm(white_step - u_white @ (u_white.T @ white_step)))
    contract_residual = float(np.linalg.norm(a @ step))
    if not all(np.isfinite(value).all() for value in (step, coefficients, residual, singular)):
        raise FloatingPointError("nonfinite edit solve; no usable step returned")
    return EditResult(
        step=step, coefficients=coefficients, basis=basis,
        residual_squared=residual_squared,
        normalized_residual=None if target_squared == 0 else residual_squared / target_squared,
        singular_values=singular, model_rank=model_rank,
        preserving_rank=basis.shape[1], edit_rank=rank, step_norm=step_norm,
        radius=None if radius is None else float(radius), lagrange_multiplier=multiplier,
        stationarity_residual=stationarity, complementarity_residual=complementarity,
        contract_residual=contract_residual, range_residual=range_residual,
        iterations=iterations, converged=bool(secular_converged and kkt_ok and primal_ok),
        status=status if secular_converged else "iteration_limit", rtol=rtol, atol=atol,
        edit_rank_threshold=cutoff)


def nested_gap(model_jacobian, reference_jacobian, contract, edit_jacobian, target,
               *, output_metric=None, edit_metric=None, radius=None,
               rtol=1e-10, atol=0., solver_tol=1e-12, max_iterations=200):
    """Nonnegative residual gap ONLY after checking nested preserving spaces.

    Both solves share observable coordinates, contract, W, Q, d and radius.
    A nonnested rig/ARAP reference is a valid comparator but is refused here.
    This is a linearized relaxation gap, not proof of finite physical feasibility.
    """
    model_v = reachable_basis(model_jacobian, contract, output_metric=output_metric,
                              rtol=rtol, atol=atol)
    reference_v = reachable_basis(reference_jacobian, contract, output_metric=output_metric,
                                  rtol=rtol, atol=atol)
    if model_v.shape[0] != reference_v.shape[0]:
        raise ValueError("nested spaces must use the same output coordinates")
    w, whitening = _metric(output_metric, model_v.shape[0], "output_metric")
    remainder = model_v - reference_v @ (reference_v.T @ w @ model_v)
    nesting_error = _norm2(whitening @ remainder)
    if nesting_error > max(10 * rtol, 100 * np.finfo(float).eps):
        raise ValueError(f"reference does not contain the preserving model space: {nesting_error:g}")
    kwargs = dict(output_metric=output_metric, edit_metric=edit_metric, radius=radius,
                  rtol=rtol, atol=atol, solver_tol=solver_tol, max_iterations=max_iterations)
    model = analyze_edit(model_jacobian, contract, edit_jacobian, target, **kwargs)
    reference = analyze_edit(reference_jacobian, contract, edit_jacobian, target, **kwargs)
    if not model.converged or not reference.converged:
        raise RuntimeError("a nonnegative gap requires both numerical solves to converge")
    raw = model.residual_squared - reference.residual_squared
    tolerance = max(100 * solver_tol, 10 * rtol) * max(
        1., model.residual_squared, reference.residual_squared)
    if raw < -tolerance:
        raise RuntimeError("negative nested gap beyond numerical tolerance; inspect rank/solve")
    return GapResult(model, reference, max(0., raw), raw, nesting_error)


def backtrack_accept(point, step, accept: Callable[[np.ndarray], bool], *,
                      retract=None, shrink=.5, max_backtracks=10):
    """Bounded finite-step acceptance using a caller-supplied full contract.

    `accept(candidate)` must evaluate ORIGINAL nonlinear constraints, edit
    progress, and cumulative budgets as appropriate; there is no hidden merit
    function. Optional `retract(point, scaled_step)` can decode admissible model
    controls. Adding an observable-space tangent alone does not certify decoder
    reachability. This helper does not replace that model-specific retraction.
    Callback errors propagate; all-rejected returns the unchanged point and an
    explicit false status. No cross-step/continuous-time guarantees are implied.
    """
    x = np.asarray(point, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("point must be a vector")
    x = _vector(x, x.size, "point")
    velocity = _vector(step, x.size, "step")
    if not np.isfinite(shrink) or not 0 < shrink < 1:
        raise ValueError("shrink must be in (0,1)")
    if not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer")
    history = []
    for attempt in range(max_backtracks + 1):
        scale = float(shrink ** attempt)
        candidate = (x + scale * velocity if retract is None else
                     np.asarray(retract(x.copy(), scale * velocity), dtype=np.float64))
        if candidate.shape != x.shape:
            raise ValueError("retraction must preserve the point shape")
        finite = bool(np.isfinite(candidate).all())
        verdict = accept(candidate.copy()) if finite else False
        if not isinstance(verdict, (bool, np.bool_)):
            raise ValueError("accept must return a boolean, not a merit value")
        history.append({"scale": scale, "finite": finite, "accepted": bool(verdict)})
        if verdict:
            return AcceptanceResult(candidate.copy(), candidate - x, scale, True,
                                    len(history), tuple(history))
    return AcceptanceResult(x.copy(), np.zeros_like(x), 0., False, len(history), tuple(history))


def constructed_demo():
    """Two analytic falsifiers; output-space baseline is deliberately allowed to win."""
    nested = nested_gap(np.array([[1.], [0.], [0.]]), np.eye(3),
                        np.array([[0., 0., 1.]]), np.eye(3), [0., 1., 0.], radius=1.)
    args = (np.ones((1, 1)), np.zeros((0, 1)), np.array([[.001]]), [1.])
    unlimited = analyze_edit(*args)
    limited = analyze_edit(*args, radius=1.)
    return {
        "evidence_scope": "constructed numerical controls; not 4D efficacy",
        "missing_direction": {
            "model_residual_squared": nested.model.residual_squared,
            "output_residual_squared": nested.reference.residual_squared,
            "nested_gap": nested.gap,
            "explanation": "model exposes e1 only; action permits e1 and e2; requested edit is e2",
        },
        "trust_budget": {
            "unbounded_residual_squared": unlimited.residual_squared,
            "bounded_residual_squared": limited.residual_squared,
            "unbounded_output_step_norm": unlimited.step_norm,
            "bounded_output_step_norm": limited.step_norm,
            "explanation": "gain .001 needs displacement 1000 for unit edit; budget is 1",
        },
    }


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run constructed numerical controls")
    args = parser.parse_args()
    if not args.demo:
        parser.error("use --demo; this module does not run a 4D generator")
    print(json.dumps(constructed_demo(), indent=2, allow_nan=False))
