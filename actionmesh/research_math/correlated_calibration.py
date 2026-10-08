"""C03 fixed-design affine calibration, source authored / Local unexecuted.

These are development-surrogate solvers, not native metric optimizers. The
caller must qualify correspondence, coordinate policy, family separation and
development provenance BEFORE supplying labels. Inference accepts only frozen
coefficients and observable same-context residuals; it never accepts GT.
No clipping, correspondence estimation, ICP or model execution is hidden here.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np


SCHEMA = "c03-affine-fit-v1"
MODES = ("full", "diagonal", "intercept")
LOSSES = ("squared", "smoothed_unsquared")


class CalibrationFailure(ValueError):
    """A rejected solve; diagnostics survive but no fit is returned."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _array(value, name, shape=None):
    value = np.asarray(value)
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name}: expected real numeric values")
    value = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: values must be finite")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {value.shape}")
    return value


def _positive(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name}: expected a finite numeric scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name}: expected {'nonnegative' if allow_zero else 'positive'} finite value")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name}: expected integer >= {minimum}")
    return int(value)


def _inputs(reference_residual, labeled_error, weights):
    reference = _array(reference_residual, "reference_residual")
    if reference.ndim != 2 or reference.shape[1] != 3 or not len(reference):
        raise ValueError("reference_residual: expected nonempty [N,3]")
    error = _array(labeled_error, "labeled_error", reference.shape)
    weight = _array(weights, "weights", (len(reference),))
    if (weight < 0).any() or not (weight > 0).any():
        raise ValueError("weights: nonnegative with positive total mass required")
    if not np.isfinite(weight.sum()):
        raise ValueError("weights: total mass overflow")
    return reference, error, weight


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _data_digest(reference, error, weights):
    digest = hashlib.sha256()
    for array in (reference, error, weights):
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def _loss_gradient(reference, error, weight, intercept, matrix, loss, ridge, tau, mode):
    residual = intercept + reference @ matrix.T - error
    square = np.einsum("ij,ij->i", residual, residual)
    if loss == "squared":
        radial_weight = weight
        data_objective = .5 * np.dot(weight, square)
    else:
        radial = np.sqrt(square + tau * tau)
        radial_weight = weight / radial
        data_objective = np.dot(weight, radial)
    weighted = radial_weight[:, None] * residual
    intercept_gradient = weighted.sum(axis=0)
    matrix_gradient = weighted.T @ reference + ridge * matrix
    if mode == "diagonal":
        matrix_gradient = np.diag(np.diag(matrix_gradient))
    elif mode == "intercept":
        matrix_gradient = np.zeros((3, 3))
    objective = float(data_objective + .5 * ridge * np.sum(matrix * matrix))
    gradient_inf = float(max(np.max(np.abs(intercept_gradient)), np.max(np.abs(matrix_gradient))))
    if not np.isfinite(objective) or not np.isfinite(gradient_inf):
        raise FloatingPointError("objective or exact gradient overflow")
    return objective, gradient_inf, radial_weight


def _weighted_solve(reference, error, q, mode, ridge):
    """Minimize .5 sum q ||E-b-Br||^2 + .5 ridge ||B||^2.

    In the diagonal restriction, the surrogate separates by output coordinate.
    For unsquared loss q remains the shared 3D residual-norm weight, never three
    independent scalar robust losses. The intercept is never regularized.
    """
    matrix = np.zeros((3, 3), dtype=np.float64)
    if mode == "intercept":
        intercept = (q[:, None] * error).sum(axis=0) / q.sum()
        condition = 1.0
    elif mode == "full":
        design = np.column_stack((np.ones(len(reference)), reference))
        hessian = design.T @ (q[:, None] * design) + np.diag([0., ridge, ridge, ridge])
        rhs = design.T @ (q[:, None] * error)
        if not np.isfinite(hessian).all() or not np.isfinite(rhs).all():
            raise FloatingPointError("normal equation overflow")
        coefficients = np.linalg.solve(hessian, rhs)
        intercept, matrix = coefficients[0], coefficients[1:].T
        condition = float(np.linalg.cond(hessian))
    else:
        intercept = np.empty(3)
        condition = 1.0
        for axis in range(3):
            design = np.column_stack((np.ones(len(reference)), reference[:, axis]))
            hessian = design.T @ (q[:, None] * design) + np.diag([0., ridge])
            rhs = design.T @ (q * error[:, axis])
            if not np.isfinite(hessian).all() or not np.isfinite(rhs).all():
                raise FloatingPointError("normal equation overflow")
            coefficients = np.linalg.solve(hessian, rhs)
            intercept[axis], matrix[axis, axis] = coefficients
            condition = max(condition, float(np.linalg.cond(hessian)))
    if not np.isfinite(intercept).all() or not np.isfinite(matrix).all() or not np.isfinite(condition):
        raise FloatingPointError("nonfinite affine solve")
    return intercept, matrix, condition


def fit_affine(reference_residual, labeled_error, weights, *, mode, loss,
               ridge, tau, max_iterations, gradient_tolerance, objective_tolerance):
    """Fit a fixed, explicitly weighted development objective.

    Squared uses .5 sum w||E-b-Br||^2 + .5 ridge||B||^2. Unsquared
    uses sum w sqrt(||E-b-Br||^2+tau^2) + .5 ridge||B||^2. Weights
    are NOT normalized, so their declared scale relative to ridge is retained.
    Only the exact free-parameter gradient infinity norm declares convergence;
    small coefficient steps or objective changes never do. Objective increases
    beyond objective_tolerance*(1+abs(previous)) reject the solve. Both controls
    are prospective inputs, not adjusted on failed attempts. A fit digest detects
    accidental changes but is not provenance authentication or admission.
    """
    reference, error, weight = _inputs(reference_residual, labeled_error, weights)
    if mode not in MODES or loss not in LOSSES:
        raise ValueError("unknown C03 mode/loss")
    ridge, tau = _positive(ridge, "ridge"), _positive(tau, "tau")
    maximum = _integer(max_iterations, "max_iterations")
    tolerance = _positive(gradient_tolerance, "gradient_tolerance")
    objective_tolerance = _positive(objective_tolerance, "objective_tolerance", allow_zero=True)
    history = []
    diagnostics = {"status": "running", "converged": False, "history": history,
                   "iterations": 0, "gradient_tolerance": tolerance,
                   "objective_tolerance": objective_tolerance,
                   "max_iterations": maximum, "sample_count": len(reference),
                   "positive_weight_count": int(np.count_nonzero(weight)),
                   "weight_sum": float(weight.sum()), "stopping_rule": "exact_free_gradient_inf"}
    intercept, matrix = np.zeros(3), np.zeros((3, 3))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            objective, gradient, q = _loss_gradient(reference, error, weight, intercept, matrix, loss, ridge, tau, mode)
            history.append({"iteration": 0, "objective": objective, "gradient_inf": gradient})
            for iteration in range(1, maximum + 1):
                if gradient <= tolerance:
                    break
                new_intercept, new_matrix, condition = _weighted_solve(reference, error, q, mode, ridge)
                surrogate_before, _, _ = _loss_gradient(reference, error, q,
                    intercept, matrix, "squared", ridge, tau, mode)
                surrogate_after, surrogate_gradient, _ = _loss_gradient(reference, error, q,
                    new_intercept, new_matrix, "squared", ridge, tau, mode)
                new_objective, new_gradient, new_q = _loss_gradient(reference, error, weight, new_intercept, new_matrix, loss, ridge, tau, mode)
                history.append({"iteration": iteration, "objective": new_objective,
                                "gradient_inf": new_gradient, "normal_condition": condition,
                                "surrogate_before": surrogate_before,
                                "surrogate_after": surrogate_after,
                                "surrogate_gradient_inf": surrogate_gradient})
                diagnostics["iterations"] = iteration
                if surrogate_after > surrogate_before + objective_tolerance * (1. + abs(surrogate_before)):
                    diagnostics["status"] = "rejected_surrogate_increase"
                    raise CalibrationFailure("C03 quadratic surrogate increased beyond frozen tolerance", diagnostics)
                if new_objective > objective + objective_tolerance * (1. + abs(objective)):
                    diagnostics["status"] = "rejected_objective_increase"
                    raise CalibrationFailure("C03 objective increased beyond the frozen tolerance", diagnostics)
                intercept, matrix = new_intercept, new_matrix
                objective, gradient, q = new_objective, new_gradient, new_q
                if loss == "squared":
                    break  # An exact quadratic solve needs no numerical re-solving.
            if gradient > tolerance:
                diagnostics["status"] = "rejected_nonconvergence"
                raise CalibrationFailure("C03 exact gradient criterion not satisfied", diagnostics)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        diagnostics["status"] = "rejected_numerical_failure"
        diagnostics["reason"] = str(exc)
        raise CalibrationFailure("C03 arithmetic or linear solve failed", diagnostics) from exc
    diagnostics["status"], diagnostics["converged"] = "converged", True
    result = {"schema": SCHEMA, "mode": mode, "loss": loss, "ridge": ridge, "tau": tau,
              "intercept": intercept.tolist(), "matrix": matrix.tolist(),
              "data_sha256": _data_digest(reference, error, weight), "diagnostics": diagnostics}
    result["fit_sha256"] = _digest(result)
    return result


def validate_fit(fit):
    """Validate serialized structure/constraints. Use verify_fit for data replay."""
    if not isinstance(fit, Mapping):
        raise ValueError("fit must be a mapping")
    try:
        value = json.loads(json.dumps(dict(fit), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("fit must be finite JSON") from exc
    required = {"schema", "mode", "loss", "ridge", "tau", "intercept", "matrix", "data_sha256", "diagnostics", "fit_sha256"}
    if set(value) != required or value["schema"] != SCHEMA or value["mode"] not in MODES or value["loss"] not in LOSSES:
        raise ValueError("invalid frozen fit schema")
    for key in ("fit_sha256", "data_sha256"):
        if not isinstance(value[key], str) or len(value[key]) != 64 or any(c not in "0123456789abcdef" for c in value[key]):
            raise ValueError(f"invalid {key}")
    if _digest({k: v for k, v in value.items() if k != "fit_sha256"}) != value["fit_sha256"]:
        raise ValueError("frozen fit digest mismatch")
    _positive(value["ridge"], "ridge")
    _positive(value["tau"], "tau")
    _array(value["intercept"], "intercept", (3,))
    matrix = _array(value["matrix"], "matrix", (3, 3))
    if value["mode"] == "intercept" and np.any(matrix != 0):
        raise ValueError("intercept-only fit has a matrix")
    if value["mode"] == "diagonal" and np.any(matrix != np.diag(np.diag(matrix))):
        raise ValueError("diagonal fit has off-diagonal terms")
    diagnostics = value["diagnostics"]
    if not isinstance(diagnostics, dict) or diagnostics.get("status") != "converged" or diagnostics.get("converged") is not True:
        raise ValueError("only converged fits may be frozen")
    if diagnostics.get("stopping_rule") != "exact_free_gradient_inf":
        raise ValueError("unrecognized stopping rule")
    tolerance = _positive(diagnostics.get("gradient_tolerance"), "gradient_tolerance")
    objective_tolerance = _positive(diagnostics.get("objective_tolerance"), "objective_tolerance", allow_zero=True)
    maximum = _integer(diagnostics.get("max_iterations"), "max_iterations")
    iterations = _integer(diagnostics.get("iterations"), "iterations", minimum=0)
    count = _integer(diagnostics.get("sample_count"), "sample_count")
    positive = _integer(diagnostics.get("positive_weight_count"), "positive_weight_count")
    _positive(diagnostics.get("weight_sum"), "weight_sum")
    history = diagnostics.get("history")
    if positive > count or iterations > maximum or not isinstance(history, list) or len(history) != iterations + 1:
        raise ValueError("inconsistent fit diagnostics")
    previous = None
    for i, row in enumerate(history):
        if not isinstance(row, dict) or _integer(row.get("iteration"), "history iteration", minimum=0) != i:
            raise ValueError("invalid fit history sequence")
        objective = _positive(row.get("objective"), "objective", allow_zero=True)
        _positive(row.get("gradient_inf"), "gradient_inf", allow_zero=True)
        if i:
            _positive(row.get("normal_condition"), "normal_condition")
            before = _positive(row.get("surrogate_before"), "surrogate_before", allow_zero=True)
            after = _positive(row.get("surrogate_after"), "surrogate_after", allow_zero=True)
            _positive(row.get("surrogate_gradient_inf"), "surrogate_gradient_inf", allow_zero=True)
            if after > before + objective_tolerance * (1 + abs(before)):
                raise ValueError("nonmonotone frozen surrogate history")
        if previous is not None and objective > previous + objective_tolerance * (1 + abs(previous)):
            raise ValueError("nonmonotone frozen fit history")
        previous = objective
    if history[-1]["gradient_inf"] > tolerance:
        raise ValueError("frozen fit fails exact gradient criterion")
    return value


def verify_fit(fit, reference_residual, labeled_error, weights):
    """Replay final objective/gradient with actual development inputs, no refit."""
    fit = validate_fit(fit)
    reference, error, weight = _inputs(reference_residual, labeled_error, weights)
    if _data_digest(reference, error, weight) != fit["data_sha256"]:
        raise ValueError("frozen fit development inputs changed")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            objective, gradient, _ = _loss_gradient(reference, error, weight,
                np.array(fit["intercept"]), np.array(fit["matrix"]),
                fit["loss"], fit["ridge"], fit["tau"], fit["mode"])
    except FloatingPointError as exc:
        raise ValueError("frozen fit replay overflow") from exc
    diagnostics = fit["diagnostics"]
    final = diagnostics["history"][-1]
    if (diagnostics["sample_count"] != len(reference)
            or diagnostics["positive_weight_count"] != int(np.count_nonzero(weight))
            or diagnostics["weight_sum"] != float(weight.sum())
            or not np.isclose(objective, final["objective"], rtol=1e-12, atol=1e-14)
            or not np.isclose(gradient, final["gradient_inf"], rtol=1e-10, atol=1e-14)
            or gradient > diagnostics["gradient_tolerance"]):
        raise ValueError("frozen fit final diagnostics fail development replay")
    return {"objective": objective, "gradient_inf": gradient, "data_sha256": fit["data_sha256"]}


def _native_inputs(raw_sequence, reference_residual, anchor):
    raw = _array(raw_sequence, "raw_sequence")
    if raw.ndim != 3 or raw.shape[0] != 16 or raw.shape[2] != 3 or raw.shape[1] == 0:
        raise ValueError("raw_sequence must retain exactly [16,V,3]")
    residual = _array(reference_residual, "reference_residual", raw.shape[1:])
    anchor = _array(anchor, "anchor", raw.shape[1:])
    return raw, residual, anchor


def apply_frozen_fits(raw_sequence, reference_residual, anchor, fits):
    """Apply fifteen frozen affine fits to frames 1..15; frame 0 is exact anchor.

    Arrays are returned as float64 with original row identities. Source anchor
    values are represented exactly (including every float32 value). The caller
    owns native export dtype checks and unchanged faces/identity validation.
    """
    raw, residual, anchor = _native_inputs(raw_sequence, reference_residual, anchor)
    if not isinstance(fits, (list, tuple)) or len(fits) != 15:
        raise ValueError("exactly fifteen nonanchor frozen fits are required")
    fits = [validate_fit(fit) for fit in fits]
    if len({(fit["mode"], fit["loss"], fit["ridge"], fit["tau"]) for fit in fits}) != 1:
        raise ValueError("framewise fits must share a frozen estimator configuration")
    output = raw.copy()
    try:
        with np.errstate(over="raise", invalid="raise"):
            for frame, fit in enumerate(fits, start=1):
                output[frame] = raw[frame] - np.array(fit["intercept"]) - residual @ np.array(fit["matrix"]).T
    except FloatingPointError as exc:
        raise ValueError("C03 application overflow") from exc
    output[0] = anchor
    if not np.isfinite(output).all():
        raise ValueError("C03 application produced nonfinite coordinates")
    return output


def unit_c01(raw_sequence, reference_residual, anchor):
    """Parameter-free unit-matrix control with identical raw input and anchor."""
    raw, residual, anchor = _native_inputs(raw_sequence, reference_residual, anchor)
    try:
        with np.errstate(over="raise", invalid="raise"):
            output = raw - residual[None]
    except FloatingPointError as exc:
        raise ValueError("unit C01 application overflow") from exc
    output[0] = anchor
    if not np.isfinite(output).all():
        raise ValueError("unit C01 produced nonfinite coordinates")
    return output
