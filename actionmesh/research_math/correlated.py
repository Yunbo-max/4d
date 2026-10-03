"""Constructed prototype A: shared-bias covariance regularization.

Generated views are conditioned on real inputs: this is a model-based
regularizer, not an independent-observation Bayesian likelihood. Bias budgets
are caller-declared, never estimated from GT or claimed calibrated. This module
solves a frozen affine-residual surrogate; renderer/nonlinear feasibility must
be checked separately by a future caller. No Torch, models, or GPU are loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _array(value, name, ndim):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or array.ndim not in ndim:
        raise ValueError(f'{name}: expected real numeric array with dimensions {ndim}')
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(name+': values must be finite')
    return array


def _scalar(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name+': expected a finite scalar')
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(name+': expected a finite scalar') from exc
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(name+': must be finite and '+('positive' if positive else 'nonnegative'))
    return value


def apply_covariance_inverse(rhs, projected_modes, variances, *, sigma2):
    """Apply (sigma2 I + A diag(variances) A.T)^-1 without a dense inverse.

    rhs: [m] or [m,q]; projected_modes A: [m,k]; variances: [k]>=0.
    Zero-rank and zero-variance modes are valid. Inputs are not modified.
    sigma2 is the variance itself, not its standard deviation.
    """
    rhs = _array(rhs, 'rhs', (1, 2))
    modes = _array(projected_modes, 'projected_modes', (2,))
    budget = _array(variances, 'variances', (1,))
    sigma2 = _scalar(sigma2, 'sigma2', positive=True)
    if not len(rhs) or modes.shape != (len(rhs), len(budget)):
        raise ValueError('Covariance row/rank dimensions disagree or have no rows')
    if (budget < 0).any():
        raise ValueError('variances must be nonnegative')
    # sqrt formulation supports singular Lambda without an inverse of Lambda.
    factor = modes * np.sqrt(budget)
    gram = sigma2*np.eye(len(budget)) + factor.T @ factor
    if not np.isfinite(gram).all():
        raise ValueError('Covariance arithmetic overflow')
    correction = factor @ np.linalg.solve(gram, factor.T @ rhs)
    result = (rhs-correction)/sigma2
    if not np.isfinite(result).all():
        raise ValueError('Covariance inverse produced nonfinite values')
    return result


def solve_correction(observed_residual, observed_jacobian,
                     synthetic_residual, synthetic_jacobian, *,
                     bias_modes, bias_variances, sigma2, synthetic_weight,
                     ridge, trust_radius, group_ids=None, group_caps=None,
                     max_backtracks=40):
    """Return a feasible descent correction for a frozen quadratic surrogate.

    Residuals are r + J a. Constraints are ||a||<=trust_radius and, for every
    actual-observation group, squared residual <= original-zero error + cap.
    The unconstrained quadratic minimizer is the proposal; radial trust clipping
    and backtracking only shorten it. If its direction is infeasible, return
    original zero. This deliberately does NOT certify a constrained optimum;
    another feasible direction might exist. Covariance remains fixed throughout.
    """
    ro = _array(observed_residual, 'observed_residual', (1,))
    jo = _array(observed_jacobian, 'observed_jacobian', (2,))
    rs = _array(synthetic_residual, 'synthetic_residual', (1,))
    js = _array(synthetic_jacobian, 'synthetic_jacobian', (2,))
    modes = _array(bias_modes, 'bias_modes', (2,))
    budget = _array(bias_variances, 'bias_variances', (1,))
    sigma2 = _scalar(sigma2, 'sigma2', positive=True)
    weight = _scalar(synthetic_weight, 'synthetic_weight')
    ridge = _scalar(ridge, 'ridge', positive=True)
    radius = _scalar(trust_radius, 'trust_radius')
    p = jo.shape[1]
    if not p or not len(ro) or not len(rs) or jo.shape[0] != len(ro) or js.shape != (len(rs), p):
        raise ValueError('Residual/Jacobian dimensions disagree or are empty')
    if modes.shape != (p, len(budget)) or (budget < 0).any():
        raise ValueError('Bias mode/budget dimensions disagree or variance is negative')
    if isinstance(max_backtracks, bool) or not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 0:
        raise ValueError('max_backtracks must be a nonnegative integer')
    ids = np.zeros(len(ro), dtype=np.int64) if group_ids is None else np.asarray(group_ids)
    if ids.shape != (len(ro),) or ids.dtype.kind not in 'iu' or (ids < 0).any():
        raise ValueError('group_ids must be nonnegative integer labels, one per real residual')
    ids = ids.astype(np.int64)
    groups = np.unique(ids)
    if not np.array_equal(groups, np.arange(len(groups))):
        raise ValueError('group_ids must cover consecutive groups starting at zero')
    caps = np.zeros(len(groups)) if group_caps is None else _array(group_caps, 'group_caps', (1,))
    if caps.shape != (len(groups),) or (caps < 0).any():
        raise ValueError('group_caps must be one nonnegative allowance per group')
    projected = js @ modes
    # Calculate C^-1[J_s,r_s] once. No changing budget or covariance in line search.
    whitened = apply_covariance_inverse(np.column_stack((js, rs)), projected, budget, sigma2=sigma2)
    cjs, crs = whitened[:, :p], whitened[:, p]
    hessian = jo.T @ jo + weight*(js.T @ cjs) + ridge*np.eye(p)
    hessian = (hessian+hessian.T)/2
    gradient = jo.T @ ro + weight*(js.T @ crs)
    if not np.isfinite(hessian).all() or not np.isfinite(gradient).all():
        raise ValueError('Quadratic arithmetic produced nonfinite values')
    baseline_groups = np.bincount(ids, weights=ro*ro, minlength=len(groups))
    baseline_objective = float(.5*(ro @ ro) + .5*weight*(rs @ crs))
    if not np.isfinite(baseline_objective) or not np.isfinite(baseline_groups).all():
        raise ValueError('Initial objective/error is nonfinite')
    zero = np.zeros(p)
    correction, after_groups = zero.copy(), baseline_groups.copy()
    objective_after, backtracks = baseline_objective, 0
    proposal = -np.linalg.solve(hessian, gradient)
    proposal_norm = float(np.linalg.norm(proposal))
    if not np.isfinite(proposal_norm):
        raise ValueError('Quadratic proposal is nonfinite')
    status, step_scale = 'rejected_constraints', 0.
    if proposal_norm == 0:
        status = 'unchanged_stationary'
    elif radius == 0:
        status = 'unchanged_zero_radius'
    else:
        clipped = proposal * min(1., radius/proposal_norm)
        for backtracks in range(max_backtracks+1):
            step_scale = 2.**(-backtracks)
            candidate = step_scale*clipped
            actual = ro + jo @ candidate
            errors = np.bincount(ids, weights=actual*actual, minlength=len(groups))
            change = float(gradient @ candidate + .5*candidate @ hessian @ candidate)
            if (np.linalg.norm(candidate) <= radius*(1+8*np.finfo(float).eps)
                    and np.isfinite(errors).all() and np.all(errors <= baseline_groups+caps)
                    and np.isfinite(change) and change < 0):
                correction, after_groups = candidate, errors
                objective_after = baseline_objective+change
                status = 'accepted'
                break
        if status != 'accepted':
            step_scale = 0.
    return {
        'correction': correction, 'status': status, 'feasible': True,
        'constrained_optimum_certified': False,
        'solver': 'unconstrained quadratic proposal; trust clip; feasible descent backtracking; zero fallback',
        'backtracks': int(backtracks), 'step_scale': step_scale,
        'objective_before': baseline_objective, 'objective_after': float(objective_after),
        'group_errors_before': baseline_groups.tolist(), 'group_errors_after': after_groups.tolist(),
        'group_caps': caps.tolist(), 'correction_norm': float(np.linalg.norm(correction)),
        'trust_radius': radius, 'sigma2': sigma2, 'synthetic_weight': weight,
        'bias_variances': budget.tolist(), 'covariance_frozen': True,
        'budget_source': 'caller-declared; not estimated from GT or generator seeds',
        'uncertainty_calibrated': False, 'nonlinear_feasibility_checked': False,
        'interpretation': 'model-based regularizer for dependent generated evidence; not an independent likelihood',
    }


def constructed_benchmark(seed=42):
    """Seeded affine fixtures with GT used solely for construction/evaluation.

    These are intentionally mixed positive/negative controls, not natural 4D
    episodes. No selection of favorable seeds, cases or failed arms is performed.
    """
    rng = np.random.default_rng(seed)
    cases = []
    repeats = 16
    jo = np.diag([1., .1])
    js = np.tile(np.eye(2), (repeats, 1))
    for name in ('shared_hidden_bias', 'helpful_hidden_prior',
                 'misspecified_bias_subspace', 'synthetic_self_render', 'zero_residual'):
        truth = np.array([1., 1.]) + rng.normal(0., .01, 2)
        bias = np.array([0., 1.]) if name in ('shared_hidden_bias', 'misspecified_bias_subspace') else np.zeros(2)
        mode = np.array([[1.], [0.]]) if name == 'misspecified_bias_subspace' else np.array([[0.], [1.]])
        ro = -jo @ truth
        target = js @ (truth+bias)
        if name == 'synthetic_self_render':
            target = np.zeros(len(js))
        if name == 'zero_residual':
            truth, ro, target = np.zeros(2), np.zeros(2), np.zeros(len(js))
        arms = {}
        # The count/cap policies count generated view blocks, not scalar residuals.
        weights = {'source_only': 0., 'equal_weight': 1., 'inverse_count': 1/repeats,
                   'fixed_cap': min(1., 2/repeats), 'covariance': 1.}
        for arm, weight in weights.items():
            use_covariance = arm == 'covariance'
            result = solve_correction(ro, jo, -target, js,
                bias_modes=mode if use_covariance else np.zeros((2, 0)),
                bias_variances=np.array([4.]) if use_covariance else np.zeros(0),
                sigma2=1., synthetic_weight=weight, ridge=.1, trust_radius=3.,
                group_ids=np.array([0, 1]), group_caps=np.zeros(2))
            result['parameter_error'] = float(np.linalg.norm(result['correction']-truth))
            result['correction'] = result['correction'].tolist()
            arms[arm] = result
        cases.append({'name': name, 'constructed': True, 'truth_for_evaluation': truth.tolist(),
                      'synthetic_bias_for_construction': bias.tolist(), 'arms': arms})
    return {'schema_version': 1, 'seed': int(seed), 'status': 'completed',
            'scope': 'Constructed affine residual surrogate; no 4D renderer/model or natural episodes',
            'natural_efficacy_claim': False, 'novelty_pass': False,
            'GT_used_by_solver': False, 'generated_view_blocks': repeats,
            'declared_bias_variance': 4., 'budget_fit_from_GT': False,
            'cases': cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = constructed_benchmark(args.seed)
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    with args.benchmark_output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'completed', 'output': str(args.benchmark_output),
                      'constructed_cases': len(report['cases']), 'natural_efficacy_claim': False}))


if __name__ == '__main__':
    main()
