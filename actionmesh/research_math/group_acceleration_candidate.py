"""C13 anchored group-l2 acceleration repair source draft.

The mathematical object is

    min_Y  .5 ||Y-Yhat||_M^2 + lambda sum_(t,v) ||(D2 Y)[t,v,:]||_2
    subject to complete pinned frames Y[p] = X[p].

``M`` is an explicit temporal SPD matrix shared across vertices and xyz, i.e.
``M kron I_(V*3)``.  This keeps off-diagonal temporal/anchor cross terms while
remaining matrix-free in the mesh dimension.  Explicit anchors use float32
native-coordinate values; the native specialization uses the recorded identity
metric and sets ``X=Yhat[0]``.  It consumes only a complete
predicted sequence; no GT, event labels, scorer state, cameras, latent state or
learned weights enter.

This module is generated source, not Natural Gate 0/IPCG admission, Local
verification, native scoring, or evidence that sparse acceleration is useful.
Run it only through the reviewed research-autopilot harness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .quadratic_acceleration_control import timestamp_second_difference


CANDIDATE_ID = '4d-math-20261006-c13'
OBJECTIVE = ('0.5||Y-Yhat||_(M kron I)^2 + '
             'group_weight*sum_(t,v)||D2_timestamp Y[t,v,:]||_2; '
             'Y[pinned_frames]=X[pinned_frames] '
             '(native specialization X=Yhat[pinned_frames])')


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _vertices(value) -> np.ndarray:
    vertices = np.asarray(value)
    if (vertices.ndim != 3 or vertices.shape[-1] != 3 or vertices.shape[0] < 3
            or vertices.shape[1] < 3 or vertices.dtype != np.dtype(np.float32)
            or not np.isfinite(vertices).all()):
        raise ValueError('Finite float32 vertices[T,V,3] with T,V >= 3 required')
    return vertices


def _positive_scalar(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f'{name} must be a finite positive scalar')
    value = float(value)
    if not np.isfinite(value) or value <= 0.:
        raise ValueError(f'{name} must be a finite positive scalar')
    return value


def _positive_integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be a positive integer')
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _pinned_frames(value, frames: int) -> np.ndarray:
    pins = np.asarray(value)
    if (pins.ndim != 1 or not len(pins) or not np.issubdtype(pins.dtype, np.integer)
            or np.issubdtype(pins.dtype, np.bool_)):
        raise ValueError('At least one integer pinned frame required')
    pins = pins.astype(np.int64, copy=False)
    if (np.any(pins < 0) or np.any(pins >= frames)
            or (len(pins) > 1 and np.any(np.diff(pins) <= 0))):
        raise ValueError('Pinned frames must be strictly increasing and in range')
    if len(pins) == frames:
        raise ValueError('At least one free frame required')
    return pins


def _observation_metric(value, frames: int) -> tuple[np.ndarray, dict]:
    metric = np.asarray(value, dtype=np.float64)
    if metric.shape != (frames, frames) or not np.isfinite(metric).all():
        raise ValueError('Finite temporal observation metric M[T,T] required')
    symmetry_error = float(np.max(np.abs(metric - metric.T)))
    symmetry_tolerance = float(100 * frames * np.finfo(np.float64).eps
                               * max(1., np.max(np.abs(metric))))
    if symmetry_error > symmetry_tolerance:
        raise ValueError('Observation metric must be symmetric')
    metric = .5 * (metric + metric.T)
    try:
        np.linalg.cholesky(metric)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Observation metric must be positive definite') from exc
    condition = float(np.linalg.cond(metric))
    condition_max = float(1. / np.sqrt(np.finfo(np.float64).eps))
    if not np.isfinite(condition) or condition > condition_max:
        raise ValueError(
            'Observation metric failed numerical admission: '
            f'condition_2={condition:.17g}, condition_2_max={condition_max:.17g}')
    offdiagonal = metric - np.diag(np.diag(metric))
    return metric, {
        'metric_scope': 'shared_temporal_spd_kron_identity_vertex_xyz',
        'metric_symmetry_error_linf': symmetry_error,
        'metric_symmetry_tolerance': symmetry_tolerance,
        'metric_condition_2': condition,
        'metric_condition_2_max': condition_max,
        'metric_offdiagonal_linf': float(np.max(np.abs(offdiagonal))),
    }


def form_anchored_group_trend_problem(vertices, timesteps, observation_metric,
                                      pinned_frames=(0,), anchor_values=None) -> dict:
    """Form the exact reduced affine-anchor problem from the reviewed card.

    The supported scalable metric class is an explicit temporal SPD block
    shared over vertex-coordinate columns.  It includes non-diagonal temporal
    terms and their pinned/free cross terms; it is not a diagonal shortcut.
    """
    source = _vertices(vertices)
    operator = timestamp_second_difference(timesteps)
    if operator.shape[1] != len(source):
        raise ValueError('One supplied timestamp is required for every frame')
    metric, metric_diagnostics = _observation_metric(observation_metric, len(source))
    pins = _pinned_frames(pinned_frames, len(source))
    if anchor_values is None:
        anchors = source[pins].astype(np.float64)
    else:
        native_anchors = np.asarray(anchor_values)
        if (native_anchors.shape != (len(pins), source.shape[1], 3)
                or native_anchors.dtype != np.dtype(np.float32)
                or not np.isfinite(native_anchors).all()):
            raise ValueError(
                'Finite float32 anchor_values[P,V,3] matching pinned_frames required')
        anchors = native_anchors.astype(np.float64)
    free = np.setdiff1d(np.arange(len(source), dtype=np.int64), pins,
                        assume_unique=True)
    source64 = source.astype(np.float64)
    particular = np.zeros_like(source64)
    particular[pins] = anchors
    error_at_origin = particular - source64
    reduced_metric = metric[np.ix_(free, free)]
    reduced_operator = operator[:, free]
    affine_acceleration = np.einsum('rt,tvc->rvc', operator, particular)
    reduced_linear = np.einsum('ft,tvc->fvc', metric[free], error_at_origin)
    reduced_condition = float(np.linalg.cond(reduced_metric))
    if (not np.isfinite(reduced_condition)
            or reduced_condition > metric_diagnostics['metric_condition_2_max']):
        raise ValueError('Free observation metric failed numerical admission')
    return {
        'source': source64,
        'operator': operator,
        'metric': metric,
        'pins': pins,
        'anchor_values': anchors,
        'free': free,
        'particular': particular,
        'error_at_origin': error_at_origin,
        'reduced_metric': reduced_metric,
        'reduced_operator': reduced_operator,
        'affine_acceleration': affine_acceleration,
        'reduced_linear': reduced_linear,
        'metric_diagnostics': {
            **metric_diagnostics,
            'free_metric_condition_2': reduced_condition,
        },
    }


def form_group_dual_balls(value, radius: float) -> np.ndarray:
    """Project each final-axis 3-vector onto its Euclidean dual ball."""
    radius = _positive_scalar('radius', radius)
    vectors = np.asarray(value, dtype=np.float64)
    if vectors.ndim != 3 or vectors.shape[-1] != 3 or not np.isfinite(vectors).all():
        raise ValueError('Finite group array [R,V,3] required')
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scale = np.minimum(1., radius / np.maximum(norms, np.finfo(np.float64).tiny))
    return vectors * scale


def _group_shrink(value: np.ndarray, threshold: float) -> np.ndarray:
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    scale = np.maximum(0., 1. - threshold / np.maximum(
        norms, np.finfo(np.float64).tiny))
    return value * scale


def _certificate(problem: dict, y: np.ndarray, dual: np.ndarray,
                 group_weight: float) -> tuple[dict, dict]:
    """Return a dual-feasible certificate and its objective diagnostics."""
    metric = problem['metric']
    operator = problem['operator']
    free = problem['free']
    hessian = problem['reduced_metric']
    reduced_operator = problem['reduced_operator']
    reduced_linear = problem['reduced_linear']
    affine_acceleration = problem['affine_acceleration']
    source = problem['source']
    particular = problem['particular']

    feasible_dual = form_group_dual_balls(dual, group_weight)
    acceleration = np.einsum('rt,tvc->rvc', operator, y)
    error = y - source
    data_term = .5 * float(np.einsum('tvc,ts,svc->', error, metric, error))
    group_term = group_weight * float(np.linalg.norm(acceleration, axis=-1).sum())
    primal = data_term + group_term

    origin_error = particular - source
    constant = .5 * float(np.einsum(
        'tvc,ts,svc->', origin_error, metric, origin_error))
    dual_linear = reduced_linear + np.einsum(
        'rf,rvc->fvc', reduced_operator, feasible_dual)
    dual_center = np.linalg.solve(
        hessian, dual_linear.reshape(len(free), -1)).reshape(dual_linear.shape)
    lower = (constant
             + float(np.sum(feasible_dual * affine_acceleration))
             - .5 * float(np.sum(dual_linear * dual_center)))
    raw_gap = primal - lower
    scale = max(1., abs(primal), abs(lower))
    negative_tolerance = 1000 * np.finfo(np.float64).eps * scale * max(1, y.size)
    if not np.isfinite(raw_gap) or raw_gap < -negative_tolerance:
        raise ValueError(
            'Invalid primal/dual certificate: '
            f'raw_gap={raw_gap:.17g}, negative_tolerance={negative_tolerance:.17g}')
    gap = max(0., float(raw_gap))
    stationarity = (np.einsum('ft,tvc->fvc', metric[free], error)
                    + np.einsum('rf,rvc->fvc', reduced_operator, feasible_dual))
    diagnostics = {
        'primal_objective': primal,
        'dual_lower_bound': float(lower),
        'primal_dual_gap': gap,
        'primal_dual_gap_raw': float(raw_gap),
        'primal_dual_gap_relative': gap / scale,
        'data_term': data_term,
        'group_term': group_term,
        'dual_group_norm_max': float(np.max(np.linalg.norm(feasible_dual, axis=-1))),
        'dual_feasibility_violation': float(max(
            0., np.max(np.linalg.norm(feasible_dual, axis=-1)) - group_weight)),
        'stationarity_l2': float(np.linalg.norm(stationarity)),
        'stationarity_linf': float(np.max(np.abs(stationarity))),
    }
    certificate = {
        'dual': feasible_dual,
        'second_difference': acceleration,
        'observation_metric': metric,
        'pinned_frames': problem['pins'],
        'anchor_values': problem['anchor_values'],
    }
    return certificate, diagnostics


def solve_affine_anchor_stationarity(problem: dict, *, group_weight: float,
                                     rho: float, absolute_tolerance: float,
                                     relative_tolerance: float,
                                     gap_tolerance: float,
                                     max_iterations: int):
    """Solve the reduced convex problem by scaled ADMM and certify its dual.

    The returned dual is projected onto the exact 3D group balls before the
    Fenchel lower bound is evaluated.  A merely small ADMM residual is not
    treated as a primal/dual certificate.
    """
    group_weight = _positive_scalar('group_weight', group_weight)
    rho = _positive_scalar('rho', rho)
    absolute_tolerance = _positive_scalar('absolute_tolerance', absolute_tolerance)
    relative_tolerance = _positive_scalar('relative_tolerance', relative_tolerance)
    gap_tolerance = _positive_scalar('gap_tolerance', gap_tolerance)
    max_iterations = _positive_integer('max_iterations', max_iterations)

    source = problem['source']
    particular = problem['particular']
    free = problem['free']
    hessian = problem['reduced_metric']
    reduced_operator = problem['reduced_operator']
    affine_acceleration = problem['affine_acceleration']
    reduced_linear = problem['reduced_linear']
    system = hessian + rho * (reduced_operator.T @ reduced_operator)
    system_condition = float(np.linalg.cond(system))
    condition_max = float(1. / np.sqrt(np.finfo(np.float64).eps))
    if not np.isfinite(system_condition) or system_condition > condition_max:
        raise ValueError(
            'ADMM free system failed numerical admission: '
            f'condition_2={system_condition:.17g}, condition_2_max={condition_max:.17g}')

    y = particular.copy()
    y[free] = source[free]
    acceleration = np.einsum('rt,tvc->rvc', problem['operator'], y)
    auxiliary = acceleration.copy()
    scaled_dual = np.zeros_like(auxiliary)
    threshold = group_weight / rho
    n_constraints = auxiliary.size
    n_free = len(free) * source.shape[1] * source.shape[2]
    diagnostics = None
    certificate = None
    termination_reason = 'maximum_iterations'

    for iteration in range(1, max_iterations + 1):
        right_hand = (-reduced_linear + rho * np.einsum(
            'rf,rvc->fvc', reduced_operator,
            auxiliary - scaled_dual - affine_acceleration))
        free_solution = np.linalg.solve(
            system, right_hand.reshape(len(free), -1)).reshape(right_hand.shape)
        y = particular.copy()
        y[free] = free_solution
        acceleration = np.einsum('rt,tvc->rvc', problem['operator'], y)
        previous_auxiliary = auxiliary
        auxiliary = _group_shrink(acceleration + scaled_dual, threshold)
        scaled_dual = scaled_dual + acceleration - auxiliary

        primal_residual = acceleration - auxiliary
        dual_residual_reduced = rho * np.einsum(
            'rf,rvc->fvc', reduced_operator, auxiliary - previous_auxiliary)
        primal_norm = float(np.linalg.norm(primal_residual))
        dual_norm = float(np.linalg.norm(dual_residual_reduced))
        primal_threshold = (np.sqrt(n_constraints) * absolute_tolerance
                            + relative_tolerance * max(
                                float(np.linalg.norm(acceleration)),
                                float(np.linalg.norm(auxiliary))))
        unscaled_dual = rho * scaled_dual
        dual_threshold = (np.sqrt(n_free) * absolute_tolerance
                          + relative_tolerance * float(np.linalg.norm(np.einsum(
                              'rf,rvc->fvc', reduced_operator, unscaled_dual))))
        certificate, objective = _certificate(
            problem, y, unscaled_dual, group_weight)
        diagnostics = {
            **problem['metric_diagnostics'], **objective,
            'iteration': iteration,
            'primal_residual_l2': primal_norm,
            'dual_residual_l2': dual_norm,
            'primal_residual_threshold': float(primal_threshold),
            'dual_residual_threshold': float(dual_threshold),
            'rho': rho,
            'group_weight': group_weight,
            'absolute_tolerance': absolute_tolerance,
            'relative_tolerance': relative_tolerance,
            'gap_tolerance': gap_tolerance,
            'max_iterations': max_iterations,
            'free_system_condition_2': system_condition,
            'free_system_condition_2_max': condition_max,
        }
        if (primal_norm <= primal_threshold and dual_norm <= dual_threshold
                and objective['primal_dual_gap_relative'] <= gap_tolerance):
            termination_reason = 'converged'
            break

    diagnostics['termination_reason'] = termination_reason
    anchor_residual = y[problem['pins']] - problem['anchor_values']
    diagnostics['anchor_residual_linf'] = float(np.max(np.abs(anchor_residual)))
    diagnostics['anchor_exact_float64'] = bool(np.array_equal(
        y[problem['pins']], problem['anchor_values']))
    if termination_reason != 'converged':
        raise RuntimeError(
            'Group acceleration solver did not meet frozen residual/gap tolerances: '
            + json.dumps(diagnostics, sort_keys=True, allow_nan=False))
    if not diagnostics['anchor_exact_float64']:
        raise RuntimeError('Affine anchor was not exact')
    if not all(np.isfinite(value) for value in diagnostics.values()
               if isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise RuntimeError('Nonfinite group acceleration diagnostic')
    return y, certificate, diagnostics


def repair_group_acceleration(vertices, timesteps, observation_metric, *,
                              pinned_frames=(0,), anchor_values=None,
                              group_weight: float, rho: float,
                              absolute_tolerance: float, relative_tolerance: float,
                              gap_tolerance: float, max_iterations: int):
    """Return a certified full trajectory and exact-anchor diagnostics."""
    source = _vertices(vertices)
    problem = form_anchored_group_trend_problem(
        source, timesteps, observation_metric, pinned_frames, anchor_values)
    repaired64, certificate, diagnostics = solve_affine_anchor_stationarity(
        problem, group_weight=group_weight, rho=rho,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        gap_tolerance=gap_tolerance, max_iterations=max_iterations)
    repaired = repaired64.astype(np.float32)
    exported_anchors = problem['anchor_values'].astype(np.float32)
    repaired[problem['pins']] = exported_anchors
    if not np.isfinite(repaired).all():
        raise RuntimeError('Group acceleration solve emitted nonfinite vertices')
    diagnostics['anchor_exact_float32'] = bool(np.array_equal(
        repaired[problem['pins']], exported_anchors))
    diagnostics['displacement_l2'] = float(np.linalg.norm(repaired64 - problem['source']))
    diagnostics['frames'] = int(source.shape[0])
    diagnostics['vertices'] = int(source.shape[1])
    diagnostics['pinned_frames'] = problem['pins'].tolist()
    # The native scorer consumes the float32 export, not the internal float64
    # iterate. Recompute the complete certificate at the exact promoted bytes.
    solver_objective = {key: value for key, value in diagnostics.items()
                        if key in ('primal_objective', 'dual_lower_bound',
                                   'primal_dual_gap', 'primal_dual_gap_raw',
                                   'primal_dual_gap_relative', 'data_term',
                                   'group_term', 'stationarity_l2',
                                   'stationarity_linf')}
    certificate, exported_objective = _certificate(
        problem, repaired.astype(np.float64), certificate['dual'], group_weight)
    for key, value in solver_objective.items():
        diagnostics['solver_float64_' + key] = value
    diagnostics.update(exported_objective)
    diagnostics['certificate_scope'] = 'exact_exported_float32_promoted_to_float64'
    diagnostics['float32_export_delta_l2'] = float(np.linalg.norm(
        repaired.astype(np.float64) - repaired64))
    if diagnostics['primal_dual_gap_relative'] > gap_tolerance:
        raise RuntimeError(
            'Exported float32 sequence failed primal/dual gap tolerance: '
            f"gap={diagnostics['primal_dual_gap_relative']:.17g}, "
            f'tolerance={gap_tolerance:.17g}')
    return repaired, certificate, diagnostics


def _load_metric(path: Path | None, expected_sha256: str | None,
                 frames: int) -> tuple[np.ndarray, dict]:
    if path is None:
        if expected_sha256 is not None:
            raise ValueError('Identity metric must not carry an external metric hash')
        return np.eye(frames, dtype=np.float64), {
            'mode': 'identity',
            'scope': 'shared_temporal_spd_kron_identity_vertex_xyz',
        }
    path = Path(path)
    if path.name in ('', '.', '..') or not expected_sha256:
        raise ValueError('Explicit metric file and expected SHA256 required')
    actual = _digest(path)
    if actual != expected_sha256:
        raise ValueError('Observation metric differs from explicitly pinned input')
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {'observation_metric'}:
            raise ValueError('Metric NPZ must contain only observation_metric')
        metric = saved['observation_metric'].copy()
    _observation_metric(metric, frames)
    return metric, {
        'mode': 'explicit_npz', 'path': str(path.resolve()), 'sha256': actual,
        'array': 'observation_metric',
        'scope': 'shared_temporal_spd_kron_identity_vertex_xyz',
    }


def export_group_candidate(source_case: Path, output: Path, *, uid: str,
                           expected_sequence_sha256: str,
                           metric_path: Path | None,
                           expected_metric_sha256: str | None,
                           group_weight: float, rho: float,
                           absolute_tolerance: float,
                           relative_tolerance: float,
                           gap_tolerance: float, max_iterations: int):
    """Export one complete C13 arm consumable by the official adapter."""
    started = time.monotonic()
    source_case, output = Path(source_case), Path(output)
    if output.exists():
        raise FileExistsError('Candidate output is single-use')
    if not uid or Path(uid).name != uid or uid in ('.', '..'):
        raise ValueError('Invalid native UID')
    # Validate every frozen parameter before creating any output directory.
    group_weight = _positive_scalar('group_weight', group_weight)
    rho = _positive_scalar('rho', rho)
    absolute_tolerance = _positive_scalar('absolute_tolerance', absolute_tolerance)
    relative_tolerance = _positive_scalar('relative_tolerance', relative_tolerance)
    gap_tolerance = _positive_scalar('gap_tolerance', gap_tolerance)
    max_iterations = _positive_integer('max_iterations', max_iterations)
    source_path = source_case / 'sequence.npz'
    actual_sequence_sha256 = _digest(source_path)
    if actual_sequence_sha256 != expected_sequence_sha256:
        raise ValueError('Sequence differs from explicitly pinned input')
    source_report_path = source_case / 'report.json'
    source_report = json.loads(source_report_path.read_text())
    if (source_report.get('status') != 'completed' or source_report.get('uid') != uid
            or source_report.get('sha256', {}).get('sequence.npz')
            != actual_sequence_sha256):
        raise ValueError('Completed same-UID source report with current sequence hash required')
    seed = source_report.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Source inference seed must be an integer')
    with np.load(source_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    required = {'vertices', 'faces', 'timesteps', 'frame_indices', 'query_vertex_ids'}
    if not required.issubset(arrays):
        raise ValueError('Full native mesh metadata required')
    vertices = _vertices(arrays['vertices'])
    if (len(vertices) != 16 or not np.issubdtype(arrays['frame_indices'].dtype, np.integer)
            or not np.array_equal(arrays['frame_indices'], np.arange(16))):
        raise ValueError('Exactly 16 original frames in original order required')
    timestamp_second_difference(arrays['timesteps'])
    if len(arrays['timesteps']) != len(vertices):
        raise ValueError('One supplied timestamp is required for every frame')
    faces = arrays['faces']
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0
            or faces.max() >= vertices.shape[1]):
        raise ValueError('Invalid shared triangle topology')
    if (not np.issubdtype(arrays['query_vertex_ids'].dtype, np.integer)
            or not np.array_equal(arrays['query_vertex_ids'], np.arange(vertices.shape[1]))):
        raise ValueError('Original identity vertex mapping required')
    for key, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f'Non-numeric/nonfinite native array: {key}')
    metric, metric_identity = _load_metric(
        metric_path, expected_metric_sha256, len(vertices))
    # Forming the problem now rejects malformed SPD/anchor inputs before output.
    form_anchored_group_trend_problem(vertices, arrays['timesteps'], metric, [0])

    output.mkdir(parents=True, exist_ok=False)
    arm = output / 'group_acceleration'
    arm.mkdir()
    _write_json(output / 'manifest.json', {
        'cases': [{'case_id': uid + '-group_acceleration', 'uid': uid,
                   'case_dir': 'group_acceleration'}],
        'scope': ('C13 generated_unexecuted candidate arm only; B0/B*/Gaussian/quadratic '
                  'remain separately hash-bound controls; no scientific admission'),
    })
    report = {
        'uid': uid, 'seed': seed, 'candidate_id': CANDIDATE_ID,
        'candidate_arm': 'group_acceleration', 'objective': OBJECTIVE,
        'source_sequence_sha256': actual_sequence_sha256,
        'source_report_sha256': _digest(source_report_path),
        'implementation_sha256': _digest(Path(__file__)),
        'observation_metric': metric_identity,
        'anchor': {'mode': 'exact_complete_frames', 'pinned_frames': [0]},
        'timestamp_units': ('supplied_sequence_units; current native 0..15 loader clock is '
                            'not claimed to be physical video wall-clock time'),
        'group_weight_units': 'coordinate_unit*supplied_sequence_unit^2',
        'solver': {
            'name': 'scaled_admm_exact_reduced_y_update',
            'group_weight': group_weight, 'rho': rho,
            'absolute_tolerance': absolute_tolerance,
            'relative_tolerance': relative_tolerance,
            'gap_tolerance': gap_tolerance,
            'max_iterations': max_iterations,
        },
        'native_qualified': False, 'scientific_admission': False,
        'local_method_verified': False,
        'source_delivery_status': 'generated_unexecuted_at_authoring',
        'information': ('Full predicted sequence plus explicit SPD metric only; no GT, event '
                        'labels, scorer state, cameras, latent state, or learned weights'),
    }
    try:
        repaired, certificate, diagnostics = repair_group_acceleration(
            vertices, arrays['timesteps'], metric, pinned_frames=[0],
            group_weight=group_weight, rho=rho,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            gap_tolerance=gap_tolerance, max_iterations=max_iterations)
        if not np.array_equal(repaired[0], vertices[0]):
            raise RuntimeError('Candidate changed exact frame-zero anchor')
        np.savez_compressed(arm / 'sequence.npz', **{**arrays, 'vertices': repaired})
        np.savez_compressed(arm / 'certificate.npz', **certificate)
        report.update(status='completed', diagnostics=diagnostics,
                      sha256={'sequence.npz': _digest(arm / 'sequence.npz'),
                              'certificate.npz': _digest(arm / 'certificate.npz')})
    except Exception as error:
        report.update(status='error', exception_type=type(error).__name__, error=str(error))
    report['elapsed_seconds'] = time.monotonic() - started
    report['timing_scope'] = ('Adapter I/O + CPU solver only; excludes generation, controls, '
                              'official scorer and collection')
    _write_json(arm / 'report.json', report)
    result = {
        'status': 'completed' if report['status'] == 'completed' else 'incomplete',
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'uid': uid, 'seed': seed, 'arms': [report],
        'candidate_id': CANDIDATE_ID,
        'native_qualified': False, 'scientific_admission': False,
        'local_method_verified': False,
        'source_delivery_status': 'generated_unexecuted_at_authoring',
        'elapsed_seconds': time.monotonic() - started,
    }
    _write_json(output / 'candidate.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-case', type=Path)
    source.add_argument('--source-sequence', type=Path)
    metric = parser.add_mutually_exclusive_group(required=True)
    metric.add_argument('--identity-observation-metric', action='store_true')
    metric.add_argument('--observation-metric', type=Path)
    parser.add_argument('--expected-metric-sha256')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--expected-sequence-sha256', required=True)
    parser.add_argument('--group-weight', required=True, type=float)
    parser.add_argument('--rho', required=True, type=float)
    parser.add_argument('--absolute-tolerance', required=True, type=float)
    parser.add_argument('--relative-tolerance', required=True, type=float)
    parser.add_argument('--gap-tolerance', required=True, type=float)
    parser.add_argument('--max-iterations', required=True, type=int)
    args = parser.parse_args()
    if args.source_sequence is not None and args.source_sequence.name != 'sequence.npz':
        parser.error('--source-sequence must name sequence.npz')
    if args.identity_observation_metric and args.expected_metric_sha256 is not None:
        parser.error('Identity metric must not carry --expected-metric-sha256')
    if args.observation_metric is not None and args.expected_metric_sha256 is None:
        parser.error('Explicit metric requires --expected-metric-sha256')
    source_case = args.source_case if args.source_case is not None else args.source_sequence.parent
    result = export_group_candidate(
        source_case, args.output, uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256,
        metric_path=args.observation_metric,
        expected_metric_sha256=args.expected_metric_sha256,
        group_weight=args.group_weight, rho=args.rho,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        gap_tolerance=args.gap_tolerance,
        max_iterations=args.max_iterations)
    print(json.dumps({
        'status': result['status'], 'candidate_id': CANDIDATE_ID,
        'native_qualified': False, 'scientific_admission': False,
        'local_method_verified': False,
    }))
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
