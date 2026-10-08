"""Quadratic supplied-timestamp acceleration control required by C13.

This is a strong simple comparator, not the C13 group-trend candidate. It uses
only a complete predicted mesh sequence and keeps frame zero exactly fixed.
No ground truth, scorer state, learned weights, or event labels enter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


OBJECTIVE = '0.5||Y-Yhat||_F^2 + 0.5*weight||D2_timestamp Y||_F^2; Y[0]=Yhat[0]'


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def timestamp_second_difference(timesteps) -> np.ndarray:
    """Return the nonuniform-time centered second-derivative operator."""
    times = np.asarray(timesteps, dtype=np.float64)
    if times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all():
        raise ValueError('At least three finite one-dimensional timesteps required')
    gaps = np.diff(times)
    if np.any(gaps <= 0.):
        raise ValueError('Timesteps must be strictly increasing')
    operator = np.zeros((len(times) - 2, len(times)), dtype=np.float64)
    for row, (left, right) in enumerate(zip(gaps[:-1], gaps[1:])):
        operator[row, row] = 2. / (left * (left + right))
        operator[row, row + 1] = -2. / (left * right)
        operator[row, row + 2] = 2. / (right * (left + right))
    return operator


def _vertices(value) -> np.ndarray:
    vertices = np.asarray(value)
    if (vertices.ndim != 3 or vertices.shape[-1] != 3 or vertices.shape[0] < 3
            or vertices.shape[1] < 3 or vertices.dtype != np.dtype(np.float32)
            or not np.isfinite(vertices).all()):
        raise ValueError('Finite float32 vertices[T,V,3] with T,V >= 3 required')
    return vertices


def repair_quadratic_acceleration(vertices, timesteps, weight: float):
    """Use a direct dense solve for the anchored identity-metric control."""
    source = _vertices(vertices)
    if isinstance(weight, (bool, np.bool_)) or not np.isfinite(weight) or weight <= 0.:
        raise ValueError('Finite positive acceleration weight required')
    operator = timestamp_second_difference(timesteps)
    if operator.shape[1] != source.shape[0]:
        raise ValueError('One supplied timestamp is required for every frame')
    source64 = source.astype(np.float64)
    system = np.eye(len(source64), dtype=np.float64) + weight * (operator.T @ operator)
    free = system[1:, 1:]
    right_hand = source64[1:].reshape(len(source64) - 1, -1)
    right_hand -= system[1:, :1] @ source64[:1].reshape(1, -1)
    repaired64 = np.empty_like(source64)
    repaired64[0] = source64[0]
    repaired64[1:] = np.linalg.solve(free, right_hand).reshape(source64[1:].shape)
    solution = repaired64[1:].reshape(len(source64) - 1, -1)
    residual = free @ solution - right_hand
    scale = (np.linalg.norm(free, ord=np.inf) * np.linalg.norm(solution, ord=np.inf)
             + np.linalg.norm(right_hand, ord=np.inf))
    backward_error = float(np.linalg.norm(residual, ord=np.inf) / max(scale, np.finfo(np.float64).tiny))
    stationarity_tolerance = float(100 * len(source64) * np.finfo(np.float64).eps)
    condition = float(np.linalg.cond(free))
    condition_max = float(1. / np.sqrt(np.finfo(np.float64).eps))
    if (not np.isfinite(backward_error) or backward_error > stationarity_tolerance
            or not np.isfinite(condition) or condition > condition_max):
        raise ValueError(
            'Quadratic control linear system failed numerical admission: '
            f'backward_error={backward_error:.17g}, '
            f'stationarity_tolerance={stationarity_tolerance:.17g}, '
            f'condition_2={condition:.17g}, condition_2_max={condition_max:.17g}')
    before = operator @ source64.reshape(len(source64), -1)
    after = operator @ repaired64.reshape(len(source64), -1)
    repaired = repaired64.astype(source.dtype)
    repaired[0] = source[0]
    if not np.isfinite(repaired).all():
        raise ValueError('Quadratic acceleration solve emitted nonfinite vertices')
    diagnostics = {
        'weight': float(weight),
        'frames': int(len(source)),
        'vertices': int(source.shape[1]),
        'acceleration_l2_before': float(np.linalg.norm(before)),
        'acceleration_l2_after': float(np.linalg.norm(after)),
        'displacement_l2': float(np.linalg.norm(repaired64 - source64)),
        'linear_residual_linf': float(np.max(np.abs(residual))),
        'relative_backward_error': backward_error,
        'stationarity_tolerance': stationarity_tolerance,
        'free_system_condition_2': condition,
        'condition_2_max': condition_max,
        'anchor_exact': bool(np.array_equal(repaired[0], source[0])),
    }
    if not all(np.isfinite(value) for key, value in diagnostics.items()
               if key not in ('frames', 'vertices', 'anchor_exact')):
        raise ValueError('Nonfinite quadratic solve diagnostic')
    return repaired, diagnostics


def export_quadratic_control(source_case: Path, output: Path, *, uid: str,
                             expected_sequence_sha256: str, weight: float):
    """Export one full control arm consumable by the official score adapter."""
    started = time.monotonic()
    source_case, output = Path(source_case), Path(output)
    if not uid or Path(uid).name != uid or uid in ('.', '..'):
        raise ValueError('Invalid native UID')
    source_path = source_case / 'sequence.npz'
    actual_sha = _digest(source_path)
    if actual_sha != expected_sequence_sha256:
        raise ValueError('Sequence differs from explicitly pinned input')
    source_report_path = source_case / 'report.json'
    source_report = json.loads(source_report_path.read_text())
    if (source_report.get('status') != 'completed' or source_report.get('uid') != uid
            or source_report.get('sha256', {}).get('sequence.npz') != actual_sha):
        raise ValueError('Completed same-UID source report with current sequence hash required')
    seed = source_report.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Source inference seed must be an integer')
    with np.load(source_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    required = {'vertices', 'faces', 'timesteps', 'frame_indices', 'query_vertex_ids'}
    if not required.issubset(arrays):
        raise ValueError('Full native mesh metadata required')
    vertices = _vertices(arrays['vertices'])
    if (len(vertices) != 16 or not np.issubdtype(arrays['frame_indices'].dtype, np.integer)
            or not np.array_equal(arrays['frame_indices'], np.arange(16))):
        raise ValueError('Exactly 16 original frames in original order required')
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
    repaired, diagnostics = repair_quadratic_acceleration(vertices, arrays['timesteps'], weight)
    output.mkdir(parents=True, exist_ok=False)
    arm = output / 'quadratic_acceleration'
    arm.mkdir()
    np.savez_compressed(arm / 'sequence.npz', **{**arrays, 'vertices': repaired})
    arm_report = {
        'status': 'completed', 'uid': uid, 'seed': seed,
        'baseline_arm': 'quadratic_acceleration', 'objective': OBJECTIVE,
        'observation_metric': 'identity',
        'timestamp_units': 'supplied_sequence_units',
        'weight_units': 'supplied_sequence_units^4',
        'source_sequence_sha256': actual_sha,
        'source_report_sha256': _digest(source_report_path),
        'implementation_sha256': _digest(Path(__file__)),
        'sha256': {'sequence.npz': _digest(arm / 'sequence.npz')},
        'diagnostics': diagnostics,
        'native_qualified': False, 'candidate_method': False,
        'information': 'Full predicted sequence only; no GT, scorer state, labels, cameras, or learned weights',
    }
    _write_json(arm / 'report.json', arm_report)
    cases = [{'case_id': uid + '-quadratic_acceleration', 'uid': uid,
              'case_dir': 'quadratic_acceleration'}]
    _write_json(output / 'manifest.json', {
        'cases': cases,
        'scope': 'C13 strong simple comparator; not the group-trend candidate or scientific qualification',
    })
    result = {
        'status': 'completed', 'recorded_at': datetime.now(timezone.utc).isoformat(),
        'uid': uid, 'seed': seed, 'arms': [arm_report],
        'native_qualified': False, 'candidate_methods_tested': False,
        'elapsed_seconds': time.monotonic() - started,
        'timing_scope': 'Adapter I/O + quadratic solve only; excludes generation, official scorer and collection',
    }
    _write_json(output / 'controls.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-case', type=Path)
    source.add_argument('--source-sequence', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--expected-sequence-sha256', required=True)
    parser.add_argument('--weight', required=True, type=float)
    args = parser.parse_args()
    if args.source_sequence is not None and args.source_sequence.name != 'sequence.npz':
        parser.error('--source-sequence must name sequence.npz')
    source_case = args.source_case if args.source_case is not None else args.source_sequence.parent
    result = export_quadratic_control(source_case, args.output, uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256, weight=args.weight)
    print(json.dumps({'status': result['status'], 'native_qualified': False,
                      'candidate_methods_tested': False}))
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
