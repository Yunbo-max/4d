"""Ordinary mesh baselines for qualification; no new candidate or scorer.

Offline Gaussian smoothing in world coordinates and after classical, uniform
vertex-weight Procrustes pose factoring. Uses only a full predicted sequence.
No GT, camera, evaluator transform, video labels, or learned parameters enter.
These baselines may erase real articulation; native qualification is separate.
Run this module only inside the reviewed research-autopilot execution harness.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np


def gaussian_weights(frames: int, sigma: float) -> np.ndarray:
    """Truncated (4 sigma), row-normalized discrete Gaussian; no wrapping/padding."""
    if frames < 1 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('Positive frame count and finite positive sigma required')
    delta = np.arange(frames)[:, None] - np.arange(frames)[None, :]
    weights = np.exp(-.5 * (delta / sigma)**2)
    weights[np.abs(delta) > 4.*sigma] = 0.
    return weights / weights.sum(axis=1, keepdims=True)


def _vertices(sequence):
    array = np.asarray(sequence)
    if array.ndim != 3 or array.shape[-1] != 3 or array.shape[0] < 1 or array.shape[1] < 3:
        raise ValueError('Expected vertices[T,V,3] with at least three vertices')
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError('Finite floating-point vertices required')
    return array


def smooth_world(sequence, sigma: float):
    """Smooth anchored displacements, then restore frame zero exactly."""
    source = _vertices(sequence)
    weights = gaussian_weights(len(source), sigma)
    anchor = source[0].astype(np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        residual = source.astype(np.float64)-anchor
        out = (anchor + np.einsum('ts,svc->tvc', weights, residual)).astype(source.dtype)
    out[0] = source[0]
    if not np.isfinite(out).all(): raise ValueError('Nonfinite world reconstruction')
    return out


def smooth_body(sequence, sigma: float):
    """Fit proper rigid poses to frame zero; smooth only body residuals.

    Row convention: Y_t ~= X @ R_t + c_t. SO(3) fit minimizes uniform vertex
    squared error. No scale fit; fitted centroids/rotations remain unchanged.
    Require rank >=2 of each cross-covariance (relative tolerance 1e-8).
    Reflection correction also requires a unique smallest singular direction.
    Degenerate/ambiguous fits fail explicitly, without a hidden fallback.
    """
    source = _vertices(sequence)
    weights = gaussian_weights(len(source), sigma)
    y = source.astype(np.float64)
    centroids = y.mean(axis=1)
    centered = y-centroids[:, None, :]
    anchor = centered[0]
    rotations, singular_values = [], []
    for frame in range(len(source)):
        u, s, vt = np.linalg.svd(anchor.T @ centered[frame])
        if s[0] <= np.finfo(np.float64).tiny or s[1] <= 1e-8*s[0]:
            raise ValueError(f'Ambiguous rank-deficient body pose at frame {frame}')
        orientation = np.linalg.det(u @ vt)
        if orientation < 0. and s[1]-s[2] <= 1e-8*s[0]:
            raise ValueError(f'Ambiguous reflection-corrected body pose at frame {frame}')
        signs = np.ones(3); signs[-1] = 1. if orientation >= 0. else -1.
        rotations.append((u*signs) @ vt)
        singular_values.append(s)
    rotations = np.stack(rotations)
    rotations[0] = np.eye(3)  # Anchor frame is the body coordinate definition.
    body = np.einsum('tvi,tji->tvj', centered, rotations)
    residual = body-anchor
    residual[0] = 0.
    filtered = np.einsum('ts,svc->tvc', weights, residual)
    out = (np.einsum('tvi,tij->tvj', anchor+filtered, rotations)+centroids[:, None, :]).astype(source.dtype)
    out[0] = source[0]
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite body reconstruction')
    return out, {'rotation_rows': rotations, 'centroids': centroids, 'singular_values': np.stack(singular_values)}


def _digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def _json(path: Path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def export_controls(source_case: Path, output: Path, *, uid: str, expected_sequence_sha256: str, sigma: float):
    """Export one asset's three full mesh arms, retaining missing/failed arms.

    A matching hash validates input identity, not the source's scientific
    qualification. The generated manifest is directly readable by the existing
    research_census_eval.py. Output directory is single use.
    """
    started = time.monotonic()
    source_case, output = Path(source_case), Path(output)
    if not uid or Path(uid).name != uid or uid in ('.', '..'):
        raise ValueError('Invalid native UID')
    source_path = source_case/'sequence.npz'
    actual_sha = _digest(source_path)
    if actual_sha != expected_sequence_sha256:
        raise ValueError('Sequence differs from explicitly pinned input')
    source_report_path = source_case/'report.json'
    source_report = json.loads(source_report_path.read_text())
    if source_report.get('status') != 'completed' or source_report.get('uid') != uid:
        raise ValueError('Source report must be completed for the same native UID')
    if source_report.get('sha256', {}).get('sequence.npz') != actual_sha:
        raise ValueError('Sequence differs from generator report')
    seed = source_report.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Source inference seed must be an integer')
    with np.load(source_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    required = {'vertices', 'faces', 'timesteps', 'frame_indices', 'query_vertex_ids'}
    if not required.issubset(arrays): raise ValueError('Full native mesh metadata required')
    vertices = _vertices(arrays['vertices'])
    faces = arrays['faces']
    if vertices.shape[0] != 16 or not np.issubdtype(arrays['frame_indices'].dtype, np.integer) or not np.array_equal(arrays['frame_indices'], np.arange(16)):
        raise ValueError('Exactly 16 original frames in original order required')
    if not np.array_equal(arrays['timesteps'], np.arange(16)):
        raise ValueError('Expected original directory-loader frame timestamps 0..15')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= vertices.shape[1]:
        raise ValueError('Invalid shared triangle topology')
    if not np.issubdtype(arrays['query_vertex_ids'].dtype, np.integer) or not np.array_equal(arrays['query_vertex_ids'], np.arange(vertices.shape[1])):
        raise ValueError('Original identity vertex mapping required')
    for key, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f'Non-numeric/nonfinite native array: {key}')
    gaussian_weights(16, sigma)  # Reject parameters before creating outputs.
    output.mkdir(parents=True, exist_ok=False)
    names = ['native', 'world_gaussian', 'body_gaussian']
    _json(output/'manifest.json', {'cases': [{'case_id': uid+'-'+name, 'uid': uid, 'case_dir': name} for name in names],
                                  'scope': 'Baseline qualification assets; not independent arms-as-assets or confirmation'})
    records = []
    for name in names:
        arm_start = time.monotonic()
        directory = output/name; directory.mkdir()
        report = {'uid': uid, 'seed': seed, 'baseline_arm': name, 'sigma_frames': sigma,
                  'source_sequence_sha256': actual_sha, 'source_report_sha256': _digest(source_report_path),
                  'implementation_sha256': _digest(Path(__file__)), 'native_qualified': False,
                  'information': 'Full predicted mesh sequence only; no GT, scorer ICP, labels, cameras, or learned weights'}
        try:
            if name == 'native':
                shutil.copyfile(source_path, directory/'sequence.npz')
            else:
                if name == 'world_gaussian': repaired = smooth_world(vertices, sigma)
                else:
                    repaired, poses = smooth_body(vertices, sigma)
                    np.savez_compressed(directory/'poses.npz', **poses)
                    report['sha256'] = {'poses.npz': _digest(directory/'poses.npz')}
                if repaired.shape != vertices.shape or not np.array_equal(repaired[0], vertices[0]):
                    raise ValueError('Control changed topology/frame count or fixed anchor')
                np.savez_compressed(directory/'sequence.npz', **{**arrays, 'vertices': repaired})
            report.setdefault('sha256', {})['sequence.npz'] = _digest(directory/'sequence.npz')
            report['status'] = 'completed'
        except Exception as error:
            report.update(status='error', exception_type=type(error).__name__, error=str(error))
        report['elapsed_seconds'] = time.monotonic()-arm_start
        _json(directory/'report.json', report)
        records.append(report)
    summary = {'status': 'completed' if all(r['status'] == 'completed' for r in records) else 'incomplete',
               'recorded_at': datetime.now(timezone.utc).isoformat(), 'uid': uid, 'seed': seed,
               'arms': records, 'native_qualified': False, 'candidate_methods_tested': False,
               'elapsed_seconds': time.monotonic()-started,
               'timing_scope': 'Adapter I/O + mesh repair only; excludes generation, scorer and collection; NOT full multi-arm cost'}
    _json(output/'controls.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-case', type=Path)
    source.add_argument('--source-sequence', type=Path, help='Explicit pinned file path for harness staging')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--uid', required=True)
    parser.add_argument('--expected-sequence-sha256', required=True)
    parser.add_argument('--sigma', required=True, type=float)
    args = parser.parse_args()
    if args.source_sequence is not None and args.source_sequence.name != 'sequence.npz':
        parser.error('--source-sequence must name sequence.npz')
    source_case = args.source_case if args.source_case is not None else args.source_sequence.parent
    result = export_controls(source_case, args.output, uid=args.uid,
                             expected_sequence_sha256=args.expected_sequence_sha256, sigma=args.sigma)
    print(json.dumps({'status': result['status'], 'native_qualified': False}))
    return 0 if result['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
