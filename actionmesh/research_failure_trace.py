"""Record contributions of the unchanged official ActionBench scorer.

This is instrumentation, not a replacement metric. Every scalar is returned by
the original function. Motion contributions reuse its exact first-frame KDTree
matching convention and must reduce to that original scalar within float error.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib

import numpy as np

KEYS = ('cd_3d', 'cd_4d', 'cd_motion')


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


@contextmanager
def trace_official(benchmark, folder, frames=16):
    """Wrap actual native calls without changing their arguments or returns."""
    originals = {name: getattr(benchmark, name) for name in (
        'compute_chamfer_score', 'compute_motion_chamfer_score',
        '_compute_per_frame_icp', '_compute_unified_icp')}
    trace = {'geometry_calls': [], 'motion': None, 'matrices': {}}

    def shape(*args, **kwargs):
        value = originals['compute_chamfer_score'](*args, **kwargs)
        trace['geometry_calls'].append(float(value))
        return value

    def motion(*args, **kwargs):
        from scipy.spatial import KDTree
        value = originals['compute_motion_chamfer_score'](*args, **kwargs)
        preds = np.asarray(kwargs['preds'] if 'preds' in kwargs else args[0])
        gts = np.asarray(kwargs['gts'] if 'gts' in kwargs else args[1])
        if preds.shape != (frames, 100000, 3) or gts.shape != preds.shape:
            raise ValueError('Native synchronized motion sampling contract changed')
        _, mu = KDTree(preds[0]).query(gts[0])
        _, nu = KDTree(gts[0]).query(preds[0])
        # Native CD-M averages across time, then points. Only expose the time
        # contributions; do not call these velocity/acceleration errors.
        first = np.linalg.norm(preds[:, mu, :] - gts, axis=-1).mean(axis=1)
        second = np.linalg.norm(gts[:, nu, :] - preds, axis=-1).mean(axis=1)
        per_frame = first + second
        if not np.isclose(per_frame.mean(), value, rtol=1e-6, atol=1e-6):
            raise ValueError('Motion contribution reduction differs from native scalar')
        path = folder / 'motion-matches.npz'
        np.savez_compressed(path, gt_to_pred=mu, pred_to_gt=nu)
        trace['motion'] = {
            'per_frame': per_frame.tolist(), 'gt_to_pred': first.tolist(),
            'pred_to_gt': second.tolist(), 'native_scalar': float(value),
            'reduction_abs_difference': abs(float(per_frame.mean()) - float(value)),
            'sampled_pred_sha256': array_hash(preds),
            'gt_sha256': array_hash(gts), 'matches_file': path.name,
            'scope': 'Native CD-M position-distance contributions, not pure motion error',
        }
        return value

    def per_frame_icp(*args, **kwargs):
        result = originals['_compute_per_frame_icp'](*args, **kwargs)
        trace['matrices']['per_frame'] = np.concatenate([
            item.get_matrix().detach().cpu().numpy() for item in result], axis=0)
        return result

    def unified_icp(*args, **kwargs):
        result = originals['_compute_unified_icp'](*args, **kwargs)
        trace['matrices']['unified'] = result.get_matrix().detach().cpu().numpy()
        return result

    benchmark.compute_chamfer_score = shape
    benchmark.compute_motion_chamfer_score = motion
    benchmark._compute_per_frame_icp = per_frame_icp
    benchmark._compute_unified_icp = unified_icp
    try:
        yield trace
    finally:
        for name, original in originals.items():
            setattr(benchmark, name, original)


def finish_trace(trace, metrics, folder, frames=16):
    calls = trace.pop('geometry_calls')
    matrices = trace.pop('matrices')
    if len(calls) != 2 * frames or trace['motion'] is None:
        raise ValueError('Expected exactly 16 CD3D, 16 CD4D calls and one CD-M call')
    if matrices['per_frame'].shape != (frames, 4, 4) or matrices['unified'].shape != (1, 4, 4):
        raise ValueError('Native ICP matrix timeline changed')
    per_frame = {'cd_3d': calls[:frames], 'cd_4d': calls[frames:],
                 'cd_motion': trace['motion']['per_frame']}
    reductions = {}
    for name, values in per_frame.items():
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Nonfinite/negative native frame contribution')
        reductions[name] = abs(float(np.mean(values)) - metrics[name])
        if not np.isclose(np.mean(values), metrics[name], rtol=1e-6, atol=1e-6):
            raise ValueError('Per-frame reduction differs from native metric: ' + name)
    np.savez_compressed(folder / 'icp-matrices.npz', **matrices)
    return dict(trace, per_frame=per_frame, reduction_abs_differences=reductions), matrices['unified']
