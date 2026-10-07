"""Explicit engineering-only compatibility backend for PyTorch3D KNN.

Keep the installed CUDA forward, neighbor indices and official loss unchanged.
Dispatch only KNN backward to the installed upstream serial CPU primitive.
This changes floating-point accumulation order and is NOT an assertion of
equivalence to an arbitrary atomic CUDA run or an accepted benchmark protocol.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path


@contextmanager
def cpu_knn_backward():
    import torch
    from pytorch3d import _C

    original = _C.knn_points_backward
    metadata = {
        'kind': 'experimental-knn-backward-cpu-compatibility',
        'scope': 'engineering-only; native qualification remains false',
        'forward': 'unchanged installed PyTorch3D CUDA KNN',
        'backward': 'installed upstream PyTorch3D CPU KNN primitive',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'extension_sha256': hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest(),
        'cpu_backward_calls': 0,
    }

    def backward(p1, p2, lengths1, lengths2, idx, norm, grad_dists):
        if not p1.is_cuda:
            return original(p1, p2, lengths1, lengths2, idx, norm, grad_dists)
        if norm not in (1, 2) or p1.dtype != torch.float32 or p2.dtype != torch.float32:
            raise ValueError('Unsupported PyTorch3D KNN backward contract')
        if not all(x.device == p1.device for x in (p2, lengths1, lengths2, idx, grad_dists)):
            raise ValueError('All CUDA KNN arguments must use the same device')
        grads = original(p1.cpu(), p2.cpu(), lengths1.cpu(), lengths2.cpu(),
                         idx.cpu(), norm, grad_dists.cpu())
        metadata['cpu_backward_calls'] += 1
        return tuple(x.to(p1.device) for x in grads)

    _C.knn_points_backward = backward
    try:
        yield metadata
    finally:
        _C.knn_points_backward = original


def run_census():
    """Run the unchanged scorer under strict guards and record the extra patch."""
    import json
    import os
    import sys
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import research_census_eval
    torch.use_deterministic_algorithms(True)
    from research_math.knn_backend_checks import check
    regression = check()
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    with cpu_knn_backward() as metadata:
        metadata['regression_checks'] = regression
        try:
            return research_census_eval.main()
        finally:
            if output.exists():
                report = json.loads(output.read_text())
                report['additional_runtime_compatibility'] = metadata
                output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    raise SystemExit(run_census())
