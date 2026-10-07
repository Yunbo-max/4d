"""Small real-CUDA regression, executed only inside the engineering harness."""
import torch
from pytorch3d import _C
from pytorch3d.ops import knn_points
from research_math.deterministic_knn import cpu_knn_backward


def check():
    original = _C.knn_points_backward
    # Two source points map to the same target: exercise target-gradient accumulation.
    source = torch.tensor([[[1., 2., 3.], [2., 3., 4.]]], device='cuda')
    target = torch.tensor([[[0., 0., 0.], [100., 100., 100.]]], device='cuda')
    raw_forward = knn_points(source, target).dists.detach().clone()
    torch.use_deterministic_algorithms(True)
    for norm in (1, 2):
        repeated = []
        for _ in range(3):
            a, b = source.clone().requires_grad_(), target.clone().requires_grad_()
            with cpu_knn_backward() as metadata:
                out = knn_points(a, b, norm=norm)
                out.dists.sum().backward()
                assert metadata['cpu_backward_calls'] == 1
                if norm == 2:
                    assert torch.equal(out.dists, raw_forward), 'CUDA forward changed'
            assert _C.knn_points_backward is original, 'Backend patch leaked'
            expected_a = (2*source if norm == 2 else torch.ones_like(source))
            expected_b = torch.zeros_like(target)
            expected_b[:, 0] = -expected_a.sum(dim=1)
            assert torch.equal(a.grad, expected_a), 'Wrong source-point derivative'
            assert torch.equal(b.grad, expected_b), 'Wrong colliding target derivative'
            repeated.append((out.dists.detach(), a.grad, b.grad))
        assert all(all(torch.equal(x,y) for x,y in zip(repeated[0],r)) for r in repeated[1:])
    try:
        with cpu_knn_backward():
            raise RuntimeError('test restoration')
    except RuntimeError:
        pass
    assert _C.knn_points_backward is original, 'Exception path leaked backend patch'
    return {'analytic_gradients': 'passed norms 1 and 2',
            'forward_unchanged': True, 'bitwise_repeats': 3,
            'restores_backend_on_exception': True}


if __name__ == '__main__':
    import json
    print(json.dumps(check()))
