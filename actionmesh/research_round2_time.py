#!/usr/bin/env python3
"""Temporal and fixed-spatial-support action disagreement, using only PyTorch.

This module scores an existing local/global edit decision. It does not select
coverage, calibrate uncertainty, modify the animation, or call a neural model.
Rotations use row vectors throughout: x_b = x_a @ R_ab + translation.
"""
from __future__ import annotations

import math

import torch


def _rotation(kabsch_rows, source, target):
    result = kabsch_rows(source, target)
    return result[0] if isinstance(result, (tuple, list)) else result


def _support_splits(count, device):
    """Three unique, overlapping half supports; no bootstrap independence claim."""
    if count < 8 or count % 4:
        raise ValueError("Support size must be at least 8 and divisible by 4")
    positions = torch.arange(count, device=device)
    return torch.stack((positions[::2], positions[:count // 2],
                        positions[(positions % 4) < 2]))


@torch.no_grad()
def build_cache(observed, query, fit_indices, global_fit, local_rot, global_rot,
                kabsch_rows):
    """Cache independently fitted paths and alternative spatial support fits.

    Args:
        observed: Floating tensor [T,N,3], on the desired compute device.
        query: Vertex IDs [Q]; identifies the rows represented by local_rot.
        fit_indices: Material-neighbor IDs [Q,K], ordinarily K=32.
        global_fit: Global material support IDs [G], ordinarily G=512.
        local_rot: Already fitted direct 0->t rotations [T,Q,3,3].
        global_rot: Already fitted direct 0->t rotations [T,3,3].
        kabsch_rows: Callable returning a rotation or (rotation, ...).

    Every m->t edge is fitted directly to observed geometry. The 0->m edge
    reuses the supplied independent direct fit, not a synchronized pose graph.
    Invalid m=t paths and all frame-zero paths are excluded by temporal_valid.
    Out-of-range intermediate frames are omitted, permitting shorter clips.

    The fixed spatial splits use even support positions, the first half, and
    positions mod 4 < 2. They overlap and are not independent bootstrap samples.
    All variants use three alternatives when available. Exact extra Kabsch
    counts differ at excluded temporal self-edges and are exposed in fit_counts.
    """
    if observed.ndim != 3 or observed.shape[-1] != 3 or not observed.is_floating_point():
        raise ValueError("observed must be a floating tensor [T,N,3]")
    frames, vertices, _ = observed.shape
    if frames < 1:
        raise ValueError("observed must contain an anchor frame")
    device, dtype = observed.device, observed.dtype
    query = torch.as_tensor(query, dtype=torch.long, device=device)
    fit_indices = torch.as_tensor(fit_indices, dtype=torch.long, device=device)
    global_fit = torch.as_tensor(global_fit, dtype=torch.long, device=device)
    if query.ndim != 1 or query.numel() < 1:
        raise ValueError("query must be nonempty [Q]")
    queries = query.numel()
    if fit_indices.ndim != 2 or fit_indices.shape[0] != queries:
        raise ValueError("fit_indices must have shape [Q,K]")
    if global_fit.ndim != 1:
        raise ValueError("global_fit must have shape [G]")
    for name, indices in (("query", query), ("fit_indices", fit_indices),
                          ("global_fit", global_fit)):
        if indices.numel() and (bool((indices < 0).any()) or bool((indices >= vertices).any())):
            raise ValueError(f"{name} contains out-of-range vertex IDs")
    if local_rot.shape != (frames, queries, 3, 3):
        raise ValueError("local_rot must have shape [T,Q,3,3]")
    if global_rot.shape != (frames, 3, 3):
        raise ValueError("global_rot must have shape [T,3,3]")
    if any(r.device != device or r.dtype != dtype for r in (local_rot, global_rot)):
        raise ValueError("observed and supplied rotations must share dtype/device")

    local_positions = _support_splits(fit_indices.shape[1], device)
    global_positions = _support_splits(global_fit.numel(), device)
    local_split_indices = fit_indices[:, local_positions].permute(1, 0, 2)
    global_split_indices = global_fit[global_positions]
    intermediates = tuple(m for m in (4, 8, 12) if m < frames)
    path_count = len(intermediates)
    identity = torch.eye(3, device=device, dtype=dtype)
    temporal_local = identity.expand(frames, path_count, queries, 3, 3).clone()
    temporal_global = identity.expand(frames, path_count, 3, 3).clone()
    temporal_valid = torch.zeros(frames, path_count, device=device, dtype=torch.bool)
    temporal_pairs = 0
    for path, intermediate in enumerate(intermediates):
        local_source = observed[intermediate, fit_indices]
        global_source = observed[intermediate, global_fit]
        for target in range(1, frames):
            if target == intermediate:
                continue
            local_edge = _rotation(kabsch_rows, local_source, observed[target, fit_indices])
            global_edge = _rotation(kabsch_rows, global_source, observed[target, global_fit])
            temporal_local[target, path] = local_rot[intermediate] @ local_edge
            temporal_global[target, path] = global_rot[intermediate] @ global_edge
            temporal_valid[target, path] = True
            temporal_pairs += 1

    spatial_local = identity.expand(frames, 3, queries, 3, 3).clone()
    spatial_global = identity.expand(frames, 3, 3, 3).clone()
    for support in range(3):
        local_ids, global_ids = local_split_indices[support], global_split_indices[support]
        local_source, global_source = observed[0, local_ids], observed[0, global_ids]
        for target in range(1, frames):
            spatial_local[target, support] = _rotation(
                kabsch_rows, local_source, observed[target, local_ids]
            )
            spatial_global[target, support] = _rotation(
                kabsch_rows, global_source, observed[target, global_ids]
            )
    return {
        "local_rot": local_rot.detach(), "global_rot": global_rot.detach(),
        "temporal_local": temporal_local, "temporal_global": temporal_global,
        "temporal_valid": temporal_valid, "intermediates": intermediates,
        "spatial_local": spatial_local, "spatial_global": spatial_global,
        "query": query, "local_split_indices": local_split_indices,
        "global_split_indices": global_split_indices,
        "fit_counts": {
            "temporal_local_batched_calls": temporal_pairs,
            "temporal_local_rotations": temporal_pairs * queries,
            "temporal_global_calls": temporal_pairs,
            "spatial_local_batched_calls": 3 * (frames - 1),
            "spatial_local_rotations": 3 * (frames - 1) * queries,
            "spatial_global_calls": 3 * (frames - 1),
            "direct_rotations_reused": True,
        },
    }


def _median(values, valid):
    """Conventional median across support axis 1, excluding invalid paths."""
    if values.shape[1] == 0:
        return values.new_zeros((values.shape[0], values.shape[2]))
    ordered = values.masked_fill(~valid[..., None], float("inf")).sort(dim=1).values
    counts = valid.sum(dim=1)
    safe_count = counts.clamp_min(1)
    lo, hi = (safe_count - 1) // 2, safe_count // 2
    lo = lo[:, None, None].expand(-1, 1, values.shape[2])
    hi = hi[:, None, None].expand(-1, 1, values.shape[2])
    median = 0.5 * (ordered.gather(1, lo) + ordered.gather(1, hi)).squeeze(1)
    return torch.where(counts[:, None] > 0, median, torch.zeros_like(median))


def _action_difference(d0, local_delta, global_delta):
    local = torch.einsum("qi,tmqij->tmqj", d0, local_delta).norm(dim=-1)
    global_ = torch.einsum("qi,tmij->tmqj", d0, global_delta).norm(dim=-1)
    magnitude = d0.norm(dim=-1)
    denominator = magnitude.clamp_min(torch.finfo(d0.dtype).tiny)
    active = magnitude > 0
    return (torch.where(active, local / denominator, torch.zeros_like(local)),
            torch.where(active, global_ / denominator, torch.zeros_like(global_)))


@torch.no_grad()
def compute(cache, d0):
    """Return signed [T,Q] scores: larger positive values favor global fallback.

    temporal_action and spatial_action compare direction-projected rotation
    disagreement, normalized by edit magnitude. temporal_matrix compares full
    Frobenius rotation disagreement without consulting the edit direction.
    All use median(local disagreement) - median(global disagreement). No score
    is a probability, confidence bound, or calibrated estimate of edit error.
    """
    local_rot, global_rot = cache["local_rot"], cache["global_rot"]
    frames, queries = local_rot.shape[:2]
    if d0.shape != (queries, 3):
        raise ValueError("d0 must have shape [Q,3]")
    if d0.device != local_rot.device or d0.dtype != local_rot.dtype:
        raise ValueError("d0 and cached rotations must share dtype/device")
    temporal_local_delta = local_rot[:, None] - cache["temporal_local"]
    temporal_global_delta = global_rot[:, None] - cache["temporal_global"]
    local_action, global_action = _action_difference(
        d0, temporal_local_delta, temporal_global_delta
    )
    valid = cache["temporal_valid"]
    temporal_action = _median(local_action, valid) - _median(global_action, valid)
    local_matrix = temporal_local_delta.square().sum(dim=(-1, -2)).sqrt()
    global_matrix = temporal_global_delta.square().sum(dim=(-1, -2)).sqrt()
    temporal_matrix = _median(local_matrix, valid) - _median(
        global_matrix[..., None].expand(-1, -1, queries), valid
    )
    local_spatial, global_spatial = _action_difference(
        d0, local_rot[:, None] - cache["spatial_local"],
        global_rot[:, None] - cache["spatial_global"]
    )
    spatial_valid = torch.ones(frames, 3, dtype=torch.bool, device=d0.device)
    spatial_action = _median(local_spatial, spatial_valid) - _median(global_spatial, spatial_valid)
    result = {
        "temporal_action": temporal_action, "temporal_matrix": temporal_matrix,
        "spatial_action": spatial_action,
    }
    for score in result.values():
        score[0] = 0
    return result


def selftest():
    """CPU-only rigid control and hand-derived nontrivial score checks."""
    torch.manual_seed(1729)
    dtype = torch.float64
    points = torch.randn(128, 3, dtype=dtype)

    def rotation(axis, angle):
        c, s = math.cos(angle), math.sin(angle)
        if axis == 0:
            return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=dtype)
        if axis == 1:
            return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=dtype)
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=dtype)

    def kabsch_rows(x, y):
        xc, yc = x - x.mean(-2, keepdim=True), y - y.mean(-2, keepdim=True)
        u, singular, vh = torch.linalg.svd(xc.transpose(-1, -2) @ yc)
        signs = torch.ones_like(singular)
        signs[..., -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0)
        return (u * signs.unsqueeze(-2)) @ vh, None, singular

    rotations = torch.stack([
        rotation(2, t * 0.09) @ rotation(0, math.sin(t * 0.2) * 0.4)
        for t in range(16)
    ])
    translations = torch.arange(16, dtype=dtype)[:, None] * torch.tensor(
        [[0.02, -0.01, 0.03]], dtype=dtype
    )
    observed = points[None] @ rotations + translations[:, None]
    query = torch.arange(8)
    fit = torch.stack([torch.arange(32) + i * 3 for i in range(len(query))])
    global_fit = torch.arange(96)
    local_rot = torch.stack([
        kabsch_rows(observed[0, fit], observed[t, fit])[0] for t in range(16)
    ])
    global_rot = torch.stack([
        kabsch_rows(observed[0, global_fit], observed[t, global_fit])[0]
        for t in range(16)
    ])
    cache = build_cache(observed, query, fit, global_fit, local_rot, global_rot, kabsch_rows)
    d0 = torch.randn(len(query), 3, dtype=dtype)
    scores = compute(cache, d0)
    assert set(scores) == {"temporal_action", "temporal_matrix", "spatial_action"}
    for name, score in scores.items():
        assert score.shape == (16, len(query)), (name, score.shape)
        assert torch.isfinite(score).all(), name
        assert score.abs().max() < 1e-11, (name, score.abs().max().item())
        assert torch.count_nonzero(score[0]) == 0, name
    assert cache["temporal_valid"][4].sum().item() == 2
    assert cache["temporal_valid"][1].sum().item() == 3

    # Break caught: including the invalid m=t path would lower this median;
    # lower-median selection would also give the wrong value for two paths.
    fixture = {key: value.clone() if isinstance(value, torch.Tensor) else value
               for key, value in cache.items()}
    identity = torch.eye(3, dtype=dtype)
    fixture["local_rot"][4] = identity
    fixture["global_rot"][4] = identity
    fixture["temporal_global"][4] = identity
    fixture["temporal_local"][4, 0] = identity  # Invalid path m=t=4.
    fixture["temporal_local"][4, 1] = rotation(0, math.pi / 2)
    fixture["temporal_local"][4, 2] = rotation(0, math.pi)
    fixture["spatial_global"][4] = identity
    fixture["spatial_local"][4, 0] = identity
    fixture["spatial_local"][4, 1] = rotation(0, math.pi / 2)
    fixture["spatial_local"][4, 2] = rotation(0, math.pi)
    y_edit = torch.tensor([[0.0, 1.0, 0.0]], dtype=dtype).expand(len(query), -1)
    checked = compute(fixture, y_edit)
    assert torch.allclose(checked["temporal_action"][4],
                          torch.full((len(query),), (math.sqrt(2) + 2) / 2, dtype=dtype))
    assert torch.allclose(checked["temporal_matrix"][4],
                          torch.full((len(query),), 1 + math.sqrt(2), dtype=dtype))
    assert torch.allclose(checked["spatial_action"][4],
                          torch.full((len(query),), math.sqrt(2), dtype=dtype))
    x_edit = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype).expand(len(query), -1)
    checked_x = compute(fixture, x_edit)
    assert torch.count_nonzero(checked_x["temporal_action"][4]) == 0
    assert torch.count_nonzero(checked_x["spatial_action"][4]) == 0
    assert torch.equal(checked_x["temporal_matrix"], checked["temporal_matrix"])
    zero_edit = compute(fixture, torch.zeros_like(d0))
    assert torch.count_nonzero(zero_edit["temporal_action"]) == 0
    assert torch.count_nonzero(zero_edit["spatial_action"]) == 0
    return {
        "status": "passed", "device": "cpu", "dtype": str(dtype),
        "rigid_max_abs": {name: score.abs().max().item() for name, score in scores.items()},
        "checks": ["noncommuting_rigid", "m_equals_t_exclusion", "two_path_median",
                   "spatial_support_median", "edit_axis_sensitivity", "zero_edit",
                   "frame_zero_exact"],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2))
