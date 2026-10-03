#!/usr/bin/env python3
"""Directional rotation-sensitivity scores for the round-two geometry probe.

The self-test runs on CPU only. No model, training, renderer, or ground truth
is used. The operational covariance is not a calibrated posterior covariance.
"""
from __future__ import annotations

import argparse
import json
import math

import torch


def _information(points):
    """Reference-basis rotation information from centered row-vector points."""
    centered = points - points.mean(dim=-2, keepdim=True)
    scatter = centered.transpose(-1, -2) @ centered
    trace = scatter.diagonal(dim1=-2, dim2=-1).sum(-1)
    information = trace[..., None, None] * torch.eye(3, dtype=points.dtype, device=points.device) - scatter
    values, vectors = torch.linalg.eigh(information)
    values = values.clamp_min(0.)
    radius_squared = trace / points.shape[-2]
    return values, vectors, radius_squared


def _covariance(values, vectors, noise_squared, null_relative_tolerance, cap):
    tiny = torch.finfo(values.dtype).tiny
    threshold = null_relative_tolerance * values.sum(-1, keepdim=True)
    null = values <= threshold
    variance = (noise_squared[..., None] / values.clamp_min(tiny)).clamp(max=cap)
    # A pseudoinverse would set the exact null mode to zero, making an
    # unobservable twist look certain. Retain finite ambiguity at zero noise.
    variance = torch.where(null, cap, variance)
    return (vectors * variance.unsqueeze(-2)) @ vectors.transpose(-1, -2)


def _action_variance(covariance, directions):
    norm = torch.linalg.vector_norm(directions, dim=-1)
    unit = directions / norm.clamp_min(torch.finfo(directions.dtype).tiny).unsqueeze(-1)
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    along = torch.einsum("qi,tqij,qj->tq", unit, covariance, unit)
    # tr([d]x C [d]x.T)/||d||^2 = tr(C) - d_hat.T C d_hat.
    return torch.where(norm > 0., (trace - along).clamp_min(0.), 0.)


def _shuffle_active_directions(directions, shift):
    """Permute active unit directions, retaining each query's norm/support."""
    norm = torch.linalg.vector_norm(directions, dim=-1)
    active = torch.where(norm > 0.)[0]
    shuffled = torch.zeros_like(directions)
    if active.numel():
        unit = directions[active] / norm[active, None]
        shuffled[active] = unit.roll(int(shift), dims=0) * norm[active, None]
    return shuffled


@torch.no_grad()
def compute(
    observed,
    d0,
    query,
    fit_indices,
    heldout,
    global_fit,
    local_rot,
    global_rot,
    scale,
    *,
    noise_floor_fraction=1e-3,
    null_relative_tolerance=1e-12,
    angular_variance_cap=(math.pi / 2) ** 2,
    weak_support_threshold=.01,
    shuffle_shift=1,
):
    """Return local-minus-global rotation sensitivity scores and a census.

    Args use row-vector rotations: ``points @ rotation``. Expected shapes are
    observed [T,N,3], d0 [Q,3], query [Q], fit_indices [Q,K], heldout [Q,H],
    global_fit [G], local_rot [T,Q,3,3], and global_rot [T,3,3]. Typical K/H/G
    are 32/8/512; other positive counts are supported. Fit and heldout indices
    should be disjoint. The caller owns fitting and that split.

    Outputs directional_covariance, scalar_covariance and shuffled_direction
    are float64 [T,Q] tensors on observed.device. Larger means greater local
    sensitivity relative to global. Scores do not select a transport policy.
    census contains only JSON-compatible numbers, computed from frame zero.
    shuffled_direction rolls unit directions among active queries only, while
    preserving every query's original edit norm and zero-edit support.

    Information and covariance remain in the reference coordinate basis.
    Noise is a per-coordinate mean-square residual of held-out spokes anchored
    at each query, evaluated separately for local/global rotations. That
    residual contains model mismatch and correlated anchor noise; it is an
    operational proxy, not an unbiased noise estimate. Global noise is thus
    query-specific despite a shared global information matrix. The squared
    noise floor is noise_floor_fraction**2 times each fit support's reference
    mean-square radius. No true targets or future-frame aggregation are used.

    Null modes receive angular_variance_cap even at exactly zero residual.
    The first-frame scores are evaluated by the same rule; an application
    should separately enforce its exact first-frame edit/anchor constraint.
    """
    if not torch.is_tensor(observed) or observed.ndim != 3 or observed.shape[-1] != 3:
        raise ValueError("observed must be a tensor shaped [T,N,3]")
    t, n, _ = observed.shape
    if t < 1 or n < 1 or not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("observed must be nonempty and scale finite and positive")
    if not math.isfinite(noise_floor_fraction) or noise_floor_fraction < 0:
        raise ValueError("noise_floor_fraction must be finite and nonnegative")
    if not math.isfinite(null_relative_tolerance) or not 0 <= null_relative_tolerance < 1:
        raise ValueError("null_relative_tolerance must be in [0,1)")
    if not math.isfinite(angular_variance_cap) or angular_variance_cap <= 0:
        raise ValueError("angular_variance_cap must be finite and positive")
    if not math.isfinite(weak_support_threshold) or weak_support_threshold < 0:
        raise ValueError("weak_support_threshold must be finite and nonnegative")
    device = observed.device
    observed = observed.to(dtype=torch.float64)
    d0 = torch.as_tensor(d0, dtype=torch.float64, device=device)
    local_rot = torch.as_tensor(local_rot, dtype=torch.float64, device=device)
    global_rot = torch.as_tensor(global_rot, dtype=torch.float64, device=device)
    indices = []
    for name, index in (("query", query), ("fit_indices", fit_indices), ("heldout", heldout), ("global_fit", global_fit)):
        raw = torch.as_tensor(index, device=device)
        if raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError(f"{name} must contain integer indices")
        index = raw.long()
        if index.numel() == 0 or bool(((index < 0) | (index >= n)).any()):
            raise ValueError(f"{name} must contain nonempty, in-range indices")
        indices.append(index)
    query, fit_indices, heldout, global_fit = indices
    q = query.numel()
    if query.ndim != 1 or global_fit.ndim != 1:
        raise ValueError("query and global_fit must be one-dimensional")
    if fit_indices.ndim != 2 or heldout.ndim != 2 or fit_indices.shape[0] != q or heldout.shape[0] != q:
        raise ValueError("fit_indices and heldout must have shapes [Q,K] and [Q,H]")
    if d0.shape != (q, 3) or local_rot.shape != (t, q, 3, 3) or global_rot.shape != (t, 3, 3):
        raise ValueError("d0 or rotation tensor has an incompatible shape")
    if not all(bool(torch.isfinite(x).all()) for x in (observed, d0, local_rot, global_rot)):
        raise ValueError("coordinates, edits, and rotations must be finite")

    local_values, local_vectors, local_radius_sq = _information(observed[0, fit_indices])
    global_values, global_vectors, global_radius_sq = _information(observed[0, global_fit])
    spokes = observed[:, heldout] - observed[:, query, None, :]
    reference = spokes[0]
    local_residual = torch.einsum("qki,tqij->tqkj", reference, local_rot) - spokes
    global_residual = torch.einsum("qki,tij->tqkj", reference, global_rot) - spokes
    local_noise_sq = local_residual.square().mean(dim=(-1, -2))
    global_noise_sq = global_residual.square().mean(dim=(-1, -2))
    local_noise_sq = torch.maximum(local_noise_sq, noise_floor_fraction ** 2 * local_radius_sq)
    global_noise_sq = torch.maximum(global_noise_sq, noise_floor_fraction ** 2 * global_radius_sq)
    local_covariance = _covariance(local_values, local_vectors, local_noise_sq, null_relative_tolerance, angular_variance_cap)
    global_covariance = _covariance(global_values, global_vectors, global_noise_sq, null_relative_tolerance, angular_variance_cap)
    directional = _action_variance(local_covariance, d0) - _action_variance(global_covariance, d0)
    scalar = (local_covariance - global_covariance).diagonal(dim1=-2, dim2=-1).sum(-1)
    shuffled = _shuffle_active_directions(d0, shuffle_shift)
    shuffled_score = _action_variance(local_covariance, shuffled) - _action_variance(global_covariance, shuffled)

    tiny = torch.finfo(observed.dtype).tiny
    local_ratio = local_values[:, 0] / local_values.sum(-1).clamp_min(tiny)
    global_ratio = global_values[0] / global_values.sum().clamp_min(tiny)
    census = {
        "queries": q,
        "fit_neighbors": fit_indices.shape[1],
        "heldout_neighbors": heldout.shape[1],
        "global_fit_points": global_fit.numel(),
        "weak_support_threshold": float(weak_support_threshold),
        "weak_support_fraction": float((local_ratio < weak_support_threshold).double().mean()),
        "null_support_fraction": float((local_ratio <= null_relative_tolerance).double().mean()),
        "local_min_information_ratio_median": float(local_ratio.median()),
        "global_min_information_ratio": float(global_ratio),
        "local_reference_radius_median_over_scale": float(local_radius_sq.sqrt().median() / scale),
        "global_reference_radius_over_scale": float(global_radius_sq.sqrt() / scale),
        "noise_floor_fraction": float(noise_floor_fraction),
        "angular_variance_cap": float(angular_variance_cap),
    }
    return {"directional_covariance": directional, "scalar_covariance": scalar, "shuffled_direction": shuffled_score, "census": census}


def selftest():
    """Catch null-mode false certainty, planar false degeneracy, and wrong basis."""
    dtype = torch.float64
    directions = torch.tensor([[2., 0., 0.], [0., 3., 0.], [0., 0., 0.]], dtype=dtype)
    eye = torch.eye(3, dtype=dtype)

    def fixture(planar=False):
        if planar:
            fit_angles = torch.arange(32, dtype=dtype) * (2 * math.pi / 32)
            heldout_angles = (torch.arange(8, dtype=dtype) + .25) * (2 * math.pi / 8)
            fit_points = torch.stack((fit_angles.cos(), fit_angles.sin(), torch.zeros(32, dtype=dtype)), -1)
            heldout_points = torch.stack((heldout_angles.cos(), heldout_angles.sin(), torch.zeros(8, dtype=dtype)), -1)
        else:
            fit_points = torch.zeros(32, 3, dtype=dtype)
            fit_points[:, 0] = torch.linspace(-1, 1, 32, dtype=dtype)
            heldout_points = torch.zeros(8, 3, dtype=dtype)
            heldout_points[:, 0] = torch.linspace(-.8, .8, 8, dtype=dtype)
        cube = torch.tensor([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)], dtype=dtype)
        points = torch.cat((torch.zeros(1, 3, dtype=dtype), fit_points, heldout_points, cube))
        observed = points.unsqueeze(0).repeat(2, 1, 1)
        return dict(
            observed=observed, d0=directions.clone(), query=torch.zeros(3, dtype=torch.long),
            fit_indices=torch.arange(1, 33).repeat(3, 1), heldout=torch.arange(33, 41).repeat(3, 1),
            global_fit=torch.arange(41, 49), local_rot=eye.repeat(2, 3, 1, 1),
            global_rot=eye.repeat(2, 1, 1), scale=4., noise_floor_fraction=0.,
        )

    line = fixture()
    result = compute(**line)
    cap = (math.pi / 2) ** 2
    expected = torch.tensor([[0., cap, 0.], [0., cap, 0.]], dtype=dtype)
    torch.testing.assert_close(result["directional_covariance"], expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result["scalar_covariance"], torch.full((2, 3), cap, dtype=dtype), atol=1e-12, rtol=1e-12)
    shuffled_expected = torch.tensor([[cap, 0., 0.], [cap, 0., 0.]], dtype=dtype)
    torch.testing.assert_close(result["shuffled_direction"], shuffled_expected, atol=1e-12, rtol=1e-12)
    shuffled_edits = _shuffle_active_directions(directions, 1)
    torch.testing.assert_close(shuffled_edits, torch.tensor([[0., 2., 0.], [3., 0., 0.], [0., 0., 0.]], dtype=dtype))
    torch.testing.assert_close(shuffled_edits.norm(dim=-1), directions.norm(dim=-1))
    torch.testing.assert_close(_shuffle_active_directions(torch.zeros_like(directions), 1), torch.zeros_like(directions))
    single = torch.tensor([[0., 0., 0.], [0., 3., 0.], [0., 0., 0.]], dtype=dtype)
    torch.testing.assert_close(_shuffle_active_directions(single, 1), single)
    assert result["census"]["weak_support_fraction"] == 1.

    plane = compute(**fixture(planar=True))
    assert plane["census"]["weak_support_fraction"] == 0., "Planar rigid patches must not be classed as rank-one patches"
    torch.testing.assert_close(plane["directional_covariance"], torch.zeros(2, 3, dtype=dtype), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(plane["scalar_covariance"], torch.zeros(2, 3, dtype=dtype), atol=1e-12, rtol=1e-12)

    # A common change of world basis must rotate edit directions too. This
    # catches covariance accidentally represented in the current-frame basis.
    change = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=dtype)
    changed = dict(line)
    changed["observed"] = line["observed"] @ change + torch.tensor([4., -2., 7.], dtype=dtype)
    changed["d0"] = directions @ change
    rotated = compute(**changed)
    for key in ("directional_covariance", "scalar_covariance", "shuffled_direction"):
        torch.testing.assert_close(rotated[key], result[key], atol=1e-12, rtol=1e-12)

    # Frame-one motion is distinct from the common coordinate transform.
    moved = dict(line)
    moved["observed"] = line["observed"].clone()
    moved["observed"][1] = line["observed"][1] @ change + 3.
    moved["local_rot"] = line["local_rot"].clone()
    moved["global_rot"] = line["global_rot"].clone()
    moved["local_rot"][1] = change
    moved["global_rot"][1] = change
    motion_result = compute(**moved)
    for key in ("directional_covariance", "scalar_covariance", "shuffled_direction"):
        torch.testing.assert_close(motion_result[key], result[key], atol=1e-12, rtol=1e-12)

    # Changing a later observed frame cannot affect an earlier score.
    changed_future = dict(line)
    changed_future["observed"] = line["observed"].clone()
    changed_future["observed"][1, 33:41, 1] += .3
    future_result = compute(**changed_future)
    for key in ("directional_covariance", "scalar_covariance", "shuffled_direction"):
        torch.testing.assert_close(future_result[key][0], result[key][0], atol=1e-12, rtol=1e-12)
        assert torch.isfinite(future_result[key]).all()
    assert not torch.equal(future_result["directional_covariance"][1], result["directional_covariance"][1])
    return {"status": "passed", "device": "cpu", "checks": ["line_direction", "scalar_control", "active_direction_roll", "shuffle_preserves_norms_and_support", "empty_and_single_active_shuffle", "planar_valid", "world_basis", "frame_motion", "no_future_leakage"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="Run deterministic CPU checks")
    options = parser.parse_args()
    if not options.selftest:
        parser.error("Import compute(), or supply --selftest")
    print(json.dumps(selftest(), indent=2))
