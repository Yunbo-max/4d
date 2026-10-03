#!/usr/bin/env python3
"""Output-space edit transport diagnostic; not native ActionMesh steering.

No models, weights, renderer, or benchmark GT are loaded. Existing vertex
trajectories provide geometry. Synthetic rotations provide *diagnostic* edit
targets; natural generated trajectories have no quality ground truth here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch


CASES = {
    "kangaroo": "input-modes/kangaroo-video/deformations.npz",
    "octopus": "applications-20261002/text-octopus/deformations.npz",
    "mushroom": "applications-20261002/image-mushroom/deformations.npz",
    "panda": "applications-20261002/mesh-panda/deformations.npz",
}


def average_ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def auc(score, label):
    score, label = np.asarray(score), np.asarray(label, dtype=bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if not n1 or not n0:
        return None
    ranks = average_ranks(score)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def finite_mean(values, mask=None):
    if mask is not None:
        values = values[..., mask]
    return float(values.mean().item()) if values.numel() else None


def row_rotation_y(angle):
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    # Row-vector convention: points @ R.
    return torch.stack((c, zero, -s, zero, one, zero, s, zero, c), dim=-1).reshape(*angle.shape, 3, 3)


def kabsch_rows(x, y):
    """Best proper rotation for row points, with residual from fit points."""
    xc, yc = x - x.mean(dim=-2, keepdim=True), y - y.mean(dim=-2, keepdim=True)
    h = xc.transpose(-1, -2) @ yc
    u, singular, vh = torch.linalg.svd(h)
    sign = torch.ones_like(singular)
    sign[..., -1] = torch.where(torch.linalg.det(u @ vh) < 0, -1.0, 1.0)
    rotation = (u * sign.unsqueeze(-2)) @ vh
    residual = ((xc @ rotation - yc).square().sum(-1).mean(-1)).sqrt()
    return rotation, residual, singular


def neighbor_indices(x, count):
    distance = torch.cdist(x, x)
    distance.fill_diagonal_(float("inf"))
    return distance.topk(count, largest=False).indices


def transports(v, d0, fit_neighbors, scale, gate):
    global_r, local_r, residuals, rank_ratios = [], [], [], []
    x_local = v[0, fit_neighbors]
    for frame in v:
        rg, _, _ = kabsch_rows(v[0], frame)
        rl, residual, singular = kabsch_rows(x_local, frame[fit_neighbors])
        global_r.append(rg)
        local_r.append(rl)
        residuals.append(residual / scale)
        # Rank-two support suffices for a planar rigid patch; do not demand rank 3.
        rank_ratios.append(singular[:, 1] / singular[:, 0].clamp_min(1e-12))
    rg, rl = torch.stack(global_r), torch.stack(local_r)
    risk, rank_ratio = torch.stack(residuals), torch.stack(rank_ratios)
    world = d0.unsqueeze(0).expand(len(v), -1, -1).clone()
    global_offset = torch.einsum("pi,tij->tpj", d0, rg)
    local_offset = torch.einsum("pi,tpij->tpj", d0, rl)
    fallback = risk > gate
    gated = torch.where(fallback[..., None], global_offset, local_offset)
    return {
        "world_offset": world,
        "global_rigid_transport": global_offset,
        "local_rigid_transport": local_offset,
        "residual_gated_local_global": gated,
    }, {"fit_residual_over_scale": risk, "rank2_ratio": rank_ratio, "fallback": fallback}


def descriptive_metrics(v, offset, d0, evaluation_neighbors, scale, edited):
    q = v + offset
    src_edges = v[:, evaluation_neighbors] - v[:, :, None]
    edited_edges = q[:, evaluation_neighbors] - q[:, :, None]
    effect = (edited_edges.norm(dim=-1) - src_edges.norm(dim=-1)) / scale
    active_edges = edited[:, None] | edited[evaluation_neighbors]
    edge_drift = (effect - effect[:1])[:, active_edges]
    dv, dq = torch.diff(v, dim=0), torch.diff(q, dim=0)
    src_speed = dv[:, edited].norm(dim=-1)
    new_speed = dq[:, edited].norm(dim=-1)
    denom = src_speed.mean()
    amplitude_error = (offset.norm(dim=-1) - d0.norm(dim=-1)[None]).abs() / scale
    return {
        "heldout_edge_effect_drift_rms": float(edge_drift.square().mean().sqrt().item()),
        "heldout_edge_effect_drift_mean_abs": float(edge_drift.abs().mean().item()),
        "edited_velocity_change_mean_over_scale_per_step": finite_mean((dq - dv).norm(dim=-1) / scale, edited),
        "edited_speed_ratio": float((new_speed.mean() / denom).item()) if denom > 1e-12 else None,
        "edit_amplitude_error_max_over_scale_SANITY": float(amplitude_error.max().item()),
        "unedited_shift_max_over_scale_BY_CONSTRUCTION": float(offset[:, ~edited].norm(dim=-1).max().item()) if (~edited).any() else None,
        "finite": bool(torch.isfinite(q).all().item()),
    }


def synthetic_sequence(x, d0, frames, regime, scale, generator, noise):
    phase = torch.linspace(0, 1, frames, device=x.device, dtype=x.dtype)
    frequency = 2 if regime == "fast_two_region_articulation" else 1
    angle = (math.pi / 3) * torch.sin(2 * math.pi * frequency * phase)
    if regime in ("global_rigid", "global_rigid_noisy_observations"):
        r = row_rotation_y(angle)
        center = x.mean(0)
        clean = torch.einsum("pi,tij->tpj", x - center, r) + center
        truth_offset = torch.einsum("pi,tij->tpj", d0, r)
    else:
        group = x[:, 0] >= x[:, 0].median()
        center = torch.stack([x[~group].mean(0), x[group].mean(0)])[group.long()]
        direction = torch.where(group, 1.0, -1.0)
        r = row_rotation_y(angle[:, None] * direction[None])
        clean = torch.einsum("pi,tpij->tpj", x - center, r) + center
        truth_offset = torch.einsum("pi,tpij->tpj", d0, r)
    observed = clean.clone()
    if regime == "global_rigid_noisy_observations":
        perturbation = torch.randn(observed.shape, generator=generator, device=x.device, dtype=x.dtype)
        observed[1:] += noise * scale * perturbation[1:]
    return observed, clean, truth_offset


@torch.inference_mode()
def run_case(name, path, args, device):
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as arrays:
        original = arrays["vertices"]
        faces = arrays["faces"]
        if original.ndim != 3 or original.shape[-1] != 3 or not np.isfinite(original).all():
            raise ValueError(f"Malformed/nonfinite trajectory: {path}")
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(original.shape[1], min(args.points, original.shape[1]), replace=False))
        v = torch.as_tensor(original[:, indices].copy(), dtype=torch.float32, device=device)
        full_count = original.shape[1]
        full_scale = float(np.linalg.norm(np.ptp(original[0].astype(np.float64), axis=0)))
        face_count = len(faces)
    if full_scale <= 0 or args.neighbors + args.eval_neighbors >= v.shape[1]:
        raise ValueError("Degenerate scale or too few points for disjoint fit/evaluation neighbors")
    scale = full_scale
    neighbors = neighbor_indices(v[0], args.neighbors + args.eval_neighbors)
    fit, heldout = neighbors[:, :args.neighbors], neighbors[:, args.neighbors:]
    height = v[0, :, 2]
    low, high = torch.quantile(height, torch.tensor([0.65, 0.9], device=device))
    weight = ((height - low) / (high - low).clamp_min(1e-8)).clamp(0, 1)
    edited = weight > 0
    direction = torch.tensor([1.0, 0.3, 0.2], device=device)
    direction /= direction.norm()
    d0 = args.edit_amplitude * scale * weight[:, None] * direction
    boundary = (v[0, :, 0] - v[0, :, 0].median()).abs() <= 0.1 * torch.ptp(v[0, :, 0]) if hasattr(torch, "ptp") else (v[0, :, 0] - v[0, :, 0].median()).abs() <= 0.1 * (v[0, :, 0].max() - v[0, :, 0].min())
    report = {
        "input": str(path.resolve()), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_frames": len(v), "source_vertices": full_count, "source_faces": face_count,
        "sampled_vertices": len(indices), "sampled_vertex_indices": indices.tolist(),
        "scale_policy": "full original first-frame bounding-box diagonal; fixed for every variant",
        "scale": scale, "region_policy": "soft upper-z quantile edit; geometric region, not an anatomical label",
        "edited_points": int(edited.sum().item()), "edited_boundary_points": int((edited & boundary).sum().item()),
        "fit_neighbors": args.neighbors, "heldout_neighbors": args.eval_neighbors,
        "natural_track_diagnostics": {}, "synthetic_controls": {},
    }
    offsets, diagnostics = transports(v, d0, fit, scale, args.risk_threshold)
    for method, offset in offsets.items():
        report["natural_track_diagnostics"][method] = descriptive_metrics(v, offset, d0, heldout, scale, edited)
    report["natural_track_diagnostics"]["interpretation"] = {
        "quality_ground_truth": False,
        "heldout_edges": "not used in Kabsch fitting, but still a local-rigidity diagnostic, not correctness GT",
        "velocity_caveat": "constant world offset preserves velocity by construction yet can be a wrong rotating material edit",
        "fallback_fraction_edited": finite_mean(diagnostics["fallback"].float(), edited),
        "rank2_ratio_median": float(diagnostics["rank2_ratio"].median().item()),
    }
    save_arrays = {"indices": indices, "source": v.cpu().numpy(), "initial_offset": d0.cpu().numpy()}
    if args.save_points:
        save_arrays.update({f"natural__{m}": (v + d).cpu().numpy() for m, d in offsets.items()})
    generator = torch.Generator(device=device).manual_seed(args.seed)
    regimes = ["global_rigid", "two_region_articulation", "fast_two_region_articulation", "global_rigid_noisy_observations"]
    for regime in regimes:
        observed, clean, truth = synthetic_sequence(v[0], d0, len(v), regime, scale, generator, args.observation_noise)
        offsets, diagnostics = transports(observed, d0, fit, scale, args.risk_threshold)
        rows, errors = {}, {}
        for method, offset in offsets.items():
            error = (offset - truth).norm(dim=-1) / scale
            errors[method] = error
            rows[method] = {
                "edit_offset_epe_over_scale": finite_mean(error[1:], edited),
                "boundary_edit_offset_epe_over_scale": finite_mean(error[1:], edited & boundary),
                "interior_edit_offset_epe_over_scale": finite_mean(error[1:], edited & ~boundary),
                "edited_target_epe_over_scale": finite_mean((observed + offset - clean - truth).norm(dim=-1)[1:] / scale, edited),
                **descriptive_metrics(observed, offset, d0, heldout, scale, edited),
            }
        local = errors["local_rigid_transport"][1:, edited].cpu().numpy().ravel()
        global_error = errors["global_rigid_transport"][1:, edited].cpu().numpy().ravel()
        harm = local - global_error
        valid = np.abs(harm) > 1e-6
        risk = diagnostics["fit_residual_over_scale"][1:, edited].cpu().numpy().ravel()
        report["synthetic_controls"][regime] = {
            "target_source": "analytical injected rigid transforms on original generated first-frame points; diagnostic GT only",
            "methods": rows,
            "local_risk_auc_predicting_harm_vs_global": auc(risk[valid], harm[valid] > 0),
            "auc_informative_points": int(valid.sum()),
            "local_harm_fraction_vs_global": float((harm[valid] > 0).mean()) if valid.any() else None,
            "fallback_fraction_edited": finite_mean(diagnostics["fallback"].float()[1:], edited),
        }
        if args.save_points:
            save_arrays[f"{regime}__source"] = observed.cpu().numpy()
            save_arrays[f"{regime}__target"] = (clean + truth).cpu().numpy()
            save_arrays.update({f"{regime}__{m}": (observed + d).cpu().numpy() for m, d in offsets.items()})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report["elapsed_seconds"] = time.perf_counter() - started
    if args.save_points:
        np.savez_compressed(args.output / f"{name}-sampled-points.npz", **save_arrays)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--eval-neighbors", type=int, default=8)
    parser.add_argument("--edit-amplitude", type=float, default=0.03, help="fraction of fixed first-frame scale")
    parser.add_argument("--observation-noise", type=float, default=0.005, help="synthetic coordinate-noise std / scale")
    parser.add_argument("--risk-threshold", type=float, default=0.005, help="local fit RMS / scale, fixed fallback rule")
    parser.add_argument("--save-points", action="store_true", help="save sampled diagnostic point tracks; these are not full meshes")
    args = parser.parse_args()
    if args.points < 8 or args.neighbors < 3 or args.eval_neighbors < 1 or args.edit_amplitude <= 0:
        parser.error("invalid sampling/edit configuration")
    if args.output.exists():
        parser.error("output already exists; use a fresh directory to preserve prior trials")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; no automatic fallback")
    args.output.mkdir(parents=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = {
        "scope": "exploratory output-space geometry postprocessing; not native activation/latent steering, not natural-GT benchmark",
        "training_performed": False, "model_inference_performed": False,
        "device": str(device), "seed": args.seed,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "falsifiers": [
            "Local transport must improve known articulation targets over world/global controls; global-rigid success alone is algebraic sanity.",
            "Noisy rigid observations may favor global transport; local-fit residual must predict actual offset harm, not just be called uncertainty.",
            "Gating must beat at least its best ungated alternative across the declared regimes; otherwise uncertainty-rule benefit is unsupported.",
            "Natural heldout-edge diagnostics cannot certify source-action or semantic edit correctness without independent targets.",
        ],
        "cases": {}, "failures": {},
    }
    for name, relative in CASES.items():
        try:
            result["cases"][name] = run_case(name, args.input_root / relative, args, device)
        except Exception as exc:
            result["failures"][name] = {"type": type(exc).__name__, "message": str(exc)}
        (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        result["device_name"] = torch.cuda.get_device_name(device)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["status"] = "completed_with_failures" if result["failures"] else "completed_diagnostic_only"
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]), "failures": result["failures"], "elapsed_seconds": result["elapsed_seconds"]}))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
