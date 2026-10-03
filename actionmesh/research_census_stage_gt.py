"""Shared-anchor Stage-I/Stage-II geometry diagnostic against ActionBench GT.

Fit the official 24-start, 200-step ICP only between Stage-II frame zero (the
cleaned anchor) and GT frame zero. Save its actual PyTorch3D 4x4 row-vector matrix
and apply that same transform to BOTH stages at frames 0, 8 and 15. No fit uses
future frames; no extra normalization, camera transform, cropping or fallback.

Distances use 100k surface samples, seed 44+t for actual frame t, and the official
10k directional query subsets (seeds 44/45). Thus sum_unsquared is the official
per-frame CD definition. Independent Stage-I topology has no material tracks:
these are geometry diagnostics, not CD-M or proof of a correspondence failure.
The raw Stage-I vs Stage-II discrepancy also measures canonical geometry without
alignment. Frame-zero Stage-I/II differences can reflect native mesh cleanup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import traceback

from research_census_eval import digest, load_arrays, load_official
from research_three_ideas import Resources, json_write

FRAMES = (0, 8, 15)
POINTS = 100000
ICP_POINTS = 10000
SAMPLE_SEED = 44


def now():
    return datetime.now(timezone.utc).isoformat()


def transform_numpy(points, matrix):
    """Independent NumPy check of a saved PyTorch3D row-vector homogeneous matrix."""
    import numpy as np
    points, matrix = np.asarray(points), np.asarray(matrix)
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    if points.shape[-1] != 3 or matrix.shape != (4, 4):
        raise ValueError("Expected [...,3] points and [4,4] matrix")
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
    transformed = homogeneous @ matrix
    if not np.all(np.isfinite(transformed)) or np.any(np.abs(transformed[..., 3]) < 1e-12):
        raise ValueError("Nonfinite transform or invalid homogeneous denominator")
    return transformed[..., :3] / transformed[..., 3:]


def distance(first, second, first_seed=44, second_seed=45):
    """Directional components of exact official unsquared CD query convention.

first->second samples first with first_seed; second->first samples second with
second_seed. Use first=prediction, second=GT to match compute_chamfer_score.
Nearest-neighbor trees always contain all 100,000 samples.
"""
    import numpy as np
    from scipy.spatial import KDTree
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != (POINTS, 3) or second.shape != (POINTS, 3):
        raise ValueError("Distance requires two full 100k point sets")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Nonfinite sampled points")
    idx_first = np.random.RandomState(first_seed).permutation(len(first))[:10000]
    idx_second = np.random.RandomState(second_seed).permutation(len(second))[:10000]
    forward = KDTree(second).query(first[idx_first])[0]
    backward = KDTree(first).query(second[idx_second])[0]
    a, b = float(forward.mean()), float(backward.mean())
    return {"first_to_second_mean_unsquared": a, "second_to_first_mean_unsquared": b,
            "sum_unsquared": a + b, "half_sum_unsquared": .5 * (a + b),
            "first_to_second_p95_unsquared": float(np.quantile(forward, .95)),
            "second_to_first_p95_unsquared": float(np.quantile(backward, .95)),
            "tree_points_per_set": POINTS, "query_points_per_direction": 10000,
            "first_query_seed": first_seed, "second_query_seed": second_seed}


def validate_raw_mesh(path, expected_frame):
    import numpy as np
    import trimesh
    with np.load(path, allow_pickle=False) as saved:
        vertices, faces = saved["vertices"].copy(), saved["faces"].copy()
        frame = int(saved["frame"])
    if frame != expected_frame:
        raise ValueError("Probe mesh frame identity mismatch")
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Invalid Stage-I mesh vertices")
    if faces.ndim != 2 or faces.shape[-1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Invalid Stage-I triangle indices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Stage-I face index out of bounds")
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Stage-I surface has no positive finite area")
    return mesh


def run(args, torch, state, started):
    import numpy as np
    import trimesh

    def checkpoint(stage):
        state["stage"] = stage
        state["elapsed_seconds"] = time.monotonic() - started
        json_write(args.output / "progress.json", state)
        print(f"STAGE_GT {stage} elapsed={state['elapsed_seconds']:.2f}s", flush=True)
        if state["elapsed_seconds"] > args.max_seconds:
            raise TimeoutError("Stage-check deadline reached; no adaptive fallback")

    checkpoint("validating_source_identity")
    generation = json.loads((args.case_dir / "report.json").read_text())
    probe = json.loads((args.probe_dir / "report.json").read_text())
    probe_provenance = json.loads((args.probe_dir / "provenance.json").read_text())
    if generation.get("status") != "completed" or probe.get("status") != "completed":
        raise ValueError("Both native generation and shape probe must be completed")
    uid, seed = generation["uid"], generation["seed"]
    if (probe.get("uid"), probe.get("seed")) != (uid, seed):
        raise ValueError("Shape probe and generation UID/seed differ")
    if (probe_provenance.get("source_uid"), probe_provenance.get("source_seed")) != (uid, seed):
        raise ValueError("Shape probe provenance UID/seed differ")
    sources = {}
    for filename in ("denoised.npz", "sequence.npz", "report.json"):
        actual = digest(args.case_dir / filename)
        if actual != probe_provenance.get("source_hashes", {}).get(filename):
            raise ValueError(f"Source differs from shape probe: {filename}")
        if filename != "report.json" and actual != generation.get("sha256", {}).get(filename):
            raise ValueError(f"Source differs from generation hash: {filename}")
        sources[str(args.case_dir / filename)] = actual
    for filename in ("report.json", "provenance.json"):
        sources[str(args.probe_dir / filename)] = digest(args.probe_dir / filename)
    gt_path = args.gt_dir / uid / "surfaces.npy"
    sources[str(gt_path)] = digest(gt_path)
    vertices, faces, gt_positions = load_arrays(args.case_dir / "sequence.npz", gt_path)
    _, official = load_official(args.root / "repo", cpu_rng_fix=True)
    from chamfer import compute_chamfer_score
    from icp import gradient_icp
    from sample_mesh import sample_points
    from sample_point_cloud import sample_point_cloud
    from pytorch3d.transforms import Transform3d
    meshes = {}
    for frame in FRAMES:
        frame_dir = args.probe_dir / f"frame_{frame:02d}"
        raw_path = frame_dir / "stageI-raw-mesh.npz"
        measurement = json.loads((frame_dir / "measurement.json").read_text())
        if measurement.get("frame") != frame or measurement.get("sha256", {}).get("stageI-raw-mesh.npz") != digest(raw_path):
            raise ValueError(f"Stage-I raw mesh provenance mismatch at frame {frame}")
        sources[str(raw_path)] = digest(raw_path)
        sources[str(frame_dir / "measurement.json")] = digest(frame_dir / "measurement.json")
        meshes[("stageI", frame)] = validate_raw_mesh(raw_path, frame)
        meshes[("stageII", frame)] = trimesh.Trimesh(vertices[frame], faces, process=False)
    protocol = {"frames": list(FRAMES), "surface_points": POINTS, "surface_seed": "44 + actual frame index",
        "icp_points": ICP_POINTS, "icp_subset_seed": SAMPLE_SEED, "icp_rotations": 24,
        "icp_iterations": 200, "icp_lr": .01,
        "alignment": "One official ICP: Stage-II cleaned anchor frame0 to GT0; identical transform on BOTH stages and ALL selected frames",
        "historical_evaluator_transform_reused": False,
        "alignment_recomputation_reason": "Historical evaluator saved scalar metrics only; this diagnostic refits frame0 using the same official samples and optimizer, then saves its matrix. Bitwise identity with the unsaved historical fit is not claimed.",
        "transform_convention": "PyTorch3D row-vector homogeneous matrix; [x,y,z,1] @ M",
        "icp_allows_anisotropic_scale": True, "future_frame_fit": False,
        "normalization": "none beyond official shared anchor ICP; raw cross-stage distances also reported",
        "metric": "Unsquared Euclidean directional mean sum, official 10k query subsets44/45 against full100k trees",
        "units": "GT coordinate units for aligned metrics; native canonical units for raw cross-stage metrics",
        "interpretation": "Surface geometry only. Stage-I independent topology has no valid material correspondence; no Stage-I CD-M. Frame0 includes cleanup/decimation/floater-removal differences. Selected-frame average is not the full16-frame benchmark average.",
        "attribution": "StageI better than StageII suggests a geometry change associated with temporal decoding; does not identify its cause. Both shape scores good but CD-M poor is only suggestive of correspondence problems, since frame0 nearest matches can be wrong.",
        "limits": "Anisotropic anchor fitting may mask scale errors. Shared anchor alignment can favor the cleaned anchor; report frame0 offset and all frame values, never fit stages independently.",
        "memory_fraction": args.memory_fraction, "device": str(args.device), "physical_monitor_gpu_index": args.gpu_index,
        "memory_cap_scope": "PyTorch caching allocator fraction, not a hard limit on all driver allocations",
        "timeout_scope": "Stage checks only; external process timeout required for an in-flight CUDA call"}
    json_write(args.output / "protocol.json", protocol)
    checkpoint("sampling_anchor_only")
    # This is exactly the first element of official sample_meshes(all16, seed44).
    anchor = sample_points(meshes[("stageII", 0)], POINTS, seed=SAMPLE_SEED)
    gt0 = torch.from_numpy(gt_positions[0])
    pred_icp = sample_point_cloud(anchor[None], ICP_POINTS, seed=SAMPLE_SEED)[0].to(args.device)
    gt_icp = sample_point_cloud(gt0[None], ICP_POINTS, seed=SAMPLE_SEED)[0].to(args.device)
    checkpoint("fitting_frame0_only")
    torch.cuda.synchronize(args.device)
    fit_started = time.monotonic()
    alignment = gradient_icp(pc_pred=pred_icp, pc_gt=gt_icp, lr=.01, n_iter=200)
    torch.cuda.synchronize(args.device)
    fit_seconds = time.monotonic() - fit_started
    matrix = alignment.get_matrix().detach().cpu().numpy()
    if matrix.shape != (1, 4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid official alignment matrix")
    np.save(args.output / "shared-anchor-transform.npy", matrix, allow_pickle=False)
    # Round-trip the saved actual matrix, avoiding a guessed scale/rotation decomposition.
    cpu_transform = Transform3d(matrix=torch.from_numpy(np.load(args.output / "shared-anchor-transform.npy")))
    reference = transform_numpy(anchor.numpy(), matrix)
    transformed_anchor = cpu_transform.transform_points(anchor[None])[0].numpy()
    matrix_error = float(np.max(np.abs(reference - transformed_anchor)))
    if not np.allclose(reference, transformed_anchor, rtol=2e-6, atol=2e-6):
        raise ValueError("Saved-matrix convention check failed")
    np.savez_compressed(args.output / "anchor-icp-samples.npz",
                        stageII_anchor_icp=pred_icp.detach().cpu().numpy(), gt0_icp=gt_icp.detach().cpu().numpy())
    del pred_icp, gt_icp, alignment
    torch.cuda.empty_cache()
    checkpoint("frame0_alignment_saved")
    state.update(uid=uid, seed=seed, alignment_fit_seconds=fit_seconds,
                 saved_matrix_numpy_max_abs_difference=matrix_error,
                 shared_matrix_sha256=digest(args.output / "shared-anchor-transform.npy"))
    for frame in FRAMES:
        checkpoint(f"frame_{frame:02d}_sampling")
        # Do not sample [0,8,15] as a three-element list: seeds must be44,52,59.
        stage1 = sample_points(meshes[("stageI", frame)], POINTS, seed=SAMPLE_SEED + frame)
        stage2 = anchor if frame == 0 else sample_points(meshes[("stageII", frame)], POINTS, seed=SAMPLE_SEED + frame)
        aligned1 = cpu_transform.transform_points(stage1[None])[0].numpy()
        aligned2 = cpu_transform.transform_points(stage2[None])[0].numpy()
        gt = gt_positions[frame]
        stage1_gt = distance(aligned1, gt)
        stage2_gt = distance(aligned2, gt)
        # Independent dispatch to the official scalar function catches formula drift.
        official1 = float(compute_chamfer_score(pred=torch.from_numpy(aligned1), gt=torch.from_numpy(gt)))
        official2 = float(compute_chamfer_score(pred=torch.from_numpy(aligned2), gt=torch.from_numpy(gt)))
        if not math.isclose(stage1_gt["sum_unsquared"], official1, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Stage-I directional sum differs from official CD")
        if not math.isclose(stage2_gt["sum_unsquared"], official2, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Stage-II directional sum differs from official CD")
        frame_dir = args.output / f"frame_{frame:02d}"
        frame_dir.mkdir()
        np.savez_compressed(frame_dir / "clouds.npz", stageI_raw=stage1.numpy(), stageII_raw=stage2.numpy(),
                            stageI_aligned=aligned1, stageII_aligned=aligned2, ground_truth=gt,
                            frame=np.asarray(frame), surface_seed=np.asarray(SAMPLE_SEED + frame))
        metrics = {"frame": frame, "surface_seed": SAMPLE_SEED + frame,
                   "stageI_aligned_to_gt": stage1_gt, "stageII_aligned_to_gt": stage2_gt,
                   "stageII_minus_stageI_gt_cd": official2 - official1,
                   "stageI_vs_stageII_raw_canonical": distance(stage1.numpy(), stage2.numpy()),
                   "stageI_vs_stageII_shared_aligned": distance(aligned1, aligned2),
                   "stageI_raw_to_gt_without_alignment": distance(stage1.numpy(), gt),
                   "stageII_raw_to_gt_without_alignment": distance(stage2.numpy(), gt),
                   "unaligned_gt_warning": "Raw-to-GT values mix coordinate mismatch and geometry; calibration diagnostics only",
                   "official_cd_scalar_crosscheck": {"stageI": official1, "stageII": official2},
                   "clouds_sha256": digest(frame_dir / "clouds.npz")}
        json_write(frame_dir / "metrics.json", metrics)
        state["frames"].append(metrics)
        checkpoint(f"frame_{frame:02d}_completed")
    provenance = {"source_files_sha256": sources, "official": official,
                  "diagnostic_script_sha256": digest(Path(__file__)),
                  "sampling_module_sha256": digest(args.root / "repo/actionbench/sample_mesh.py"),
                  "shared_matrix_sha256": state["shared_matrix_sha256"],
                  "protocol_sha256": digest(args.output / "protocol.json"),
                  "anchor_samples_sha256": digest(args.output / "anchor-icp-samples.npz"),
                  "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "trimesh", "pytorch3d")}}
    json_write(args.output / "provenance.json", provenance)
    state.update(status="completed", geometry_only=True, future_frame_alignment=False,
                 material_motion_metric_computed=False, method_intervention=False,
                 selected_frame_average_stageI_gt=sum(row["stageI_aligned_to_gt"]["sum_unsquared"] for row in state["frames"]) / 3,
                 selected_frame_average_stageII_gt=sum(row["stageII_aligned_to_gt"]["sum_unsquared"] for row in state["frames"]) / 3,
                 selected_average_is_not_full_benchmark=True)


def self_test(with_torch=False):
    import numpy as np
    # Scale xyz by(2,3,4), then +90deg about z, then translate(5,-2,7).
    # Hand-calculated expected points distinguish R vs R.T and S@R vs R@S.
    matrix = np.array([[0., 2., 0., 0.], [-3., 0., 0., 0.], [0., 0., 4., 0.], [5., -2., 7., 1.]])
    points = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 2., 3.]])
    expected = np.array([[5., 0., 7.], [2., -2., 7.], [5., -2., 11.], [-1., 0., 19.]])
    assert np.array_equal(transform_numpy(points, matrix), expected)
    sequence = np.stack((points, points + [0., 1., 0.], points + [1., 0., 1.]))
    transformed = transform_numpy(sequence, matrix)
    assert np.array_equal(transformed[1] - transformed[0], np.tile([-3., 0., 0.], (4, 1)))
    assert np.array_equal(transformed[2] - transformed[0], np.tile([0., 2., 4.], (4, 1)))
    assert not np.allclose(transform_numpy(points, matrix.T), expected)
    checks = ["known90deg rotation/anisotropic scale/translation match handcalculated coordinates",
              "same transform preserves analytically expected motion across all frames",
              "transposed homogeneous convention is detected"]
    if with_torch:
        import torch
        from pytorch3d.transforms import Transform3d
        actual = Transform3d(matrix=torch.tensor(matrix[None], dtype=torch.float32)).transform_points(torch.tensor(sequence, dtype=torch.float32))
        assert np.allclose(actual.numpy(), transformed, atol=1e-6, rtol=0)
        checks.append("actual PyTorch3DTransform3d agrees on all3frames")
    return {"status": "pass", "checks": checks, "torch_cuda_used": False,
            "pytorch3d_checked": with_torch, "scope": "matrix/motion geometry fixtures, no ICP or generated geometry run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/rivermind-data/actionmesh-repro"))
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0", help="Explicit CUDA device, e.g. cuda:0")
    parser.add_argument("--gpu-index", default="0", help="Physical nvidia-smi monitor index; may differ under CUDA_VISIBLE_DEVICES")
    parser.add_argument("--memory-fraction", type=float, default=.08)
    parser.add_argument("--max-seconds", type=float, default=300.)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-torch", action="store_true")
    args = parser.parse_args()
    if args.self_test or args.self_test_torch:
        print(json.dumps(self_test(args.self_test_torch), indent=2))
        return 0
    if any(getattr(args, key) is None for key in ("case_dir", "probe_dir", "gt_dir", "output")):
        parser.error("--case-dir --probe-dir --gt-dir --output required")
    if not 0 < args.memory_fraction <= 1 or not 0 < args.max_seconds < math.inf:
        parser.error("Invalid memory fraction or deadline")
    for key in ("root", "case_dir", "probe_dir", "gt_dir", "output"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    started, monitor = time.monotonic(), None
    state = {"status": "running", "started_utc": now(), "frames": [], "stage": "setup"}
    json_write(args.output / "command.json", {"argv": sys.argv, "script_sha256": digest(Path(__file__)),
               "device": args.device, "memory_fraction": args.memory_fraction, "max_seconds": args.max_seconds})
    try:
        import torch
        torch.set_num_threads(1)
        device = torch.device(args.device)
        if device.type != "cuda" or device.index is None:
            raise ValueError("Provide explicit CUDA index, e.g. --device cuda:0")
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise RuntimeError("Selected CUDA device unavailable")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=device)
        args.device = device
        monitor = Resources(args.output, torch, args.gpu_index)
        monitor.start()
        run(args, torch, state, started)
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        state.update(ended_utc=now(), elapsed_seconds=time.monotonic() - started)
        if monitor:
            try:
                state["resources"] = monitor.finish()
            except Exception as exc:
                state["resource_monitor_error"] = str(exc)
        json_write(args.output / "report.json", state)
        json_write(args.output / "progress.json", state)
    return int(state["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
