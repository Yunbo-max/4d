"""CPU-only auxiliary motion readouts on hash-qualified cached trajectories.

Measure displacement, adjacent velocity and second-difference acceleration
errors with the original two frame0 nearest-neighbour maps frozen. These are
not official CD-M, not summary-score subtraction and not correspondence-free
motion errors. No models, resampling, ICP, CUDA or nearest-neighbour fitting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

from research_census_eval import digest
from research_census_motion_controls import ALLOWED_UIDS, array_hash, source_hash
from research_census_time_direction_eval import qualify_cache
from research_three_ideas import json_write

T, N, Q = 16, 100000, 10000
ZERO_EPSILON = 1e-12


def ratio_record(predicted, ground_truth):
    """A ratio of mean vector magnitudes, never a mean of pointwise ratios."""
    zero_gt = ground_truth <= ZERO_EPSILON
    return {"predicted_mean_magnitude": float(predicted),
            "gt_mean_magnitude": float(ground_truth),
            "predicted_to_gt_magnitude_ratio": float(predicted / ground_truth) if not zero_gt else None,
            "zero_gt_motion": bool(zero_gt),
            "spurious_prediction_motion_when_gt_zero": bool(zero_gt and predicted > ZERO_EPSILON)}


def direction_record(predicted, ground_truth):
    import numpy as np
    error = float(np.linalg.norm(predicted - ground_truth, axis=-1).mean())
    return {"mean_endpoint_error": error, **ratio_record(
        float(np.linalg.norm(predicted, axis=-1).mean()),
        float(np.linalg.norm(ground_truth, axis=-1).mean()))}


def vector_measurements(predicted, ground_truth, kind):
    """Float64 differences preserve source precision and avoid float32 reduction."""
    import numpy as np
    p, g = predicted.astype(np.float64), ground_truth.astype(np.float64)
    if kind == "displacement":
        return p - p[0], g - g[0], list(range(len(p)))
    order = {"velocity": 1, "acceleration": 2}[kind]
    return np.diff(p, n=order, axis=0), np.diff(g, n=order, axis=0), list(range(order, len(p)))


def motion_readout(pred, gt, nu, mu, query_pred, query_gt):
    """Reusable for a future arm ONLY after exact native anchor/map qualification.

    nu maps each predicted sample to GT0, mu maps each GT sample to predicted0.
    Query subsets are applied on the corresponding source side before matching;
    both maps remain unchanged across every frame and every derivative order.
    """
    import numpy as np
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape[0] != gt.shape[0] or pred.shape[-1] != 3 or gt.shape[-1] != 3:
        raise ValueError("Require two [T,N,3] tracked clouds with equal frame count")
    for values, expected, upper in ((nu, pred.shape[1], gt.shape[1]), (mu, gt.shape[1], pred.shape[1])):
        if values.shape != (expected,) or values.dtype.kind not in "iu" or values.min() < 0 or values.max() >= upper:
            raise ValueError("Invalid frozen nearest-neighbour map")
    for values, upper in ((query_pred, pred.shape[1]), (query_gt, gt.shape[1])):
        if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu" or values.min() < 0 or values.max() >= upper or len(np.unique(values)) != len(values):
            raise ValueError("Invalid query index subset")
    trajectories = {
        "pred_to_gt": (pred[:, query_pred], gt[:, nu[query_pred]]),
        "gt_to_pred": (pred[:, mu[query_gt]], gt[:, query_gt]),
    }
    if any(not np.isfinite(array).all() for pair in trajectories.values() for array in pair):
        raise ValueError("Nonfinite queried trajectories")
    result = {}
    for kind in ("displacement", "velocity", "acceleration"):
        directional = {name: vector_measurements(*pair, kind) for name, pair in trajectories.items()}
        frames = directional["pred_to_gt"][2]
        rows = []
        for row_index, frame in enumerate(frames):
            row = {"frame": frame}
            for direction, (p, g, _) in directional.items():
                row[direction] = direction_record(p[row_index], g[row_index])
            row["sum_directional_mean_endpoint_error"] = sum(row[d]["mean_endpoint_error"] for d in trajectories)
            rows.append(row)
        # The displacement row at frame0 is identically zero by construction.
        primary = rows[1:] if kind == "displacement" else rows
        summary = {"frames": [row["frame"] for row in primary], "frame_count": len(primary)}
        for direction in trajectories:
            summary[direction] = {
                "mean_endpoint_error": sum(row[direction]["mean_endpoint_error"] for row in primary) / len(primary),
                **ratio_record(
                    sum(row[direction]["predicted_mean_magnitude"] for row in primary) / len(primary),
                    sum(row[direction]["gt_mean_magnitude"] for row in primary) / len(primary))}
        summary["sum_directional_mean_endpoint_error"] = sum(summary[d]["mean_endpoint_error"] for d in trajectories)
        result[kind] = {"per_frame": rows, "primary_summary": summary}
        if kind == "displacement":
            result[kind]["including_frame0_mean_error_for_transparency"] = sum(row["sum_directional_mean_endpoint_error"] for row in rows) / len(rows)
    return result


def protocol():
    return {
        "name": "Conditional fixed-anchor-NN motion readout; auxiliary, not official CD-M",
        "frames": list(range(T)), "tracked_surface_points": N,
        "query_count_each_direction": Q, "query_rng": "numpy.random.RandomState(seed).permutation(100000)[:10000]",
        "query_seeds": {"predicted": 44, "gt": 45},
        "sampling": "Exact cached synchronized seed44 faces/barycentric trajectories, all16frames; no independent framewise sample used here",
        "alignment": "Exact saved Stage-GT frame0 matrix already applied in cached native cloud; no transformation refit or normalization",
        "maps": "nu[i]=NN_GT0(P0[i]); mu[j]=NN_P0(G0[j]); reuse exact original full100k maps without rematching",
        "displacement": {
            "pred_to_gt": "mean_i_in_I norm((Pt[i]-P0[i])-(Gt[nu[i]]-G0[nu[i]]))",
            "gt_to_pred": "mean_j_in_J norm((Pt[mu[j]]-P0[mu[j]])-(Gt[j]-G0[j]))",
            "primary_frames": list(range(1, T)), "units": "aligned dataset coordinate units"},
        "velocity": {"operator": "X[t]-X[t-1]", "primary_frames": list(range(1, T)), "units": "coordinate units per frame"},
        "acceleration": {"operator": "X[t]-2*X[t-1]+X[t-2]", "primary_frames": list(range(2, T)), "units": "coordinate units per frame squared"},
        "aggregation": "Unsquared Euclidean vector error, mean queries in each direction, then sum two direction means; temporal mean over valid frames only",
        "amplitude_ratio": "Mean predicted vector magnitude divided by matched GT mean vector magnitude, separately by direction and order. Sequence ratio divides aggregate means, never averages pointwise/per-frame ratios",
        "zero_gt_epsilon": ZERO_EPSILON,
        "zero_gt_policy": "Ratio is null at GT mean <= epsilon; report zero-GT and spurious-prediction flags; never substitute zero error or infinity",
        "precision": "float32 cached source clouds; selected trajectories promoted to float64 before all differences, norms and means",
        "interpretation": [
            "Subtracting each paired trajectory's initial position removes constant positional offsets only conditional on the fixed pairing",
            "Wrong first-frame NN/material pairing, anisotropic alignment, sampling and occlusion remain confounds; this is not geometry-independent ground-truth motion",
            "Official CD-M uses absolute tracked point positions and full100k queries; this auxiliary readout uses differences and10k queries. Do not subtract or compare their scalar summaries as an algebraic decomposition",
            "Velocity and acceleration use frame indices; unknown physical frame rate prevents claims in units per second",
            "Magnitude agreement does not establish direction, phase or material correctness; inspect vector errors alongside ratios",
            "GT is used only for evaluation. No GT-guided steering, oracle generation, model inference or training is performed",
            "Two frozen seed42 assets only; frames, derivatives and query samples are not extra independent assets"],
        "future_arm_comparison": "Qualify arm completion/source hashes and exact native frame0/topology/query identity first. Use identical saved material sampling, shared matrix, maps and query IDs; do not fit a new map for the arm",
        "not_recomputed": ["CD3D", "CD4D", "CD-M", "nearest-neighbour maps", "ICP", "surface sampling"],
    }


def original_report_hash(provenance, path):
    suffix = "/" + path.parent.name + "/" + path.name
    candidates = [value for key, value in provenance["source_files_sha256"].items() if key.endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError("Ambiguous original report provenance: " + suffix)
    return candidates[0]


def run(args, state, started):
    import numpy as np

    def check(stage):
        state.update(stage=stage, elapsed_seconds=time.monotonic() - started)
        json_write(args.output / "progress.json", state)
        if state["elapsed_seconds"] > args.max_seconds:
            raise TimeoutError("CPU readout budget exceeded")
        print("MOTION_READOUT", stage, round(state["elapsed_seconds"], 3), flush=True)

    check("qualifying_cached_identity_and_hashes")
    generation_path = args.case_dir / "report.json"
    stage_report_path = args.stage_gt_dir / "report.json"
    stage_provenance_path = args.stage_gt_dir / "provenance.json"
    generation = json.loads(generation_path.read_text())
    stage = json.loads(stage_report_path.read_text())
    stage_provenance = json.loads(stage_provenance_path.read_text())
    if generation.get("status") != "completed" or generation.get("uid") not in ALLOWED_UIDS or generation.get("seed") != 42:
        raise ValueError("Require a completed frozen first-two-assets native seed42 case")
    if stage.get("status") != "completed" or (stage.get("uid"), stage.get("seed")) != (generation["uid"], 42):
        raise ValueError("Shared Stage-GT alignment incomplete or mismatched UID/seed")
    state.update(uid=generation["uid"], seed=42, independent_assets=1)
    sequence_path = args.case_dir / "sequence.npz"
    gt_path = args.gt_dir / generation["uid"] / "surfaces.npy"
    matrix_path = args.stage_gt_dir / "shared-anchor-transform.npy"
    if digest(sequence_path) != generation.get("sha256", {}).get("sequence.npz"):
        raise ValueError("Native sequence differs from generation report")
    if digest(matrix_path) != stage.get("shared_matrix_sha256") or digest(matrix_path) != stage_provenance.get("shared_matrix_sha256"):
        raise ValueError("Saved alignment matrix differs from Stage-GT reports")
    for path in (sequence_path, gt_path):
        if digest(path) != source_hash(stage_provenance, path.name):
            raise ValueError("Stage-GT source mismatch: " + path.name)
    cache_report, cache_provenance, native_metrics, shapes, maps = qualify_cache(args, generation, gt_path, matrix_path)
    for path in (generation_path, stage_report_path, stage_provenance_path):
        if digest(path) != original_report_hash(cache_provenance, path):
            raise ValueError("Native cache report provenance mismatch: " + str(path))
    with np.load(sequence_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["frame_indices"], np.arange(T)) or not np.array_equal(saved["timesteps"], np.arange(T)):
            raise ValueError("Native sequence is not all16 physical frames0..15")
    cloud_path = args.native_cache_dir / "aligned-material-clouds.npz"
    # Both frozen assets have these clouds; absence is not permission to resample.
    with np.load(cloud_path, allow_pickle=False) as saved:
        pred = saved["native_aligned"].copy()
    if pred.shape != (T, N, 3) or pred.dtype != np.float32 or not np.isfinite(pred).all():
        raise ValueError("Cached native cloud must be finite float32[16,100000,3]")
    if array_hash(pred) != cache_provenance.get("native_material_cloud_sha256"):
        raise ValueError("Native cloud bytes do not match original motion-control provenance")
    gt_raw = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    if gt_raw.shape != (T, N, 6):
        raise ValueError("GT must contain all16 tracked100k surfaces and normals")
    gt = np.asarray(gt_raw[:, :, :3], dtype=np.float32)
    if not np.isfinite(gt).all():
        raise ValueError("Nonfinite GT positions")
    query_pred = np.random.RandomState(44).permutation(N)[:Q]
    query_gt = np.random.RandomState(45).permutation(N)[:Q]
    if not np.array_equal(query_pred, maps["query_predicted_ids"]):
        raise ValueError("Native cache predicted query mismatch")
    source_paths = [generation_path, sequence_path, stage_report_path, stage_provenance_path, matrix_path, gt_path, cloud_path]
    source_paths += [args.native_cache_dir / name for name in ("report.json", "provenance.json", "protocol.json", "controls.json", "per-frame-shape.json", "sampling-and-maps.npz")]
    initial_hashes = {str(path): digest(path) for path in source_paths}
    check("computing_new_conditional_trajectory_readouts")
    readouts = motion_readout(pred, gt, maps["pred_to_gt_firstframe"], maps["gt_to_pred_firstframe"], query_pred, query_gt)
    json_write(args.output / "motion-readouts.json", readouts)
    np.savez_compressed(args.output / "query-ids.npz", predicted=query_pred, gt=query_gt)
    reference = {"official_formula_CD_M_cached": native_metrics["cd_motion_official_formula"],
        "shared_alignment_CD4D_cached": shapes["shared_alignment_cd4d_16frame_mean"],
        "first_frame_NN_ties_cached": cache_report.get("frame0_nearest_ties"),
        "scope": "Unchanged cached reference metrics; no recomputation and no algebraic subtraction from auxiliary readouts"}
    json_write(args.output / "cached-reference.json", reference)
    check("final_source_revalidation")
    if {str(path): digest(path) for path in source_paths} != initial_hashes:
        raise ValueError("An input changed during CPU evaluation")
    helpers = ("research_census_eval.py", "research_census_motion_controls.py", "research_census_time_direction_eval.py", "research_three_ideas.py")
    json_write(args.output / "provenance.json", {
        "script_sha256": digest(Path(__file__)),
        "helper_sha256": {name: digest(Path(__file__).parent / name) for name in helpers},
        "source_files_sha256": initial_hashes, "cached_official_source_provenance": cache_provenance.get("official"),
        "native_cloud_sha256": array_hash(pred), "query_ids_sha256": digest(args.output / "query-ids.npz"),
        "readouts_sha256": digest(args.output / "motion-readouts.json"), "numpy_version": np.__version__,
        "torch_imported": "torch" in sys.modules, "cuda_used": False,
        "core_metrics_recomputed": False, "NN_maps_recomputed": False, "surface_sampling_run": False, "ICP_run": False})
    if "torch" in sys.modules:
        raise RuntimeError("NumPy-only readout unexpectedly imported Torch")
    state.update(status="completed", summaries={key: value["primary_summary"] for key, value in readouts.items()},
        cached_reference=reference, native_cache_qualified=True, all16frames_retained=True,
        core_metrics_recomputed=False, cuda_used=False, ICP_run=False, GT_use="evaluation only",
        scientific_scope="Auxiliary motion readout conditional on original nearest pairing and shared alignment; not a new method or geometry-independent error")


def self_test():
    import numpy as np
    # Four well-separated particles translate and accelerate identically.
    base = np.array([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.], [30., 0., 0.]])
    times = np.arange(4, dtype=np.float64)
    gt = base[None] + np.stack((times, times * times, np.zeros(4)), axis=-1)[:, None]
    pred = gt + np.array([0., 0., .25])
    ids = np.arange(4)
    matched = motion_readout(pred, gt, ids, ids, ids, ids)
    for kind in matched:
        assert matched[kind]["primary_summary"]["sum_directional_mean_endpoint_error"] == 0.
        assert matched[kind]["primary_summary"]["pred_to_gt"]["predicted_to_gt_magnitude_ratio"] == 1.
    assert float(np.linalg.norm(pred - gt, axis=-1).mean()) == .25
    static = np.broadcast_to(pred[0], pred.shape).copy()
    stopped = motion_readout(static, gt, ids, ids, ids, ids)
    # Analytic acceleration of y=t^2 is2; static prediction gives sum2+2=4.
    assert stopped["acceleration"]["primary_summary"]["sum_directional_mean_endpoint_error"] == 4.
    assert stopped["velocity"]["per_frame"][0]["sum_directional_mean_endpoint_error"] == 2 * np.sqrt(2)
    assert stopped["displacement"]["primary_summary"]["sum_directional_mean_endpoint_error"] > 0
    translated = motion_readout(static + [40., -8., 2.], gt + [40., -8., 2.], ids, ids, ids, ids)
    assert translated == stopped
    # Common time-varying translation preserves vector errors, not magnitude ratios.
    drift = np.stack((times**2, times**3, times), axis=-1)[:, None]
    drifted = motion_readout(static + drift, gt + drift, ids, ids, ids, ids)
    for kind in stopped:
        assert drifted[kind]["primary_summary"]["sum_directional_mean_endpoint_error"] == stopped[kind]["primary_summary"]["sum_directional_mean_endpoint_error"]
    crossing_gt = np.zeros((3, 2, 3))
    crossing_gt[:, :, 0] = [[-1, 1], [0, 0], [1, -1]]
    crossing_pred = crossing_gt + np.array([[2., 0., 0.], [-2., 0., 0.]])[None]
    semantic = np.arange(2); nearest = np.array([1, 0])
    correct = motion_readout(crossing_pred, crossing_gt, semantic, semantic, semantic, semantic)
    wrong = motion_readout(crossing_pred, crossing_gt, nearest, nearest, semantic, semantic)
    assert correct["displacement"]["primary_summary"]["sum_directional_mean_endpoint_error"] == 0.
    assert wrong["displacement"]["primary_summary"]["sum_directional_mean_endpoint_error"] == 6.
    assert wrong["velocity"]["primary_summary"]["sum_directional_mean_endpoint_error"] == 4.
    still = motion_readout(static, static, ids, ids, ids, ids)
    assert still["displacement"]["primary_summary"]["pred_to_gt"]["predicted_to_gt_magnitude_ratio"] is None
    spurious = motion_readout(gt, static, ids, ids, ids, ids)
    assert spurious["velocity"]["primary_summary"]["pred_to_gt"]["spurious_prediction_motion_when_gt_zero"]
    assert ratio_record(3., 2.)["predicted_to_gt_magnitude_ratio"] == 1.5
    # Asymmetric query subsets must act on each direction's own source indices.
    subset_pred = static.copy(); subset_pred[1:, 0, 2] += 2
    subset = motion_readout(subset_pred, static, ids, ids, np.array([0]), np.array([1]))
    assert subset["displacement"]["primary_summary"]["pred_to_gt"]["mean_endpoint_error"] == 2.
    assert subset["displacement"]["primary_summary"]["gt_to_pred"]["mean_endpoint_error"] == 0.
    return {"status": "pass", "checks": ["constant anchor offset with matching motion gives zero errors", "static prediction misses linear and accelerating GT", "common constant translation leaves all readouts invariant", "common time-dependent translation preserves errors, not ratios", "crossing wrong-NN counterexample remains nonzero despite semantically matching motion", "zero GT motion yields null ratios and explicit spurious-motion flags", "each query subset acts on its own NN direction"],
        "crossing_counterexample": {"correct_map_displacement_error": 0., "NN_map_displacement_error": 6., "NN_map_velocity_error": 4.},
        "scope": "Analytic NumPy fixtures; no model, GT file, CUDA or metric fit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case-dir", "stage-gt-dir", "native-cache-dir", "gt-dir", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--max-seconds", type=float, default=180.)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    names = ("case_dir", "stage_gt_dir", "native_cache_dir", "gt_dir", "output")
    if any(getattr(args, name) is None for name in names):
        parser.error("All five path arguments required")
    if not 0 < args.max_seconds <= 180:
        parser.error("CPU wall-clock budget must be in (0,180] seconds")
    for name in names:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    json_write(args.output / "command.json", {"argv": sys.argv, "script_sha256": digest(Path(__file__)), "CPU_only": True, "max_seconds": args.max_seconds})
    json_write(args.output / "protocol.json", protocol())
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    def deadline(_signal, _frame):
        raise TimeoutError("CPU readout wall-clock budget exceeded")
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    try:
        run(args, state, started)
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        state.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
            peak_process_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_process_rss_units="bytes" if platform.system() == "Darwin" else "KiB")
        json_write(args.output / "report.json", state)
        json_write(args.output / "progress.json", state)
    return int(state["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
