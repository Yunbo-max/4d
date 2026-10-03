"""CPU-only controls for cached ActionMesh motion scores; no model, ICP or guidance.

Two distinct official sampling branches are preserved: independently sampled
100k surface points (seed44+t) for framewise CD, and one frame0 face/barycentric
sample (seed44) shared across16frames for CD-M. A saved Stage-GT anchor transform
is reused. Native scores are diagnostic recomputations, not a claim of bitwise
identity with the historical evaluator whose transform was not saved.

The GT-motion-transfer oracle uses future GT for evaluation only and is NOT an
error floor. A separate conservative fixed-assignment lower bound is explicitly
derived in protocol.json. Controls are trajectories, not additional assets or
new fixed-topology mesh reconstructions. No model weights are loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

from research_census_eval import digest, load_arrays, load_official
from research_three_ideas import json_write

N = 100000
T = 16
CONTROL_SEED = 20261002
ALLOWED_UIDS = {
    "000-037_1358c424008a43cbaa35eba5e58551ac",
    "000-043_061697e330d44524bd11f8cf95772e2d",
}


def array_hash(array):
    import hashlib
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def permutation(size, subset, seed):
    """Permute only within query indices and within their complement."""
    import numpy as np
    subset = np.asarray(subset, dtype=np.int64)
    mask = np.ones(size, dtype=bool)
    mask[subset] = False
    other = np.flatnonzero(mask)
    rng = np.random.RandomState(seed)
    result = np.arange(size)
    result[subset] = rng.permutation(subset)
    result[other] = rng.permutation(other)
    if not np.array_equal(np.sort(result), np.arange(size)) or not np.array_equal(np.sort(result[subset]), np.sort(subset)):
        raise ValueError("Control permutation is not a query-preserving bijection")
    return result


def maps(pred0, gt0):
    import numpy as np
    from scipy.spatial import KDTree
    # The metric source uses scipy.spatial.KDTree, not an approximate matcher.
    gt_tree, pred_tree = KDTree(gt0), KDTree(pred0)
    _, nu = gt_tree.query(pred0)  # Exact k=1 convention used by official CD-M.
    _, mu = pred_tree.query(gt0)
    d_nu, _ = gt_tree.query(pred0, k=2)
    d_mu, _ = pred_tree.query(gt0, k=2)
    return nu, mu, {
        "pred_to_gt_first_second_distance_ties": int(np.sum(np.abs(d_nu[:, 0] - d_nu[:, 1]) <= 1e-12)),
        "gt_to_pred_first_second_distance_ties": int(np.sum(np.abs(d_mu[:, 0] - d_mu[:, 1]) <= 1e-12)),
        "absolute_tie_tolerance": 1e-12,
    }


def motion_table(pred, gt, nu, mu):
    import numpy as np
    rows = []
    for frame in range(len(pred)):
        a, b = pred[frame].astype(np.float64), gt[frame].astype(np.float64)
        forward = float(np.linalg.norm(a - b[nu], axis=-1).mean())
        backward = float(np.linalg.norm(a[mu] - b, axis=-1).mean())
        rows.append({"frame": frame, "pred_to_gt_fixed_nn": forward,
                     "gt_to_pred_fixed_nn": backward, "sum_unsquared": forward + backward})
    return rows


def amplitude(sequence):
    import numpy as np
    first = sequence[0].astype(np.float64)
    rows = []
    for frame in range(len(sequence)):
        current = sequence[frame].astype(np.float64)
        rows.append({"frame": frame,
                     "mean_displacement_from_frame0": float(np.linalg.norm(current - first, axis=-1).mean()),
                     "mean_step_displacement": float(np.linalg.norm(current - sequence[frame - 1], axis=-1).mean()) if frame else 0.})
    return rows


def oracle_reference(pred, gt, nu):
    import numpy as np
    out = np.empty_like(pred)
    out[0] = pred[0]  # Exact anchor identity; avoid cancellation changing nearest ties.
    for frame in range(1, len(pred)):
        out[frame] = pred[0] + (gt[frame, nu] - gt[0, nu])
    return out


def pairing_lower_bound(pred, gt, nu, mu, anchor_cost):
    """Triangle bound for equal P=Q, fixed frame0 and fixed NN assignments."""
    import numpy as np
    if pred.shape[1] != gt.shape[1]:
        raise ValueError("Lower-bound implementation requires equal sampling counts")
    size = pred.shape[1]
    rows = []
    for frame in range(len(pred)):
        per_gt = np.linalg.norm(gt[frame, nu[mu]].astype(np.float64) - gt[frame].astype(np.float64), axis=-1)
        group_max = np.zeros(size, dtype=np.float64)
        np.maximum.at(group_max, mu, per_gt)
        rows.append({"frame": frame, "cycle_separation_mean": float(per_gt.mean()),
                     "fixed_pairing_lower_bound": float(group_max.sum() / size)})
    sequence_bound = (anchor_cost + sum(r["fixed_pairing_lower_bound"] for r in rows[1:])) / len(pred)
    return {"per_frame": rows, "sequence_lower_bound_with_fixed_anchor": sequence_bound,
            "actual_fixed_anchor_cost": anchor_cost,
            "cycle_index_mismatch_fraction": float(np.mean(nu[mu] != np.arange(size))),
            "scope": "Conservative lower bound for fixed first-frame assignments; arbitrary disconnected future points allowed. Not all anchor-geometry error and not necessarily attainable by a mesh."}


def source_hash(provenance, filename):
    values = [value for path, value in provenance.get("source_files_sha256", {}).items() if Path(path).name == filename]
    if len(values) != 1:
        raise ValueError(f"Expected one Stage-GT provenance hash for {filename}")
    return values[0]


def run(args, state, started):
    import numpy as np
    import torch
    import trimesh
    torch.set_num_threads(1)

    def checkpoint(stage):
        state["stage"] = stage
        state["elapsed_seconds"] = time.monotonic() - started
        json_write(args.output / "progress.json", state)
        print(f"MOTION_CONTROL {stage} elapsed={state['elapsed_seconds']:.2f}s", flush=True)
        if state["elapsed_seconds"] > args.max_seconds:
            raise TimeoutError("CPU diagnostic stage deadline exceeded")

    checkpoint("validating_sources")
    generation = json.loads((args.case_dir / "report.json").read_text())
    stage_gt = json.loads((args.stage_gt_dir / "report.json").read_text())
    stage_gt_provenance = json.loads((args.stage_gt_dir / "provenance.json").read_text())
    if generation.get("status") != "completed" or stage_gt.get("status") != "completed":
        raise ValueError("Generation and shared-alignment diagnostic must both be completed")
    if (generation.get("uid"), generation.get("seed")) != (stage_gt.get("uid"), stage_gt.get("seed")):
        raise ValueError("UID/seed mismatch between generation and Stage-GT")
    if generation.get("uid") not in ALLOWED_UIDS or generation.get("seed") != 42:
        raise ValueError("This bounded diagnostic is limited to the first two frozen cohort assets, seed42")
    state.update(uid=generation["uid"], seed=generation["seed"], independent_assets=1,
                 control_conditions_are_not_extra_assets=True)
    gt_path = args.gt_dir / generation["uid"] / "surfaces.npy"
    sequence_path = args.case_dir / "sequence.npz"
    matrix_path = args.stage_gt_dir / "shared-anchor-transform.npy"
    for path in (sequence_path, gt_path):
        if digest(path) != source_hash(stage_gt_provenance, path.name):
            raise ValueError(f"Stage-GT source mismatch: {path.name}")
    if digest(sequence_path) != generation.get("sha256", {}).get("sequence.npz"):
        raise ValueError("Sequence differs from generator report")
    if digest(matrix_path) != stage_gt.get("shared_matrix_sha256") or digest(matrix_path) != stage_gt_provenance.get("shared_matrix_sha256"):
        raise ValueError("Saved alignment matrix hash mismatch")
    vertices, faces, gt = load_arrays(sequence_path, gt_path)
    matrix = np.load(matrix_path, allow_pickle=False)
    if matrix.shape != (1, 4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid saved shared alignment matrix")
    _, official = load_official(args.root / "repo", cpu_rng_fix=True)
    from chamfer import compute_chamfer_score, compute_motion_chamfer_score
    import sample_mesh
    from pytorch3d.transforms import Transform3d
    transform = Transform3d(matrix=torch.from_numpy(matrix))
    meshes = [trimesh.Trimesh(frame, faces, process=False) for frame in vertices]
    checkpoint("official_synchronized_sampling")
    sample_faces, barycentric = sample_mesh.get_baryc_sampling_mesh(
        sample_mesh.trimesh_to_pytorch3d(meshes[0]), num_samples=N, seed=44)
    raw = sample_mesh.apply_baryc_sampling_on_meshes(
        sample_mesh.join_meshes_as_batch([sample_mesh.trimesh_to_pytorch3d(mesh) for mesh in meshes]),
        sample_faces, barycentric)
    pred = transform.transform_points(raw).numpy()
    if pred.shape != (T, N, 3) or not np.isfinite(pred).all():
        raise ValueError("Bad aligned material sample shape")
    query_ids = np.random.RandomState(44).permutation(N)[:10000]
    nu, mu, tie_report = maps(pred[0], gt[0])
    native_rows = motion_table(pred, gt, nu, mu)
    native_cdm = float(compute_motion_chamfer_score(torch.from_numpy(pred), torch.from_numpy(gt)))
    if not math.isclose(native_cdm, sum(r["sum_unsquared"] for r in native_rows) / T, abs_tol=2e-6, rel_tol=2e-6):
        raise ValueError("Per-frame native CD-M disagrees with official scalar")
    protocol = {
        "scope": "CPU cached-output diagnostic; one asset per invocation; fixed first two assets designated by parent cohort",
        "frames": list(range(T)), "points": N,
        "shape_sampling": "Independent per-frame official100k surface samples, seeds44+t; official CD uses10k queries seeds44/45",
        "motion_sampling": "Separate official frame0 face/barycentric draw seed44, applied unchanged over16frames",
        "alignment": "Reuse saved Stage-GT4x4matrix, no ICP/refitting and no future-frame fit",
        "historical_cd_m_bitwise_identity_claimed": False,
        "control_seed": CONTROL_SEED,
        "negative_control": "Frame0 unchanged; independent per-frame row permutations within fixed10kqueryset and its complement. Every pointset/queryset preserved; not a mesh reconstruction.",
        "positive_control": "One common query-block-preserving permutation acrossALLframes includingframe0; recompute initial NNmaps. Same trajectories relabeled; ties can make correspondence ambiguous.",
        "oracle": "O[t,i]=P[0,i]+G[t,nu(i)]-G[0,nu(i)], O[0]=P[0] exactly. Uses futureGT ONLYfor diagnostic; NEVERguidance, hyperparameter selection, or learned/publishablemethod result.",
        "oracle_is_error_floor": False,
        "gt_correspondence_assumption": "GT point index is material identity over time, as assumed by the official ActionBench evaluator",
        "oracle_limitation": "The bidirectional nearest maps need not be inverses. Other trajectory compromises can outperform this oracle, so no percent-of-possible-improvement claim.",
        "pairing_lower_bound": "For equalN, L[t]=(1/N)sum_i max_{j:mu(j)=i}||G[t,nu(i)]-G[t,j]|| (emptymax0). Triangle inequality bounds any fixed-assignmentCDM. SequenceLB=(actualC0+sum_t>0 L[t])/T. Conservative relaxed bound, not uniquely material-motion error.",
        "surface_gap": "Matched synchronizedcloud full-query100k nearest-surfaceCD, subtracted from samecloud fixed-assignmentCDM. Nonnegative gap is assignment-retention cost, not pure correspondence error.",
        "units": "Unsquared distances inGTcoordinateunits after fixed anisotropicanchorfit",
        "max_seconds": args.max_seconds, "timeout_scope": "Stage checks; caller can impose external CPU wall deadline",
    }
    json_write(args.output / "protocol.json", protocol)
    checkpoint("control_trajectories")
    negative = np.empty_like(pred)
    negative[0] = pred[0]
    permutation_hashes = {}
    for frame in range(1, T):
        perm = permutation(N, query_ids, CONTROL_SEED + frame)
        negative[frame] = pred[frame, perm]
        permutation_hashes[str(frame)] = array_hash(perm)
    common = permutation(N, query_ids, CONTROL_SEED)
    positive = pred[:, common]
    positive_nu, positive_mu, positive_ties = maps(positive[0], gt[0])
    oracle = oracle_reference(pred, gt, nu)
    controls = {"native": (pred, nu, mu), "negative_scrambled": (negative, nu, mu),
                "positive_consistent_relabeling": (positive, positive_nu, positive_mu),
                "oracle_gt_motion_transfer_NOT_floor": (oracle, nu, mu)}
    summaries = {}
    for name, (cloud, forward, backward) in controls.items():
        checkpoint("control_metric_" + name)
        table = motion_table(cloud, gt, forward, backward)
        score = native_cdm if name == "native" else float(compute_motion_chamfer_score(torch.from_numpy(cloud), torch.from_numpy(gt)))
        if not math.isclose(score, sum(row["sum_unsquared"] for row in table) / T, abs_tol=2e-6, rel_tol=2e-6):
            raise ValueError(f"Official CD-M differs from fixed-map expansion: {name}")
        summaries[name] = {"cd_motion_official_formula": score, "per_frame_motion": table,
                           "trajectory_amplitude": amplitude(cloud)}
    expected_anchor_residual = native_rows[0]["pred_to_gt_fixed_nn"]
    oracle_forward_error = max(abs(row["pred_to_gt_fixed_nn"] - expected_anchor_residual)
                               for row in summaries["oracle_gt_motion_transfer_NOT_floor"]["per_frame_motion"])
    if oracle_forward_error > 2e-6:
        raise ValueError("GT-motion oracle violated constant forward anchor-residual invariant")
    bound = pairing_lower_bound(pred, gt, nu, mu, native_rows[0]["sum_unsquared"])
    lower = bound["sequence_lower_bound_with_fixed_anchor"]
    if native_cdm + 2e-6 < lower or summaries["oracle_gt_motion_transfer_NOT_floor"]["cd_motion_official_formula"] + 2e-6 < lower:
        raise ValueError("Derived conservative pairing lower bound violated")
    shape_rows = []
    for frame in range(T):
        checkpoint(f"frame_{frame:02d}_shape_sampling_and_metrics")
        independent = sample_mesh.sample_points(meshes[frame], N, seed=44 + frame)
        independent_aligned = transform.transform_points(independent[None])[0]
        ground_truth = torch.from_numpy(gt[frame])
        official_cd = float(compute_chamfer_score(pred=independent_aligned, gt=ground_truth))
        material_cd = float(compute_chamfer_score(pred=torch.from_numpy(pred[frame]), gt=ground_truth))
        negative_cd = float(compute_chamfer_score(pred=torch.from_numpy(negative[frame]), gt=ground_truth))
        positive_cd = float(compute_chamfer_score(pred=torch.from_numpy(positive[frame]), gt=ground_truth))
        full_cd = float(compute_chamfer_score(pred=torch.from_numpy(pred[frame]), gt=ground_truth, n=0))
        shape_difference = max(abs(material_cd - negative_cd), abs(material_cd - positive_cd))
        if shape_difference > 1e-10:
            raise ValueError("Shape-preserving control changed matched-query surface CD")
        gap = native_rows[frame]["sum_unsquared"] - full_cd
        if gap < -2e-6:
            raise ValueError("Fixed assignments beat unconstrained nearest-surface distance")
        shape_rows.append({"frame": frame, "independent_sample_seed": 44 + frame,
            "official_framewise_cd_shared_alignment": official_cd,
            "material_cloud_query10k_cd_native": material_cd,
            "material_cloud_query10k_cd_negative": negative_cd,
            "material_cloud_query10k_cd_positive": positive_cd,
            "max_control_shape_cd_difference": shape_difference,
            "material_cloud_full100k_query_cd": full_cd,
            "fixed_assignment_minus_full_query_shape_gap": gap})
    positive_difference = summaries["positive_consistent_relabeling"]["cd_motion_official_formula"] - native_cdm
    n_ties = tie_report["pred_to_gt_first_second_distance_ties"] + tie_report["gt_to_pred_first_second_distance_ties"]
    if abs(positive_difference) > 2e-6 and n_ties == 0:
        raise ValueError("Harmless consistent relabeling changed CD-M without detected anchor ties")
    np.savez_compressed(args.output / "sampling-and-maps.npz", material_face_indices=sample_faces.numpy(),
        material_barycentric=barycentric.numpy(), pred_to_gt_firstframe=nu, gt_to_pred_firstframe=mu,
        query_predicted_ids=query_ids, positive_common_permutation=common,
        shared_alignment_matrix=matrix)
    if args.cache_clouds:
        np.savez_compressed(args.output / "aligned-material-clouds.npz", native_aligned=pred)
    json_write(args.output / "controls.json", summaries)
    json_write(args.output / "per-frame-shape.json", {"frames": shape_rows,
        "shared_alignment_cd4d_16frame_mean": sum(r["official_framewise_cd_shared_alignment"] for r in shape_rows) / T,
        "historical_evaluator_bitwise_identity_claimed": False})
    json_write(args.output / "pairing-lower-bound.json", bound)
    json_write(args.output / "provenance.json", {"script_sha256": digest(Path(__file__)), "official": official,
        "source_files_sha256": {str(path): digest(path) for path in
            (sequence_path, gt_path, matrix_path, args.case_dir / "report.json", args.stage_gt_dir / "report.json", args.stage_gt_dir / "provenance.json")},
        "sampling_and_maps_sha256": digest(args.output / "sampling-and-maps.npz"),
        "negative_permutation_sha256_by_frame": permutation_hashes,
        "native_material_cloud_sha256": array_hash(pred), "common_permutation_sha256": array_hash(common),
        "torch_cuda_initialized": torch.cuda.is_initialized(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")})
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization in CPU-only diagnostic")
    state.update(status="completed", computed_metrics=list(summaries),
        native_cd_motion=native_cdm, negative_cd_motion=summaries["negative_scrambled"]["cd_motion_official_formula"],
        negative_minus_native_cd_motion=summaries["negative_scrambled"]["cd_motion_official_formula"] - native_cdm,
        positive_minus_native_cd_motion=positive_difference,
        positive_control_status="invariant" if abs(positive_difference) <= 2e-6 else "ambiguous_anchor_ties",
        oracle_cd_motion_NOT_floor=summaries["oracle_gt_motion_transfer_NOT_floor"]["cd_motion_official_formula"],
        oracle_constant_forward_residual_max_abs_error=oracle_forward_error,
        conservative_pairing_lower_bound=lower, frame0_nearest_ties=tie_report,
        positive_frame0_nearest_ties=positive_ties, gt_trajectory_amplitude=amplitude(gt),
        cuda_used=False, model_loaded=False, icp_run=False,
        future_gt_used_to_construct_trajectories="oracle_only; all metrics also use held-out GT for evaluation",
        oracle_is_method=False, oracle_is_error_floor=False,
        caution="Scrambling need not worsen an already poor motion estimate; report all assets, including failed sensitivity. Lower CD-M alone does not prove better material motion.")


def self_test():
    import numpy as np
    # Separated particles move rigidly. The known correspondences are analytic.
    gt0 = np.array([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.], [30., 0., 0.]])
    gt = np.stack((gt0, gt0 + [0., 1., 0.], gt0 + [0., 2., 0.]))
    pred = gt + [0., 0., .25]
    nu = mu = np.arange(4)
    native = motion_table(pred, gt, nu, mu)
    common = np.array([1, 0, 3, 2])
    inverse = np.argsort(common)
    relabeled = motion_table(pred[:, common], gt, common, inverse)
    assert all(abs(a["sum_unsquared"] - b["sum_unsquared"]) < 1e-12 for a, b in zip(native, relabeled))
    scrambled = pred.copy(); scrambled[1:] = scrambled[1:, common]
    bad = motion_table(scrambled, gt, nu, mu)
    assert sum(r["sum_unsquared"] for r in bad) > sum(r["sum_unsquared"] for r in native)
    for frame in range(3):
        assert np.array_equal(np.sort(scrambled[frame], axis=0), np.sort(pred[frame], axis=0))
    oracle = oracle_reference(pred, gt, nu)
    assert np.array_equal(oracle[0], pred[0]) and np.array_equal(oracle, pred)
    bound = pairing_lower_bound(pred, gt, nu, mu, native[0]["sum_unsquared"])
    assert abs(bound["sequence_lower_bound_with_fixed_anchor"] - .5 / 3) < 1e-12
    for seed in (4, 5, 6):
        perm = permutation(100, np.array([2, 7, 21, 88]), seed)
        assert set(perm[[2, 7, 21, 88]]) == {2, 7, 21, 88}
    # Noninvertible matching yields a nonzero, provably conservative future bound.
    collapsed = np.array([[[0., 0., 0.], [.1, 0., 0.]], [[0., 0., 0.], [.1, 0., 0.]]])
    moving = np.array([[[0., 0., 0.], [1., 0., 0.]], [[0., 0., 0.], [3., 0., 0.]]])
    small_nu, small_mu = np.array([0, 0]), np.array([0, 1])
    rows = motion_table(collapsed, moving, small_nu, small_mu)
    small_bound = pairing_lower_bound(collapsed, moving, small_nu, small_mu, rows[0]["sum_unsquared"])
    assert small_bound["sequence_lower_bound_with_fixed_anchor"] <= sum(r["sum_unsquared"] for r in rows) / 2 + 1e-12
    return {"status": "pass", "checks": ["consistent relabeling preserves analytic CD-M",
        "time-varying relabeling changes motion but preserves pointsets", "query block permutations preserve subset",
        "GT-motion oracle preserves anchor and constant residual", "conservative pairing bound with noninvertible maps"],
        "scope": "NumPy analytic fixtures, no actual model/official sampling run; no torch/CUDA used"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/rivermind-data/actionmesh-repro"))
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--stage-gt-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float, default=300.)
    parser.add_argument("--cache-clouds", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    if any(getattr(args, name) is None for name in ("case_dir", "stage_gt_dir", "gt_dir", "output")):
        parser.error("--case-dir --stage-gt-dir --gt-dir --output required")
    if not 0 < args.max_seconds < math.inf:
        parser.error("max-seconds must be positive and finite")
    for name in ("root", "case_dir", "stage_gt_dir", "gt_dir", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    # Force CPU before any import of torch/PyTorch3D; never touch GPU admission.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "stage": "setup"}
    json_write(args.output / "command.json", {"argv": sys.argv, "script_sha256": digest(Path(__file__)),
        "CPU_only": True, "max_seconds": args.max_seconds, "cache_clouds": args.cache_clouds})
    try:
        run(args, state, started)
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        state.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                     peak_process_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     peak_process_rss_units="bytes" if platform.system() == "Darwin" else "KiB")
        json_write(args.output / "report.json", state)
        json_write(args.output / "progress.json", state)
    return int(state["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
