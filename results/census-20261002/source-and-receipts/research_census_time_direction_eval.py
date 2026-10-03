"""CPU evaluation of qualified frozen Stage-II time-coordinate diagnostic arms.

Reuse the exact saved Stage-GT alignment and qualified motion-control native
metrics. Never run ICP, a model, or CUDA. Independently verify forward bitwise
reproduction and timestamp-preserving row-permutation tolerances before scoring
the row-permutation and time-reversal arms. All arm records remain in output.
GT is evaluation-only; direction dependence is not a causal error/uncertainty or
new-method claim. Native normal/degeneration diagnostics are new measurements;
cached native CD4D/CD-M/amplitude are never recomputed.
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
import tempfile
import time
import traceback

from research_census_eval import digest, load_arrays, load_official
from research_census_motion_controls import ALLOWED_UIDS, amplitude, array_hash, motion_table, source_hash
from research_census_stage_gt import transform_numpy
from research_census_time_direction import array_digest
from research_three_ideas import json_write

ARMS = ("forward", "row_permutation", "time_reversal")
RMS_LIMIT, MAX_LIMIT = 1e-4, 1e-3
N, T = 100000, 16


def exact(first, second):
    return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()


def difference(candidate, native, diagonal):
    import numpy as np
    delta = candidate[1:].astype(np.float64) - native[1:].astype(np.float64)
    return {"bitwise_equal": exact(candidate, native),
            "moving_rms_xyz_over_D": float(np.sqrt(np.mean(delta * delta)) / diagonal),
            "moving_max_abs_coordinate_over_D": float(np.abs(delta).max() / diagonal)}


def validate_sequence(path, report, native, native_faces, native_ids, arm):
    import numpy as np
    actual = digest(path)
    if actual != report.get("sequence_sha256") or actual != report.get("sha256", {}).get("sequence.npz"):
        raise ValueError(f"Arm sequence hash mismatch: {arm}")
    with np.load(path, allow_pickle=False) as saved:
        vertices, faces = saved["vertices"].copy(), saved["faces"].copy()
        ids = saved["query_vertex_ids"].copy()
        frames, times, clocks = saved["frame_indices"], saved["timesteps"], saved["decoder_clock_times"]
        expected_clocks = 15 - np.arange(16) if arm == "time_reversal" else np.arange(16)
        if not np.array_equal(frames, np.arange(16)) or not np.array_equal(times, np.arange(16)) or not np.array_equal(clocks, expected_clocks):
            raise ValueError(f"Physical chronology/clock mismatch: {arm}")
    if vertices.shape != native.shape or not np.isfinite(vertices).all():
        raise ValueError(f"Bad arm geometry shape or nonfinite values: {arm}")
    if not exact(vertices[0], native[0]) or not exact(faces, native_faces) or not exact(ids, native_ids):
        raise ValueError(f"Anchor/topology/query identity mismatch: {arm}")
    mapping = report.get("mapping", {})
    expected_rows = list(range(15, -1, -1)) if arm == "row_permutation" else list(range(16))
    expected_source = 15 if arm == "time_reversal" else 0
    expected_target = list(range(14, -1, -1)) if arm == "time_reversal" else list(range(1, 16))
    expected_row_clocks = list(range(15, -1, -1)) if arm in ("row_permutation", "time_reversal") else list(range(16))
    if mapping.get("latent_row_physical_ids") != expected_rows or mapping.get("source_physical_frame") != 0:
        raise ValueError(f"Latent/source physical mapping mismatch: {arm}")
    if mapping.get("source_clock_time") != expected_source or mapping.get("target_clock_times") != expected_target:
        raise ValueError(f"Source/target clock mapping mismatch: {arm}")
    if mapping.get("latent_row_clock_times") != expected_row_clocks or mapping.get("target_physical_frames") != list(range(1, 16)):
        raise ValueError(f"Latent row clock or target physical identity mismatch: {arm}")
    return vertices


def face_geometry(vertices, faces, matrix):
    """Normals from transformed triangles handle anisotropic alignment correctly."""
    import numpy as np
    aligned = transform_numpy(vertices, matrix)
    triangles = aligned[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=-1)
    normal = np.zeros_like(cross)
    valid = area2 > 0
    normal[valid] = cross[valid] / area2[valid, None]
    return area2, normal


def geometry_diagnostics(vertices, native, faces, matrix):
    import numpy as np
    diagonal = float(np.linalg.norm(np.ptp(transform_numpy(native[0], matrix), axis=0)))
    threshold = 1e-12 * diagonal * diagonal
    rows, normals = [], []
    for frame in range(T):
        area2, normal = face_geometry(vertices[frame], faces, matrix)
        native_area2, native_normal = face_geometry(native[frame], faces, matrix)
        valid = (area2 > threshold) & (native_area2 > threshold)
        cosine = np.einsum("ij,ij->i", normal[valid], native_normal[valid])
        rows.append({"frame": frame, "faces": len(faces),
            "near_degenerate_faces": int(np.sum(area2 <= threshold)),
            "near_degenerate_fraction": float(np.mean(area2 <= threshold)),
            "twice_area_threshold": threshold, "total_surface_area": float(area2.sum() / 2),
            "minimum_triangle_area": float(area2.min() / 2),
            "face_normal_cosine_to_native_mean": float(cosine.mean()) if len(cosine) else None,
            "face_normal_opposed_to_native_fraction": float(np.mean(cosine < 0)) if len(cosine) else None,
            "normal_comparison_valid_faces": int(valid.sum())})
        normals.append(normal)
    return {"frames": rows, "normal_comparison": "Same physical frame and fixed face IDs vs native; orientation change is not a claim of triangle inversion/self-intersection",
            "degeneration_definition": "Twice triangle area <=1e-12 times fixed aligned-anchor diagonal squared"}, normals


def normal_consistency(points, sample_normals, gt_positions, gt_normals):
    import numpy as np
    from scipy.spatial import KDTree
    ids_p = np.random.RandomState(44).permutation(N)[:10000]
    ids_g = np.random.RandomState(45).permutation(N)[:10000]
    _, p_to_g = KDTree(gt_positions).query(points[ids_p])
    _, g_to_p = KDTree(points).query(gt_positions[ids_g])

    def directional(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        valid = (na > 1e-12) & (nb > 1e-12)
        if not valid.any():
            return {"mean_absolute_cosine": None, "mean_signed_cosine": None, "valid_queries": 0}
        dot = np.einsum("ij,ij->i", a[valid], b[valid]) / (na[valid] * nb[valid])
        dot = np.clip(dot, -1, 1)
        return {"mean_absolute_cosine": float(np.abs(dot).mean()), "mean_signed_cosine": float(dot.mean()),
                "valid_queries": int(valid.sum())}

    forward = directional(sample_normals[ids_p], gt_normals[p_to_g])
    backward = directional(sample_normals[g_to_p], gt_normals[ids_g])
    values = [item["mean_absolute_cosine"] for item in (forward, backward)]
    return {"pred_to_gt": forward, "gt_to_pred": backward,
            "symmetric_mean_absolute_cosine": sum(values) / 2 if all(v is not None for v in values) else None,
            "definition": "Auxiliary nearest-surface normal consistency, two direction means averaged; not an official ActionBench metric; flat face normals after shared affine transform"}


def descriptive_summary(metrics):
    normal = [r["symmetric_mean_absolute_cosine"] for r in metrics["normal_consistency"]
              if r["symmetric_mean_absolute_cosine"] is not None]
    amplitude_rows = metrics["trajectory_amplitude"]
    geometry_rows = metrics["geometry"]["frames"]
    return {"cd4d": metrics["cd_4d_shared_alignment"], "cd_motion": metrics["cd_motion"],
            "mean_normal_absolute_cosine": sum(normal) / len(normal) if normal else None,
            "normal_valid_frames": len(normal),
            "mean_displacement_from_frame0": sum(r["mean_displacement_from_frame0"] for r in amplitude_rows) / T,
            "endpoint_displacement_from_frame0": amplitude_rows[-1]["mean_displacement_from_frame0"],
            "mean_step_displacement_frames1_to15": sum(r["mean_step_displacement"] for r in amplitude_rows[1:]) / (T - 1),
            "mean_near_degenerate_face_fraction": sum(r["near_degenerate_fraction"] for r in geometry_rows) / T,
            "maximum_near_degenerate_face_count": max(r["near_degenerate_faces"] for r in geometry_rows)}


def qualify_cache(args, generation, gt_path, matrix_path):
    import numpy as np
    base = args.native_cache_dir
    cache_report = json.loads((base / "report.json").read_text())
    provenance = json.loads((base / "provenance.json").read_text())
    protocol = json.loads((base / "protocol.json").read_text())
    controls = json.loads((base / "controls.json").read_text())
    shapes = json.loads((base / "per-frame-shape.json").read_text())
    if cache_report.get("status") != "completed" or (cache_report.get("uid"), cache_report.get("seed")) != (generation["uid"], 42):
        raise ValueError("Native metric cache incomplete or different UID/seed")
    for path in (args.case_dir / "sequence.npz", gt_path, matrix_path):
        if digest(path) != source_hash(provenance, path.name):
            raise ValueError("Native cache source mismatch: " + path.name)
    if protocol.get("points") != N or protocol.get("frames") != list(range(T)):
        raise ValueError("Native metric cache sampling protocol differs")
    map_path = base / "sampling-and-maps.npz"
    if digest(map_path) != provenance.get("sampling_and_maps_sha256"):
        raise ValueError("Native sampling-map cache hash mismatch")
    with np.load(map_path, allow_pickle=False) as saved:
        maps = {key: saved[key].copy() for key in ("material_face_indices", "material_barycentric", "pred_to_gt_firstframe", "gt_to_pred_firstframe", "query_predicted_ids", "shared_alignment_matrix")}
    if not exact(maps["shared_alignment_matrix"], np.load(matrix_path, allow_pickle=False)):
        raise ValueError("Native cache alignment matrix bytes differ")
    if not np.array_equal(maps["query_predicted_ids"], np.random.RandomState(44).permutation(N)[:10000]):
        raise ValueError("Native cache query subset differs from official seed44")
    native = controls["native"]
    if native["cd_motion_official_formula"] != cache_report.get("native_cd_motion"):
        raise ValueError("Native cached scalar disagrees between reports")
    for rows in (native["per_frame_motion"], native["trajectory_amplitude"], shapes["frames"]):
        if [row["frame"] for row in rows] != list(range(T)):
            raise ValueError("Native cache is missing physical frames")
    if [row.get("independent_sample_seed") for row in shapes["frames"]] != list(range(44, 60)):
        raise ValueError("Native framewise seeds differ")
    if not math.isclose(sum(row["sum_unsquared"] for row in native["per_frame_motion"]) / T, native["cd_motion_official_formula"], rel_tol=2e-6, abs_tol=2e-6):
        raise ValueError("Native motion cache per-frame aggregate mismatch")
    shape_mean = sum(row["official_framewise_cd_shared_alignment"] for row in shapes["frames"]) / T
    if not math.isclose(shape_mean, shapes["shared_alignment_cd4d_16frame_mean"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Native shape cache aggregate mismatch")
    if cache_report.get("positive_control_status") != "invariant":
        raise ValueError("Native cache identity control was not invariant")
    return cache_report, provenance, native, shapes, maps


def run(args, state, started):
    import numpy as np
    import torch
    import trimesh
    torch.set_num_threads(1)

    def check(stage):
        state["stage"] = stage
        state["elapsed_seconds"] = time.monotonic() - started
        json_write(args.output / "progress.json", state)
        if state["elapsed_seconds"] > args.max_seconds:
            raise TimeoutError("CPU evaluation stage budget exceeded")
        print("TIME_DIRECTION_EVAL", stage, round(state["elapsed_seconds"], 2), flush=True)

    check("identity_and_control_validation")
    generation = json.loads((args.case_dir / "report.json").read_text())
    direction = json.loads((args.direction_dir / "report.json").read_text())
    stage_gt = json.loads((args.stage_gt_dir / "report.json").read_text())
    if generation.get("status") != "completed" or generation.get("uid") not in ALLOWED_UIDS or generation.get("seed") != 42:
        raise ValueError("Require a completed first-two-assets seed42 native case")
    state.update(uid=generation["uid"], seed=42, independent_assets=1)
    if direction.get("status") != "completed" or not direction.get("controls_passed"):
        raise ValueError("Direction run did not complete both control gates; no arm metric evaluation")
    if direction.get("GT_read") is not False or direction.get("training") is not False:
        raise ValueError("Direction diagnostic must declare no GT access or training")
    if stage_gt.get("status") != "completed":
        raise ValueError("Shared alignment diagnostic incomplete")
    for report in (direction, stage_gt):
        if (report.get("uid"), report.get("seed")) != (generation["uid"], 42):
            raise ValueError("UID/seed mismatch in parent diagnostics")
    for filename, expected in direction.get("source_hashes", {}).items():
        if digest(args.case_dir / filename) != expected:
            raise ValueError("Direction parent source hash mismatch: " + filename)
    for required in ("prepared.npz", "denoised.npz", "sequence.npz", "report.json"):
        if required not in direction.get("source_hashes", {}):
            raise ValueError("Missing direction parent source identity: " + required)
    gt_path = args.gt_dir / generation["uid"] / "surfaces.npy"
    matrix_path = args.stage_gt_dir / "shared-anchor-transform.npy"
    if digest(matrix_path) != stage_gt.get("shared_matrix_sha256"):
        raise ValueError("Stage-GT alignment hash mismatch")
    native, faces, gt = load_arrays(args.case_dir / "sequence.npz", gt_path)
    if digest(args.case_dir / "sequence.npz") != generation.get("sha256", {}).get("sequence.npz"):
        raise ValueError("Native sequence no longer matches generation report")
    with np.load(args.case_dir / "sequence.npz", allow_pickle=False) as saved:
        ids = saved["query_vertex_ids"].copy()
    if not np.array_equal(ids, np.arange(native.shape[1])) or array_digest(ids) != direction.get("query_vertex_ids_sha256"):
        raise ValueError("Direction source query identity differs")
    diagonal = float(np.linalg.norm(np.ptp(native[0].astype(np.float64), axis=0)))
    variants = {}
    arm_reports = {}
    for arm in ARMS:
        folder = args.direction_dir / "variants" / arm
        report = json.loads((folder / "report.json").read_text())
        arm_reports[arm] = report
        state["arms"][arm]["source_report_status"] = report.get("status")
        if report.get("status") != "completed" or report.get("uid") != generation["uid"] or report.get("seed") != 42:
            raise ValueError("Arm incomplete or identity mismatch: " + arm)
        variants[arm] = validate_sequence(folder / "sequence.npz", report, native, faces, ids, arm)
        state["arms"][arm]["independent_discrepancy"] = difference(variants[arm], native, diagonal)
    if not exact(variants["forward"], native):
        raise ValueError("Independent forward bitwise gate failed")
    perm = difference(variants["row_permutation"], variants["forward"], diagonal)
    if perm["moving_rms_xyz_over_D"] > RMS_LIMIT or perm["moving_max_abs_coordinate_over_D"] > MAX_LIMIT:
        raise ValueError("Independent timestamp-preserving row-permutation gate failed")
    if not arm_reports["forward"].get("control_pass") or not arm_reports["row_permutation"].get("control_pass"):
        raise ValueError("Runner control flags disagree with independent checks")
    state["independent_control_gates_passed"] = True
    cache_report, cache_provenance, cached_native, cached_shape, cached_maps = qualify_cache(args, generation, gt_path, matrix_path)
    matrix = np.load(matrix_path, allow_pickle=False)
    _, official = load_official(args.root / "repo", cpu_rng_fix=True)
    if official["sha256"] != cache_provenance.get("official", {}).get("sha256"):
        raise ValueError("Official metric/sampling source differs from native cache")
    from chamfer import compute_chamfer_score, compute_motion_chamfer_score
    import sample_mesh
    from pytorch3d.transforms import Transform3d
    transform = Transform3d(matrix=torch.from_numpy(matrix))
    sample_faces = torch.from_numpy(cached_maps["material_face_indices"])
    barycentric = torch.from_numpy(cached_maps["material_barycentric"])

    def material_cloud(vertices):
        meshes = [trimesh.Trimesh(frame, faces, process=False) for frame in vertices]
        sample = sample_mesh.apply_baryc_sampling_on_meshes(
            sample_mesh.join_meshes_as_batch([sample_mesh.trimesh_to_pytorch3d(mesh) for mesh in meshes]),
            sample_faces, barycentric)
        return transform.transform_points(sample).numpy()

    check("native_metric_cache_qualification")
    cloud_path = args.native_cache_dir / "aligned-material-clouds.npz"
    if cloud_path.is_file():
        with np.load(cloud_path, allow_pickle=False) as saved:
            native_cloud = saved["native_aligned"].copy()
        cloud_source = "cached_cloud"
    else:
        native_cloud = material_cloud(native)
        cloud_source = "reconstructed_from_cached_faces_and_barycentric_weights_no_metric_rerun"
    if array_hash(native_cloud) != cache_provenance.get("native_material_cloud_sha256"):
        raise ValueError("Native material-cloud bytes do not match qualified cache")
    nu, mu = cached_maps["pred_to_gt_firstframe"], cached_maps["gt_to_pred_firstframe"]
    gt_raw = np.load(gt_path, mmap_mode="r", allow_pickle=False)
    protocol = {"scope": "Independent CPU evaluation of three retained Stage-II diagnostic arms on one asset; no new method claim",
        "frames": list(range(T)), "samples": N, "framewise_sampling_seeds": list(range(44, 60)),
        "motion_sampling": "Reuse exact official synchronized frame0 face/barycentric seed44 sample from qualified native cache",
        "shape_query_seeds": {"predicted": 44, "gt": 45}, "alignment": "Exact saved Stage-GT matrix for all arms; no new fit",
        "native_metrics": "CD4D/CD-M/per-frame motion/amplitude reused from hash-qualified cache; no native core metric recomputation",
        "native_geometry": "Normal consistency and triangle degeneration were not cached; measured here for fair comparison",
        "control_gates": {"forward": "independent bitwise equality", "permutation_rms_over_D_limit": RMS_LIMIT, "permutation_max_over_D_limit": MAX_LIMIT},
        "interpretation": "Changes are associated with decoder time coordinates; no claim of causal reconstruction error, useful uncertainty, or reverse-diffusion effect",
        "normal_metric": "Auxiliary symmetric nearest-surface absolute normal cosine, 10k queries per direction, GTworld normals; predicted flatface normals computed after anisotropic alignment",
        "degeneration": "Fixed face identities; area threshold scaled once by aligned anchor diagonal; normal opposition to native is not a self-intersection test",
        "budget_seconds": args.max_seconds, "GT_use": "evaluation only; not used by the direction runner"}
    json_write(args.output / "protocol.json", protocol)
    native_geometry, native_normals = geometry_diagnostics(native, native, faces, matrix)
    native_normal_rows = []

    def sample_with_normals(vertices, frame, normals):
        mesh = trimesh.Trimesh(vertices[frame], faces, process=False)
        sampled = sample_mesh.sample_points(mesh, N, seed=44 + frame)
        check_points, face_ids = trimesh.sample.sample_surface(mesh, count=N, seed=44 + frame)
        if not np.array_equal(sampled.numpy(), np.asarray(check_points, dtype=np.float32)):
            raise ValueError("Face-id sampler differs from official point sampler")
        aligned = transform.transform_points(sampled[None])[0].numpy()
        return aligned, normals[frame][face_ids]

    for frame in range(T):
        check(f"native_new_normal_metric_frame{frame}")
        positions, normal_samples = sample_with_normals(native, frame, native_normals)
        native_normal_rows.append(dict(frame=frame, **normal_consistency(positions, normal_samples, gt[frame], gt_raw[frame, :, 3:])))
    baseline = {"cd_4d_shared_alignment": cached_shape["shared_alignment_cd4d_16frame_mean"],
                "cd_motion": cached_native["cd_motion_official_formula"],
                "per_frame_shape": cached_shape["frames"], "per_frame_motion": cached_native["per_frame_motion"],
                "trajectory_amplitude": cached_native["trajectory_amplitude"],
                "normal_consistency": native_normal_rows, "geometry": native_geometry,
                "core_metrics_reused": True, "native_cloud_source": cloud_source}
    json_write(args.output / "native-reference.json", baseline)
    baseline_summary = descriptive_summary(baseline)
    state["native_reference_summary"] = baseline_summary
    state["gt_trajectory_amplitude_reference"] = cache_report.get("gt_trajectory_amplitude")
    state["arms"]["forward"].update(status="evaluated_from_bitwise_native_cache", metrics=baseline,
                                       summary=baseline_summary, delta_cd4d_vs_native=0., delta_cdm_vs_native=0.)
    for arm in ("row_permutation", "time_reversal"):
        check(arm + "_motion_metric")
        vertices = variants[arm]
        cloud = material_cloud(vertices)
        if not exact(cloud[0], native_cloud[0]):
            raise ValueError("Material-sampled anchor changed despite mesh identity")
        geometry, normals = geometry_diagnostics(vertices, native, faces, matrix)
        cdm = float(compute_motion_chamfer_score(torch.from_numpy(cloud), torch.from_numpy(gt)))
        motion = motion_table(cloud, gt, nu, mu)
        if not math.isclose(cdm, sum(row["sum_unsquared"] for row in motion) / T, rel_tol=2e-6, abs_tol=2e-6):
            raise ValueError("Official arm CD-M disagrees with cached first-frame assignments")
        shape_rows, normal_rows = [], []
        for frame in range(T):
            check(f"{arm}_frame{frame}_shape_normal")
            positions, normal_samples = sample_with_normals(vertices, frame, normals)
            cd = float(compute_chamfer_score(pred=torch.from_numpy(positions), gt=torch.from_numpy(gt[frame])))
            shape_rows.append({"frame": frame, "sampling_seed": 44 + frame, "cd4d_contribution": cd,
                               "delta_vs_native": cd - cached_shape["frames"][frame]["official_framewise_cd_shared_alignment"]})
            normal_rows.append(dict(frame=frame, **normal_consistency(positions, normal_samples, gt[frame], gt_raw[frame, :, 3:])))
        cd4d = sum(row["cd4d_contribution"] for row in shape_rows) / T
        metrics = {"cd_4d_shared_alignment": cd4d, "cd_motion": cdm, "per_frame_shape": shape_rows,
                   "per_frame_motion": motion, "trajectory_amplitude": amplitude(cloud),
                   "normal_consistency": normal_rows, "geometry": geometry, "core_metrics_reused": False}
        folder = args.output / arm
        folder.mkdir()
        json_write(folder / "metrics.json", metrics)
        if args.cache_clouds:
            np.savez_compressed(folder / "aligned-material-clouds.npz", aligned_material_cloud=cloud)
        summary = descriptive_summary(metrics)
        changes = {key: value - baseline_summary[key] for key, value in summary.items()
                   if value is not None and baseline_summary[key] is not None and key != "normal_valid_frames"}
        state["arms"][arm].update(status="evaluated", metrics=metrics, summary=summary,
            delta_summary_vs_native=changes,
            delta_cd4d_vs_native=cd4d - baseline["cd_4d_shared_alignment"],
            delta_cdm_vs_native=cdm - baseline["cd_motion"])
        json_write(args.output / "report.json", state)
    source_paths = [args.case_dir / "report.json", args.case_dir / "sequence.npz", gt_path, matrix_path,
                    args.direction_dir / "report.json", args.stage_gt_dir / "report.json"]
    source_paths += [args.native_cache_dir / name for name in ("report.json", "provenance.json", "protocol.json", "controls.json", "per-frame-shape.json", "sampling-and-maps.npz")]
    source_paths += [args.direction_dir / "variants" / arm / name for arm in ARMS for name in ("report.json", "sequence.npz")]
    json_write(args.output / "provenance.json", {"script_sha256": digest(Path(__file__)), "official": official,
        "source_files_sha256": {str(path): digest(path) for path in source_paths},
        "same_matrix_sha256": digest(matrix_path), "native_core_metrics_recomputed": False,
        "native_cache_qualified": True, "torch_cuda_initialized": torch.cuda.is_initialized()})
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only evaluation unexpectedly initialized CUDA")
    state.update(status="completed", all_arms_retained=True, cuda_used=False, icp_run=False,
                 native_core_metrics_recomputed=False, GT_used_only_for_evaluation=True,
                 scientific_scope="One paired asset diagnostic; all16frames are repeated observations, not16 independent assets. No causal-error or new-method claim.")


def self_test():
    import numpy as np
    # Geometry exposes collapse and orientation changes missed by average displacement.
    native = np.tile(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])[None], (16, 1, 1)).astype(np.float32)
    faces = np.array([[0, 1, 2]])
    matrix = np.eye(4)[None]
    altered = native.copy(); altered[8, 2] = altered[8, 0]; altered[15, 2] = [0, -1, 0]
    geom, _ = geometry_diagnostics(altered, native, faces, matrix)
    assert geom["frames"][8]["near_degenerate_faces"] == 1
    assert geom["frames"][15]["face_normal_opposed_to_native_fraction"] == 1.
    scaled = np.eye(4); scaled[:3, :3] = np.diag([2., 3., 4.])
    area2, normal = face_geometry(native[0], faces, scaled)
    assert np.allclose(area2, [6.]) and np.array_equal(normal, [[0., 0., 1.]])
    # Forward identity and frozen numerical gate distinguish permitted noise from a true departure.
    assert difference(native.copy(), native, 1.)["bitwise_equal"]
    small = native.copy(); small[1:, :, 2] += 1e-5
    large = native.copy(); large[1:, :, 2] += .1
    assert difference(small, native, 1.)["moving_rms_xyz_over_D"] < RMS_LIMIT
    assert difference(large, native, 1.)["moving_max_abs_coordinate_over_D"] > MAX_LIMIT
    with tempfile.TemporaryDirectory(prefix="direction-eval-fixture-") as temporary:
        path = Path(temporary) / "sequence.npz"
        ids = np.arange(3)
        def save_fixture(vertices, clocks):
            np.savez(path, vertices=vertices, faces=faces, query_vertex_ids=ids,
                     frame_indices=np.arange(16), timesteps=np.arange(16), decoder_clock_times=clocks)
        save_fixture(native, 15 - np.arange(16))
        report = {"sequence_sha256": digest(path), "sha256": {"sequence.npz": digest(path)},
                  "mapping": {"latent_row_physical_ids": list(range(16)), "source_physical_frame": 0,
                              "source_clock_time": 15, "target_clock_times": list(range(14, -1, -1)),
                              "latent_row_clock_times": list(range(15, -1, -1)), "target_physical_frames": list(range(1, 16))}}
        assert exact(validate_sequence(path, report, native, faces, ids, "time_reversal"), native)
        bad_anchor = native.copy(); bad_anchor[0, 0, 0] += .01
        save_fixture(bad_anchor, 15 - np.arange(16))
        report["sequence_sha256"] = report["sha256"]["sequence.npz"] = digest(path)
        try:
            validate_sequence(path, report, native, faces, ids, "time_reversal")
        except ValueError as exc:
            assert "Anchor/topology/query" in str(exc)
        else:
            raise AssertionError("Changed anchor accepted despite updated file hash")
        save_fixture(native, np.arange(16))
        report["sequence_sha256"] = report["sha256"]["sequence.npz"] = digest(path)
        try:
            validate_sequence(path, report, native, faces, ids, "time_reversal")
        except ValueError as exc:
            assert "chronology/clock" in str(exc)
        else:
            raise AssertionError("Wrong physical clock convention accepted")
    return {"status": "pass", "checks": ["face collapse detected at physicalframe8", "opposed normal detected at physicalframe15",
        "anisotropic alignment correctly changes triangle area", "forward identity and numerical gate distinguish noise from largechange",
        "valid reversal schema accepted", "updated hash cannot hide changed anchor", "wrong reversed clock rejected"],
        "scope": "NumPy geometry/gate fixtures, no official metrics/model/CUDA run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--direction-dir", type=Path)
    parser.add_argument("--stage-gt-dir", type=Path)
    parser.add_argument("--native-cache-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float, default=300.)
    parser.add_argument("--cache-clouds", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return 0
    names = ("root", "case_dir", "direction_dir", "stage_gt_dir", "native_cache_dir", "gt_dir", "output")
    if any(getattr(args, name) is None for name in names):
        parser.error("All seven path arguments required")
    if not 0 < args.max_seconds <= 300:
        parser.error("CPU stage budget must be in(0,300]seconds")
    for name in names:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
             "arms": {arm: {"status": "not_evaluated"} for arm in ARMS}, "independent_control_gates_passed": False}
    json_write(args.output / "command.json", {"argv": sys.argv, "script_sha256": digest(Path(__file__)),
               "CPU_only": True, "max_seconds": args.max_seconds, "timeout_scope": "Stage checks; external deadline recommended"})
    started = time.monotonic()
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
