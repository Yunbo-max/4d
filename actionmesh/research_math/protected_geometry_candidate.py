"""C02 protected geometry repair over a complete predicted 4D mesh.

The method first computes one declared, shared geometry-only ARAP repair from
the predicted sequence.  It then projects that *same* desired update under a
fixed area metric so that anchor-frame positions and area-weighted patch-centroid
velocities are unchanged.  The module also exports the unprotected repair and
the equal-W-norm scalar control required by the reviewed mathematics.

Patches, weights and the repair are inferred only from the predicted mesh.
Ground truth, scorer state, cameras, video labels and model internals are not
inputs.  This is generated source; Local acceptance and scientific admission
remain separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np

from research_ten.m01_elasticity import arap_fit, mesh_edges


CANDIDATE_ID = "4d-math-20261006-c02"
METHOD_ID = CANDIDATE_ID
ROLES = ("geometry_only", "strength_matched_blend", "protected_step")
FLOAT32_RELATIVE_UPDATE_ERROR_CAP = 5e-3
FLOAT32_RELATIVE_STRENGTH_NORM2_MISMATCH_CAP = 1e-2


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _positive(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def _positive_integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_native_arrays(arrays: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"vertices", "faces", "timesteps", "frame_indices", "query_vertex_ids"}
    if not required.issubset(arrays):
        raise ValueError("Complete native mesh metadata required")
    vertices = np.asarray(arrays["vertices"])
    faces = np.asarray(arrays["faces"])
    times = np.asarray(arrays["timesteps"])
    if (vertices.ndim != 3 or vertices.shape[0] != 16 or vertices.shape[-1] != 3
            or vertices.shape[1] < 4 or vertices.dtype != np.dtype(np.float32)
            or not np.isfinite(vertices).all()):
        raise ValueError("Exactly 16 finite float32 native mesh frames required")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= vertices.shape[1])):
        raise ValueError("Valid shared integer triangle topology required")
    if (times.shape != (16,) or not np.issubdtype(times.dtype, np.number)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Sixteen finite strictly increasing timestamps required")
    if (not np.issubdtype(arrays["frame_indices"].dtype, np.integer)
            or not np.array_equal(arrays["frame_indices"], np.arange(16))):
        raise ValueError("Original frame order 0..15 required")
    if (not np.issubdtype(arrays["query_vertex_ids"].dtype, np.integer)
            or not np.array_equal(arrays["query_vertex_ids"], np.arange(vertices.shape[1]))):
        raise ValueError("Original identity vertex mapping required")
    for name, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError("Non-numeric or nonfinite native array: " + name)
    return vertices, faces.astype(np.int64, copy=False), times.astype(np.float64)


def vertex_area_weights(anchor: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return strictly positive barycentric vertex areas from the anchor mesh."""
    triangles = anchor[faces]
    doubled = np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    scale = max(1.0, float(np.max(np.linalg.norm(anchor, axis=1))))
    tolerance = 100.0 * np.finfo(np.float64).eps * scale * scale
    if not np.isfinite(doubled).all() or np.any(doubled <= tolerance):
        raise ValueError("Degenerate/nonfinite anchor triangles are unsupported")
    area = np.zeros(len(anchor), dtype=np.float64)
    for corner in range(3):
        np.add.at(area, faces[:, corner], doubled / 6.0)
    if np.any(area <= tolerance):
        raise ValueError("Every native vertex must have positive incident area")
    return area


def _adjacency(anchor: np.ndarray, edges: np.ndarray) -> list[list[tuple[int, float]]]:
    result: list[list[tuple[int, float]]] = [[] for _ in range(len(anchor))]
    lengths = np.linalg.norm(anchor[edges[:, 0]] - anchor[edges[:, 1]], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("Positive finite anchor edge lengths required")
    for (left, right), length in zip(edges, lengths):
        result[int(left)].append((int(right), float(length)))
        result[int(right)].append((int(left), float(length)))
    return result


def _distances(adjacency: list[list[tuple[int, float]]], sources: list[int]) -> np.ndarray:
    distance = np.full(len(adjacency), np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for source in sorted(sources):
        distance[source] = 0.0
        heapq.heappush(queue, (0.0, source))
    while queue:
        value, vertex = heapq.heappop(queue)
        if value != distance[vertex]:
            continue
        for neighbor, length in adjacency[vertex]:
            candidate = value + length
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distance


def deterministic_patches(anchor: np.ndarray, faces: np.ndarray,
                          patch_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic farthest-point patches with deterministic vertex-ID tie breaks."""
    patch_count = _positive_integer("patch_count", patch_count)
    if patch_count > len(anchor):
        raise ValueError("patch_count cannot exceed native vertex count")
    edges = mesh_edges(faces, len(anchor))
    adjacency = _adjacency(anchor.astype(np.float64), edges)
    unseen = set(range(len(anchor)))
    components: list[list[int]] = []
    while unseen:
        seed = min(unseen)
        stack, component = [seed], []
        unseen.remove(seed)
        while stack:
            vertex = stack.pop(); component.append(vertex)
            for neighbor, _ in adjacency[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor); stack.append(neighbor)
        components.append(sorted(component))
    if patch_count < len(components):
        raise ValueError("patch_count must seed every connected component")
    seeds = [component[0] for component in components]
    while len(seeds) < patch_count:
        distance = _distances(adjacency, seeds)
        if not np.isfinite(distance).all():
            raise ValueError("Mesh contains an unseeded disconnected component")
        maximum = float(np.max(distance))
        seeds.append(int(np.flatnonzero(distance == maximum)[0]))
    per_seed = np.stack([_distances(adjacency, [seed]) for seed in seeds])
    labels = np.argmin(per_seed, axis=0).astype(np.int64)
    if set(labels.tolist()) != set(range(patch_count)):
        raise ValueError("Deterministic patch construction produced an empty patch")
    return labels, np.asarray(seeds, dtype=np.int64)


def protected_projection(desired_step: np.ndarray, area: np.ndarray,
                         patch_labels: np.ndarray,
                         anchor_frame: int = 0) -> tuple[np.ndarray, dict]:
    """Project into the patch-centroid-velocity nullspace under area metric.

    With the complete anchor-frame update fixed to zero, zero adjacent velocity
    change is equivalent to zero area-weighted patch centroid update at every
    frame.  The W-orthogonal projection therefore subtracts each patch/frame's
    weighted mean without forming C or W densely.
    """
    desired = np.asarray(desired_step, dtype=np.float64)
    weights = np.asarray(area, dtype=np.float64)
    labels = np.asarray(patch_labels)
    if (desired.ndim != 3 or desired.shape[-1] != 3 or not np.isfinite(desired).all()
            or weights.shape != (desired.shape[1],) or np.any(weights <= 0)
            or not np.isfinite(weights).all() or labels.shape != weights.shape
            or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("Finite desired[T,V,3], positive area[V], integer labels[V] required")
    if not isinstance(anchor_frame, (int, np.integer)) or not 0 <= anchor_frame < len(desired):
        raise ValueError("anchor_frame outside sequence")
    if np.max(np.abs(desired[anchor_frame])) != 0.0:
        raise ValueError("Desired repair must preserve the complete anchor frame exactly")
    projected = desired.copy()
    centroids = np.zeros((len(desired), int(labels.max()) + 1, 3), dtype=np.float64)
    patch_masses = []
    for patch in range(centroids.shape[1]):
        mask = labels == patch
        mass = float(weights[mask].sum())
        if not np.any(mask) or mass <= 0:
            raise ValueError("Every protected patch requires positive area")
        patch_masses.append(mass)
        centroids[:, patch] = np.einsum("v,tvc->tc", weights[mask], desired[:, mask]) / mass
        projected[:, mask] -= centroids[:, patch, None, :]
    projected[anchor_frame] = 0.0
    protected_centroids = np.zeros_like(centroids)
    for patch in range(centroids.shape[1]):
        mask = labels == patch
        protected_centroids[:, patch] = np.einsum(
            "v,tvc->tc", weights[mask], projected[:, mask]) / weights[mask].sum()
    velocity_residual = np.diff(protected_centroids, axis=0)
    removed = desired - projected
    desired_norm2 = float(np.einsum("v,tvc,tvc->", weights, desired, desired))
    projected_norm2 = float(np.einsum("v,tvc,tvc->", weights, projected, projected))
    removed_norm2 = float(np.einsum("v,tvc,tvc->", weights, removed, removed))
    orthogonality = float(np.einsum("v,tvc,tvc->", weights, projected, removed))
    scale = max(1.0, desired_norm2)
    tolerance = 1000.0 * np.finfo(np.float64).eps * scale * desired.size
    if abs(desired_norm2 - projected_norm2 - removed_norm2) > tolerance:
        raise ValueError("Weighted projection failed Pythagorean certificate")
    if abs(orthogonality) > tolerance:
        raise ValueError("Weighted projection failed orthogonality certificate")
    if float(np.max(np.abs(velocity_residual), initial=0.0)) > tolerance:
        raise ValueError("Protected patch-centroid velocity residual exceeds tolerance")
    rho = 0.0 if desired_norm2 == 0.0 else removed_norm2 / desired_norm2
    rho = float(np.clip(rho, 0.0, 1.0))
    scalar = float(np.sqrt(max(0.0, 1.0 - rho)))
    matched = scalar * desired
    rank = 3 * (len(desired) - 1) * centroids.shape[1]
    free_dimension = 3 * (len(desired) - 1) * desired.shape[1]
    nullity = free_dimension - rank
    if nullity <= 0:
        raise ValueError("Protected operator must retain a nontrivial free nullspace")
    return projected, {
        "metric": "anchor-frame barycentric area diagonal, repeated over frames/xyz",
        "constraint": "fixed patch-centroid adjacent-frame velocity plus complete anchor-frame pin",
        "protection_observation_is_linear": True,
        "nonlinear_remainder_hessian_bound": 0.0,
        "nonlinear_remainder_bound": 0.0,
        "patch_masses": patch_masses,
        "free_coordinate_dimension": free_dimension,
        "constraint_rank": rank,
        "constraint_nullity": nullity,
        "desired_w_norm_squared": desired_norm2,
        "projected_w_norm_squared": projected_norm2,
        "removed_w_norm_squared": removed_norm2,
        "weighted_orthogonality": orthogonality,
        "rho": rho,
        "strength_matched_scale": scalar,
        "strength_matched_w_norm_squared": float(np.einsum(
            "v,tvc,tvc->", weights, matched, matched)),
        "protected_centroid_velocity_residual_linf": float(
            np.max(np.abs(velocity_residual), initial=0.0)),
        "numerical_tolerance": tolerance,
        "unconstrained_local_gain": 0.5 * desired_norm2,
        "protected_local_gain": 0.5 * projected_norm2,
        "strength_matched_local_gain": (scalar - 0.5 * scalar * scalar) * desired_norm2,
    }


def compute_common_geometry_repair(vertices: np.ndarray, faces: np.ndarray,
                                   times: np.ndarray, *, arap_weight: float,
                                   temporal_weight: float, iterations: int,
                                   cg_tolerance: float,
                                   cg_max_iterations: int) -> tuple[np.ndarray, dict]:
    """Compute the one common matrix-free ARAP repair shared by all C02 arms."""
    arap_weight = _positive("arap_weight", arap_weight)
    temporal_weight = _positive("temporal_weight", temporal_weight)
    iterations = _positive_integer("iterations", iterations)
    cg_tolerance = _positive("cg_tolerance", cg_tolerance)
    cg_max_iterations = _positive_integer("cg_max_iterations", cg_max_iterations)
    edges = mesh_edges(faces, vertices.shape[1])
    result = arap_fit(
        vertices[0], vertices, edges, arap_weight, data_weight=1.0,
        iterations=iterations, anchor_frame=0, times=times,
        temporal_weight=temporal_weight, tol=cg_tolerance,
        cg_maxiter=cg_max_iterations)
    if not result.converged:
        raise RuntimeError("Common geometry-only ARAP repair did not converge")
    repaired = np.asarray(result.trajectories, dtype=np.float64)
    if repaired.shape != vertices.shape or not np.isfinite(repaired).all():
        raise ValueError("Common geometry repair returned invalid geometry")
    repaired[0] = vertices[0]
    desired = repaired - vertices.astype(np.float64)
    desired[0] = 0.0
    return desired, {
        "operator": "matrix-free local/global ARAP with correction-velocity regularization",
        "arap_weight": arap_weight,
        "temporal_weight": temporal_weight,
        "outer_iterations_requested": iterations,
        "outer_iterations_observed": len(result.energies),
        "cg_tolerance": cg_tolerance,
        "cg_max_iterations": cg_max_iterations,
        "cg_iterations": result.cg_iterations,
        "energies": result.energies,
        "converged": bool(result.converged),
    }


def export_candidate(source_case: Path, output: Path, *, uid: str,
                     expected_sequence_sha256: str, patch_count: int,
                     arap_weight: float, temporal_weight: float,
                     iterations: int, cg_tolerance: float,
                     cg_max_iterations: int) -> dict:
    """Export all three operation-isolating C02 arms as complete native meshes."""
    started = time.monotonic()
    source_case, output = Path(source_case), Path(output)
    if not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    source_path = source_case / "sequence.npz"
    report_path = source_case / "report.json"
    if digest(source_path) != expected_sequence_sha256:
        raise ValueError("Source sequence differs from explicit pin")
    source_report = json.loads(report_path.read_text())
    if (source_report.get("status") != "completed" or source_report.get("uid") != uid
            or source_report.get("sha256", {}).get("sequence.npz") != expected_sequence_sha256):
        raise ValueError("Completed source report and matching sequence required")
    seed = source_report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer inference seed required")
    with np.load(source_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    vertices, faces, times = validate_native_arrays(arrays)
    area = vertex_area_weights(vertices[0].astype(np.float64), faces)
    labels, patch_seeds = deterministic_patches(vertices[0].astype(np.float64), faces,
                                                 patch_count)
    desired, repair = compute_common_geometry_repair(
        vertices, faces, times, arap_weight=arap_weight,
        temporal_weight=temporal_weight, iterations=iterations,
        cg_tolerance=cg_tolerance, cg_max_iterations=cg_max_iterations)
    protected, projection = protected_projection(desired, area, labels)
    matched = projection["strength_matched_scale"] * desired
    updates = {
        "geometry_only": desired,
        "strength_matched_blend": matched,
        "protected_step": protected,
    }
    native_vertices = {
        role: (vertices.astype(np.float64) + update).astype(np.float32)
        for role, update in updates.items()
    }
    for converted in native_vertices.values():
        converted[0] = vertices[0]
    # These two diagnostics describe the *actual native float32 artifact*, not
    # merely a cast of the update.  Adding the update to a nonzero float32
    # source and then casting can round differently from casting the update in
    # isolation, so freeze-time recomputation must use this same construction.
    protected_vertices32 = native_vertices["protected_step"]
    protected32 = protected_vertices32.astype(np.float64) - vertices.astype(np.float64)
    matched32 = (native_vertices["strength_matched_blend"].astype(np.float64)
                 - vertices.astype(np.float64))
    float32_centroid_residual = 0.0
    for patch in range(int(labels.max()) + 1):
        mask = labels == patch
        centroids32 = np.einsum(
            "v,tvc->tc", area[mask], protected32[:, mask]) / area[mask].sum()
        float32_centroid_residual = max(
            float32_centroid_residual,
            float(np.max(np.abs(np.diff(centroids32, axis=0)), initial=0.0)))
    coordinate_scale = max(
        1.0, float(np.max(np.abs(vertices), initial=0.0)),
        float(np.max(np.abs(protected_vertices32), initial=0.0)))
    float32_tolerance = 64.0 * np.finfo(np.float32).eps * coordinate_scale
    if float32_centroid_residual > float32_tolerance:
        raise ValueError("Float32 native export violates protected centroid velocity tolerance")
    projection["float32_protected_centroid_velocity_residual_linf"] = float32_centroid_residual
    projection["float32_protection_tolerance"] = float(float32_tolerance)
    projection["float32_protection_tolerance_units"] = (
        "native coordinate units per retained timestep index")
    projection["float32_projection_change_w_norm"] = float(np.sqrt(np.einsum(
        "v,tvc,tvc->", area, protected32 - protected, protected32 - protected)))
    protected32_norm2 = float(np.einsum(
        "v,tvc,tvc->", area, protected32, protected32))
    matched32_norm2 = float(np.einsum("v,tvc,tvc->", area, matched32, matched32))
    protected_error2 = float(np.einsum(
        "v,tvc,tvc->", area, protected32 - protected, protected32 - protected))
    matched_error2 = float(np.einsum(
        "v,tvc,tvc->", area, matched32 - matched, matched32 - matched))
    strength_tolerance = (
        2.0 * np.sqrt(projection["projected_w_norm_squared"] * protected_error2)
        + protected_error2
        + 2.0 * np.sqrt(projection["strength_matched_w_norm_squared"] * matched_error2)
        + matched_error2 + projection["numerical_tolerance"])
    strength_delta = abs(protected32_norm2 - matched32_norm2)
    if strength_delta > strength_tolerance:
        raise ValueError("Float32 native arms violate strength-matching tolerance")
    ideal_norm2 = projection["projected_w_norm_squared"]
    if ideal_norm2 <= projection["numerical_tolerance"]:
        if max(protected32_norm2, matched32_norm2) > 4.0 * projection["numerical_tolerance"]:
            raise ValueError("Near-zero ideal update gained material float32 native strength")
        protected_relative_error = matched_relative_error = 0.0
        strength_relative_mismatch = 0.0
    else:
        protected_relative_error = float(np.sqrt(protected_error2 / ideal_norm2))
        matched_relative_error = float(np.sqrt(matched_error2 / ideal_norm2))
        strength_relative_mismatch = float(strength_delta / ideal_norm2)
        if (protected_relative_error > FLOAT32_RELATIVE_UPDATE_ERROR_CAP
                or matched_relative_error > FLOAT32_RELATIVE_UPDATE_ERROR_CAP
                or strength_relative_mismatch
                > FLOAT32_RELATIVE_STRENGTH_NORM2_MISMATCH_CAP):
            raise ValueError("Float32 quantization materially destroys strength matching")
    projection.update(
        float32_protected_w_norm_squared=protected32_norm2,
        float32_strength_matched_w_norm_squared=matched32_norm2,
        float32_strength_match_abs_delta=strength_delta,
        float32_strength_match_tolerance=float(strength_tolerance),
        float32_strength_match_tolerance_basis=(
            "weighted norm perturbation bound from both measured float32 quantization errors"),
        float32_protected_relative_update_error=protected_relative_error,
        float32_strength_matched_relative_update_error=matched_relative_error,
        float32_strength_norm2_relative_mismatch=strength_relative_mismatch,
        float32_relative_update_error_cap=FLOAT32_RELATIVE_UPDATE_ERROR_CAP,
        float32_strength_norm2_relative_mismatch_cap=(
            FLOAT32_RELATIVE_STRENGTH_NORM2_MISMATCH_CAP),
        float32_near_zero_rule=(
            "if ideal W-norm squared <= numerical_tolerance, each actual W-norm squared "
            "must be <= 4*numerical_tolerance"),
    )
    output.mkdir(parents=True, exist_ok=False)
    certificate = output / "certificate.npz"
    np.savez_compressed(certificate, vertex_area=area, patch_labels=labels,
                        patch_seed_vertex_ids=patch_seeds, desired_step=desired,
                        protected_step=protected, strength_matched_step=matched)
    records = []
    for role in ROLES:
        role_started = time.monotonic()
        directory = output / role; directory.mkdir()
        repaired = vertices.astype(np.float64) + updates[role]
        repaired[0] = vertices[0]
        report = {
            "status": "started", "candidate_id": CANDIDATE_ID,
            "method_id": CANDIDATE_ID if role == "protected_step" else "c02-control-" + role,
            "candidate_role": role, "uid": uid, "seed": seed,
            "source_sequence_sha256": expected_sequence_sha256,
            "source_report_sha256": digest(report_path),
            "implementation_sha256": digest(Path(__file__)),
            "repair": repair, "projection": projection,
            "patch_count": int(patch_count),
            "patch_seed_vertex_ids": patch_seeds.tolist(),
            "sha256": {"certificate.npz": digest(certificate)},
            "information": "predicted full mesh/timestamps/topology only; no GT/scorer/camera/labels/model state",
            "generated_unexecuted_source": True,
            "native_qualified": False, "scientific_verdict": "not_computed",
        }
        try:
            converted = native_vertices[role]
            if not np.isfinite(converted).all() or not np.array_equal(converted[0], vertices[0]):
                raise ValueError("Float32 export changed anchor or produced nonfinite geometry")
            np.savez_compressed(directory / "sequence.npz",
                                **{**arrays, "vertices": converted})
            report["sha256"]["sequence.npz"] = digest(directory / "sequence.npz")
            report["status"] = "completed"
        except Exception as error:
            report.update(status="error", exception_type=type(error).__name__, error=str(error))
        report["elapsed_seconds"] = time.monotonic() - role_started
        write_json(directory / "report.json", report)
        records.append(report)
    manifest = {
        "kind": "c02-candidate-artifact-manifest", "version": 1,
        "candidate_id": CANDIDATE_ID, "uid": uid, "inference_seed": seed,
        "roles": list(ROLES),
        "cases": [{"case_id": uid + "-" + role, "uid": uid,
                   "case_dir": role, "role": role} for role in ROLES],
        "certificate_sha256": digest(certificate),
        "scope": "candidate artifacts only; no scorer, admission, confidence interval or verdict",
    }
    write_json(output / "manifest.json", manifest)
    summary = {
        "kind": "c02-candidate-artifacts", "version": 1,
        "candidate_id": CANDIDATE_ID, "method_id": METHOD_ID,
        "status": "completed" if all(row["status"] == "completed" for row in records) else "incomplete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "uid": uid, "inference_seed": seed, "roles": records,
        "manifest_sha256": digest(output / "manifest.json"),
        "certificate_sha256": digest(certificate),
        "elapsed_seconds": time.monotonic() - started,
        "native_qualified": False, "local_method_verified": False,
        "scientific_verdict": "not_computed", "dispatch_ready": False,
        "timing_scope": "CPU artifact repair/export only; excludes generation/scorer/collection",
    }
    write_json(output / "candidate.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-case", type=Path)
    source.add_argument("--source-sequence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--expected-sequence-sha256", required=True)
    parser.add_argument("--patch-count", type=int, required=True)
    parser.add_argument("--arap-weight", type=float, required=True)
    parser.add_argument("--temporal-weight", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--cg-tolerance", type=float, required=True)
    parser.add_argument("--cg-max-iterations", type=int, required=True)
    args = parser.parse_args()
    if args.source_sequence is not None and args.source_sequence.name != "sequence.npz":
        parser.error("--source-sequence must name sequence.npz")
    case = args.source_case if args.source_case is not None else args.source_sequence.parent
    result = export_candidate(
        case, args.output, uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256,
        patch_count=args.patch_count, arap_weight=args.arap_weight,
        temporal_weight=args.temporal_weight, iterations=args.iterations,
        cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations)
    print(json.dumps({"status": result["status"], "candidate_id": CANDIDATE_ID,
                      "native_qualified": False, "scientific_verdict": "not_computed"}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
