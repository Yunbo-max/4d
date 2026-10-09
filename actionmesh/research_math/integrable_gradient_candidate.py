"""C10 common differential target and pinned integrable reconstruction.

The input is one receipt-bound, complete native predicted mesh sequence.  A
fixed geometry-only rule constructs an oriented edge correction field ``h``:
each edge is moved toward the mean of the two incident-vertex median edge
lengths, with an explicit strength and relative-change cap.  Faces, predicted
coordinates and topology are the only inputs; GT, ActionBench, text/event
labels, cameras, scorer state and learned parameters are never read.

The same byte-identical ``h`` and positive edge weights are used by all arms:

* ``direct_common_lift`` integrates ``h`` along a deterministic spanning forest;
* ``independent_local_repair`` is a qualified face-local ARAP comparator: it
  fits one proper rotation per predicted triangle to the same ``h``/``W`` and
  averages overlapping local proposals without a global compatibility solve;
* ``pinned_integrable_solve`` minimizes ``.5 ||G delta-h||_W^2`` by a sparse,
  matrix-free conjugate-gradient Poisson solve.

The lowest numbered vertex of every connected component is pinned.  The code
does not claim to prevent flips or self intersections.  It is generated source,
not Local verification, scientific admission, native scoring or a GPU resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np


CANDIDATE_ID = "4d-math-20261006-c10"
ROLES = (
    "direct_common_lift",
    "independent_local_repair",
    "pinned_integrable_solve",
)
METHOD_IDS = {
    "direct_common_lift": "c10-control-direct-common-lift-v1",
    "independent_local_repair": "c10-control-qualified-independent-local-arap-v1",
    "pinned_integrable_solve": CANDIDATE_ID,
}
ROLE_CONTRACTS = {
    "direct_common_lift": "direct_common_lift",
    "independent_local_repair": "qualified_arap_common_target",
    "pinned_integrable_solve": "pinned_integrable_solve",
}
INTEGRABILITY_RATIO_TOLERANCE = 1e-5


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _artifact_member_paths(output: Path, reports: list[dict]) -> list[Path]:
    members = [output / "candidate.json", output / "manifest.json",
               output / "common-target.npz"]
    for report in reports:
        directory = output / report["candidate_arm"]
        members.append(directory / "report.json")
        if report["status"] == "completed":
            members.extend((directory / "sequence.npz", directory / "certificate.npz"))
    return sorted(members, key=lambda path: path.relative_to(output).as_posix())


def _write_deterministic_archive(output: Path, reports: list[dict],
                                 max_artifact_bytes: int) -> dict:
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    if max_artifact_bytes < 10240:
        raise ValueError("max_artifact_bytes must allow one tar record")
    members = _artifact_member_paths(output, reports)
    estimated = 1024
    refs = []
    for path in members:
        size = path.stat().st_size
        estimated += 512 + ((size + 511) // 512) * 512
        refs.append({"path": path.relative_to(output).as_posix(),
                     "sha256": _digest(path), "size_bytes": size})
    estimated = ((estimated + 10239) // 10240) * 10240
    if estimated > max_artifact_bytes:
        raise RuntimeError(
            f"Deterministic artifact archive upper bound {estimated} exceeds "
            f"max_artifact_bytes={max_artifact_bytes}")
    temporary = output / "artifact.tar.tmp"
    archive = output / "artifact.tar"
    with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
        for path, ref in zip(members, refs):
            info = tarfile.TarInfo(ref["path"])
            info.size = ref["size_bytes"]
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as stream:
                bundle.addfile(info, stream)
    if temporary.stat().st_size > max_artifact_bytes:
        raise RuntimeError("Written artifact archive exceeds max_artifact_bytes")
    temporary.replace(archive)
    record = {
        "kind": "c10-terminal-artifact-archive", "version": 1,
        "archive": {"path": "artifact.tar", "sha256": _digest(archive),
                    "size_bytes": archive.stat().st_size},
        "members": refs, "max_artifact_bytes": max_artifact_bytes,
        "terminal_candidate_status": json.loads((output / "candidate.json").read_text())["status"],
    }
    _write_json(output / "artifact-archive.json", record)
    return record


def materialize_candidate_archive(archive_path: Path, destination: Path, *,
                                  max_artifact_bytes: int,
                                  max_member_bytes: int) -> list[dict]:
    """Safely materialize a retained terminal artifact without tar.extract."""
    archive_path, destination = Path(archive_path), Path(destination)
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    max_member_bytes = _positive_integer("max_member_bytes", max_member_bytes)
    if archive_path.stat().st_size > max_artifact_bytes:
        raise ValueError("Artifact archive exceeds frozen byte ceiling")
    record_path = archive_path.with_name("artifact-archive.json")
    record = json.loads(record_path.read_text())
    if (record.get("archive") != {
            "path": "artifact.tar", "sha256": _digest(archive_path),
            "size_bytes": archive_path.stat().st_size}
            or record.get("max_artifact_bytes", max_artifact_bytes) > max_artifact_bytes):
        raise ValueError("Artifact archive record/hash exceeds materialization contract")
    if destination.exists():
        raise FileExistsError("Archive destination is single-use")
    rows = []
    with tarfile.open(archive_path, mode="r:") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate artifact archive member")
        total = 0
        for member in members:
            relative = Path(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or member.size > max_member_bytes):
                raise ValueError("Unsafe/nonregular/oversize artifact member")
            total += member.size
            if total > max_artifact_bytes:
                raise ValueError("Expanded artifact exceeds frozen byte ceiling")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("Regular artifact member could not be read")
            digest = hashlib.sha256()
            remaining = member.size
            with target.open("xb") as sink:
                while remaining:
                    block = source.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError("Truncated artifact member")
                    sink.write(block)
                    digest.update(block)
                    remaining -= len(block)
                if source.read(1):
                    raise ValueError("Artifact member grew beyond declared size")
            rows.append({"path": member.name, "sha256": digest.hexdigest(),
                         "size_bytes": member.size})
    if rows != record.get("members"):
        raise ValueError("Materialized member inventory differs from archive record")
    for source, target in ((archive_path, destination / "artifact.tar"),
                           (record_path, destination / "artifact-archive.json")):
        if source.stat().st_size > max_artifact_bytes:
            raise ValueError("Archive support file exceeds byte ceiling")
        remaining = source.stat().st_size
        with source.open("rb") as stream, target.open("xb") as sink:
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("Truncated archive support file")
                sink.write(block)
                remaining -= len(block)
            if stream.read(1):
                raise ValueError("Archive support file grew while copying")
    return rows


def _positive(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def _unit_interval(name: str, value: float) -> float:
    value = _positive(name, value)
    if value > 1.0:
        raise ValueError(f"{name} must be at most one")
    return value


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_native_arrays(arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    """Validate the exact native mesh interface and return vertices/faces."""
    required = {"vertices", "faces", "timesteps", "frame_indices", "query_vertex_ids"}
    if not required.issubset(arrays):
        raise ValueError("Full native mesh metadata required")
    for key, array in arrays.items():
        array = np.asarray(array)
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError(f"Non-numeric or nonfinite native array: {key}")
    vertices = np.asarray(arrays["vertices"])
    if (vertices.dtype != np.dtype(np.float32) or vertices.ndim != 3
            or vertices.shape[0] != 16 or vertices.shape[-1] != 3
            or vertices.shape[1] < 3):
        raise ValueError("Finite float32 vertices[16,V,3], V>=3 required")
    faces = np.asarray(arrays["faces"])
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= vertices.shape[1])):
        raise ValueError("Nonempty in-range integer triangle topology required")
    if np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 0] == faces[:, 2]) \
            or np.any(faces[:, 1] == faces[:, 2]):
        raise ValueError("Degenerate triangle indices are not supported")
    frame_indices = np.asarray(arrays["frame_indices"])
    if (not np.issubdtype(frame_indices.dtype, np.integer)
            or not np.array_equal(frame_indices, np.arange(16))):
        raise ValueError("Exactly 16 original frames in original order required")
    vertex_ids = np.asarray(arrays["query_vertex_ids"])
    if (not np.issubdtype(vertex_ids.dtype, np.integer)
            or not np.array_equal(vertex_ids, np.arange(vertices.shape[1]))):
        raise ValueError("Original identity vertex mapping required")
    timesteps = np.asarray(arrays["timesteps"])
    if (timesteps.shape != (16,) or np.any(np.diff(timesteps.astype(np.float64)) <= 0.0)):
        raise ValueError("One strictly increasing timestamp per original frame required")
    return vertices, faces.astype(np.int64, copy=False)


def mesh_edges(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    """Return deterministic unique edges oriented from lower to higher ID."""
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= vertex_count)):
        raise ValueError("Valid triangle faces required")
    pairs = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    pairs = np.sort(pairs.astype(np.int64, copy=False), axis=1)
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("Self edges are not supported")
    return np.unique(pairs, axis=0)


def connected_components(vertex_count: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic component labels and one lowest-ID pin per component."""
    if isinstance(vertex_count, bool) or not isinstance(vertex_count, (int, np.integer)) \
            or vertex_count < 1:
        raise ValueError("Positive vertex_count required")
    adjacency = [[] for _ in range(int(vertex_count))]
    for first, second in np.asarray(edges, dtype=np.int64):
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    labels = np.full(int(vertex_count), -1, dtype=np.int64)
    pins = []
    component = 0
    for root in range(int(vertex_count)):
        if labels[root] >= 0:
            continue
        pins.append(root)
        labels[root] = component
        queue = [root]
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for neighbor in sorted(adjacency[current]):
                if labels[neighbor] < 0:
                    labels[neighbor] = component
                    queue.append(neighbor)
        component += 1
    return labels, np.asarray(pins, dtype=np.int64)


def _edge_support(faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
    index = {tuple(edge): position for position, edge in enumerate(edges.tolist())}
    support = np.zeros(len(edges), dtype=np.float64)
    for face in faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            support[index[tuple(sorted((int(first), int(second))))]] += 1.0
    if np.any(support <= 0.0):
        raise RuntimeError("Every mesh edge must have positive face support")
    return support


def generate_common_differential_target(vertices: np.ndarray, faces: np.ndarray, *,
                                        target_strength: float,
                                        max_relative_change: float) -> dict:
    """Generate a fixed legal edge correction using predicted geometry only."""
    target_strength = _unit_interval("target_strength", target_strength)
    max_relative_change = _unit_interval("max_relative_change", max_relative_change)
    vertices = np.asarray(vertices)
    if (vertices.dtype != np.dtype(np.float32) or vertices.ndim != 3
            or vertices.shape[-1] != 3 or not np.isfinite(vertices).all()):
        raise ValueError("Finite float32 vertices[T,V,3] required")
    edges = mesh_edges(faces, vertices.shape[1])
    labels, pins = connected_components(vertices.shape[1], edges)
    edge_vectors = (vertices[:, edges[:, 1]] - vertices[:, edges[:, 0]]).astype(np.float64)
    lengths = np.linalg.norm(edge_vectors, axis=-1)
    scale = np.maximum(1.0, np.max(np.abs(vertices.astype(np.float64)), axis=(1, 2)))
    minimum_length = 100.0 * np.finfo(np.float32).eps * scale[:, None]
    if np.any(lengths <= minimum_length):
        raise ValueError("Collapsed/near-collapsed source edge prevents a defined target")
    triangles = vertices[:, np.asarray(faces, dtype=np.int64)].astype(np.float64)
    doubled_area = np.linalg.norm(np.cross(
        triangles[:, :, 1] - triangles[:, :, 0],
        triangles[:, :, 2] - triangles[:, :, 0]), axis=-1)
    face_scale = np.max(np.linalg.norm(
        triangles - triangles[:, :, :1], axis=-1), axis=-1)
    if np.any(doubled_area <= 100.0 * np.finfo(np.float32).eps
              * np.maximum(1.0, np.square(face_scale))):
        raise ValueError("Near-degenerate predicted face prevents qualified local ARAP")
    incident = [[] for _ in range(vertices.shape[1])]
    for edge_id, (first, second) in enumerate(edges):
        incident[int(first)].append(edge_id)
        incident[int(second)].append(edge_id)
    vertex_median = np.zeros((len(vertices), vertices.shape[1]), dtype=np.float64)
    for vertex_id, edge_ids in enumerate(incident):
        if edge_ids:
            vertex_median[:, vertex_id] = np.median(lengths[:, edge_ids], axis=1)
        else:
            # Isolated vertices have no differential target and are their own pinned component.
            vertex_median[:, vertex_id] = 0.0
    desired_length = 0.5 * (
        vertex_median[:, edges[:, 0]] + vertex_median[:, edges[:, 1]])
    raw_relative_change = desired_length / lengths - 1.0
    bounded_relative_change = np.clip(
        raw_relative_change, -max_relative_change, max_relative_change)
    applied_relative_change = target_strength * bounded_relative_change
    # All native arms preserve the original first frame exactly.  Freeze that
    # requirement in the common target rather than silently repairing it later.
    desired_length[0] = lengths[0]
    raw_relative_change[0] = 0.0
    applied_relative_change[0] = 0.0
    target = applied_relative_change[..., None] * edge_vectors
    support = _edge_support(np.asarray(faces, dtype=np.int64), edges)
    anchor_weights = support / np.square(lengths[0])
    anchor_weights /= np.median(anchor_weights)
    weights = anchor_weights
    if not np.isfinite(target).all() or not np.isfinite(weights).all() \
            or np.any(weights <= 0.0):
        raise RuntimeError("Common target or weights are nonfinite/nonpositive")
    return {
        "edges": edges,
        "component_labels": labels,
        "pins": pins,
        "target": target,
        "weights": weights,
        "source_edge_vectors": edge_vectors,
        "source_edge_lengths": lengths,
        "desired_edge_lengths": desired_length,
        "raw_relative_change": raw_relative_change,
        "applied_relative_change": applied_relative_change,
        "target_strength": target_strength,
        "max_relative_change": max_relative_change,
        "generator": "incident_median_edge_length_geometry_only_anchor_weight_v2",
    }


def _realized(displacement: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return displacement[:, edges[:, 1]] - displacement[:, edges[:, 0]]


def _projection_diagnostics(displacement: np.ndarray, common: dict) -> dict:
    realized = _realized(displacement, common["edges"])
    residual = realized - common["target"]
    weighted_target_sq = float(np.sum(common["weights"][None, :, None]
                                      * np.square(common["target"])))
    weighted_residual_sq = float(np.sum(common["weights"][None, :, None]
                                        * np.square(residual)))
    pin_error = float(np.max(np.abs(displacement[:, common["pins"]])))
    return {
        "weighted_target_l2": float(np.sqrt(weighted_target_sq)),
        "weighted_residual_l2": float(np.sqrt(weighted_residual_sq)),
        "relative_inconsistent_component": float(
            np.sqrt(weighted_residual_sq) / max(np.sqrt(weighted_target_sq),
                                                np.finfo(np.float64).tiny)),
        "pin_residual_linf": pin_error,
        "displacement_l2": float(np.linalg.norm(displacement)),
        "realized_differential_l2": float(np.linalg.norm(realized)),
    }


def direct_common_lift(common: dict, vertex_count: int) -> tuple[np.ndarray, dict]:
    """Integrate the common target only along a deterministic spanning forest."""
    edges = common["edges"]
    edge_index = {tuple(edge): index for index, edge in enumerate(edges.tolist())}
    adjacency = [[] for _ in range(vertex_count)]
    for first, second in edges:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    displacement = np.zeros((len(common["target"]), vertex_count, 3), dtype=np.float64)
    tree_edges = []
    visited = np.zeros(vertex_count, dtype=bool)
    for pin in common["pins"]:
        pin = int(pin)
        visited[pin] = True
        queue = [pin]
        cursor = 0
        while cursor < len(queue):
            parent = queue[cursor]
            cursor += 1
            for child in sorted(adjacency[parent]):
                if visited[child]:
                    continue
                visited[child] = True
                queue.append(child)
                low, high = sorted((parent, child))
                edge_id = edge_index[(low, high)]
                directed = common["target"][:, edge_id]
                step = directed if parent == low else -directed
                displacement[:, child] = displacement[:, parent] + step
                tree_edges.append(edge_id)
    if not visited.all():
        raise RuntimeError("Spanning forest did not cover every vertex")
    diagnostics = _projection_diagnostics(displacement, common)
    diagnostics.update({
        "construction": "deterministic_bfs_spanning_forest_path_integral",
        "tree_edge_count": len(tree_edges),
        "tree_edge_ids": tree_edges,
    })
    return displacement, diagnostics


def independent_local_repair(common: dict, faces: np.ndarray,
                             vertex_count: int) -> tuple[np.ndarray, dict]:
    """Fit proper face rotations to common h/W, then average local proposals."""
    edge_index = {tuple(edge): index for index, edge in enumerate(common["edges"].tolist())}
    proposals = np.zeros((len(common["target"]), vertex_count, 3), dtype=np.float64)
    counts = np.zeros(vertex_count, dtype=np.int64)
    determinant_values = []
    rotation_orthogonality = []
    arap_target_residual_sq = 0.0
    for face in np.asarray(faces, dtype=np.int64):
        local_rhs = np.zeros((len(common["target"]), 3, 3), dtype=np.float64)
        local_system = np.zeros((3, 3), dtype=np.float64)
        local_pairs = ((0, 1), (0, 2), (1, 2))
        edge_ids = []
        edge_signs = []
        for local_first, local_second in local_pairs:
            first, second = int(face[local_first]), int(face[local_second])
            low, high = sorted((first, second))
            edge_id = edge_index[(low, high)]
            edge_ids.append(edge_id)
            edge_signs.append(1.0 if first == low else -1.0)
        edge_ids_array = np.asarray(edge_ids, dtype=np.int64)
        signs = np.asarray(edge_signs, dtype=np.float64)
        face_weights = common["weights"][edge_ids_array]
        source_edges = (common["source_edge_vectors"][:, edge_ids_array]
                        * signs[None, :, None])
        target_corrections = (common["target"][:, edge_ids_array]
                              * signs[None, :, None])
        desired_edges = source_edges + target_corrections
        arap_corrections = np.empty_like(target_corrections)
        for frame in range(len(common["target"])):
            covariance = (source_edges[frame].T
                          @ (face_weights[:, None] * desired_edges[frame]))
            left, _, right_t = np.linalg.svd(covariance, full_matrices=True)
            orientation = np.eye(3, dtype=np.float64)
            orientation[-1, -1] = np.sign(np.linalg.det(left @ right_t))
            rotation = left @ orientation @ right_t
            determinant_values.append(float(np.linalg.det(rotation)))
            rotation_orthogonality.append(float(np.max(np.abs(
                rotation.T @ rotation - np.eye(3)))))
            rotated = source_edges[frame] @ rotation
            arap_corrections[frame] = rotated - source_edges[frame]
            arap_target_residual_sq += float(np.sum(
                face_weights[:, None] * np.square(rotated - desired_edges[frame])))
        for pair_id, (local_first, local_second) in enumerate(local_pairs):
            desired = arap_corrections[:, pair_id]
            weight = float(face_weights[pair_id])
            local_rhs[:, local_first] -= weight * desired
            local_rhs[:, local_second] += weight * desired
            local_system[local_first, local_first] += weight
            local_system[local_second, local_second] += weight
            local_system[local_first, local_second] -= weight
            local_system[local_second, local_first] -= weight
        # Adding the projector onto constants fixes only the local gauge.  The
        # three-by-three systems remain independent; no global solve is hidden.
        local_system += np.ones((3, 3), dtype=np.float64) / 3.0
        local_solution = np.stack([
            np.linalg.solve(local_system, local_rhs[frame])
            for frame in range(len(common["target"]))])
        for local_id, vertex_id in enumerate(face):
            proposals[:, int(vertex_id)] += local_solution[:, local_id]
            counts[int(vertex_id)] += 1
    displacement = np.zeros_like(proposals)
    used = counts > 0
    displacement[:, used] = proposals[:, used] / counts[used][None, :, None]
    # Fix the same per-component gauges without altering within-component edges.
    for pin in common["pins"]:
        label = common["component_labels"][pin]
        members = common["component_labels"] == label
        displacement[:, members] -= displacement[:, int(pin), None, :]
    diagnostics = _projection_diagnostics(displacement, common)
    diagnostics.update({
        "construction": "independent_face_local_proper_rotation_arap_then_vertex_average",
        "local_problem_count": int(len(faces)),
        "unreferenced_vertex_count": int(np.count_nonzero(~used)),
        "proper_rotation_determinant_min": min(determinant_values),
        "proper_rotation_determinant_max": max(determinant_values),
        "rotation_orthogonality_linf": max(rotation_orthogonality),
        "weighted_arap_target_residual_l2": float(np.sqrt(arap_target_residual_sq)),
        "arap_scope": "face_local_rotation_fit_no_global_integrability_solve",
    })
    return displacement, diagnostics


def _laplacian_matvec(free_value: np.ndarray, *, free: np.ndarray, pins: np.ndarray,
                      vertex_count: int, edges: np.ndarray,
                      weights: np.ndarray) -> np.ndarray:
    full = np.zeros(vertex_count, dtype=np.float64)
    full[free] = free_value
    difference = full[edges[:, 0]] - full[edges[:, 1]]
    result = np.zeros(vertex_count, dtype=np.float64)
    np.add.at(result, edges[:, 0], weights * difference)
    np.add.at(result, edges[:, 1], -weights * difference)
    result[pins] = 0.0
    return result[free]


def _conjugate_gradient(rhs: np.ndarray, matvec, *, absolute_tolerance: float,
                        relative_tolerance: float, max_iterations: int) -> tuple[np.ndarray, dict]:
    solution = np.zeros_like(rhs, dtype=np.float64)
    residual = rhs.astype(np.float64, copy=True)
    direction = residual.copy()
    residual_sq = float(np.dot(residual, residual))
    rhs_norm = float(np.linalg.norm(rhs))
    threshold = absolute_tolerance + relative_tolerance * rhs_norm
    if np.sqrt(residual_sq) <= threshold:
        return solution, {"iterations": 0, "residual_l2": float(np.sqrt(residual_sq)),
                          "residual_threshold": threshold}
    for iteration in range(1, max_iterations + 1):
        action = matvec(direction)
        denominator = float(np.dot(direction, action))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("Pinned Laplacian lost positive definiteness")
        alpha = residual_sq / denominator
        solution += alpha * direction
        residual -= alpha * action
        next_residual_sq = float(np.dot(residual, residual))
        if np.sqrt(next_residual_sq) <= threshold:
            return solution, {
                "iterations": iteration,
                "residual_l2": float(np.sqrt(next_residual_sq)),
                "residual_threshold": threshold,
            }
        direction = residual + (next_residual_sq / residual_sq) * direction
        residual_sq = next_residual_sq
    raise RuntimeError(
        "Pinned integrable solve did not meet frozen CG tolerance: "
        f"residual={np.sqrt(residual_sq):.17g}, threshold={threshold:.17g}, "
        f"iterations={max_iterations}")


def pinned_integrable_solve(common: dict, vertex_count: int, *,
                            absolute_tolerance: float,
                            relative_tolerance: float,
                            max_iterations: int) -> tuple[np.ndarray, dict]:
    """Solve the pinned weighted range projection without a dense VxV matrix."""
    absolute_tolerance = _positive("absolute_tolerance", absolute_tolerance)
    relative_tolerance = _positive("relative_tolerance", relative_tolerance)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    pins = common["pins"]
    free = np.setdiff1d(np.arange(vertex_count, dtype=np.int64), pins,
                        assume_unique=True)
    displacement = np.zeros((len(common["target"]), vertex_count, 3), dtype=np.float64)
    solves = []
    for frame in range(len(common["target"])):
        weights = common["weights"]
        rhs = np.zeros((vertex_count, 3), dtype=np.float64)
        weighted_target = weights[:, None] * common["target"][frame]
        np.add.at(rhs, common["edges"][:, 0], -weighted_target)
        np.add.at(rhs, common["edges"][:, 1], weighted_target)
        for coordinate in range(3):
            solution, diagnostics = _conjugate_gradient(
                rhs[free, coordinate],
                lambda value, current_weights=weights: _laplacian_matvec(
                    value, free=free, pins=pins, vertex_count=vertex_count,
                    edges=common["edges"], weights=current_weights),
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                max_iterations=max_iterations)
            displacement[frame, free, coordinate] = solution
            solves.append({"frame": frame, "coordinate": coordinate, **diagnostics})
    realized = _realized(displacement, common["edges"])
    weighted_residual = common["weights"][None, :, None] * (realized - common["target"])
    normal_residual = np.zeros_like(displacement)
    np.add.at(normal_residual, (slice(None), common["edges"][:, 0]), -weighted_residual)
    np.add.at(normal_residual, (slice(None), common["edges"][:, 1]), weighted_residual)
    normal_residual[:, pins] = 0.0
    diagnostics = _projection_diagnostics(displacement, common)
    diagnostics.update({
        "construction": "matrix_free_pinned_weighted_poisson_cg",
        "normal_equation_residual_l2": float(np.linalg.norm(normal_residual[:, free])),
        "normal_equation_residual_linf": float(np.max(np.abs(normal_residual[:, free]))),
        "cg_absolute_tolerance": absolute_tolerance,
        "cg_relative_tolerance": relative_tolerance,
        "cg_max_iterations": max_iterations,
        "cg_iteration_max": max(item["iterations"] for item in solves),
        "cg_solve_count": len(solves),
    })
    return displacement, diagnostics


def construct_all_arms(vertices: np.ndarray, faces: np.ndarray, common: dict, *,
                       absolute_tolerance: float, relative_tolerance: float,
                       max_iterations: int) -> dict:
    """Construct all three fair arms from one in-memory common target."""
    direct, direct_diagnostics = direct_common_lift(common, vertices.shape[1])
    local, local_diagnostics = independent_local_repair(common, faces, vertices.shape[1])
    integrable, integrable_diagnostics = pinned_integrable_solve(
        common, vertices.shape[1], absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance, max_iterations=max_iterations)
    values = {
        "direct_common_lift": (direct, direct_diagnostics),
        "independent_local_repair": (local, local_diagnostics),
        "pinned_integrable_solve": (integrable, integrable_diagnostics),
    }
    return values


def _construct_role(role: str, vertices: np.ndarray, faces: np.ndarray, common: dict, *,
                    absolute_tolerance: float, relative_tolerance: float,
                    max_iterations: int) -> tuple[np.ndarray, dict]:
    if role == "direct_common_lift":
        return direct_common_lift(common, vertices.shape[1])
    if role == "independent_local_repair":
        return independent_local_repair(common, faces, vertices.shape[1])
    if role == "pinned_integrable_solve":
        return pinned_integrable_solve(
            common, vertices.shape[1], absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance, max_iterations=max_iterations)
    raise ValueError(f"Unknown C10 role: {role}")


def _bound_diagnostics(vertices: np.ndarray, lower: float, upper: float,
                       policy: str) -> dict:
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("Finite ordered coordinate bounds required")
    if policy not in ("preserve_and_report", "reject"):
        raise ValueError("bounds_policy must be preserve_and_report or reject")
    below = int(np.count_nonzero(vertices < lower))
    above = int(np.count_nonzero(vertices > upper))
    diagnostics = {
        "policy": policy, "coordinate_min": float(vertices.min()),
        "coordinate_max": float(vertices.max()), "declared_lower": lower,
        "declared_upper": upper, "below_count": below, "above_count": above,
        "coordinates_modified_by_bounds_policy": False,
    }
    if policy == "reject" and (below or above):
        raise RuntimeError("Export exceeded declared coordinate bounds under reject policy")
    return diagnostics


def _exported_projection_diagnostics(source: np.ndarray, exported: np.ndarray,
                                     common: dict, role: str,
                                     absolute_tolerance: float,
                                     relative_tolerance: float) -> dict:
    """Recompute the certificate at exact float32 bytes consumed by scoring."""
    displacement = exported.astype(np.float64) - source.astype(np.float64)
    realized = _realized(displacement, common["edges"])
    residual = realized - common["target"]
    weighted_residual = common["weights"][None, :, None] * residual
    normal = np.zeros_like(displacement)
    np.add.at(normal, (slice(None), common["edges"][:, 0]), -weighted_residual)
    np.add.at(normal, (slice(None), common["edges"][:, 1]), weighted_residual)
    free = np.setdiff1d(np.arange(source.shape[1], dtype=np.int64), common["pins"],
                        assume_unique=True)
    normal_free = normal[:, free]
    target_energy = .5 * float(np.sum(
        common["weights"][None, :, None] * np.square(common["target"])))
    residual_energy = .5 * float(np.sum(
        common["weights"][None, :, None] * np.square(residual)))
    rhs_scale = float(np.sqrt(2.0 * target_energy))
    export_roundoff = (1000.0 * np.finfo(np.float32).eps
                       * max(1.0, rhs_scale) * np.sqrt(max(1, normal_free.size)))
    orthogonality_tolerance = max(
        export_roundoff,
        10.0 * np.sqrt(16.0 * 3.0)
        * (absolute_tolerance + relative_tolerance * rhs_scale))
    diagnostics = {
        "scope": "exact_exported_float32_promoted_to_float64",
        "weighted_target_energy": target_energy,
        "weighted_projection_residual_energy": residual_energy,
        "weighted_projection_residual_l2": float(np.sqrt(2.0 * residual_energy)),
        "normal_equation_residual_l2": float(np.linalg.norm(normal_free)),
        "normal_equation_residual_linf": float(np.max(np.abs(normal_free))),
        "normal_equation_tolerance_l2": orthogonality_tolerance,
        "normal_equation_within_tolerance": bool(
            np.linalg.norm(normal_free) <= orthogonality_tolerance),
        "pin_residual_linf": float(np.max(np.abs(displacement[:, common["pins"]]))),
        "pins_exact_float32": bool(np.array_equal(
            exported[:, common["pins"]], source[:, common["pins"]])),
        "frame_zero_exact_float32": bool(np.array_equal(exported[0], source[0])),
        "weight_scope": "anchor_frame_fixed_positive_edge_weights_reused_all_frames",
        "common_target_modified_for_arm": False,
    }
    if not diagnostics["pins_exact_float32"] or not diagnostics["frame_zero_exact_float32"]:
        raise RuntimeError(f"{role} changed an exact native anchor")
    if role == "pinned_integrable_solve" \
            and not diagnostics["normal_equation_within_tolerance"]:
        raise RuntimeError(
            "Exported pinned solve failed weighted orthogonality tolerance: "
            f"residual={diagnostics['normal_equation_residual_l2']:.17g}, "
            f"tolerance={orthogonality_tolerance:.17g}")
    return diagnostics


def _common_from_npz(path: Path) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    required = {"source_vertices", "faces", "edges", "target", "weights",
                "source_edge_vectors",
                "component_labels", "pins", "source_edge_lengths",
                "desired_edge_lengths", "raw_relative_change",
                "applied_relative_change", "target_strength", "max_relative_change"}
    if set(arrays) != required:
        raise ValueError("Unexpected common-target certificate members")
    common = {key: arrays[key] for key in required}
    common["target_strength"] = float(np.asarray(common["target_strength"]).item())
    common["max_relative_change"] = float(np.asarray(common["max_relative_change"]).item())
    common["generator"] = "incident_median_edge_length_geometry_only_anchor_weight_v2"
    return arrays, common


def _validate_artifact_archive(output: Path, reports: list[dict]) -> dict:
    record_path = output / "artifact-archive.json"
    archive_path = output / "artifact.tar"
    record = json.loads(record_path.read_text())
    expected_members = [{"path": path.relative_to(output).as_posix(),
                         "sha256": _digest(path), "size_bytes": path.stat().st_size}
                        for path in _artifact_member_paths(output, reports)]
    if (record.get("kind") != "c10-terminal-artifact-archive"
            or record.get("version") != 1
            or record.get("members") != expected_members
            or record.get("terminal_candidate_status")
            != json.loads((output / "candidate.json").read_text()).get("status")
            or not isinstance(record.get("max_artifact_bytes"), int)
            or record["max_artifact_bytes"] < archive_path.stat().st_size
            or record.get("archive") != {
                "path": "artifact.tar", "sha256": _digest(archive_path),
                "size_bytes": archive_path.stat().st_size}):
        raise ValueError("Terminal artifact archive record mismatch")
    with tarfile.open(archive_path, mode="r:") as bundle:
        members = bundle.getmembers()
        if [member.name for member in members] != [row["path"] for row in expected_members]:
            raise ValueError("Artifact archive member inventory/order mismatch")
        for member, expected in zip(members, expected_members):
            if (not member.isfile() or member.size != expected["size_bytes"]
                    or member.mode != 0o644 or member.uid != 0 or member.gid != 0
                    or member.mtime != 0 or member.uname or member.gname):
                raise ValueError("Artifact archive metadata is not canonical")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("Artifact archive regular member is unreadable")
            digest = hashlib.sha256()
            remaining = member.size
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("Artifact archive member is truncated")
                digest.update(block)
                remaining -= len(block)
            if stream.read(1) or digest.hexdigest() != expected["sha256"]:
                raise ValueError("Artifact archive member bytes mismatch")
    return {
        "record": {"path": record_path, "sha256": _digest(record_path)},
        "archive": {"path": archive_path, "sha256": _digest(archive_path)},
    }


def validate_candidate_artifact(root: Path, candidate_path: Path) -> dict:
    """Recompute and validate a complete three-arm artifact for comparison freeze."""
    root = Path(root).resolve()
    candidate_path = Path(candidate_path).resolve()
    candidate_path.relative_to(root)
    artifact = candidate_path.parent
    candidate = json.loads(candidate_path.read_text())
    if (candidate.get("status") not in ("completed", "incomplete")
            or candidate.get("candidate_id") != CANDIDATE_ID):
        raise ValueError("Terminal C10 candidate artifact required")
    if tuple(candidate.get("roles", ())) != ROLES:
        raise ValueError("Exact ordered C10 role set required")
    if (candidate.get("native_qualified") is not False
            or candidate.get("scientific_admission") is not False
            or candidate.get("local_method_verified") is not False
            or candidate.get("source_delivery_status")
            != "generated_unexecuted_at_authoring"):
        raise ValueError("Candidate overstates generated-unexecuted evidence")
    source_refs = candidate.get("source_refs", {})
    if set(source_refs) != {"sequence", "report"}:
        raise ValueError("Exact source sequence/report references required")
    for ref in source_refs.values():
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str)
                or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in ref["sha256"])):
            raise ValueError("Canonical source path/SHA256 references required")
    if (candidate.get("source_sequence_sha256")
            != source_refs.get("sequence", {}).get("sha256")
            or candidate.get("source_report_sha256")
            != source_refs.get("report", {}).get("sha256")):
        raise ValueError("Candidate source digest aliases disagree with exact refs")
    source_candidate = root / source_refs["sequence"]["path"]
    report_candidate = root / source_refs["report"]["path"]
    for candidate_path_part in (source_candidate, report_candidate):
        if candidate_path_part.is_symlink() or any(
                parent.is_symlink() for parent in candidate_path_part.parents
                if parent != root.parent):
            raise ValueError("Symlinked source evidence is not accepted")
    source_path = source_candidate.resolve()
    source_report_path = report_candidate.resolve()
    source_path.relative_to(root)
    source_report_path.relative_to(root)
    if (source_path.name != "sequence.npz"
            or source_report_path != source_path.with_name("report.json")
            or not source_path.is_file() or not source_report_path.is_file()
            or source_path.is_symlink() or source_report_path.is_symlink()):
        raise ValueError("Exact colocated regular source sequence/report pair required")
    if (_digest(source_path) != source_refs["sequence"]["sha256"]
            or _digest(source_report_path) != source_refs["report"]["sha256"]):
        raise ValueError("Current receipt-bound source files differ from candidate refs")
    implementation_sha256 = _digest(Path(__file__))
    if candidate.get("implementation_sha256") != implementation_sha256:
        raise ValueError("Candidate implementation identity differs from installed core")
    source_report = json.loads(source_report_path.read_text())
    if (source_report.get("status") != "completed"
            or source_report.get("uid") != candidate.get("uid")
            or source_report.get("seed") != candidate.get("seed")
            or source_report.get("sha256", {}).get("sequence.npz")
            != source_refs["sequence"]["sha256"]):
        raise ValueError("Source receipt identity/hash mismatch")
    with np.load(source_path, allow_pickle=False) as saved:
        source_arrays = {key: saved[key].copy() for key in saved.files}
    exact_source_vertices, exact_source_faces = validate_native_arrays(source_arrays)
    common_path = artifact / "common-target.npz"
    common_ref = candidate.get("common_target", {})
    if common_ref.get("path") != "common-target.npz" \
            or common_ref.get("sha256") != _digest(common_path):
        raise ValueError("Common target reference/hash mismatch")
    common_arrays, stored_common = _common_from_npz(common_path)
    source_vertices = common_arrays["source_vertices"]
    faces = common_arrays["faces"]
    if (not np.array_equal(source_vertices, exact_source_vertices)
            or not np.array_equal(faces, exact_source_faces)):
        raise ValueError("Common target source snapshot differs from exact native source")
    recomputed = generate_common_differential_target(
        source_vertices, faces, target_strength=stored_common["target_strength"],
        max_relative_change=stored_common["max_relative_change"])
    for key in ("edges", "target", "weights", "source_edge_vectors",
                "component_labels", "pins",
                "source_edge_lengths", "desired_edge_lengths", "raw_relative_change",
                "applied_relative_change"):
        if not np.array_equal(common_arrays[key], recomputed[key]):
            raise ValueError(f"Common target member failed exact recomputation: {key}")
    solver = candidate.get("solver", {})
    expected_target_parameters = {
        "target_strength": recomputed["target_strength"],
        "max_relative_change": recomputed["max_relative_change"],
    }
    if candidate.get("target_parameters") != expected_target_parameters:
        raise ValueError("Candidate target parameters differ from common target")
    if (set(solver) != {"absolute_tolerance", "relative_tolerance", "max_iterations"}
            or _positive("absolute_tolerance", solver.get("absolute_tolerance"))
            != solver["absolute_tolerance"]
            or _positive("relative_tolerance", solver.get("relative_tolerance"))
            != solver["relative_tolerance"]
            or _positive_integer("max_iterations", solver.get("max_iterations"))
            != solver["max_iterations"]):
        raise ValueError("Exact valid solver parameters required")
    bounds = candidate.get("bounds", {})
    if (set(bounds) != {"policy", "lower", "upper"}
            or bounds.get("policy") not in ("preserve_and_report", "reject")
            or not np.isfinite(bounds.get("lower", np.nan))
            or not np.isfinite(bounds.get("upper", np.nan))
            or bounds["lower"] >= bounds["upper"]):
        raise ValueError("Exact valid candidate bounds required")
    records = []
    disk_reports = []
    for role in ROLES:
        directory = artifact / role
        report_path = directory / "report.json"
        sequence_path = directory / "sequence.npz"
        certificate_path = directory / "certificate.npz"
        report = json.loads(report_path.read_text())
        disk_reports.append(report)
        if (report.get("status") not in ("completed", "error")
                or report.get("candidate_arm") != role):
            raise ValueError(f"Terminal report required for {role}")
        expected_report_solver = {
            "name": "matrix_free_pinned_weighted_poisson_cg", **solver}
        if (report.get("candidate_id") != CANDIDATE_ID
                or report.get("method_id") != METHOD_IDS[role]
                or report.get("role_contract") != ROLE_CONTRACTS[role]
                or report.get("target_parameters") != expected_target_parameters
                or report.get("solver") != expected_report_solver
                or report.get("pins") != recomputed["pins"].tolist()
                or report.get("component_count") != len(recomputed["pins"])
                or report.get("native_qualified") is not False
                or report.get("scientific_admission") is not False
                or report.get("local_method_verified") is not False
                or report.get("source_delivery_status")
                != "generated_unexecuted_at_authoring"):
            raise ValueError(f"{role} report method/parameters/evidence flags mismatch")
        if report.get("common_target_sha256") != common_ref["sha256"]:
            raise ValueError(f"{role} does not bind the common target")
        if (report.get("implementation_sha256") != implementation_sha256
                or report.get("source_refs") != source_refs
                or report.get("uid") != candidate["uid"]
                or report.get("seed") != candidate["seed"]):
            raise ValueError(f"{role} source/implementation identity mismatch")
        if report["status"] == "error":
            if sequence_path.exists() or certificate_path.exists():
                raise ValueError(f"Failed {role} must not expose scoreable artifacts")
            if (not isinstance(report.get("exception_type"), str)
                    or not report["exception_type"].strip()
                    or len(report["exception_type"]) > 256
                    or not isinstance(report.get("error"), str)
                    or not report["error"].strip()
                    or len(report["error"]) > 4096):
                raise ValueError(f"Failed {role} lacks retained exception evidence")
            records.append({
                "role": role, "status": "error", "sequence": None,
                "certificate": None,
                "report": {"path": report_path.relative_to(root).as_posix(),
                           "sha256": _digest(report_path)},
                "exception_type": report["exception_type"],
            })
            continue
        expected_hashes = {
            "sequence.npz": _digest(sequence_path),
            "certificate.npz": _digest(certificate_path),
        }
        if report.get("sha256") != expected_hashes:
            raise ValueError(f"{role} output hashes mismatch")
        with np.load(sequence_path, allow_pickle=False) as saved:
            sequence = {key: saved[key].copy() for key in saved.files}
        output_vertices, output_faces = validate_native_arrays(sequence)
        if not np.array_equal(output_faces, faces):
            raise ValueError(f"{role} changed topology")
        if set(sequence) != set(source_arrays):
            raise ValueError(f"{role} changed native array inventory")
        for key in sequence:
            if key != "vertices" and not np.array_equal(sequence[key], source_arrays[key]):
                raise ValueError(f"{role} changed native metadata array: {key}")
        expected_displacement, _ = _construct_role(
            role, source_vertices, faces, recomputed,
            absolute_tolerance=float(solver["absolute_tolerance"]),
            relative_tolerance=float(solver["relative_tolerance"]),
            max_iterations=int(solver["max_iterations"]))
        expected_vertices = (source_vertices.astype(np.float64)
                             + expected_displacement).astype(np.float32)
        expected_vertices[:, recomputed["pins"]] = source_vertices[:, recomputed["pins"]]
        if not np.array_equal(output_vertices, expected_vertices):
            raise ValueError(f"{role} sequence differs from exact reconstruction")
        with np.load(certificate_path, allow_pickle=False) as saved:
            certificate = {key: saved[key].copy() for key in saved.files}
        expected_keys = {"displacement", "realized_differential", "differential_residual",
                         "edges", "pins", "component_labels"}
        if set(certificate) != expected_keys:
            raise ValueError(f"{role} certificate members mismatch")
        expected_exported_displacement = (output_vertices.astype(np.float64)
                                          - source_vertices.astype(np.float64))
        expected_realized = _realized(expected_exported_displacement, recomputed["edges"])
        checks = {
            "displacement": expected_exported_displacement,
            "realized_differential": expected_realized,
            "differential_residual": expected_realized - recomputed["target"],
            "edges": recomputed["edges"], "pins": recomputed["pins"],
            "component_labels": recomputed["component_labels"],
        }
        for key, value in checks.items():
            if not np.array_equal(certificate[key], value):
                raise ValueError(f"{role} certificate failed recomputation: {key}")
        exported_diagnostics = _exported_projection_diagnostics(
            source_vertices, output_vertices, recomputed, role,
            float(solver["absolute_tolerance"]),
            float(solver["relative_tolerance"]))
        if report.get("exported_certificate") != exported_diagnostics:
            raise ValueError(f"{role} exported certificate diagnostics mismatch")
        expected_bounds = _bound_diagnostics(
            output_vertices, float(bounds["lower"]), float(bounds["upper"]), bounds["policy"])
        if report.get("export_bounds") != expected_bounds:
            raise ValueError(f"{role} exported bounds diagnostics mismatch")
        records.append({
            "role": role, "status": "completed",
            "sequence": {"path": sequence_path.relative_to(root).as_posix(),
                         "sha256": expected_hashes["sequence.npz"]},
            "certificate": {"path": certificate_path.relative_to(root).as_posix(),
                            "sha256": expected_hashes["certificate.npz"]},
            "report": {"path": report_path.relative_to(root).as_posix(),
                       "sha256": _digest(report_path)},
        })
    expected_status = "completed" if all(
        record["status"] == "completed" for record in records) else "incomplete"
    if candidate["status"] != expected_status:
        raise ValueError("Candidate aggregate status disagrees with terminal arms")
    candidate_report = next((
        report for report in disk_reports
        if report.get("candidate_arm") == "pinned_integrable_solve"
        and report.get("status") == "completed"), None)
    if candidate_report is None:
        expected_mechanism = {
            "status": "unavailable_candidate_failed",
            "artificial_target_perturbation": False,
        }
    else:
        certificate = candidate_report["exported_certificate"]
        target_l2 = float(np.sqrt(2.0 * certificate["weighted_target_energy"]))
        residual_l2 = certificate["weighted_projection_residual_l2"]
        ratio = residual_l2 / max(target_l2, np.finfo(np.float64).tiny)
        expected_mechanism = {
            "status": ("inconclusive_already_integrable" if ratio
                       <= INTEGRABILITY_RATIO_TOLERANCE
                       else "natural_nonintegrable_component_observed_software_only"),
            "weighted_inconsistent_component_l2": residual_l2,
            "weighted_target_l2": target_l2,
            "relative_inconsistent_component": ratio,
            "near_zero_relative_threshold": INTEGRABILITY_RATIO_TOLERANCE,
            "artificial_target_perturbation": False,
            "scientific_evidence": False,
        }
    if candidate.get("mechanism_diagnostic") != expected_mechanism:
        raise ValueError("Candidate mechanism diagnostic failed exact recomputation")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_cases = [{
        "case_id": candidate["uid"] + "-" + record["role"],
        "uid": candidate["uid"], "case_dir": record["role"],
        "arm_role": record["role"],
    } for record in records if record["status"] == "completed"]
    expected_manifest = {
        "cases": expected_cases, "expected_roles": list(ROLES),
        "common_target": {"path": "common-target.npz", "sha256": common_ref["sha256"]},
        "scope": "C10 generated_unexecuted three-arm artifact; no B0/B*/scoring/admission",
    }
    if manifest != expected_manifest:
        raise ValueError("Manifest does not exactly describe validated terminal arms")
    candidate_arms = candidate.get("arms")
    if candidate_arms != disk_reports:
        raise ValueError("Candidate embedded arms differ from exact terminal reports")
    archive_refs = _validate_artifact_archive(artifact, disk_reports)
    return {
        "candidate_id": CANDIDATE_ID, "uid": candidate["uid"],
        "seed": candidate["seed"], "roles": list(ROLES),
        "source_sequence_sha256": candidate["source_sequence_sha256"],
        "common_target": {"path": common_path.relative_to(root).as_posix(),
                          "sha256": common_ref["sha256"]},
        "manifest": {"path": manifest_path.relative_to(root).as_posix(),
                     "sha256": _digest(manifest_path)},
        "artifact_archive": {
            "record": {"path": archive_refs["record"]["path"].relative_to(root).as_posix(),
                       "sha256": archive_refs["record"]["sha256"]},
            "archive": {"path": archive_refs["archive"]["path"].relative_to(root).as_posix(),
                        "sha256": archive_refs["archive"]["sha256"]},
        },
        "arms": records, "status": candidate["status"], "native_qualified": False,
        "scientific_admission": False, "local_method_verified": False,
    }


def export_integrable_candidate(source_case: Path, output: Path, *, uid: str,
                                expected_sequence_sha256: str,
                                source_sequence_ref: str,
                                source_report_ref: str,
                                target_strength: float, max_relative_change: float,
                                absolute_tolerance: float, relative_tolerance: float,
                                max_iterations: int, coordinate_lower: float,
                                coordinate_upper: float, bounds_policy: str,
                                max_artifact_bytes: int) -> dict:
    """Export three complete, common-target C10 arms and their certificates."""
    started = time.monotonic()
    source_case, output = Path(source_case), Path(output)
    if output.exists():
        raise FileExistsError("Candidate output is single-use")
    if not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    for name, value in (("source_sequence_ref", source_sequence_ref),
                        ("source_report_ref", source_report_ref)):
        ref = Path(value)
        if ref.is_absolute() or not value or ".." in ref.parts or ref.name in ("", ".", ".."):
            raise ValueError(f"{name} must be a safe repository-relative path")
    target_strength = _unit_interval("target_strength", target_strength)
    max_relative_change = _unit_interval("max_relative_change", max_relative_change)
    absolute_tolerance = _positive("absolute_tolerance", absolute_tolerance)
    relative_tolerance = _positive("relative_tolerance", relative_tolerance)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    if not np.isfinite(coordinate_lower) or not np.isfinite(coordinate_upper) \
            or coordinate_lower >= coordinate_upper:
        raise ValueError("Finite ordered coordinate bounds required")
    if bounds_policy not in ("preserve_and_report", "reject"):
        raise ValueError("Invalid bounds policy")
    source_path = source_case / "sequence.npz"
    source_report_path = source_case / "report.json"
    actual_sequence_sha256 = _digest(source_path)
    if actual_sequence_sha256 != expected_sequence_sha256:
        raise ValueError("Sequence differs from explicitly pinned input")
    source_report = json.loads(source_report_path.read_text())
    if (source_report.get("status") != "completed" or source_report.get("uid") != uid
            or source_report.get("sha256", {}).get("sequence.npz")
            != actual_sequence_sha256):
        raise ValueError("Completed same-UID source report with current sequence hash required")
    seed = source_report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer native inference seed required")
    with np.load(source_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    vertices, faces = validate_native_arrays(arrays)
    common = generate_common_differential_target(
        vertices, faces, target_strength=target_strength,
        max_relative_change=max_relative_change)

    output.mkdir(parents=True, exist_ok=False)
    common_path = output / "common-target.npz"
    np.savez_compressed(
        common_path, source_vertices=vertices, faces=faces, edges=common["edges"],
        target=common["target"], weights=common["weights"],
        source_edge_vectors=common["source_edge_vectors"],
        component_labels=common["component_labels"], pins=common["pins"],
        source_edge_lengths=common["source_edge_lengths"],
        desired_edge_lengths=common["desired_edge_lengths"],
        raw_relative_change=common["raw_relative_change"],
        applied_relative_change=common["applied_relative_change"],
        target_strength=np.asarray(target_strength, dtype=np.float64),
        max_relative_change=np.asarray(max_relative_change, dtype=np.float64))
    common_sha256 = _digest(common_path)
    base_report = {
        "uid": uid, "seed": seed, "candidate_id": CANDIDATE_ID,
        "source_sequence_sha256": actual_sequence_sha256,
        "source_report_sha256": _digest(source_report_path),
        "source_refs": {
            "sequence": {"path": source_sequence_ref,
                         "sha256": actual_sequence_sha256},
            "report": {"path": source_report_ref,
                       "sha256": _digest(source_report_path)},
        },
        "implementation_sha256": _digest(Path(__file__)),
        "common_target_sha256": common_sha256,
        "common_target_generator": common["generator"],
        "target_parameters": {
            "target_strength": target_strength,
            "max_relative_change": max_relative_change,
        },
        "pins": common["pins"].tolist(),
        "component_count": int(len(common["pins"])),
        "solver": {
            "name": "matrix_free_pinned_weighted_poisson_cg",
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "max_iterations": max_iterations,
        },
        "information": ("Predicted vertices, faces, original identities/times only; no GT, "
                        "event labels, cameras, scorer state, latent state or learned weights"),
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False,
        "source_delivery_status": "generated_unexecuted_at_authoring",
    }
    reports = []
    for role in ROLES:
        directory = output / role
        directory.mkdir()
        report = {**base_report, "candidate_arm": role,
                  "method_id": METHOD_IDS[role],
                  "role_contract": ROLE_CONTRACTS[role]}
        try:
            displacement, diagnostics = _construct_role(
                role, vertices, faces, common,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                max_iterations=max_iterations)
            exported_vertices = (vertices.astype(np.float64) + displacement).astype(np.float32)
            exported_vertices[:, common["pins"]] = vertices[:, common["pins"]]
            exported_vertices[0] = vertices[0]
            if not np.isfinite(exported_vertices).all():
                raise RuntimeError("Arm emitted nonfinite coordinates")
            bounds = _bound_diagnostics(
                exported_vertices, coordinate_lower, coordinate_upper, bounds_policy)
            exported_certificate = _exported_projection_diagnostics(
                vertices, exported_vertices, common, role,
                absolute_tolerance, relative_tolerance)
            exported_displacement = (exported_vertices.astype(np.float64)
                                     - vertices.astype(np.float64))
            realized = _realized(exported_displacement, common["edges"])
            np.savez_compressed(directory / "sequence.npz",
                                **{**arrays, "vertices": exported_vertices})
            np.savez_compressed(
                directory / "certificate.npz", displacement=exported_displacement,
                realized_differential=realized,
                differential_residual=realized - common["target"],
                edges=common["edges"], pins=common["pins"],
                component_labels=common["component_labels"])
            report.update({
                "status": "completed", "diagnostics": diagnostics,
                "exported_certificate": exported_certificate,
                "export_bounds": bounds,
                "frames": int(len(vertices)), "vertices": int(vertices.shape[1]),
                "topology_preserved": True, "identity_mapping_preserved": True,
                "sha256": {
                    "sequence.npz": _digest(directory / "sequence.npz"),
                    "certificate.npz": _digest(directory / "certificate.npz"),
                },
            })
        except Exception as error:
            # An exception may occur after one compressed member was written.
            # A terminal failed arm must never leave a scoreable partial output.
            for partial_name in ("sequence.npz", "certificate.npz"):
                (directory / partial_name).unlink(missing_ok=True)
            message = str(error).strip() or "unspecified method failure"
            report.update(status="error", exception_type=type(error).__name__[:256],
                          error=message[:4096])
        report["elapsed_seconds"] = time.monotonic() - started
        report["timing_scope"] = "CPU target/reconstruction/I-O only; excludes generation/scoring"
        _write_json(directory / "report.json", report)
        reports.append(report)
    completed = [report for report in reports if report["status"] == "completed"]
    manifest = {
        "cases": [{"case_id": uid + "-" + report["candidate_arm"], "uid": uid,
                   "case_dir": report["candidate_arm"],
                   "arm_role": report["candidate_arm"]} for report in completed],
        "expected_roles": list(ROLES),
        "common_target": {"path": "common-target.npz", "sha256": common_sha256},
        "scope": "C10 generated_unexecuted three-arm artifact; no B0/B*/scoring/admission",
    }
    _write_json(output / "manifest.json", manifest)
    result = {
        "status": "completed" if len(completed) == len(ROLES) else "incomplete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": CANDIDATE_ID, "uid": uid, "seed": seed,
        "implementation_sha256": _digest(Path(__file__)),
        "roles": list(ROLES), "arms": reports,
        "source_sequence_sha256": actual_sequence_sha256,
        "source_report_sha256": _digest(source_report_path),
        "source_refs": {
            "sequence": {"path": source_sequence_ref,
                         "sha256": actual_sequence_sha256},
            "report": {"path": source_report_ref,
                       "sha256": _digest(source_report_path)},
        },
        "common_target": {"path": "common-target.npz", "sha256": common_sha256},
        "target_parameters": {"target_strength": target_strength,
                              "max_relative_change": max_relative_change},
        "solver": {"absolute_tolerance": absolute_tolerance,
                   "relative_tolerance": relative_tolerance,
                   "max_iterations": max_iterations},
        "bounds": {"policy": bounds_policy, "lower": coordinate_lower,
                   "upper": coordinate_upper},
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False,
        "source_delivery_status": "generated_unexecuted_at_authoring",
        "elapsed_seconds": time.monotonic() - started,
    }
    candidate_report = next((report for report in reports
                             if report["candidate_arm"] == "pinned_integrable_solve"
                             and report["status"] == "completed"), None)
    if candidate_report is None:
        result["mechanism_diagnostic"] = {
            "status": "unavailable_candidate_failed",
            "artificial_target_perturbation": False,
        }
    else:
        certificate = candidate_report["exported_certificate"]
        residual = certificate["weighted_projection_residual_l2"]
        target_l2 = float(np.sqrt(2.0 * certificate["weighted_target_energy"]))
        ratio = residual / max(target_l2, np.finfo(np.float64).tiny)
        result["mechanism_diagnostic"] = {
            "status": ("inconclusive_already_integrable" if ratio
                       <= INTEGRABILITY_RATIO_TOLERANCE
                       else "natural_nonintegrable_component_observed_software_only"),
            "weighted_inconsistent_component_l2": residual,
            "weighted_target_l2": target_l2,
            "relative_inconsistent_component": ratio,
            "near_zero_relative_threshold": INTEGRABILITY_RATIO_TOLERANCE,
            "artificial_target_perturbation": False,
            "scientific_evidence": False,
        }
    _write_json(output / "candidate.json", result)
    _write_deterministic_archive(output, reports, max_artifact_bytes)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-case", type=Path)
    source.add_argument("--source-sequence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--expected-sequence-sha256", required=True)
    parser.add_argument("--source-sequence-ref", required=True)
    parser.add_argument("--source-report-ref", required=True)
    parser.add_argument("--target-strength", required=True, type=float)
    parser.add_argument("--max-relative-change", required=True, type=float)
    parser.add_argument("--absolute-tolerance", required=True, type=float)
    parser.add_argument("--relative-tolerance", required=True, type=float)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--coordinate-bounds", required=True, nargs=2, type=float,
                        metavar=("LOWER", "UPPER"))
    parser.add_argument("--bounds-policy", required=True,
                        choices=("preserve_and_report", "reject"))
    parser.add_argument("--max-artifact-bytes", required=True, type=int)
    args = parser.parse_args()
    if args.source_sequence is not None and args.source_sequence.name != "sequence.npz":
        parser.error("--source-sequence must name sequence.npz")
    source_case = args.source_case if args.source_case is not None else args.source_sequence.parent
    result = export_integrable_candidate(
        source_case, args.output, uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256,
        source_sequence_ref=args.source_sequence_ref,
        source_report_ref=args.source_report_ref,
        target_strength=args.target_strength,
        max_relative_change=args.max_relative_change,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        max_iterations=args.max_iterations,
        coordinate_lower=args.coordinate_bounds[0],
        coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy,
        max_artifact_bytes=args.max_artifact_bytes)
    print(json.dumps({
        "status": result["status"], "candidate_id": CANDIDATE_ID,
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False,
    }))
    # A retained terminal method failure is a successfully delivered artifact,
    # not a synthetic method success.  candidate.json keeps status=incomplete.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
