"""C11 rotation-preserving strain projection on complete native mesh sequences.

For every predicted face and frame, the method forms an explicit full-rank
three-dimensional map by completing the two triangle edges with their oriented
unit normal.  It rejects degenerate or reflected maps.  A proper polar factor is
then retained while only the stretch singular values are changed.  Three roles
share the same maps, face weights, pinned integrability solve and native input:

* ``arap_repair`` replaces every stretch by one;
* ``elastic_repair`` applies a frozen quadratic shrink toward one;
* ``rotation_preserving_stretch_projection`` clips stretches to ``[lower, upper]``.

The completed local edge targets are averaged with reference-area weights and
re-integrated by the repository's matrix-free pinned Laplacian solver.  Frame
zero, topology, frame IDs and vertex IDs are exact.  No GT, scorer state,
camera, text/event label or learned parameter is read.  This is generated,
unexecuted source; scientific admission and official scoring are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile
import time
import traceback

import numpy as np

from research_math import integrable_gradient_candidate as lift_module
from research_math import corotational_residual_candidate as pose_module


CANDIDATE_ID = "4d-math-20261006-c11"
ROLES = ("arap_repair", "elastic_repair",
         "rotation_preserving_stretch_projection")
METHOD_IDS = {
    "arap_repair": "c11-control-common-lift-arap-v1",
    "elastic_repair": "c11-control-common-lift-quadratic-elastic-v1",
    "rotation_preserving_stretch_projection": CANDIDATE_ID,
}
FRAME_COMPLETION = (
    "frozen whole-mesh proper Kabsch body coordinates; two oriented triangle "
    "edges plus oriented unit normal; reject reference-normal hemisphere crossing")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _positive(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def _positive_integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_native_arrays(arrays: dict) -> tuple[np.ndarray, np.ndarray]:
    return lift_module.validate_native_arrays(arrays)


def _basis(points: np.ndarray, *, degeneracy_epsilon: float) -> tuple[np.ndarray, float]:
    edge_first = points[1] - points[0]
    edge_second = points[2] - points[0]
    cross = np.cross(edge_first, edge_second)
    doubled_area = float(np.linalg.norm(cross))
    scale = max(1.0, float(np.linalg.norm(edge_first)),
                float(np.linalg.norm(edge_second)))
    if doubled_area <= degeneracy_epsilon * scale * scale:
        raise ValueError("Degenerate or near-degenerate triangle prevents a square map")
    normal = cross / doubled_area
    basis = np.column_stack((edge_first, edge_second, normal))
    determinant = float(np.linalg.det(basis))
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("Face completion is not positively oriented")
    return basis, 0.5 * doubled_area


def qualify_square_maps(vertices: np.ndarray, faces: np.ndarray, *,
                        degeneracy_epsilon: float) -> dict[str, np.ndarray]:
    """Build proper full-rank face maps and their polar/SVD evidence."""
    degeneracy_epsilon = _positive("degeneracy_epsilon", degeneracy_epsilon)
    vertices = np.asarray(vertices)
    if (not np.issubdtype(vertices.dtype, np.floating) or vertices.ndim != 3
            or vertices.shape[0] != 16 or vertices.shape[-1] != 3
            or not np.isfinite(vertices).all()):
        raise ValueError("Finite float32 vertices[16,V,3] required")
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= vertices.shape[1])):
        raise ValueError("Valid nonempty triangle topology required")
    reference_basis, reference_area = [], []
    for face in faces:
        basis, area = _basis(vertices[0, face].astype(np.float64),
                             degeneracy_epsilon=degeneracy_epsilon)
        reference_basis.append(basis)
        reference_area.append(area)
    reference_basis = np.stack(reference_basis)
    reference_area = np.asarray(reference_area, dtype=np.float64)
    inverse_reference = np.linalg.inv(reference_basis)
    maps = np.empty((16, len(faces), 3, 3), dtype=np.float64)
    left = np.empty_like(maps)
    right_t = np.empty_like(maps)
    singular = np.empty((16, len(faces), 3), dtype=np.float64)
    determinants = np.empty((16, len(faces)), dtype=np.float64)
    orthogonality = np.empty((16, len(faces)), dtype=np.float64)
    normal_alignment = np.empty((16, len(faces)), dtype=np.float64)
    current_area = np.empty((16, len(faces)), dtype=np.float64)
    for frame in range(16):
        for face_id, face in enumerate(faces):
            target_basis, area = _basis(
                vertices[frame, face].astype(np.float64),
                degeneracy_epsilon=degeneracy_epsilon)
            alignment = float(np.dot(target_basis[:, 2],
                                     reference_basis[face_id, :, 2]))
            if alignment <= 0.0:
                raise ValueError(
                    f"Reflected or large-fold ambiguous face at frame {frame}, "
                    f"face {face_id} under the frozen body-frame policy")
            deformation = target_basis @ inverse_reference[face_id]
            determinant = float(np.linalg.det(deformation))
            if not math.isfinite(determinant) or determinant <= degeneracy_epsilon:
                raise ValueError(
                    f"Reflected or near-singular local map at frame {frame}, face {face_id}")
            u, values, vt = np.linalg.svd(deformation, full_matrices=True)
            rotation = u @ vt
            if (float(np.linalg.det(rotation)) <= 0.0
                    or not np.allclose(rotation.T @ rotation, np.eye(3),
                                       atol=1e-10, rtol=1e-10)):
                raise ValueError("Polar factor is not a proper rotation")
            maps[frame, face_id] = deformation
            left[frame, face_id] = u
            right_t[frame, face_id] = vt
            singular[frame, face_id] = values
            determinants[frame, face_id] = determinant
            orthogonality[frame, face_id] = float(np.max(np.abs(
                rotation.T @ rotation - np.eye(3))))
            normal_alignment[frame, face_id] = alignment
            current_area[frame, face_id] = area
    if not np.allclose(maps[0], np.eye(3), atol=5e-7, rtol=5e-7):
        raise ValueError("Frame-zero local maps must be identity within float32 geometry error")
    return {
        "reference_basis": reference_basis,
        "reference_area": reference_area,
        "current_area": current_area,
        "maps": maps, "left": left, "right_t": right_t,
        "singular_values": singular, "determinants": determinants,
        "rotation_orthogonality_linf": orthogonality,
        "reference_normal_alignment": normal_alignment,
    }


def project_stretch_spectrum(maps: dict[str, np.ndarray], role: str, *,
                             lower: float, upper: float,
                             elastic_weight: float) -> tuple[np.ndarray, dict]:
    """Preserve the proper polar rotation and alter stretch eigenvalues only."""
    if role not in ROLES:
        raise ValueError("Unknown C11 role")
    lower, upper = _positive("lower", lower), _positive("upper", upper)
    elastic_weight = _positive("elastic_weight", elastic_weight)
    if not lower <= 1.0 <= upper or lower >= upper:
        raise ValueError("Stretch interval must satisfy 0 < lower <= 1 <= upper")
    values = maps["singular_values"]
    if role == "arap_repair":
        projected = np.ones_like(values)
        operation = "proper polar rotation with unit stretch"
    elif role == "elastic_repair":
        projected = (values + elastic_weight) / (1.0 + elastic_weight)
        operation = "quadratic stretch shrink toward identity"
    else:
        projected = np.clip(values, lower, upper)
        operation = "spectral projection onto declared stretch interval"
    transforms = np.einsum(
        "tfij,tfj,tfjk->tfik", maps["left"], projected, maps["right_t"])
    rotations_before = np.einsum("tfij,tfjk->tfik", maps["left"], maps["right_t"])
    left_after, _, right_after = np.linalg.svd(transforms)
    rotations_after = np.einsum("tfij,tfjk->tfik", left_after, right_after)
    rotation_error = float(np.max(np.abs(rotations_after - rotations_before)))
    if rotation_error > 1e-9:
        raise RuntimeError("Stretch operation changed the retained polar rotation")
    diagnostics = {
        "operation": operation,
        "singular_min_before": float(values.min()),
        "singular_max_before": float(values.max()),
        "singular_min_after": float(projected.min()),
        "singular_max_after": float(projected.max()),
        "changed_singular_values": int(np.count_nonzero(projected != values)),
        "rotation_preservation_linf": rotation_error,
        "lower": lower, "upper": upper,
        "elastic_weight": elastic_weight,
    }
    return transforms, diagnostics


def _common_target(vertices: np.ndarray, faces: np.ndarray,
                   maps: dict[str, np.ndarray], transforms: np.ndarray) -> dict:
    edges = lift_module.mesh_edges(faces, vertices.shape[1])
    labels, pins = lift_module.connected_components(vertices.shape[1], edges)
    edge_index = {tuple(edge): index for index, edge in enumerate(edges.tolist())}
    accumulated = np.zeros((16, len(edges), 3), dtype=np.float64)
    support = np.zeros(len(edges), dtype=np.float64)
    reference = vertices[0].astype(np.float64)
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        weight = float(maps["reference_area"][face_id])
        for first, second in ((face[0], face[1]), (face[1], face[2]),
                              (face[2], face[0])):
            low, high = sorted((int(first), int(second)))
            edge_id = edge_index[(low, high)]
            rest_edge = reference[high] - reference[low]
            desired = np.einsum("tfij,j->tfi", transforms[:, face_id:face_id + 1],
                                rest_edge)[:, 0]
            accumulated[:, edge_id] += weight * desired
            support[edge_id] += weight
    if np.any(support <= 0.0):
        raise RuntimeError("Every mesh edge requires positive face support")
    desired_edges = accumulated / support[None, :, None]
    source_edges = (vertices[:, edges[:, 1]] - vertices[:, edges[:, 0]]).astype(np.float64)
    target = desired_edges - source_edges
    target[0] = 0.0
    weights = support / np.median(support)
    return {
        "edges": edges, "component_labels": labels, "pins": pins,
        "target": target, "weights": weights,
        "source_edge_vectors": source_edges,
        "desired_edge_vectors": desired_edges,
    }


def construct_role(vertices: np.ndarray, faces: np.ndarray, role: str, *,
                   lower: float, upper: float, elastic_weight: float,
                   degeneracy_epsilon: float, absolute_tolerance: float,
                   relative_tolerance: float, max_iterations: int) -> tuple[np.ndarray, dict, dict]:
    pose = pose_module.fit_frozen_rigid_factors(vertices)
    body_vertices = pose["anchor"][None] + pose["body_residual"]
    maps = qualify_square_maps(body_vertices, faces,
                               degeneracy_epsilon=degeneracy_epsilon)
    transforms, projection = project_stretch_spectrum(
        maps, role, lower=lower, upper=upper,
        elastic_weight=elastic_weight)
    common = _common_target(body_vertices, faces, maps, transforms)
    displacement, lift = lift_module.pinned_integrable_solve(
        common, vertices.shape[1], absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance, max_iterations=max_iterations)
    repaired_body = body_vertices + displacement
    repaired64 = pose_module.reconstruct_with_fixed_pose(
        pose["anchor"], repaired_body - pose["anchor"][None],
        pose["rotation_rows"], pose["centroids"])
    repaired = repaired64.astype(np.float32)
    repaired[0] = vertices[0]
    repaired[:, common["pins"]] = vertices[:, common["pins"]]
    if not np.isfinite(repaired).all():
        raise ValueError("Nonfinite C11 reconstruction")
    exported_centered = repaired.astype(np.float64) - pose["centroids"][:, None, :]
    exported_body = np.einsum(
        "tvi,tji->tvj", exported_centered, pose["rotation_rows"])
    exported_displacement = exported_body - body_vertices
    realized = (exported_displacement[:, common["edges"][:, 1]]
                - exported_displacement[:, common["edges"][:, 0]])
    residual = realized - common["target"]
    weighted = common["weights"][None, :, None] * residual
    normal = np.zeros_like(exported_displacement)
    np.add.at(normal, (slice(None), common["edges"][:, 0]), -weighted)
    np.add.at(normal, (slice(None), common["edges"][:, 1]), weighted)
    free = np.setdiff1d(np.arange(vertices.shape[1], dtype=np.int64),
                        common["pins"], assume_unique=True)
    target_l2 = float(np.sqrt(np.sum(
        common["weights"][None, :, None] * np.square(common["target"]))))
    export_tolerance = max(
        1000.0 * np.finfo(np.float32).eps * max(1.0, target_l2)
        * np.sqrt(max(1, normal[:, free].size)),
        10.0 * np.sqrt(16.0 * 3.0)
        * (absolute_tolerance + relative_tolerance * target_l2))
    if not np.array_equal(repaired[:, common["pins"]],
                          vertices[:, common["pins"]]):
        raise RuntimeError("C11 exported component pins changed")
    if not np.array_equal(repaired[0], vertices[0]):
        raise RuntimeError("C11 exported frame-zero anchor changed")
    if float(np.linalg.norm(normal[:, free])) > export_tolerance:
        raise RuntimeError("C11 exported lift violates frozen normal-equation tolerance")
    certificate = {
        "singular_values_before": maps["singular_values"],
        "map_determinants": maps["determinants"],
        "reference_area": maps["reference_area"],
        "current_area": maps["current_area"],
        "reference_normal_alignment": maps["reference_normal_alignment"],
        "rotation_rows": pose["rotation_rows"],
        "centroids": pose["centroids"],
        "pose_singular_values": pose["singular_values"],
        "pose_fit_rms": pose["fit_rms"],
        "desired_edge_vectors": common["desired_edge_vectors"],
        "edge_weights": common["weights"],
        "pins": common["pins"],
        "exported_body_displacement": exported_displacement,
        "realized_edge_displacement": realized,
        "edge_residual": residual,
    }
    diagnostics = {
        "role": role, "frame_completion": FRAME_COMPLETION,
        "map_determinant_min": float(maps["determinants"].min()),
        "map_determinant_max": float(maps["determinants"].max()),
        "rotation_orthogonality_linf": float(
            maps["rotation_orthogonality_linf"].max()),
        "reference_normal_alignment_min": float(
            maps["reference_normal_alignment"].min()),
        "pose_fit_rms_max": float(pose["fit_rms"].max()),
        "projection": projection, "lift": lift,
        "float32_and_pin_export_linf": float(np.max(
            np.abs(repaired64 - repaired.astype(np.float64)))),
        "frame_zero_exact_float32": True,
        "pins_exact_float32": True,
        "pin_residual_linf": float(np.max(np.abs(
            repaired[:, common["pins"]] - vertices[:, common["pins"]]))),
        "weighted_edge_residual_l2": float(np.sqrt(np.sum(
            common["weights"][None, :, None] * np.square(residual)))),
        "normal_equation_residual_l2": float(np.linalg.norm(normal[:, free])),
        "normal_equation_tolerance_l2": export_tolerance,
        "normal_equation_within_tolerance": True,
    }
    return repaired, certificate, diagnostics


def _artifact_members(output: Path, reports: list[dict]) -> list[Path]:
    members = [output / "candidate.json", output / "manifest.json",
               output / "common-target.npz"]
    for report in reports:
        directory = output / report["candidate_arm"]
        members.append(directory / "report.json")
        if report["status"] == "completed":
            members.extend((directory / "sequence.npz", directory / "certificate.npz"))
    return sorted(members, key=lambda path: path.relative_to(output).as_posix())


def _archive(output: Path, reports: list[dict], max_artifact_bytes: int) -> dict:
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    if max_artifact_bytes < 10240:
        raise ValueError("max_artifact_bytes must allow one tar record")
    rows, estimated = [], 1024
    members = _artifact_members(output, reports)
    for path in members:
        size = path.stat().st_size
        estimated += 512 + ((size + 511) // 512) * 512
        rows.append({"path": path.relative_to(output).as_posix(),
                     "sha256": _digest(path), "size_bytes": size})
    estimated = ((estimated + 10239) // 10240) * 10240
    if estimated > max_artifact_bytes:
        raise RuntimeError("C11 artifact exceeds frozen byte ceiling")
    temporary, archive = output / "artifact.tar.tmp", output / "artifact.tar"
    with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as bundle:
        for path, row in zip(members, rows):
            info = tarfile.TarInfo(row["path"])
            info.size = row["size_bytes"]
            info.mode = 0o644
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                bundle.addfile(info, stream)
    if temporary.stat().st_size > max_artifact_bytes:
        raise RuntimeError("Written C11 artifact exceeds frozen byte ceiling")
    temporary.replace(archive)
    record = {
        "kind": "c11-terminal-artifact-archive", "version": 1,
        "archive": {"path": "artifact.tar", "sha256": _digest(archive),
                    "size_bytes": archive.stat().st_size},
        "members": rows, "max_artifact_bytes": max_artifact_bytes,
        "terminal_candidate_status": json.loads(
            (output / "candidate.json").read_text())["status"],
    }
    _write_json(output / "artifact-archive.json", record)
    return record


def materialize_candidate_archive(archive_path: Path, destination: Path, *,
                                  max_artifact_bytes: int,
                                  max_member_bytes: int) -> list[dict]:
    """Safely materialize retained C11 output without ``tar.extract``."""
    archive_path, destination = Path(archive_path), Path(destination)
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    max_member_bytes = _positive_integer("max_member_bytes", max_member_bytes)
    if archive_path.is_symlink() or archive_path.stat().st_size > max_artifact_bytes:
        raise ValueError("C11 artifact archive exceeds frozen byte ceiling")
    record_path = archive_path.with_name("artifact-archive.json")
    if record_path.is_symlink():
        raise ValueError("C11 artifact archive record must be a regular file")
    record = json.loads(record_path.read_text())
    if (record.get("kind") != "c11-terminal-artifact-archive"
            or record.get("version") != 1
            or record.get("archive") != {
                "path": "artifact.tar", "sha256": _digest(archive_path),
                "size_bytes": archive_path.stat().st_size}
            or record.get("max_artifact_bytes", max_artifact_bytes)
            > max_artifact_bytes):
        raise ValueError("C11 artifact archive record/hash mismatch")
    if destination.exists():
        raise FileExistsError("C11 archive destination is single-use")
    rows, total = [], 0
    with tarfile.open(archive_path, mode="r:") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate C11 artifact archive member")
        for member in members:
            relative = Path(member.name)
            if (not member.isfile() or relative.is_absolute()
                    or ".." in relative.parts or member.size > max_member_bytes):
                raise ValueError("Unsafe/nonregular/oversize C11 artifact member")
            total += member.size
            if total > max_artifact_bytes:
                raise ValueError("Expanded C11 artifact exceeds frozen byte ceiling")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("Regular C11 artifact member could not be read")
            digest, remaining = hashlib.sha256(), member.size
            with target.open("xb") as sink:
                while remaining:
                    block = source.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError("Truncated C11 artifact member")
                    sink.write(block)
                    digest.update(block)
                    remaining -= len(block)
                if source.read(1):
                    raise ValueError("C11 artifact member exceeds declared size")
            rows.append({"path": member.name, "sha256": digest.hexdigest(),
                         "size_bytes": member.size})
    if rows != record.get("members"):
        raise ValueError("Materialized C11 inventory differs from archive record")
    for source, target in ((archive_path, destination / "artifact.tar"),
                           (record_path, destination / "artifact-archive.json")):
        if source.stat().st_size > max_artifact_bytes:
            raise ValueError("C11 archive support file exceeds byte ceiling")
        with source.open("rb") as stream, target.open("xb") as sink:
            while True:
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                sink.write(block)
    return rows


def _validate_artifact_archive(output: Path, reports: list[dict]) -> dict:
    record_path, archive_path = (output / "artifact-archive.json",
                                 output / "artifact.tar")
    record = json.loads(record_path.read_text())
    expected = [{"path": path.relative_to(output).as_posix(),
                 "sha256": _digest(path), "size_bytes": path.stat().st_size}
                for path in _artifact_members(output, reports)]
    if (record.get("kind") != "c11-terminal-artifact-archive"
            or record.get("version") != 1
            or record.get("members") != expected
            or record.get("terminal_candidate_status")
            != json.loads((output / "candidate.json").read_text()).get("status")
            or not isinstance(record.get("max_artifact_bytes"), int)
            or record["max_artifact_bytes"] < archive_path.stat().st_size
            or record.get("archive") != {
                "path": "artifact.tar", "sha256": _digest(archive_path),
                "size_bytes": archive_path.stat().st_size}):
        raise ValueError("C11 terminal archive record mismatch")
    with tarfile.open(archive_path, mode="r:") as bundle:
        members = bundle.getmembers()
        if [member.name for member in members] != [row["path"] for row in expected]:
            raise ValueError("C11 archive member inventory/order mismatch")
        for member, row in zip(members, expected):
            if (not member.isfile() or member.size != row["size_bytes"]
                    or member.mode != 0o644 or member.uid != 0 or member.gid != 0
                    or member.mtime != 0 or member.uname or member.gname):
                raise ValueError("C11 archive metadata is not canonical")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("C11 archive regular member is unreadable")
            digest, remaining = hashlib.sha256(), member.size
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("C11 archive member is truncated")
                digest.update(block)
                remaining -= len(block)
            if stream.read(1) or digest.hexdigest() != row["sha256"]:
                raise ValueError("C11 archive member bytes mismatch")
    return {
        "archive": {"path": archive_path, "sha256": _digest(archive_path)},
        "record": {"path": record_path, "sha256": _digest(record_path)},
    }


def _load_source(source_case: Path, uid: str, expected_sequence_sha256: str):
    source_case = Path(source_case)
    sequence_path, report_path = source_case / "sequence.npz", source_case / "report.json"
    if sequence_path.name != "sequence.npz" or _digest(sequence_path) != expected_sequence_sha256:
        raise ValueError("Sequence differs from explicitly pinned input")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "completed" or report.get("uid") != uid
            or report.get("sha256", {}).get("sequence.npz") != expected_sequence_sha256):
        raise ValueError("Completed source report/hash for the same UID required")
    seed = report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer inference seed required")
    with np.load(sequence_path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    vertices, faces = validate_native_arrays(arrays)
    return arrays, vertices, faces, report, report_path, seed


def export_strain_projection_candidate(source_case: Path, output: Path, *,
                                       uid: str, expected_sequence_sha256: str,
                                       source_sequence_ref: str,
                                       source_report_ref: str,
                                       lower: float, upper: float,
                                       elastic_weight: float,
                                       degeneracy_epsilon: float,
                                       absolute_tolerance: float,
                                       relative_tolerance: float,
                                       max_iterations: int,
                                       coordinate_lower: float,
                                       coordinate_upper: float,
                                       bounds_policy: str,
                                       max_artifact_bytes: int) -> dict:
    started = time.monotonic()
    if not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    for name, value in (("source_sequence_ref", source_sequence_ref),
                        ("source_report_ref", source_report_ref)):
        relative = Path(value)
        if (not value or relative.is_absolute() or ".." in relative.parts
                or relative.name in ("", ".", "..")):
            raise ValueError(f"{name} must be a safe repository-relative path")
    lower, upper = _positive("lower", lower), _positive("upper", upper)
    if not lower <= 1.0 <= upper or lower >= upper:
        raise ValueError("Stretch interval must contain one")
    elastic_weight = _positive("elastic_weight", elastic_weight)
    degeneracy_epsilon = _positive("degeneracy_epsilon", degeneracy_epsilon)
    absolute_tolerance = _positive("absolute_tolerance", absolute_tolerance)
    relative_tolerance = _positive("relative_tolerance", relative_tolerance)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    if (not math.isfinite(coordinate_lower) or not math.isfinite(coordinate_upper)
            or coordinate_lower >= coordinate_upper):
        raise ValueError("Finite ordered coordinate bounds required")
    if bounds_policy not in ("preserve_and_report", "reject"):
        raise ValueError("Invalid bounds_policy")
    arrays, vertices, faces, _, report_path, seed = _load_source(
        Path(source_case), uid, expected_sequence_sha256)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    common_written = False
    for role in ROLES:
        arm_started = time.monotonic()
        directory = output / role
        directory.mkdir()
        report = {
            "candidate_id": CANDIDATE_ID, "candidate_arm": role,
            "method_id": METHOD_IDS[role], "uid": uid, "seed": seed,
            "source_sequence_sha256": expected_sequence_sha256,
            "source_report_sha256": _digest(report_path),
            "implementation_sha256": _digest(Path(__file__)),
            "frame_completion": FRAME_COMPLETION,
            "parameters": {
                "lower": lower, "upper": upper,
                "elastic_weight": elastic_weight,
                "degeneracy_epsilon": degeneracy_epsilon,
                "absolute_tolerance": absolute_tolerance,
                "relative_tolerance": relative_tolerance,
                "max_iterations": max_iterations,
                "coordinate_bounds": [coordinate_lower, coordinate_upper],
                "bounds_policy": bounds_policy,
            },
            "generated_unexecuted": False, "native_qualified": False,
            "scientific_admission": False, "local_method_verified": False,
            "scientific_verdict": "not_computed",
        }
        try:
            repaired, certificate, diagnostics = construct_role(
                vertices, faces, role, lower=lower, upper=upper,
                elastic_weight=elastic_weight,
                degeneracy_epsilon=degeneracy_epsilon,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                max_iterations=max_iterations)
            below = int(np.count_nonzero(repaired < coordinate_lower))
            above = int(np.count_nonzero(repaired > coordinate_upper))
            if bounds_policy == "reject" and (below or above):
                raise ValueError("C11 reconstruction violates frozen coordinate bounds")
            np.savez_compressed(directory / "sequence.npz",
                                **{**arrays, "vertices": repaired})
            np.savez_compressed(directory / "certificate.npz", **certificate)
            if not common_written:
                np.savez_compressed(
                    output / "common-target.npz",
                    source_vertices=vertices, faces=faces,
                    reference_area=certificate["reference_area"],
                    pins=certificate["pins"])
                common_written = True
            report.update(
                status="completed", diagnostics=diagnostics,
                bounds={"below": below, "above": above,
                        "policy": bounds_policy,
                        "lower": coordinate_lower, "upper": coordinate_upper},
                sha256={
                    "sequence.npz": _digest(directory / "sequence.npz"),
                    "certificate.npz": _digest(directory / "certificate.npz"),
                })
        except Exception as error:  # terminal per-role evidence
            for partial in (directory / "sequence.npz", directory / "certificate.npz"):
                partial.unlink(missing_ok=True)
            report.update(status="error", exception_type=type(error).__name__,
                          error=str(error)[:4096],
                          traceback="".join(traceback.format_exception(error))[-16384:])
        report["elapsed_seconds"] = time.monotonic() - arm_started
        _write_json(directory / "report.json", report)
        reports.append(report)
    if not common_written:
        np.savez_compressed(output / "common-target.npz",
                            source_vertices=vertices, faces=faces,
                            reference_area=np.empty((0,), dtype=np.float64),
                            pins=np.empty((0,), dtype=np.int64))
    status = "completed" if all(row["status"] == "completed" for row in reports) else "incomplete"
    record = {
        "kind": "c11-strain-projection-candidate", "version": 1,
        "candidate_id": CANDIDATE_ID, "status": status,
        "uid": uid, "seed": seed, "roles": list(ROLES), "arms": reports,
        "source_sequence_sha256": expected_sequence_sha256,
        "source_report_sha256": _digest(report_path),
        "source_refs": {
            "sequence": {"path": source_sequence_ref,
                         "sha256": expected_sequence_sha256},
            "report": {"path": source_report_ref,
                       "sha256": _digest(report_path)},
        },
        "common_target": {"path": "common-target.npz",
                          "sha256": _digest(output / "common-target.npz")},
        "implementation_sha256": _digest(Path(__file__)),
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False,
        "source_delivery_status": "generated_unexecuted_at_authoring",
        "elapsed_seconds": time.monotonic() - started,
        "timing_scope": "CPU local-map projection/integration and artifact I/O only",
    }
    _write_json(output / "candidate.json", record)
    cases = []
    for report in reports:
        if report["status"] == "completed":
            cases.append({"case_id": uid + "-" + report["candidate_arm"],
                          "uid": uid, "case_dir": report["candidate_arm"],
                          "arm_role": report["candidate_arm"]})
    _write_json(output / "manifest.json", {
        "expected_roles": list(ROLES), "cases": cases,
        "common_target": record["common_target"],
        "scope": "C11 generated_unexecuted three-arm artifact; no B0/B*/scoring/admission",
    })
    _archive(output, reports, max_artifact_bytes)
    return record


def validate_candidate_artifact(root: Path, candidate_path: Path) -> dict:
    root, candidate_path = Path(root).resolve(), Path(candidate_path).resolve()
    candidate_path.relative_to(root)
    record = json.loads(candidate_path.read_text())
    if (record.get("kind") != "c11-strain-projection-candidate"
            or record.get("version") != 1
            or record.get("candidate_id") != CANDIDATE_ID
            or record.get("roles") != list(ROLES)
            or len(record.get("arms", [])) != len(ROLES)):
        raise ValueError("Current C11 candidate record required")
    implementation_sha256 = _digest(Path(__file__))
    if (record.get("implementation_sha256") != implementation_sha256
            or record.get("native_qualified") is not False
            or record.get("scientific_admission") is not False
            or record.get("local_method_verified") is not False
            or record.get("source_delivery_status")
            != "generated_unexecuted_at_authoring"):
        raise ValueError("C11 candidate identity or evidence flags changed")
    directory = candidate_path.parent
    source_ref = record.get("source_refs", {}).get("sequence")
    report_ref = record.get("source_refs", {}).get("report")
    if not isinstance(source_ref, dict) or not isinstance(report_ref, dict):
        raise ValueError("C11 source references required")
    for ref in (source_ref, report_ref):
        if (set(ref) != {"path", "sha256"}
                or Path(ref["path"]).is_absolute()
                or ".." in Path(ref["path"]).parts
                or not isinstance(ref["sha256"], str)
                or len(ref["sha256"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in ref["sha256"])):
            raise ValueError("Exact project-relative C11 source ref required")
    source_path, source_report_path = (root / source_ref["path"],
                                       root / report_ref["path"])
    if (source_path.is_symlink() or source_report_path.is_symlink()
            or source_path.resolve().parent != source_report_path.resolve().parent):
        raise ValueError("Physical sibling C11 source pair required")
    source_path, source_report_path = source_path.resolve(), source_report_path.resolve()
    source_path.relative_to(root); source_report_path.relative_to(root)
    if source_path.name != "sequence.npz" or source_report_path != source_path.with_name("report.json"):
        raise ValueError("Exact sequence.npz/report.json C11 source pair required")
    if (_digest(source_path) != source_ref["sha256"]
            or _digest(source_report_path) != report_ref["sha256"]):
        raise ValueError("C11 source references changed")
    arrays, vertices, faces, source_report, _, seed = _load_source(
        source_path.parent, record["uid"], source_ref["sha256"])
    if (record.get("seed") != seed
            or record.get("source_sequence_sha256") != source_ref["sha256"]
            or record.get("source_report_sha256") != report_ref["sha256"]
            or source_report.get("seed") != record.get("seed")):
        raise ValueError("C11 candidate/source identity aliases changed")
    completed = []
    for role, saved_report in zip(ROLES, record["arms"]):
        report_path = directory / role / "report.json"
        report = json.loads(report_path.read_text())
        if report != saved_report or report.get("candidate_arm") != role:
            raise ValueError("C11 terminal role report changed")
        parameters = report.get("parameters")
        if (not isinstance(parameters, dict) or set(parameters) != {
                "lower", "upper", "elastic_weight", "degeneracy_epsilon",
                "absolute_tolerance", "relative_tolerance", "max_iterations",
                "coordinate_bounds", "bounds_policy"}):
            raise ValueError("Exact C11 role parameter schema required")
        lower = _positive("lower", parameters["lower"])
        upper = _positive("upper", parameters["upper"])
        coordinate_bounds = parameters["coordinate_bounds"]
        if (lower >= upper or not lower <= 1.0 <= upper
                or not isinstance(coordinate_bounds, list)
                or len(coordinate_bounds) != 2
                or not all(np.isfinite(value) for value in coordinate_bounds)
                or coordinate_bounds[0] >= coordinate_bounds[1]
                or parameters["bounds_policy"] not in ("preserve_and_report", "reject")
                or _positive("elastic_weight", parameters["elastic_weight"])
                != parameters["elastic_weight"]
                or _positive("degeneracy_epsilon", parameters["degeneracy_epsilon"])
                != parameters["degeneracy_epsilon"]
                or _positive("absolute_tolerance", parameters["absolute_tolerance"])
                != parameters["absolute_tolerance"]
                or _positive("relative_tolerance", parameters["relative_tolerance"])
                != parameters["relative_tolerance"]
                or _positive_integer("max_iterations", parameters["max_iterations"])
                != parameters["max_iterations"]):
            raise ValueError("Invalid C11 role parameters")
        if (report.get("candidate_id") != CANDIDATE_ID
                or report.get("method_id") != METHOD_IDS[role]
                or report.get("uid") != record["uid"]
                or report.get("seed") != record["seed"]
                or report.get("source_sequence_sha256") != source_ref["sha256"]
                or report.get("source_report_sha256") != report_ref["sha256"]
                or report.get("implementation_sha256") != implementation_sha256
                or report.get("frame_completion") != FRAME_COMPLETION
                or report.get("generated_unexecuted") is not False
                or report.get("native_qualified") is not False
                or report.get("scientific_admission") is not False
                or report.get("local_method_verified") is not False
                or report.get("scientific_verdict") != "not_computed"):
            raise ValueError("C11 report method/source/evidence identity changed")
        if report["status"] == "completed":
            expected, expected_certificate, expected_diagnostics = construct_role(
                vertices, faces, role,
                lower=parameters["lower"], upper=parameters["upper"],
                elastic_weight=parameters["elastic_weight"],
                degeneracy_epsilon=parameters["degeneracy_epsilon"],
                absolute_tolerance=parameters["absolute_tolerance"],
                relative_tolerance=parameters["relative_tolerance"],
                max_iterations=parameters["max_iterations"])
            sequence_path, certificate_path = (directory / role / "sequence.npz",
                                                directory / role / "certificate.npz")
            if (report["sha256"] != {
                    "sequence.npz": _digest(sequence_path),
                    "certificate.npz": _digest(certificate_path)}):
                raise ValueError("C11 completed arm hashes changed")
            with np.load(sequence_path, allow_pickle=False) as saved:
                arm = {name: saved[name].copy() for name in saved.files}
            validate_native_arrays(arm)
            if set(arm) != set(arrays) or not np.array_equal(arm["vertices"], expected):
                raise ValueError("C11 completed sequence differs from reconstruction")
            for name in set(arrays) - {"vertices"}:
                if not np.array_equal(arm[name], arrays[name]):
                    raise ValueError("C11 arm changed native identity: " + name)
            with np.load(certificate_path, allow_pickle=False) as saved:
                certificate = {name: saved[name].copy() for name in saved.files}
            if set(certificate) != set(expected_certificate):
                raise ValueError("C11 certificate inventory changed")
            for name in certificate:
                if not np.array_equal(certificate[name], expected_certificate[name]):
                    raise ValueError("C11 certificate differs from reconstruction: " + name)
            expected_bounds = {
                "below": int(np.count_nonzero(expected < coordinate_bounds[0])),
                "above": int(np.count_nonzero(expected > coordinate_bounds[1])),
                "policy": parameters["bounds_policy"],
                "lower": coordinate_bounds[0], "upper": coordinate_bounds[1],
            }
            if (report.get("diagnostics") != expected_diagnostics
                    or report.get("bounds") != expected_bounds
                    or (parameters["bounds_policy"] == "reject"
                        and (expected_bounds["below"] or expected_bounds["above"]))):
                raise ValueError("C11 diagnostics or export bounds changed")
            completed.append(role)
        elif report["status"] == "error":
            if ((directory / role / "sequence.npz").exists()
                    or (directory / role / "certificate.npz").exists()
                    or not isinstance(report.get("exception_type"), str)
                    or not report["exception_type"]
                    or len(report["exception_type"]) > 256
                    or not isinstance(report.get("error"), str)
                    or not report["error"] or len(report["error"]) > 4096):
                raise ValueError("C11 failed role lacks bounded terminal evidence")
        else:
            raise ValueError("C11 role must terminate completed or error")
    expected_status = "completed" if len(completed) == len(ROLES) else "incomplete"
    if record.get("status") != expected_status:
        raise ValueError("C11 aggregate status differs from terminal roles")
    common_ref = record.get("common_target")
    if (common_ref != {"path": "common-target.npz",
                      "sha256": _digest(directory / "common-target.npz")}):
        raise ValueError("C11 common target changed")
    with np.load(directory / "common-target.npz", allow_pickle=False) as saved:
        common = {name: saved[name].copy() for name in saved.files}
    if set(common) != {"source_vertices", "faces", "reference_area", "pins"}:
        raise ValueError("C11 common-target inventory changed")
    if (not np.array_equal(common["source_vertices"], vertices)
            or not np.array_equal(common["faces"], faces)):
        raise ValueError("C11 common target changed its native source")
    if completed:
        first_certificate = np.load(
            directory / completed[0] / "certificate.npz", allow_pickle=False)
        try:
            if (not np.array_equal(common["reference_area"],
                                   first_certificate["reference_area"])
                    or not np.array_equal(common["pins"], first_certificate["pins"])):
                raise ValueError("C11 common target differs from recomputed certificate")
        finally:
            first_certificate.close()
    elif common["reference_area"].size or common["pins"].size:
        raise ValueError("All-failed C11 artifact must have empty common target")
    expected_manifest = {
        "expected_roles": list(ROLES),
        "cases": [{"case_id": record["uid"] + "-" + role,
                   "uid": record["uid"], "case_dir": role, "arm_role": role}
                  for role in completed],
        "common_target": common_ref,
        "scope": "C11 generated_unexecuted three-arm artifact; no B0/B*/scoring/admission",
    }
    if json.loads((directory / "manifest.json").read_text()) != expected_manifest:
        raise ValueError("C11 manifest differs from validated terminal roles")
    archive_refs = _validate_artifact_archive(directory, record["arms"])
    return {
        **record, "completed_roles": completed,
        "artifact_archive": {
            key: {"path": ref["path"].relative_to(root).as_posix(),
                  "sha256": ref["sha256"]}
            for key, ref in archive_refs.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--expected-sequence-sha256", required=True)
    parser.add_argument("--source-sequence-ref", required=True)
    parser.add_argument("--source-report-ref", required=True)
    parser.add_argument("--stretch-bounds", nargs=2, type=float, required=True,
                        metavar=("LOWER", "UPPER"))
    parser.add_argument("--elastic-weight", type=float, required=True)
    parser.add_argument("--degeneracy-epsilon", type=float, required=True)
    parser.add_argument("--absolute-tolerance", type=float, required=True)
    parser.add_argument("--relative-tolerance", type=float, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--coordinate-bounds", nargs=2, type=float, required=True,
                        metavar=("LOWER", "UPPER"))
    parser.add_argument("--bounds-policy", choices=("preserve_and_report", "reject"),
                        required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    args = parser.parse_args()
    result = export_strain_projection_candidate(
        args.source_sequence.parent, args.output, uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256,
        source_sequence_ref=args.source_sequence_ref,
        source_report_ref=args.source_report_ref,
        lower=args.stretch_bounds[0], upper=args.stretch_bounds[1],
        elastic_weight=args.elastic_weight,
        degeneracy_epsilon=args.degeneracy_epsilon,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        max_iterations=args.max_iterations,
        coordinate_lower=args.coordinate_bounds[0],
        coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy,
        max_artifact_bytes=args.max_artifact_bytes)
    print(json.dumps({"status": result["status"],
                      "candidate_id": CANDIDATE_ID,
                      "native_qualified": False,
                      "scientific_admission": False}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
