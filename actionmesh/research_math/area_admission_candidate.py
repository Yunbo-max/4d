"""C12 maximal feasible triangle-area admission for complete native 4D meshes.

One predicted-mesh-only ARAP repair supplies the common proposed update.  The
three arms differ only in how one global step is admitted: a frozen scalar,
ordinary geometric backtracking, or the first true crossing of the exact face
quadratics.  Projected oriented area is only a conservative local safeguard; it
does not prove injectivity, absence of self-intersection, or physical validity.

This is generated source.  Local acceptance, native scoring and scientific
admission remain separate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import traceback

import numpy as np

from research_math.protected_geometry_candidate import (
    compute_common_geometry_repair,
    validate_native_arrays,
)


CANDIDATE_ID = "4d-math-20261006-c12"
ROLES = ("fixed_damping", "generic_backtracking",
         "exact_quadratic_admission")
METHOD_IDS = {
    "fixed_damping": "c12-control-fixed-damping",
    "generic_backtracking": "c12-control-generic-backtracking",
    "exact_quadratic_admission": CANDIDATE_ID,
}
FRAME_COMPLETION = "all 16 original frames; exact frame-zero anchor and identity mapping"


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
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return value


def _unit_interval(name: str, value, *, strict: bool = True) -> float:
    value = _positive(name, value)
    if value >= 1.0 if strict else value > 1.0:
        raise ValueError(f"{name} must be {'in (0,1)' if strict else 'in (0,1]'}")
    return value


def _positive_integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def form_exact_face_quadratics(vertices: np.ndarray, desired: np.ndarray,
                               faces: np.ndarray, *, beta: float,
                               degeneracy_epsilon: float) -> dict[str, np.ndarray]:
    """Return q(alpha)=projected_area(alpha)-beta*area(0) for every frame/face."""
    vertices = np.asarray(vertices, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    faces = np.asarray(faces)
    beta = _unit_interval("beta", beta)
    degeneracy_epsilon = _positive("degeneracy_epsilon", degeneracy_epsilon)
    if (vertices.ndim != 3 or vertices.shape[-1] != 3
            or desired.shape != vertices.shape or not np.isfinite(vertices).all()
            or not np.isfinite(desired).all()):
        raise ValueError("Finite matching [T,V,3] source/update arrays required")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= vertices.shape[1])):
        raise ValueError("Valid integer triangle topology required")
    tri = vertices[:, faces]
    delta = desired[:, faces]
    e1, e2 = tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0]
    d1, d2 = delta[:, :, 1] - delta[:, :, 0], delta[:, :, 2] - delta[:, :, 0]
    cross0 = np.cross(e1, e2)
    a0 = np.linalg.norm(cross0, axis=-1)
    coordinate_scale = max(1.0, float(np.max(np.abs(vertices))))
    threshold = degeneracy_epsilon * coordinate_scale * coordinate_scale
    if not np.isfinite(a0).all() or np.any(a0 <= threshold):
        raise ValueError("Degenerate original face encountered")
    normal = cross0 / a0[..., None]
    a1 = np.einsum("tfc,tfc->tf", normal,
                   np.cross(d1, e2) + np.cross(e1, d2))
    a2 = np.einsum("tfc,tfc->tf", normal, np.cross(d1, d2))
    q0 = (1.0 - beta) * a0
    if not all(np.isfinite(value).all() for value in (a1, a2, q0)):
        raise ValueError("Nonfinite face quadratic")
    return {
        "a0": a0, "a1": a1, "a2": a2,
        "q0": q0, "q1": a1, "q2": a2, "scale": a0,
        "beta": np.array(beta, dtype=np.float64),
    }


def _real_roots(q0: float, q1: float, q2: float):
    """Return real roots without erasing scale-small or near-tangent crossings."""
    scale = max(abs(q0), abs(q1), abs(q2))
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError("Finite nonzero polynomial required")
    q0, q1, q2 = q0 / scale, q1 / scale, q2 / scale
    if q2 == 0.0:
        if q1 == 0.0:
            return []
        return [(-q0 / q1, q1)]
    discriminant = q1 * q1 - 4.0 * q2 * q0
    roundoff = 32.0 * np.finfo(np.float64).eps * (
        abs(q1 * q1) + abs(4.0 * q2 * q0))
    if discriminant < 0.0:
        if discriminant >= -roundoff:
            raise RuntimeError("Numerically ambiguous near-tangent area polynomial")
        return []
    if discriminant == 0.0:
        root = -q1 / (2.0 * q2)
        return [(root, 0.0)]
    square = math.sqrt(max(0.0, discriminant))
    # Stable quadratic formula followed by Vieta for the second root.
    numerator = -0.5 * (q1 + math.copysign(square, q1))
    if numerator == 0.0:
        roots = [(-q1 - square) / (2.0 * q2),
                 (-q1 + square) / (2.0 * q2)]
    else:
        roots = [numerator / q2, q0 / numerator]
    return [(root, 2.0 * q2 * root + q1) for root in sorted(roots)]


def find_first_violation_interval(coefficients: dict[str, np.ndarray], *,
                                  root_tolerance: float) -> dict:
    """Find the earliest positive-to-negative root over all constraints."""
    root_tolerance = _positive("root_tolerance", root_tolerance)
    required = ("q0", "q1", "q2", "scale")
    arrays = {name: np.asarray(coefficients[name], dtype=np.float64)
              for name in required}
    shape = arrays["q0"].shape
    if (not shape or any(value.shape != shape for value in arrays.values())
            or any(not np.isfinite(value).all() for value in arrays.values())
            or np.any(arrays["q0"] <= 0.0) or np.any(arrays["scale"] <= 0.0)):
        raise ValueError("Positive finite same-shaped quadratic constraints required")
    boundary, active, tangencies, crossings = 1.0, None, 0, 0
    for index in np.ndindex(shape):
        face_scale = float(arrays["scale"][index])
        q0, q1, q2 = (float(arrays[name][index]) / face_scale
                      for name in ("q0", "q1", "q2"))
        for root, derivative in _real_roots(q0, q1, q2):
            if root < -root_tolerance or root > 1.0 + root_tolerance:
                continue
            root = min(1.0, max(0.0, root))
            if derivative == 0.0:
                tangencies += 1
                continue
            if derivative < 0.0:
                crossings += 1
                if active is None or root < boundary:
                    boundary, active = root, index
    # A missed negative endpoint indicates numerically unresolved roots.
    endpoint = arrays["q0"] + arrays["q1"] + arrays["q2"]
    endpoint_tolerance = root_tolerance * arrays["scale"]
    if np.any(endpoint < -endpoint_tolerance) and active is None:
        raise RuntimeError("Negative endpoint without resolved crossing")
    return {
        "boundary_alpha": float(boundary),
        "boundary_kind": "crossing" if active is not None else "none",
        "active_frame": int(active[0]) if active is not None else None,
        "active_face": int(active[1]) if active is not None else None,
        "crossing_count": crossings,
        "tangency_count": tangencies,
        "root_tolerance": root_tolerance,
    }


def check_all_face_area_constraints(coefficients: dict[str, np.ndarray], alpha: float,
                                    *, tolerance: float) -> tuple[bool, dict]:
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError("alpha must be a finite scalar in [0,1]")
    alpha = float(alpha)
    tolerance = _positive("tolerance", tolerance)
    if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be a finite scalar in [0,1]")
    q0 = np.asarray(coefficients["q0"], dtype=np.float64)
    q1 = np.asarray(coefficients["q1"], dtype=np.float64)
    q2 = np.asarray(coefficients["q2"], dtype=np.float64)
    scale = np.asarray(coefficients["scale"], dtype=np.float64)
    values = q0 + alpha * q1 + alpha * alpha * q2
    normalized = values / scale
    slack = tolerance * scale
    return bool(np.all(values >= -slack)), {
        "minimum_normalized_slack": float(np.min(normalized)),
        "violating_constraints": int(np.count_nonzero(values < -slack)),
        "near_boundary_constraints": int(np.count_nonzero(np.abs(values) <= slack)),
    }


def admit_role(vertices: np.ndarray, desired: np.ndarray, faces: np.ndarray,
               role: str, *, beta: float, fixed_alpha: float,
               backtracking_factor: float, backtracking_max_steps: int,
               degeneracy_epsilon: float, root_tolerance: float,
               absolute_margin: float, relative_margin: float):
    if role not in ROLES:
        raise ValueError("Unknown C12 role: " + role)
    fixed_alpha = _unit_interval("fixed_alpha", fixed_alpha, strict=False)
    backtracking_factor = _unit_interval("backtracking_factor", backtracking_factor)
    backtracking_max_steps = _positive_integer(
        "backtracking_max_steps", backtracking_max_steps)
    absolute_margin = _positive("absolute_margin", absolute_margin)
    relative_margin = _positive("relative_margin", relative_margin)
    coefficients = form_exact_face_quadratics(
        vertices, desired, faces, beta=beta,
        degeneracy_epsilon=degeneracy_epsilon)
    boundary = find_first_violation_interval(
        coefficients, root_tolerance=root_tolerance)
    check_tolerance = max(root_tolerance, 10.0 * np.finfo(np.float64).eps)
    steps = 0
    if role == "fixed_damping":
        alpha = fixed_alpha
        feasible, check = check_all_face_area_constraints(
            coefficients, alpha, tolerance=check_tolerance)
        if not feasible:
            raise RuntimeError("Frozen fixed damping violates face-area constraint")
    elif role == "generic_backtracking":
        alpha = 1.0
        while True:
            feasible, check = check_all_face_area_constraints(
                coefficients, alpha, tolerance=check_tolerance)
            if feasible:
                break
            steps += 1
            if steps > backtracking_max_steps:
                raise RuntimeError("Generic backtracking exhausted frozen step budget")
            alpha *= backtracking_factor
    else:
        raw = boundary["boundary_alpha"]
        if boundary["boundary_kind"] == "none":
            alpha = 1.0
        else:
            margin = absolute_margin + relative_margin * max(1.0, abs(raw))
            alpha = max(0.0, raw - margin)
        feasible, check = check_all_face_area_constraints(
            coefficients, alpha, tolerance=check_tolerance)
        repairs = 0
        while not feasible and alpha > 0.0 and repairs < 64:
            alpha = float(np.nextafter(alpha, 0.0))
            feasible, check = check_all_face_area_constraints(
                coefficients, alpha, tolerance=check_tolerance)
            repairs += 1
        if not feasible:
            raise RuntimeError("Inward-margined exact step is not numerically feasible")
        steps = repairs
    updated64 = np.asarray(vertices, dtype=np.float64) + alpha * np.asarray(
        desired, dtype=np.float64)
    updated = updated64.astype(np.float32)
    updated[0] = np.asarray(vertices)[0]
    if not np.isfinite(updated).all():
        raise ValueError("Nonfinite C12 output")
    if not np.array_equal(updated[0], np.asarray(vertices)[0]):
        raise ValueError("Float32 export cannot retain the exact frame-zero anchor")
    exported_coefficients = form_exact_face_quadratics(
        vertices, updated.astype(np.float64) - np.asarray(vertices, dtype=np.float64),
        faces, beta=beta, degeneracy_epsilon=degeneracy_epsilon)
    export_feasible, export_check = check_all_face_area_constraints(
        exported_coefficients, 1.0, tolerance=check_tolerance)
    if not export_feasible:
        raise RuntimeError("Float32 export violates the admitted face-area constraint")
    certificate = {
        "a0": coefficients["a0"], "a1": coefficients["a1"],
        "a2": coefficients["a2"], "q0": coefficients["q0"],
        "q1": coefficients["q1"], "q2": coefficients["q2"],
        "scale": coefficients["scale"],
        "beta": coefficients["beta"],
        "admitted_alpha": np.array(alpha, dtype=np.float64),
    }
    diagnostics = {
        "role": role, "admitted_alpha": float(alpha),
        "backtracking_or_inward_steps": steps,
        "constraint_check": check, "boundary": boundary,
        "export_constraint_check": export_check,
        "frame_zero_exact_float32": True,
        "float32_update_linf": float(np.max(np.abs(updated64 - updated.astype(np.float64)))),
        "limitation": "projected oriented area only; no global injectivity/self-intersection guarantee",
    }
    return updated, certificate, diagnostics


def _load_source(source_case: Path, uid: str, expected_sequence_sha256: str):
    source_case = Path(source_case)
    sequence, report_path = source_case / "sequence.npz", source_case / "report.json"
    if sequence.is_symlink() or report_path.is_symlink() or digest(sequence) != expected_sequence_sha256:
        raise ValueError("Exact regular receipt-bound source pair required")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "completed" or report.get("uid") != uid
            or report.get("sha256", {}).get("sequence.npz") != expected_sequence_sha256):
        raise ValueError("Completed source report/hash for the same UID required")
    seed = report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer inference seed required")
    with np.load(sequence, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    vertices, faces, times = validate_native_arrays(arrays)
    return arrays, vertices, faces, times, report_path, seed


def _members(output: Path, reports: list[dict]) -> list[Path]:
    paths = [output / "candidate.json", output / "manifest.json",
             output / "common-target.npz"]
    for report in reports:
        directory = output / report["candidate_arm"]
        paths.append(directory / "report.json")
        if report["status"] == "completed":
            paths.extend((directory / "sequence.npz", directory / "certificate.npz"))
    return sorted(paths, key=lambda path: path.relative_to(output).as_posix())


def _write_archive(output: Path, reports: list[dict], max_artifact_bytes: int) -> dict:
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    if max_artifact_bytes < 10240:
        raise ValueError("max_artifact_bytes must permit one tar record")
    rows, estimate = [], 1024
    for path in _members(output, reports):
        size = path.stat().st_size
        estimate += 512 + ((size + 511) // 512) * 512
        rows.append({"path": path.relative_to(output).as_posix(),
                     "sha256": digest(path), "size_bytes": size})
    estimate = ((estimate + 10239) // 10240) * 10240
    if estimate > max_artifact_bytes:
        raise RuntimeError("C12 artifact exceeds frozen byte ceiling")
    temporary, archive = output / "artifact.tar.tmp", output / "artifact.tar"
    with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as bundle:
        for row in rows:
            path = output / row["path"]
            info = tarfile.TarInfo(row["path"])
            info.size = row["size_bytes"]; info.mode = 0o644
            info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ""
            with path.open("rb") as stream:
                bundle.addfile(info, stream)
    if temporary.stat().st_size > max_artifact_bytes:
        raise RuntimeError("Written C12 artifact exceeds frozen byte ceiling")
    temporary.replace(archive)
    record = {
        "kind": "c12-terminal-artifact-archive", "version": 1,
        "archive": {"path": "artifact.tar", "sha256": digest(archive),
                    "size_bytes": archive.stat().st_size},
        "members": rows, "max_artifact_bytes": max_artifact_bytes,
        "terminal_candidate_status": json.loads(
            (output / "candidate.json").read_text())["status"],
    }
    write_json(output / "artifact-archive.json", record)
    return record


def _validate_archive(output: Path, reports: list[dict]) -> dict:
    record_path, archive = output / "artifact-archive.json", output / "artifact.tar"
    record = json.loads(record_path.read_text())
    expected = [{"path": path.relative_to(output).as_posix(),
                 "sha256": digest(path), "size_bytes": path.stat().st_size}
                for path in _members(output, reports)]
    if (record.get("kind") != "c12-terminal-artifact-archive"
            or record.get("version") != 1 or record.get("members") != expected
            or record.get("terminal_candidate_status")
            != json.loads((output / "candidate.json").read_text()).get("status")
            or record.get("archive") != {"path": "artifact.tar",
                "sha256": digest(archive), "size_bytes": archive.stat().st_size}
            or not isinstance(record.get("max_artifact_bytes"), int)
            or archive.stat().st_size > record["max_artifact_bytes"]):
        raise ValueError("C12 terminal archive record mismatch")
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        if [member.name for member in members] != [row["path"] for row in expected]:
            raise ValueError("C12 archive inventory/order mismatch")
        for member, row in zip(members, expected):
            if (not member.isfile() or member.size != row["size_bytes"]
                    or member.mode != 0o644 or member.uid or member.gid
                    or member.mtime or member.uname or member.gname):
                raise ValueError("C12 archive metadata is not canonical")
            stream = bundle.extractfile(member)
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != row["sha256"]:
                raise ValueError("C12 archive member bytes mismatch")
    return {"archive": {"path": archive, "sha256": digest(archive)},
            "record": {"path": record_path, "sha256": digest(record_path)}}


def materialize_candidate_archive(archive_path: Path, destination: Path, *,
                                  max_artifact_bytes: int,
                                  max_member_bytes: int,
                                  max_files: int = 10_000) -> list[dict]:
    archive_path, destination = Path(archive_path), Path(destination)
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    max_member_bytes = _positive_integer("max_member_bytes", max_member_bytes)
    max_files = _positive_integer("max_files", max_files)
    if (archive_path.is_symlink() or not archive_path.is_file()
            or archive_path.stat().st_size > max_artifact_bytes):
        raise ValueError("C12 archive exceeds frozen byte ceiling")
    record_path = archive_path.with_name("artifact-archive.json")
    metadata_limit = min(max_member_bytes, 16 * 1024 * 1024)
    if (record_path.is_symlink() or not record_path.is_file()
            or record_path.stat().st_size > metadata_limit):
        raise ValueError("C12 archive record is not a bounded regular file")
    record = json.loads(record_path.read_text())
    if (record.get("kind") != "c12-terminal-artifact-archive"
            or record.get("archive") != {"path": "artifact.tar",
                "sha256": digest(archive_path), "size_bytes": archive_path.stat().st_size}
            or not isinstance(record.get("max_artifact_bytes"), int)
            or record["max_artifact_bytes"] > max_artifact_bytes):
        raise ValueError("C12 archive record/hash mismatch")
    if destination.exists():
        raise FileExistsError("C12 archive destination is single-use")
    declared = record.get("members")
    if (not isinstance(declared, list) or not declared
            or len(declared) > max_files):
        raise ValueError("C12 archive record has invalid member count")
    rows, total = [], 0
    with tarfile.open(archive_path, "r:") as bundle:
        members = bundle.getmembers()
        if (not members or len(members) > max_files
                or len({member.name for member in members}) != len(members)):
            raise ValueError("Invalid C12 archive member count or identity")
        for member in members:
            relative = Path(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or member.size > max_member_bytes):
                raise ValueError("Unsafe C12 archive member")
            total += member.size
            if total > max_artifact_bytes:
                raise ValueError("Expanded C12 archive exceeds byte ceiling")
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("Unreadable C12 archive member")
            value = hashlib.sha256()
            copied = 0
            for block in iter(lambda: source.read(1024 * 1024), b""):
                copied += len(block)
                if copied > member.size:
                    raise ValueError("C12 archive member grew while validating")
                value.update(block)
            if copied != member.size:
                raise ValueError("C12 archive member truncated while validating")
            rows.append({"path": member.name, "sha256": value.hexdigest(),
                         "size_bytes": copied})
    if rows != declared:
        raise ValueError("Materialized C12 inventory differs from record")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("Physical C12 materialization parent required")
    temporary = Path(tempfile.mkdtemp(
        prefix=destination.name + ".tmp-", dir=destination.parent))
    try:
        with tarfile.open(archive_path, "r:") as bundle:
            for member, row in zip(bundle.getmembers(), rows):
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError("Unreadable C12 archive member")
                target = temporary / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                value = hashlib.sha256(); copied = 0
                with target.open("xb") as sink:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        copied += len(block)
                        if copied > row["size_bytes"]:
                            raise ValueError("C12 archive member grew while materializing")
                        value.update(block); sink.write(block)
                if copied != row["size_bytes"] or value.hexdigest() != row["sha256"]:
                    raise ValueError("C12 materialized member differs from validation")
        for source, target in ((archive_path, temporary / "artifact.tar"),
                               (record_path, temporary / "artifact-archive.json")):
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return rows


def export_area_admission_candidate(source_case: Path, output: Path, *, uid: str,
                                    expected_sequence_sha256: str,
                                    source_sequence_ref: str,
                                    source_report_ref: str, beta: float,
                                    fixed_alpha: float,
                                    backtracking_factor: float,
                                    backtracking_max_steps: int,
                                    degeneracy_epsilon: float,
                                    root_tolerance: float,
                                    absolute_margin: float,
                                    relative_margin: float,
                                    arap_weight: float,
                                    temporal_weight: float,
                                    arap_iterations: int,
                                    cg_tolerance: float,
                                    cg_max_iterations: int,
                                    coordinate_lower: float,
                                    coordinate_upper: float,
                                    bounds_policy: str,
                                    max_artifact_bytes: int) -> dict:
    started = time.monotonic()
    if not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    for name, value in (("source_sequence_ref", source_sequence_ref),
                        ("source_report_ref", source_report_ref)):
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{name} must be a project-relative path")
    beta = _unit_interval("beta", beta)
    fixed_alpha = _unit_interval("fixed_alpha", fixed_alpha, strict=False)
    backtracking_factor = _unit_interval("backtracking_factor", backtracking_factor)
    backtracking_max_steps = _positive_integer("backtracking_max_steps", backtracking_max_steps)
    for name, value in (("degeneracy_epsilon", degeneracy_epsilon),
                        ("root_tolerance", root_tolerance),
                        ("absolute_margin", absolute_margin),
                        ("relative_margin", relative_margin),
                        ("arap_weight", arap_weight),
                        ("temporal_weight", temporal_weight),
                        ("cg_tolerance", cg_tolerance)):
        _positive(name, value)
    arap_iterations = _positive_integer("arap_iterations", arap_iterations)
    cg_max_iterations = _positive_integer("cg_max_iterations", cg_max_iterations)
    if (not math.isfinite(coordinate_lower) or not math.isfinite(coordinate_upper)
            or coordinate_lower >= coordinate_upper):
        raise ValueError("Finite ordered coordinate bounds required")
    if bounds_policy not in ("preserve_and_report", "reject"):
        raise ValueError("Invalid bounds_policy")
    arrays, vertices, faces, times, report_path, seed = _load_source(
        Path(source_case), uid, expected_sequence_sha256)
    desired, repair = compute_common_geometry_repair(
        vertices, faces, times, arap_weight=arap_weight,
        temporal_weight=temporal_weight, iterations=arap_iterations,
        cg_tolerance=cg_tolerance, cg_max_iterations=cg_max_iterations)
    if not np.array_equal(desired[0], np.zeros_like(desired[0])):
        raise ValueError("Common proposed update changed frame-zero anchor")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "common-target.npz",
                        source_vertices=vertices, faces=faces, timesteps=times,
                        desired_update=desired)
    parameters = {
        "beta": beta, "fixed_alpha": fixed_alpha,
        "backtracking_factor": backtracking_factor,
        "backtracking_max_steps": backtracking_max_steps,
        "degeneracy_epsilon": degeneracy_epsilon,
        "root_tolerance": root_tolerance,
        "absolute_margin": absolute_margin,
        "relative_margin": relative_margin,
        "arap_weight": arap_weight, "temporal_weight": temporal_weight,
        "arap_iterations": arap_iterations,
        "cg_tolerance": cg_tolerance,
        "cg_max_iterations": cg_max_iterations,
        "coordinate_bounds": [coordinate_lower, coordinate_upper],
        "bounds_policy": bounds_policy,
    }
    reports = []
    for role in ROLES:
        arm_started = time.monotonic(); directory = output / role; directory.mkdir()
        report = {
            "candidate_id": CANDIDATE_ID, "candidate_arm": role,
            "method_id": METHOD_IDS[role], "uid": uid, "seed": seed,
            "source_sequence_sha256": expected_sequence_sha256,
            "source_report_sha256": digest(report_path),
            "implementation_sha256": digest(Path(__file__)),
            "frame_completion": FRAME_COMPLETION,
            "parameters": parameters, "common_repair": repair,
            "generated_unexecuted": False, "native_qualified": False,
            "scientific_admission": False, "local_method_verified": False,
            "scientific_verdict": "not_computed",
        }
        try:
            repaired, certificate, diagnostics = admit_role(
                vertices, desired, faces, role, beta=beta,
                fixed_alpha=fixed_alpha, backtracking_factor=backtracking_factor,
                backtracking_max_steps=backtracking_max_steps,
                degeneracy_epsilon=degeneracy_epsilon,
                root_tolerance=root_tolerance, absolute_margin=absolute_margin,
                relative_margin=relative_margin)
            below = int(np.count_nonzero(repaired < coordinate_lower))
            above = int(np.count_nonzero(repaired > coordinate_upper))
            if bounds_policy == "reject" and (below or above):
                raise ValueError("C12 output violates frozen coordinate bounds")
            np.savez_compressed(directory / "sequence.npz",
                                **{**arrays, "vertices": repaired})
            np.savez_compressed(directory / "certificate.npz", **certificate)
            report.update(status="completed", diagnostics=diagnostics,
                          bounds={"below": below, "above": above,
                                  "policy": bounds_policy,
                                  "lower": coordinate_lower,
                                  "upper": coordinate_upper},
                          sha256={"sequence.npz": digest(directory / "sequence.npz"),
                                  "certificate.npz": digest(directory / "certificate.npz")})
        except Exception as error:
            for partial in (directory / "sequence.npz", directory / "certificate.npz"):
                partial.unlink(missing_ok=True)
            report.update(status="error", exception_type=type(error).__name__,
                          error=str(error)[:4096],
                          traceback="".join(traceback.format_exception(error))[-16384:])
        report["elapsed_seconds"] = time.monotonic() - arm_started
        write_json(directory / "report.json", report); reports.append(report)
    status = "completed" if all(row["status"] == "completed" for row in reports) else "incomplete"
    record = {
        "kind": "c12-area-admission-candidate", "version": 1,
        "candidate_id": CANDIDATE_ID, "status": status,
        "uid": uid, "seed": seed, "roles": list(ROLES), "arms": reports,
        "source_sequence_sha256": expected_sequence_sha256,
        "source_report_sha256": digest(report_path),
        "source_refs": {"sequence": {"path": source_sequence_ref,
                                      "sha256": expected_sequence_sha256},
                        "report": {"path": source_report_ref,
                                   "sha256": digest(report_path)}},
        "common_target": {"path": "common-target.npz",
                          "sha256": digest(output / "common-target.npz")},
        "implementation_sha256": digest(Path(__file__)),
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False,
        "source_delivery_status": "generated_unexecuted_at_authoring",
        "elapsed_seconds": time.monotonic() - started,
        "timing_scope": "CPU common ARAP repair, scalar admission and artifact I/O only",
    }
    write_json(output / "candidate.json", record)
    completed = [row["candidate_arm"] for row in reports if row["status"] == "completed"]
    write_json(output / "manifest.json", {
        "expected_roles": list(ROLES),
        "cases": [{"case_id": uid + "-" + role, "uid": uid,
                   "case_dir": role, "arm_role": role} for role in completed],
        "common_target": record["common_target"],
        "scope": "C12 generated_unexecuted three-arm artifact; no B0/B*/scoring/admission",
    })
    _write_archive(output, reports, max_artifact_bytes)
    return record


def validate_candidate_artifact(root: Path, candidate_path: Path) -> dict:
    root, candidate_path = Path(root).resolve(), Path(candidate_path).resolve()
    candidate_path.relative_to(root)
    record = json.loads(candidate_path.read_text())
    if (record.get("kind") != "c12-area-admission-candidate"
            or record.get("version") != 1 or record.get("candidate_id") != CANDIDATE_ID
            or record.get("roles") != list(ROLES)
            or len(record.get("arms", [])) != len(ROLES)):
        raise ValueError("Current C12 candidate record required")
    implementation_sha256 = digest(Path(__file__))
    if (record.get("implementation_sha256") != implementation_sha256
            or record.get("native_qualified") is not False
            or record.get("scientific_admission") is not False
            or record.get("local_method_verified") is not False
            or record.get("source_delivery_status") != "generated_unexecuted_at_authoring"):
        raise ValueError("C12 candidate identity or evidence flags changed")
    directory = candidate_path.parent
    source_ref, report_ref = (record.get("source_refs", {}).get("sequence"),
                              record.get("source_refs", {}).get("report"))
    for ref in (source_ref, report_ref):
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or Path(ref["path"]).is_absolute() or ".." in Path(ref["path"]).parts):
            raise ValueError("Exact project-relative C12 source ref required")
    source_path, source_report_path = (root / source_ref["path"], root / report_ref["path"])
    if (source_path.is_symlink() or source_report_path.is_symlink()
            or source_report_path.resolve() != source_path.resolve().with_name("report.json")
            or digest(source_path) != source_ref["sha256"]
            or digest(source_report_path) != report_ref["sha256"]):
        raise ValueError("C12 source pair changed")
    arrays, vertices, faces, times, _, seed = _load_source(
        source_path.parent, record["uid"], source_ref["sha256"])
    if (record.get("seed") != seed or record.get("source_sequence_sha256") != source_ref["sha256"]
            or record.get("source_report_sha256") != report_ref["sha256"]):
        raise ValueError("C12 source identity aliases changed")
    common_ref = {"path": "common-target.npz", "sha256": digest(directory / "common-target.npz")}
    if record.get("common_target") != common_ref:
        raise ValueError("C12 common target changed")
    completed = []
    first_parameters = record["arms"][0].get("parameters")
    if not isinstance(first_parameters, dict):
        raise ValueError("C12 frozen parameters required")
    desired, repair = compute_common_geometry_repair(
        vertices, faces, times,
        arap_weight=first_parameters["arap_weight"],
        temporal_weight=first_parameters["temporal_weight"],
        iterations=first_parameters["arap_iterations"],
        cg_tolerance=first_parameters["cg_tolerance"],
        cg_max_iterations=first_parameters["cg_max_iterations"])
    with np.load(directory / "common-target.npz", allow_pickle=False) as saved:
        common = {name: saved[name].copy() for name in saved.files}
    if (set(common) != {"source_vertices", "faces", "timesteps", "desired_update"}
            or not np.array_equal(common["source_vertices"], vertices)
            or not np.array_equal(common["faces"], faces)
            or not np.array_equal(common["timesteps"], times)
            or not np.array_equal(common["desired_update"], desired)):
        raise ValueError("C12 common target differs from reconstruction")
    for role, saved_report in zip(ROLES, record["arms"]):
        report_path = directory / role / "report.json"
        report = json.loads(report_path.read_text())
        if report != saved_report or report.get("candidate_arm") != role:
            raise ValueError("C12 terminal role report changed")
        parameters = report.get("parameters")
        if parameters != first_parameters or report.get("common_repair") != repair:
            raise ValueError("C12 roles do not share one frozen update")
        if (report.get("candidate_id") != CANDIDATE_ID
                or report.get("method_id") != METHOD_IDS[role]
                or report.get("uid") != record["uid"] or report.get("seed") != seed
                or report.get("implementation_sha256") != implementation_sha256
                or report.get("source_sequence_sha256") != source_ref["sha256"]
                or report.get("source_report_sha256") != report_ref["sha256"]
                or report.get("generated_unexecuted") is not False
                or report.get("native_qualified") is not False
                or report.get("scientific_admission") is not False
                or report.get("local_method_verified") is not False
                or report.get("scientific_verdict") != "not_computed"):
            raise ValueError("C12 report method/source/evidence identity changed")
        coordinate_bounds = parameters.get("coordinate_bounds")
        if (not isinstance(coordinate_bounds, list) or len(coordinate_bounds) != 2
                or not all(math.isfinite(value) for value in coordinate_bounds)
                or coordinate_bounds[0] >= coordinate_bounds[1]
                or parameters.get("bounds_policy") not in ("preserve_and_report", "reject")):
            raise ValueError("C12 coordinate-bound policy changed")
        replay_error = None
        try:
            expected, certificate, diagnostics = admit_role(
                vertices, desired, faces, role,
                beta=parameters["beta"], fixed_alpha=parameters["fixed_alpha"],
                backtracking_factor=parameters["backtracking_factor"],
                backtracking_max_steps=parameters["backtracking_max_steps"],
                degeneracy_epsilon=parameters["degeneracy_epsilon"],
                root_tolerance=parameters["root_tolerance"],
                absolute_margin=parameters["absolute_margin"],
                relative_margin=parameters["relative_margin"])
            expected_bounds = {
                "below": int(np.count_nonzero(expected < coordinate_bounds[0])),
                "above": int(np.count_nonzero(expected > coordinate_bounds[1])),
                "policy": parameters["bounds_policy"],
                "lower": coordinate_bounds[0], "upper": coordinate_bounds[1],
            }
            if (parameters["bounds_policy"] == "reject"
                    and (expected_bounds["below"] or expected_bounds["above"])):
                raise ValueError("C12 output violates frozen coordinate bounds")
        except Exception as error:
            replay_error = error
        if report["status"] == "completed":
            if replay_error is not None:
                raise ValueError("C12 completed arm fails deterministic replay") from replay_error
            sequence_path, certificate_path = directory / role / "sequence.npz", directory / role / "certificate.npz"
            if report.get("sha256") != {"sequence.npz": digest(sequence_path),
                                        "certificate.npz": digest(certificate_path)}:
                raise ValueError("C12 completed arm hashes changed")
            with np.load(sequence_path, allow_pickle=False) as saved:
                arm = {name: saved[name].copy() for name in saved.files}
            validate_native_arrays(arm)
            if set(arm) != set(arrays) or not np.array_equal(arm["vertices"], expected):
                raise ValueError("C12 sequence differs from reconstruction")
            for name in set(arrays) - {"vertices"}:
                if not np.array_equal(arm[name], arrays[name]):
                    raise ValueError("C12 changed native identity: " + name)
            with np.load(certificate_path, allow_pickle=False) as saved:
                observed = {name: saved[name].copy() for name in saved.files}
            if set(observed) != set(certificate) or any(
                    not np.array_equal(observed[name], certificate[name]) for name in observed):
                raise ValueError("C12 certificate differs from reconstruction")
            if report.get("diagnostics") != diagnostics:
                raise ValueError("C12 diagnostics changed")
            if report.get("bounds") != expected_bounds:
                raise ValueError("C12 export-bound evidence changed")
            completed.append(role)
        elif report["status"] == "error":
            if (replay_error is None
                    or report.get("exception_type") != type(replay_error).__name__
                    or report.get("error") != str(replay_error)[:4096]
                    or (directory / role / "sequence.npz").exists()
                    or (directory / role / "certificate.npz").exists()
                    or not isinstance(report.get("exception_type"), str)
                    or not report.get("exception_type")
                    or not isinstance(report.get("error"), str) or not report.get("error")):
                raise ValueError("C12 failed role lacks bounded terminal evidence")
        else:
            raise ValueError("C12 role must terminate completed or error")
    expected_status = "completed" if len(completed) == len(ROLES) else "incomplete"
    if record.get("status") != expected_status:
        raise ValueError("C12 aggregate status differs from roles")
    expected_manifest = {
        "expected_roles": list(ROLES),
        "cases": [{"case_id": record["uid"] + "-" + role,
                   "uid": record["uid"], "case_dir": role, "arm_role": role}
                  for role in completed],
        "common_target": common_ref,
        "scope": "C12 generated_unexecuted three-arm artifact; no B0/B*/scoring/admission",
    }
    if json.loads((directory / "manifest.json").read_text()) != expected_manifest:
        raise ValueError("C12 manifest differs from terminal roles")
    archive = _validate_archive(directory, record["arms"])
    archive_refs = {
        name: {"path": value["path"].resolve().relative_to(root).as_posix(),
               "sha256": value["sha256"]}
        for name, value in archive.items()
    }
    return {**record, "completed_roles": completed,
            "artifact_archive": archive_refs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--expected-sequence-sha256", required=True)
    parser.add_argument("--source-sequence-ref", required=True)
    parser.add_argument("--source-report-ref", required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--fixed-alpha", type=float, required=True)
    parser.add_argument("--backtracking-factor", type=float, required=True)
    parser.add_argument("--backtracking-max-steps", type=int, required=True)
    parser.add_argument("--degeneracy-epsilon", type=float, required=True)
    parser.add_argument("--root-tolerance", type=float, required=True)
    parser.add_argument("--absolute-margin", type=float, required=True)
    parser.add_argument("--relative-margin", type=float, required=True)
    parser.add_argument("--arap-weight", type=float, required=True)
    parser.add_argument("--temporal-weight", type=float, required=True)
    parser.add_argument("--arap-iterations", type=int, required=True)
    parser.add_argument("--cg-tolerance", type=float, required=True)
    parser.add_argument("--cg-max-iterations", type=int, required=True)
    parser.add_argument("--coordinate-bounds", nargs=2, type=float, required=True)
    parser.add_argument("--bounds-policy", choices=("preserve_and_report", "reject"), required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    args = parser.parse_args()
    result = export_area_admission_candidate(
        args.source_sequence.parent, args.output, uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256,
        source_sequence_ref=args.source_sequence_ref,
        source_report_ref=args.source_report_ref, beta=args.beta,
        fixed_alpha=args.fixed_alpha,
        backtracking_factor=args.backtracking_factor,
        backtracking_max_steps=args.backtracking_max_steps,
        degeneracy_epsilon=args.degeneracy_epsilon,
        root_tolerance=args.root_tolerance,
        absolute_margin=args.absolute_margin,
        relative_margin=args.relative_margin,
        arap_weight=args.arap_weight, temporal_weight=args.temporal_weight,
        arap_iterations=args.arap_iterations,
        cg_tolerance=args.cg_tolerance,
        cg_max_iterations=args.cg_max_iterations,
        coordinate_lower=args.coordinate_bounds[0],
        coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy,
        max_artifact_bytes=args.max_artifact_bytes)
    print(json.dumps({"status": result["status"], "candidate_id": CANDIDATE_ID,
                      "execution_started": True, "native_qualified": False,
                      "scientific_admission": False}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
