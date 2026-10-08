"""Receipt-bound C05 sparse spatial-mode artifact materializer.

The input is the retained same-anchor model-output bank produced by
``c05_mode_bank`` plus an independent exact B0 parity receipt.  This module
selects a prospective sparse material landmark set, clusters complete
trajectories, applies the C05 reductions, and lifts every sparse displacement
through one identical inverse-distance operator.  It accepts no labels,
evaluator state, attention tensor, or caller-supplied coordinate bank.

The source and its tests are ``generated_unexecuted`` until Local acceptance.
Scientific admission and native scoring are separate boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Mapping, Sequence

import numpy as np

from .c05_mode_bank import validate_retained_mode_bank
from .spatial_mode_candidate import (
    METHOD_IDS,
    cluster_empirical_trajectories,
    independent_top1,
    localized_mean,
    normalized_mode_separations,
    solve_joint_labels_icm,
    temperature_mean,
    union_surface_projected_mean,
    validate_native_arrays as _validate_native_arrays,
)


CANDIDATE_ID = "4d-math-20261006-c05"
ROLE_ORDER = tuple(METHOD_IDS)
MAX_LANDMARKS = 256
SCOPE = {
    "native_scientific_qualification": False,
    "scientific_effect_qualification": False,
    "local_method_verified": False,
    "dispatch_ready": False,
}
PARITY_REF_KEYS = {
    "mode_bank_result_ref": "mode_bank_result",
    "mode_bank_manifest_ref": "mode_bank_manifest",
    "mode_bank_array_ref": "mode_bank",
    "branch_zero_ref": "branch0_sequence",
    "retained_b0_sequence_ref": "b0_sequence",
    "retained_b0_report_ref": "b0_report",
}


class C05ArtifactError(RuntimeError):
    """A retained input, construction, or output invariant failed."""


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_canonical_bytes(value))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise C05ArtifactError("JSON object required: " + str(path))
    return value


def _physical(path: Path) -> Path:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise C05ArtifactError("regular physical file required: " + str(path))
    return path.resolve()


def _ref(root: Path, path: Path) -> dict[str, Any]:
    root, path = Path(root).resolve(), _physical(path)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise C05ArtifactError("artifact input must be inside project root") from error
    return {"path": relative, "sha256": _digest(path), "bytes": path.stat().st_size}


def _resolve_ref(root: Path, ref: Mapping[str, Any]) -> Path:
    if not isinstance(ref, Mapping) or set(ref) != {"path", "sha256", "bytes"}:
        raise C05ArtifactError("exact path/hash/size reference required")
    raw = ref["path"]
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise C05ArtifactError("canonical relative POSIX reference required")
    relative = PurePosixPath(raw)
    if (relative.is_absolute() or relative.as_posix() != raw
            or any(part in ("", ".", "..") for part in relative.parts)):
        raise C05ArtifactError("canonical relative POSIX reference required")
    path = _physical(Path(root).resolve() / relative)
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as error:
        raise C05ArtifactError("reference escaped project root") from error
    if (not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64
            or isinstance(ref["bytes"], bool) or not isinstance(ref["bytes"], int)
            or ref["bytes"] < 0 or path.stat().st_size != ref["bytes"]
            or _digest(path) != ref["sha256"]):
        raise C05ArtifactError("referenced bytes changed: " + raw)
    return path


def _producer_namespace_root(root: Path, mode_root: Path,
                             manifest: Mapping[str, Any]) -> Path:
    """Locate the retained producer workspace that owns manifest-relative refs."""
    root, mode_root = Path(root).resolve(), Path(mode_root).resolve()
    closures = manifest.get("producer_input_refs")
    if not isinstance(closures, list):
        raise C05ArtifactError("retained producer input closure required")
    requests = [ref for ref in closures
                if isinstance(ref, Mapping)
                and Path(str(ref.get("path", ""))).name == "request.json"]
    if len(requests) != 1:
        raise C05ArtifactError("one retained C05 producer request required")
    request_ref = requests[0]
    raw = request_ref.get("path")
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise C05ArtifactError("canonical retained producer request path required")
    relative = PurePosixPath(raw)
    if (relative.is_absolute() or relative.as_posix() != raw
            or any(part in ("", ".", "..") for part in relative.parts)
            or not isinstance(request_ref.get("sha256"), str)):
        raise C05ArtifactError("canonical retained producer request ref required")
    matches = []
    ancestor = mode_root
    while True:
        try:
            ancestor.relative_to(root)
        except ValueError:
            break
        request_path = ancestor / relative
        if (request_path.is_file() and not request_path.is_symlink()
                and _digest(request_path) == request_ref["sha256"]):
            request = _read_json(request_path)
            output_value = request.get("output_relative")
            if isinstance(output_value, str):
                output_relative = PurePosixPath(output_value)
                if (not output_relative.is_absolute()
                        and output_relative.as_posix() == output_value
                        and all(part not in ("", ".", "..")
                                for part in output_relative.parts)
                        and (ancestor / output_relative).resolve() == mode_root):
                    matches.append(ancestor)
        if ancestor == root:
            break
        ancestor = ancestor.parent
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise C05ArtifactError("unique retained C05 producer workspace required")
    return matches[0]


def _positive(name: str, value: Any) -> float:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    path = _physical(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise C05ArtifactError(
                f"exact fields required in {path.name}: {sorted(required)}")
        return {name: archive[name].copy() for name in archive.files}


_SEQUENCE_FIELDS = {"vertices", "faces", "frame_indices", "timesteps", "query_vertex_ids"}
_BANK_FIELDS = _SEQUENCE_FIELDS | {
    "equal_mass_scores", "inner_stage1_seeds", "outer_generation_seed"
}


def validate_native_arrays(mode_bank: Mapping[str, np.ndarray],
                           b0: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Validate complete same-anchor bank and exact retained B0 parity."""
    if set(mode_bank) != _BANK_FIELDS or set(b0) != _SEQUENCE_FIELDS:
        raise C05ArtifactError("complete mode-bank and B0 sequence fields required")
    vertices = np.asarray(mode_bank["vertices"])
    if vertices.ndim != 4 or vertices.shape[1] != 16:
        raise C05ArtifactError("C05 requires sampled vertices [K,16,V,3]")
    if vertices.dtype != np.float32 or np.asarray(b0["vertices"]).dtype != np.float32:
        raise C05ArtifactError("C05 current-release vertices must retain float32 identity")
    if not np.array_equal(mode_bank["frame_indices"], np.arange(16, dtype=np.int64)):
        raise C05ArtifactError("C05 current-release frame order must be exactly 0..15")
    native = _validate_native_arrays(
        vertices,
        anchor_vertices=vertices[0, 0],
        faces=mode_bank["faces"],
        target_frame_indices=mode_bank["frame_indices"],
        target_timesteps=mode_bank["timesteps"],
        query_vertex_ids=mode_bank["query_vertex_ids"],
        draw_seeds=tuple(int(value) for value in mode_bank["inner_stage1_seeds"]),
    )
    expected_mass = np.full(vertices.shape[0], 1.0 / vertices.shape[0], dtype=np.float64)
    if not np.array_equal(mode_bank["equal_mass_scores"], expected_mass):
        raise C05ArtifactError("mode-bank must retain exact empirical equal masses")
    for name in _SEQUENCE_FIELDS - {"vertices"}:
        if (np.asarray(mode_bank[name]).dtype != np.asarray(b0[name]).dtype
                or not np.array_equal(mode_bank[name], b0[name])):
            raise C05ArtifactError("B0 identity differs from mode bank: " + name)
    if (vertices[0].dtype != np.asarray(b0["vertices"]).dtype
            or not np.array_equal(vertices[0], b0["vertices"])):
        raise C05ArtifactError("branch zero is not exact retained B0")
    if not np.array_equal(vertices[:, 0], np.broadcast_to(
            b0["vertices"][0], vertices[:, 0].shape)):
        raise C05ArtifactError("mode branches do not share exact frame-zero anchor")
    if not np.issubdtype(b0["vertices"].dtype, np.floating):
        raise C05ArtifactError("native B0 vertices must be floating point")
    return native


def _validate_branch_closure(mode_root: Path, manifest: Mapping[str, Any],
                             mode_arrays: Mapping[str, np.ndarray]) -> None:
    """Prove that the aggregate bank is exactly the ordered retained branches."""

    vertices = mode_arrays["vertices"]
    seeds = mode_arrays["inner_stage1_seeds"].astype(np.int64)
    branches = manifest.get("branches")
    if (manifest.get("candidate_id") != "C05"
            or manifest.get("branch_count") != len(vertices)
            or manifest.get("frames") != 16
            or manifest.get("vertices_per_frame") != vertices.shape[2]
            or manifest.get("outer_generation_seed") !=
               int(mode_arrays["outer_generation_seed"])
            or manifest.get("inner_stage1_seeds") != seeds.tolist()
            or manifest.get("mode_semantics") !=
               "equal-mass empirical model-output modes"
            or manifest.get("posterior_correspondence_claim") is not False
            or manifest.get("calibrated_probability_claim") is not False
            or not isinstance(branches, list) or len(branches) != len(vertices)):
        raise C05ArtifactError("mode-bank semantic/identity receipt differs")
    for index, row in enumerate(branches):
        expected_role = "b0_algorithmic_path_unverified" if index == 0 else "empirical_mode"
        if (not isinstance(row, Mapping) or row.get("branch_index") != index
                or row.get("role") != expected_role
                or row.get("stage1_seed") != int(seeds[index])):
            raise C05ArtifactError("canonical retained mode branch inventory required")
        ref = row.get("sequence_ref")
        if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
            raise C05ArtifactError("retained mode branch reference required")
        branch_path = _physical(mode_root / ref["path"])
        branch = _load_npz(branch_path, _SEQUENCE_FIELDS)
        expected = {"vertices": vertices[index], **{
            name: mode_arrays[name] for name in _SEQUENCE_FIELDS - {"vertices"}}}
        for name in _SEQUENCE_FIELDS:
            if (branch[name].dtype != expected[name].dtype
                    or not np.array_equal(branch[name], expected[name])):
                raise C05ArtifactError(
                    f"aggregate mode-bank branch differs from retained branch {index}: {name}")


def _validate_b0_report(path: Path, sequence_ref: Mapping[str, Any], *,
                        uid: str, seed: int) -> dict[str, Any]:
    report = _read_json(path)
    if (report.get("status") != "completed" or report.get("uid") != uid
            or report.get("seed") != seed
            or report.get("sha256", {}).get("sequence.npz") != sequence_ref["sha256"]):
        raise C05ArtifactError("completed matching retained B0 report required")
    return report


def _validate_parity(root: Path, path: Path, *, refs: Mapping[str, Any],
                     uid: str, seed: int) -> dict[str, Any]:
    receipt = _read_json(path)
    if (receipt.get("kind") != "c05-b0-parity-receipt"
            or receipt.get("version") != 1
            or receipt.get("candidate_id") != CANDIDATE_ID
            or receipt.get("uid") != uid or receipt.get("outer_seed") != seed
            or receipt.get("matches") is not True
            or receipt.get("comparison") !=
               "exact array equality for all native sequence fields"
            or receipt.get("candidate_methods_tested") is not False
            or receipt.get("scientific_effect_qualification") is not False
            or receipt.get("native_qualified") is not False):
        raise C05ArtifactError("exact receipt-bound C05/B0 parity evidence required")
    expected_keys = {
        "kind", "version", "candidate_id", "uid", "outer_seed", "matches",
        "comparison", *PARITY_REF_KEYS, "candidate_methods_tested",
        "scientific_effect_qualification", "native_qualified",
    }
    if set(receipt) != expected_keys:
        raise C05ArtifactError("parity receipt has incomplete or extra fields")
    for receipt_name, internal_name in PARITY_REF_KEYS.items():
        expected = refs[internal_name]
        parity_ref = receipt[receipt_name]
        if parity_ref != {"path": expected["path"], "sha256": expected["sha256"]}:
            raise C05ArtifactError("parity input reference differs: " + receipt_name)
        _resolve_ref(root, expected)
    return receipt


def _farthest_landmarks(anchor: np.ndarray, query_ids: np.ndarray,
                        count: int) -> np.ndarray:
    if count < 3 or count > min(len(anchor), MAX_LANDMARKS):
        raise ValueError(f"landmark_count must be in [3,min(V,{MAX_LANDMARKS})]")
    first = int(np.argmin(query_ids))
    chosen = [first]
    minimum = np.sum((anchor - anchor[first]) ** 2, axis=1)
    while len(chosen) < count:
        maximum = float(np.max(minimum))
        if maximum <= np.finfo(np.float64).tiny:
            raise C05ArtifactError("anchor has too few distinct landmark coordinates")
        choices = np.flatnonzero(minimum == maximum)
        next_index = min((int(index) for index in choices),
                         key=lambda index: (int(query_ids[index]), index))
        chosen.append(next_index)
        minimum = np.minimum(minimum,
                             np.sum((anchor - anchor[next_index]) ** 2, axis=1))
    return np.asarray(chosen, dtype=np.int64)


def _minimum_spanning_tree(points: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
    count = len(points)
    reached = {0}
    edges: list[tuple[int, int]] = []
    while len(reached) < count:
        choices = []
        for left in sorted(reached):
            for right in range(count):
                if right in reached:
                    continue
                distance = float(np.sum((points[left] - points[right]) ** 2))
                choices.append((distance, int(query_ids[left]), int(query_ids[right]),
                                left, right))
        _, _, _, left, right = min(choices)
        edges.append((min(left, right), max(left, right)))
        reached.add(right)
    return np.asarray(sorted(edges), dtype=np.int64)


def _landmark_scales(anchor: np.ndarray, edges: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(anchor[edges[:, 0]] - anchor[edges[:, 1]], axis=1)
    if np.any(lengths <= np.finfo(np.float64).tiny):
        raise C05ArtifactError("landmark graph contains zero-length anchor edge")
    incident: list[list[float]] = [[] for _ in anchor]
    for (left, right), length in zip(edges, lengths):
        incident[int(left)].append(float(length))
        incident[int(right)].append(float(length))
    return np.asarray([np.median(values) for values in incident], dtype=np.float64)


def _proper_kabsch_hints(b0_landmarks: np.ndarray,
                         edges: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    anchor = b0_landmarks[0].astype(np.float64)
    source = anchor - anchor.mean(axis=0)
    rotations, singular_rows, residuals = [], [], []
    for frame, values in enumerate(b0_landmarks.astype(np.float64)):
        target = values - values.mean(axis=0)
        u, singular, vt = np.linalg.svd(source.T @ target)
        if singular[0] <= np.finfo(np.float64).tiny or singular[1] <= 1e-8 * singular[0]:
            raise C05ArtifactError(f"ambiguous rank-deficient B0 pose at frame {frame}")
        signs = np.ones(3)
        signs[-1] = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
        rotation = (u * signs) @ vt
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=1e-10)
                or not np.isclose(np.linalg.det(rotation), 1.0,
                                  atol=1e-10, rtol=1e-10)):
            raise C05ArtifactError("B0 Kabsch hint is not a proper rotation")
        # ``spatial_mode_candidate`` applies hints to column edge vectors;
        # transpose the row-convention Kabsch factor used for the fit above.
        rotations.append(rotation.T)
        singular_rows.append(singular)
        residuals.append(float(np.sqrt(np.mean((target - source @ rotation) ** 2))))
    rotations[0] = np.eye(3)
    hints = np.broadcast_to(np.stack(rotations)[None],
                            (len(edges), len(rotations), 3, 3)).copy()
    return hints, {
        "fit": "uniform sparse-landmark proper Kabsch from retained B0 only",
        "singular_values": np.stack(singular_rows).tolist(),
        "rms_residual": residuals,
    }


def _time_weights(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    intervals = np.diff(times)
    weights = np.empty_like(times)
    weights[0], weights[-1] = intervals[0] / 2.0, intervals[-1] / 2.0
    weights[1:-1] = (intervals[:-1] + intervals[1:]) / 2.0
    return weights / weights.sum()


def _inverse_distance_lift(anchor: np.ndarray, landmark_ids: np.ndarray,
                           scales: np.ndarray) -> np.ndarray:
    landmarks = anchor[landmark_ids]
    distance = np.linalg.norm(anchor[:, None] - landmarks[None], axis=2)
    weights = np.empty_like(distance, dtype=np.float64)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(anchor))))
    for vertex in range(len(anchor)):
        exact = np.flatnonzero(distance[vertex] <= tolerance)
        if len(exact):
            weights[vertex] = 0.0
            weights[vertex, int(exact[0])] = 1.0
        else:
            row = 1.0 / np.maximum(distance[vertex], tolerance) ** 2
            weights[vertex] = row / row.sum()
    if (not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
            or np.any(weights < 0.0)):
        raise AssertionError("invalid inverse-distance lift")
    return weights


def _lift_role(b0: np.ndarray, sparse_target: np.ndarray, landmark_ids: np.ndarray,
               lift: np.ndarray, scales: np.ndarray,
               clip_multiplier: float) -> tuple[np.ndarray, dict[str, Any]]:
    base_landmarks = b0[:, landmark_ids].astype(np.float64)
    sparse_displacement = sparse_target.astype(np.float64) - base_landmarks
    displacement = np.einsum("vl,tlc->tvc", lift, sparse_displacement)
    caps = clip_multiplier * (lift @ scales)
    norms = np.linalg.norm(displacement, axis=2)
    factors = np.minimum(1.0, np.divide(caps[None], norms,
        out=np.ones_like(norms), where=norms > 0.0))
    clipped = displacement * factors[:, :, None]
    output = (b0.astype(np.float64) + clipped).astype(b0.dtype)
    output[0] = b0[0]
    if not np.isfinite(output).all() or not np.array_equal(output[0], b0[0]):
        raise C05ArtifactError("lift changed frame zero or produced nonfinite output")
    return output, {
        "policy": "per-vertex L2 clip at multiplier times lifted landmark scale",
        "multiplier": clip_multiplier,
        "clipped_vertex_frames": int(np.count_nonzero(factors < 1.0)),
        "maximum_preclip_norm": float(np.max(norms)),
        "maximum_postclip_norm": float(np.max(np.linalg.norm(clipped, axis=2))),
    }


def _bounded_failure(error: Exception) -> dict[str, str]:
    """Return a stable, bounded terminal failure without retaining a traceback."""

    exception_type = type(error).__name__.strip() or "Exception"
    message = str(error).strip() or exception_type
    return {"exception_type": exception_type[:256], "error": message[:4096]}


def _attempt_sparse(role: str, function: Any,
                    states: dict[str, dict[str, Any]]) -> tuple[Any, ...] | None:
    """Run one independent sparse construction and retain a terminal outcome."""

    try:
        value = function()
    except Exception as error:  # terminal per-role method failure, never a fallback
        states[role] = {"status": "error", **_bounded_failure(error)}
        return None
    values = value if isinstance(value, tuple) else (value,)
    states[role] = {"status": "completed_unqualified", "sparse": values[0]}
    return values


def _construct_sparse_roles(
    bank: Any,
    *,
    gate_passed: bool,
    graph: np.ndarray,
    anchor_landmarks: np.ndarray,
    b0_landmarks: np.ndarray,
    timesteps: np.ndarray,
    sampled_surfaces: np.ndarray,
    faces: np.ndarray,
    localized_radius: float,
    temperature: float,
    unary_weight: float,
    spatial_weight: float,
    max_sweeps: int,
    face_chunk_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Construct five roles independently and retain method-local failures.

    Proper-Kabsch and ICM are dependencies only of ``joint_spatial_labels``;
    union-surface projection is a dependency only of
    ``surface_projected_mean``.  This separation preserves the fixed five-role
    failure denominator without converting one method's exception into a
    wholesale artifact failure.
    """

    states: dict[str, dict[str, Any]] = {}
    evidence: dict[str, Any] = {
        "rotation_hints": None,
        "time_weights": None,
        "localized_supports": None,
        "projection": None,
        "independent_labels": None,
        "joint": None,
    }
    if not gate_passed:
        failure = _bounded_failure(C05ArtifactError(
            "Natural Gate rejected: candidate role is not scoreable"))
        for role in ROLE_ORDER:
            states[role] = {"status": "error", **failure}
        return states, evidence

    localized = _attempt_sparse(
        "localized_mean", lambda: localized_mean(bank, localized_radius), states)
    if localized is not None:
        evidence["localized_supports"] = [list(row) for row in localized[1]]

    _attempt_sparse(
        "temperature_matched_mean", lambda: temperature_mean(bank, temperature), states)

    projected = _attempt_sparse(
        "surface_projected_mean",
        lambda: union_surface_projected_mean(
            bank, localized_radius=localized_radius,
            sampled_surfaces=sampled_surfaces, faces=faces,
            face_chunk_size=face_chunk_size),
        states)
    if projected is not None:
        evidence["projection"] = {
            name: value.tolist() for name, value in projected[1].items()}

    top1 = _attempt_sparse("independent_top1", lambda: independent_top1(bank), states)
    if top1 is not None:
        evidence["independent_labels"] = top1[1].tolist()

    def joint_construction() -> tuple[np.ndarray, Any]:
        rotations, rotation_report = _proper_kabsch_hints(b0_landmarks, graph)
        weights = _time_weights(timesteps)
        evidence["rotation_hints"] = dict(rotation_report)
        evidence["time_weights"] = weights.tolist()
        joint = solve_joint_labels_icm(
            bank, edges=graph, anchor_vertices=anchor_landmarks,
            rotations=rotations, time_weights=weights,
            unary_weight=unary_weight, spatial_weight=spatial_weight,
            max_sweeps=max_sweeps)
        evidence["joint"] = {
            "labels": list(joint.labels), "energy": joint.energy,
            "one_flip_residual": joint.one_flip_residual,
            "selected_start": joint.selected_start,
            "runs": [{
                "start_name": run.start_name,
                "initial_labels": list(run.initial_labels),
                "labels": list(run.labels),
                "objective_trace": list(run.objective_trace),
                "sweeps": run.sweeps, "converged": run.converged,
                "one_flip_residual": run.one_flip_residual,
                "energy": run.energy,
            } for run in joint.runs],
        }
        return joint.trajectories, joint

    _attempt_sparse("joint_spatial_labels", joint_construction, states)
    return states, evidence


def _solver_record(*, evidence: Mapping[str, Any],
                   states: Mapping[str, Mapping[str, Any]],
                   lift: np.ndarray | None) -> dict[str, Any]:
    """Canonical recomputable record for successes and bounded failures."""

    return {
        "kind": "c05-spatial-mode-solver-certificate", "version": 1,
        "candidate_id": CANDIDATE_ID,
        **{name: evidence[name] for name in (
            "rotation_hints", "time_weights", "localized_supports",
            "projection", "independent_labels", "joint")},
        "role_statuses": {role: states[role]["status"] for role in ROLE_ORDER},
        "role_errors": {
            role: ({name: states[role][name] for name in ("exception_type", "error")}
                   if states[role]["status"] == "error" else None)
            for role in ROLE_ORDER
        },
        "lift_sha256": (hashlib.sha256(
            np.ascontiguousarray(lift).tobytes()).hexdigest()
            if lift is not None else None),
    }


def _tar_bytes(output: Path, members: Sequence[str]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for relative in sorted(members):
            path = _physical(output / relative)
            data = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size = len(data); info.mode = 0o644; info.mtime = 0
            info.uid = info.gid = 0; info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def _output_ref(output: Path, path: Path) -> dict[str, Any]:
    path = _physical(path)
    return {"path": path.relative_to(output.resolve()).as_posix(),
            "sha256": _digest(path), "bytes": path.stat().st_size}


def _sibling_ref(path: Path) -> dict[str, Any]:
    """Reference one file from a JSON document in the same directory."""
    path = _physical(path)
    return {"path": path.name, "sha256": _digest(path), "bytes": path.stat().st_size}


def materialize_candidate(
    root: str | Path,
    mode_bank_root: str | Path,
    parity_receipt: str | Path,
    b0_sequence: str | Path,
    b0_report: str | Path,
    output_dir: str | Path,
    *,
    landmark_count: int,
    cluster_radius: float,
    natural_gate_min_fraction: float,
    localized_radius: float,
    temperature: float,
    unary_weight: float,
    spatial_weight: float,
    max_sweeps: int,
    displacement_clip_multiplier: float,
    max_artifact_bytes: int,
    face_chunk_size: int = 4096,
    producer_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize all five C05 roles or retain a Natural-Gate rejection."""
    root = Path(root).resolve()
    mode_bank_root = Path(mode_bank_root).resolve()
    parity_receipt = Path(parity_receipt).resolve()
    b0_sequence = Path(b0_sequence).resolve()
    b0_report = Path(b0_report).resolve()
    output = Path(output_dir).resolve()
    producer_root = root if producer_root is None else Path(producer_root).resolve()
    for path in (mode_bank_root, parity_receipt, b0_sequence, b0_report, output.parent):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise C05ArtifactError("all inputs/outputs must stay inside project root") from error
    try:
        mode_bank_root.relative_to(producer_root)
        parity_receipt.relative_to(producer_root)
        b0_sequence.relative_to(producer_root)
        b0_report.relative_to(producer_root)
        producer_root.relative_to(root)
    except ValueError as error:
        raise C05ArtifactError("C05 producer inputs must share one retained workspace") from error
    if output.exists():
        raise FileExistsError("C05 artifact output is append-only and must be new")
    landmark_count = _positive_integer("landmark_count", landmark_count)
    max_sweeps = _positive_integer("max_sweeps", max_sweeps)
    face_chunk_size = _positive_integer("face_chunk_size", face_chunk_size)
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    cluster_radius = _positive("cluster_radius", cluster_radius)
    localized_radius = _positive("localized_radius", localized_radius)
    temperature = _positive("temperature", temperature)
    unary_weight = _positive("unary_weight", unary_weight)
    spatial_weight = _positive("spatial_weight", spatial_weight)
    displacement_clip_multiplier = _positive(
        "displacement_clip_multiplier", displacement_clip_multiplier)
    if (isinstance(natural_gate_min_fraction, bool)
            or not isinstance(natural_gate_min_fraction, (int, float))
            or not math.isfinite(float(natural_gate_min_fraction))
            or not 0.0 < float(natural_gate_min_fraction) <= 1.0):
        raise ValueError("natural_gate_min_fraction must be in (0,1]")
    natural_gate_min_fraction = float(natural_gate_min_fraction)

    if any(item.is_symlink() for item in mode_bank_root.rglob("*")):
        raise C05ArtifactError("symlinks are forbidden in retained C05 mode bank")
    manifest = validate_retained_mode_bank(mode_bank_root, producer_root)
    bank_path = mode_bank_root / "mode-bank.npz"
    branch0_path = mode_bank_root / "sequences/branch-0000.npz"
    mode_arrays = _load_npz(bank_path, _BANK_FIELDS)
    _validate_branch_closure(mode_bank_root, manifest, mode_arrays)
    b0_arrays = _load_npz(b0_sequence, _SEQUENCE_FIELDS)
    native = validate_native_arrays(mode_arrays, b0_arrays)
    uid = manifest["provenance"]["generation_uid"]
    outer_seed = int(mode_arrays["outer_generation_seed"])
    producer_b0_ref = _ref(producer_root, b0_sequence)
    _validate_b0_report(b0_report, producer_b0_ref, uid=uid, seed=outer_seed)
    receipt_refs = {
        "mode_bank_result": _ref(producer_root, mode_bank_root / "result.json"),
        "mode_bank_manifest": _ref(producer_root, mode_bank_root / "raw-manifest.json"),
        "mode_bank": _ref(producer_root, bank_path),
        "branch0_sequence": _ref(producer_root, branch0_path),
        "b0_sequence": producer_b0_ref,
        "b0_report": _ref(producer_root, b0_report),
    }
    _validate_parity(producer_root, parity_receipt, refs=receipt_refs,
                     uid=uid, seed=outer_seed)
    parity_refs = {
        "mode_bank_result": _ref(root, mode_bank_root / "result.json"),
        "mode_bank_manifest": _ref(root, mode_bank_root / "raw-manifest.json"),
        "mode_bank": _ref(root, bank_path),
        "branch0_sequence": _ref(root, branch0_path),
        "b0_sequence": _ref(root, b0_sequence),
        "b0_report": _ref(root, b0_report),
    }

    anchor = b0_arrays["vertices"][0].astype(np.float64)
    query_ids = b0_arrays["query_vertex_ids"].astype(np.int64)
    landmark_ids = _farthest_landmarks(anchor, query_ids, landmark_count)
    landmark_query_ids = query_ids[landmark_ids]
    graph = _minimum_spanning_tree(anchor[landmark_ids], landmark_query_ids)
    scales = _landmark_scales(anchor[landmark_ids], graph)
    draws = native["sampled_surfaces"][:, :, landmark_ids, :]
    mode_bank = cluster_empirical_trajectories(
        draws, draw_seeds=native["draw_seeds"], landmark_scales=scales,
        cluster_radius=cluster_radius)
    mode_counts = np.asarray([len(item.atoms) for item in mode_bank.modes], dtype=np.int64)
    separations = normalized_mode_separations(mode_bank)
    separated = (mode_counts >= 2) & (separations > cluster_radius)
    multimodal_fraction = float(np.mean(separated))
    required_count = int(math.ceil(natural_gate_min_fraction * landmark_count))
    gate_passed = int(np.count_nonzero(separated)) >= required_count
    parameters = {
        "landmark_count": landmark_count,
        "landmark_selection": "deterministic Euclidean farthest-point from lowest material ID",
        "graph": "deterministic anchor-space Euclidean minimum spanning tree",
        "cluster_radius": cluster_radius,
        "natural_gate_min_fraction": natural_gate_min_fraction,
        "localized_radius": localized_radius,
        "temperature": temperature,
        "unary_weight": unary_weight,
        "spatial_weight": spatial_weight,
        "max_sweeps": max_sweeps,
        "displacement_clip_multiplier": displacement_clip_multiplier,
        "face_chunk_size": face_chunk_size,
        "lift": "single inverse-square-distance anchor operator for every role",
    }
    mode_bank_implementation = Path(validate_retained_mode_bank.__code__.co_filename).resolve()
    math_implementation = Path(cluster_empirical_trajectories.__code__.co_filename).resolve()
    if manifest.get("producer_code_sha256") != _digest(mode_bank_implementation):
        raise C05ArtifactError("retained mode bank does not bind the current producer source")
    input_refs = {
        **parity_refs,
        "parity_receipt": _ref(root, parity_receipt),
        "artifact_implementation": _ref(root, Path(__file__)),
        "math_implementation": _ref(root, math_implementation),
        "mode_bank_implementation": _ref(root, mode_bank_implementation),
    }
    output.mkdir(parents=True)
    certificate = {
        "kind": "c05-spatial-mode-certificate", "version": 1,
        "candidate_id": CANDIDATE_ID, "status": "passed" if gate_passed else "rejected",
        "uid": uid, "outer_seed": outer_seed,
        "landmark_vertex_indices": landmark_ids.tolist(),
        "landmark_query_vertex_ids": landmark_query_ids.tolist(),
        "graph_edges": graph.tolist(), "landmark_scales": scales.tolist(),
        "mode_counts": mode_counts.tolist(),
        "max_medoid_separations": separations.tolist(),
        "separated_landmark_count": int(np.count_nonzero(separated)),
        "multimodal_landmark_count": int(np.count_nonzero(separated)),
        "landmark_count": landmark_count,
        "multimodal_fraction": multimodal_fraction,
        "required_multimodal_count": required_count,
        "natural_gate": (
            ">=2 empirical medoid atoms with normalized full-trajectory separation "
            "> cluster_radius on frozen minimum landmark fraction"),
        "parameters": parameters, "input_refs": input_refs, **SCOPE,
    }
    certificate_path = output / "certificate.json"
    _write_json(certificate_path, certificate)
    role_refs: dict[str, Any] = {}
    role_states, solver_evidence = _construct_sparse_roles(
        mode_bank, gate_passed=gate_passed, graph=graph,
        anchor_landmarks=anchor[landmark_ids],
        b0_landmarks=b0_arrays["vertices"][:, landmark_ids],
        timesteps=b0_arrays["timesteps"],
        sampled_surfaces=native["sampled_surfaces"], faces=native["faces"],
        localized_radius=localized_radius, temperature=temperature,
        unary_weight=unary_weight, spatial_weight=spatial_weight,
        max_sweeps=max_sweeps, face_chunk_size=face_chunk_size)
    lift: np.ndarray | None = None
    if gate_passed:
        try:
            lift = _inverse_distance_lift(anchor, landmark_ids, scales)
        except Exception as error:  # the shared lift is an explicit common dependency
            failure = _bounded_failure(error)
            for role in ROLE_ORDER:
                if role_states[role]["status"] == "completed_unqualified":
                    role_states[role] = {"status": "error", **failure}
        if lift is not None:
            for role in ROLE_ORDER:
                state = role_states[role]
                if state["status"] != "completed_unqualified":
                    continue
                try:
                    vertices, clipping = _lift_role(
                        b0_arrays["vertices"], state["sparse"], landmark_ids,
                        lift, scales, displacement_clip_multiplier)
                except Exception as error:  # one export cannot erase other role evidence
                    role_states[role] = {"status": "error", **_bounded_failure(error)}
                else:
                    state["vertices"] = vertices
                    state["clipping"] = clipping

    solver_certificate = _solver_record(
        evidence=solver_evidence, states=role_states, lift=lift)
    solver_path = output / "solver-certificate.json"
    _write_json(solver_path, solver_certificate)
    for role in ROLE_ORDER:
        directory = output / "roles" / role
        directory.mkdir(parents=True)
        state = role_states[role]
        sequence_path = directory / "sequence.npz"
        if state["status"] == "completed_unqualified":
            sequence = {name: value.copy() for name, value in b0_arrays.items()}
            sequence["vertices"] = state["vertices"]
            np.savez_compressed(sequence_path, **sequence)
            sequence_ref = _sibling_ref(sequence_path)
            output_sequence_ref = _output_ref(output, sequence_path)
            exact = True
        else:
            sequence_ref = None
            output_sequence_ref = None
            exact = False
        report = {
            "kind": "c05-spatial-mode-role-report", "version": 1,
            "candidate_id": CANDIDATE_ID, "status": state["status"],
            "uid": uid, "outer_seed": outer_seed,
            "candidate_arm": role, "method_id": METHOD_IDS[role],
            "implementation_ref": input_refs["math_implementation"],
            "sequence_ref": sequence_ref,
            "source_sequence_ref": b0_sequence_ref,
            "source_report_ref": input_refs["b0_report"],
            "parity_receipt_ref": input_refs["parity_receipt"],
            "certificate_ref": _ref(root, certificate_path),
            "solver_certificate_ref": _ref(root, solver_path),
            "landmark_query_vertex_ids": landmark_query_ids.tolist(),
            "lift_policy": parameters["lift"],
            "clipping": state.get("clipping"),
            "exception_type": state.get("exception_type"),
            "error": state.get("error"),
            "exact_frame_zero": exact, "exact_faces": exact,
            "exact_material_ids": exact, "exact_times": exact,
            **SCOPE,
        }
        report_path = directory / "report.json"
        _write_json(report_path, report)
        role_refs[role] = {
            "method_id": METHOD_IDS[role],
            "preparation_status": ("completed" if exact else "error"),
            "sequence_ref": output_sequence_ref,
            "report_ref": _output_ref(output, report_path),
        }

    roles_completed = [role for role in ROLE_ORDER
                       if role_states[role]["status"] == "completed_unqualified"]
    roles_failed = [role for role in ROLE_ORDER if role not in roles_completed]

    status = "completed_unqualified" if gate_passed else "incomplete_natural_gate"
    candidate = {
        "kind": "c05-spatial-mode-candidate", "version": 1,
        "candidate_id": CANDIDATE_ID, "status": status,
        "uid": uid, "outer_seed": outer_seed,
        "method_ids": dict(METHOD_IDS), "parameters": parameters,
        "input_refs": input_refs,
        "certificate_ref": _output_ref(output, certificate_path),
        "solver_certificate_ref": _output_ref(output, solver_path),
        "role_refs": role_refs,
        "role_denominator": len(ROLE_ORDER),
        "roles_completed": roles_completed,
        "roles_failed": roles_failed,
        "natural_gate_passed": gate_passed,
        "failure_policy": (
            "fixed five-role denominator; terminal errors retain reports and no sequence; "
            "no role fallback, substitution, or retry"),
        "source_delivery_status": "generated_unexecuted",
        **SCOPE,
    }
    candidate_path = output / "candidate.json"
    _write_json(candidate_path, candidate)
    members = ["candidate.json", "certificate.json", "solver-certificate.json"]
    for role in ROLE_ORDER:
        members.append(f"roles/{role}/report.json")
        if role in roles_completed:
            members.append(f"roles/{role}/sequence.npz")
    total = sum((output / name).stat().st_size for name in members)
    if total > max_artifact_bytes:
        raise C05ArtifactError("C05 artifact members exceed prospective byte ceiling")
    archive_bytes = _tar_bytes(output, members)
    if len(archive_bytes) > max_artifact_bytes:
        raise C05ArtifactError("C05 raw archive exceeds prospective byte ceiling")
    archive_path = output / "raw-evidence.tar"
    archive_path.write_bytes(archive_bytes)
    raw_manifest = {
        "kind": "c05-spatial-mode-artifact-manifest", "version": 1,
        "candidate_id": CANDIDATE_ID, "status": status,
        "members": [_output_ref(output, output / name) for name in sorted(members)],
        "archive_ref": _output_ref(output, archive_path),
        "max_artifact_bytes": max_artifact_bytes,
        "role_denominator": len(ROLE_ORDER),
        "roles_completed": roles_completed,
        "roles_failed": roles_failed,
        "input_refs": input_refs, **SCOPE,
    }
    manifest_path = output / "raw-manifest.json"
    _write_json(manifest_path, raw_manifest)
    result = {
        "kind": "c05-spatial-mode-artifact-result", "version": 1,
        "candidate_id": CANDIDATE_ID, "status": status,
        "candidate_ref": _output_ref(output, candidate_path),
        "certificate_ref": _output_ref(output, certificate_path),
        "manifest_ref": _output_ref(output, manifest_path),
        "archive_ref": _output_ref(output, archive_path),
        "role_denominator": len(ROLE_ORDER),
        "roles_completed": roles_completed,
        "roles_failed": roles_failed,
        "natural_gate_passed": gate_passed,
        **SCOPE,
    }
    _write_json(output / "result.json", result)
    return result


def validate_candidate_artifact(root: str | Path, path: str | Path, *,
                                evidence_root: str | Path | None = None
                                ) -> dict[str, Any]:
    """Rehash the exact bounded artifact inventory and structural invariants."""
    root = Path(root).resolve()
    evidence_root = root if evidence_root is None else Path(evidence_root).resolve()
    evidence_root.relative_to(root)
    supplied = Path(path).resolve()
    supplied.relative_to(evidence_root)
    output = supplied.parent if supplied.name == "candidate.json" else supplied
    candidate_path = output / "candidate.json"
    candidate = _read_json(candidate_path)
    if (candidate.get("kind") != "c05-spatial-mode-candidate"
            or candidate.get("version") != 1
            or candidate.get("candidate_id") != CANDIDATE_ID
            or candidate.get("status") not in
               ("completed_unqualified", "incomplete_natural_gate")
            or candidate.get("method_ids") != METHOD_IDS
            or any(candidate.get(name) is not value for name, value in SCOPE.items())):
        raise C05ArtifactError("current unqualified C05 candidate required")
    result = _read_json(output / "result.json")
    manifest = _read_json(output / "raw-manifest.json")
    certificate = _read_json(output / "certificate.json")
    if (result.get("kind") != "c05-spatial-mode-artifact-result"
            or result.get("version") != 1 or result.get("candidate_id") != CANDIDATE_ID
            or manifest.get("kind") != "c05-spatial-mode-artifact-manifest"
            or manifest.get("version") != 1 or manifest.get("candidate_id") != CANDIDATE_ID
            or certificate.get("kind") != "c05-spatial-mode-certificate"
            or certificate.get("version") != 1
            or certificate.get("candidate_id") != CANDIDATE_ID
            or any(document.get(name) is not value
                   for document in (result, manifest, certificate)
                   for name, value in SCOPE.items())
            or result.get("status") != candidate["status"]
            or manifest.get("status") != candidate["status"]
            or certificate.get("status") !=
               ("passed" if candidate["status"] == "completed_unqualified" else "rejected")
            or result.get("candidate_ref") != _output_ref(output, candidate_path)
            or result.get("manifest_ref") != _output_ref(output, output / "raw-manifest.json")
            or result.get("certificate_ref") != _output_ref(output, output / "certificate.json")
            or manifest.get("input_refs") != candidate.get("input_refs")
            or certificate.get("input_refs") != candidate.get("input_refs")
            or certificate.get("parameters") != candidate.get("parameters")
            or candidate.get("role_denominator") != len(ROLE_ORDER)
            or result.get("role_denominator") != len(ROLE_ORDER)
            or manifest.get("role_denominator") != len(ROLE_ORDER)
            or result.get("roles_completed") != candidate.get("roles_completed")
            or result.get("roles_failed") != candidate.get("roles_failed")
            or manifest.get("roles_completed") != candidate.get("roles_completed")
            or manifest.get("roles_failed") != candidate.get("roles_failed")):
        raise C05ArtifactError("C05 result/manifest/certificate identity differs")
    expected_input_names = {
        "mode_bank_result", "mode_bank_manifest", "mode_bank",
        "branch0_sequence", "b0_sequence", "b0_report", "parity_receipt",
        "artifact_implementation", "math_implementation",
        "mode_bank_implementation",
    }
    if set(candidate.get("input_refs", {})) != expected_input_names:
        raise C05ArtifactError("complete exact C05 input closure required")
    for ref in candidate["input_refs"].values():
        _resolve_ref(evidence_root, ref)
    inputs = candidate["input_refs"]
    mode_bank_path = _resolve_ref(evidence_root, inputs["mode_bank"])
    mode_root = mode_bank_path.parent
    if any(item.is_symlink() for item in mode_root.rglob("*")):
        raise C05ArtifactError("symlinks are forbidden in retained C05 mode bank")
    unvalidated_manifest = _read_json(mode_root / "raw-manifest.json")
    producer_root = _producer_namespace_root(
        evidence_root, mode_root, unvalidated_manifest)
    retained_manifest = validate_retained_mode_bank(mode_root, producer_root)
    b0_path = _resolve_ref(evidence_root, inputs["b0_sequence"])
    b0_report_path = _resolve_ref(evidence_root, inputs["b0_report"])
    parity_path = _resolve_ref(evidence_root, inputs["parity_receipt"])
    mode_arrays = _load_npz(mode_bank_path, _BANK_FIELDS)
    _validate_branch_closure(mode_root, retained_manifest, mode_arrays)
    if (inputs["artifact_implementation"] != _ref(root, Path(__file__))
            or inputs["math_implementation"] != _ref(
                root, Path(cluster_empirical_trajectories.__code__.co_filename))
            or inputs["mode_bank_implementation"] != _ref(
                root, Path(validate_retained_mode_bank.__code__.co_filename))
            or retained_manifest.get("producer_code_sha256") !=
               inputs["mode_bank_implementation"]["sha256"]):
        raise C05ArtifactError("C05 implementation source closure differs")
    b0 = _load_npz(b0_path, _SEQUENCE_FIELDS)
    native = validate_native_arrays(mode_arrays, b0)
    uid = retained_manifest["provenance"]["generation_uid"]
    outer_seed = int(mode_arrays["outer_generation_seed"])
    _validate_b0_report(b0_report_path, inputs["b0_sequence"], uid=uid,
                        seed=outer_seed)
    parity_refs = {
        "mode_bank_result": _ref(producer_root, mode_root / "result.json"),
        "mode_bank_manifest": _ref(
            producer_root, mode_root / "raw-manifest.json"),
        "mode_bank": _ref(producer_root, mode_bank_path),
        "branch0_sequence": _ref(
            producer_root, mode_root / "sequences/branch-0000.npz"),
        "b0_sequence": _ref(producer_root, b0_path),
        "b0_report": _ref(producer_root, b0_report_path),
    }
    _validate_parity(producer_root, parity_path, refs=parity_refs,
                     uid=uid, seed=outer_seed)
    parameters = candidate["parameters"]
    expected_parameter_keys = {
        "landmark_count", "landmark_selection", "graph", "cluster_radius",
        "natural_gate_min_fraction", "localized_radius", "temperature",
        "unary_weight", "spatial_weight", "max_sweeps",
        "displacement_clip_multiplier", "face_chunk_size", "lift",
    }
    if set(parameters) != expected_parameter_keys:
        raise C05ArtifactError("complete frozen C05 parameter set required")
    anchor = b0["vertices"][0].astype(np.float64)
    query_ids = b0["query_vertex_ids"].astype(np.int64)
    landmark_ids = _farthest_landmarks(
        anchor, query_ids, _positive_integer("landmark_count", parameters["landmark_count"]))
    landmark_query_ids = query_ids[landmark_ids]
    graph = _minimum_spanning_tree(anchor[landmark_ids], landmark_query_ids)
    scales = _landmark_scales(anchor[landmark_ids], graph)
    bank = cluster_empirical_trajectories(
        native["sampled_surfaces"][:, :, landmark_ids, :],
        draw_seeds=native["draw_seeds"], landmark_scales=scales,
        cluster_radius=_positive("cluster_radius", parameters["cluster_radius"]))
    mode_counts = np.asarray([len(item.atoms) for item in bank.modes], dtype=np.int64)
    separations = normalized_mode_separations(bank)
    fraction = parameters["natural_gate_min_fraction"]
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not 0.0 < float(fraction) <= 1.0):
        raise C05ArtifactError("invalid frozen Natural Gate fraction")
    required_count = int(math.ceil(float(fraction) * len(landmark_ids)))
    separated = ((mode_counts >= 2)
                 & (separations > _positive("cluster_radius", parameters["cluster_radius"])))
    gate_passed = int(np.count_nonzero(separated)) >= required_count
    expected_certificate = {
        "landmark_vertex_indices": landmark_ids.tolist(),
        "landmark_query_vertex_ids": landmark_query_ids.tolist(),
        "graph_edges": graph.tolist(), "landmark_scales": scales.tolist(),
        "mode_counts": mode_counts.tolist(),
        "max_medoid_separations": separations.tolist(),
        "separated_landmark_count": int(np.count_nonzero(separated)),
        "multimodal_landmark_count": int(np.count_nonzero(separated)),
        "landmark_count": len(landmark_ids),
        "multimodal_fraction": float(np.mean(separated)),
        "required_multimodal_count": required_count,
    }
    if any(certificate.get(name) != value for name, value in expected_certificate.items()):
        raise C05ArtifactError("C05 Natural Gate certificate does not recompute")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise C05ArtifactError("bounded C05 member inventory required")
    member_names = []
    for ref in members:
        member_path = output / ref.get("path", "")
        if _output_ref(output, member_path) != ref:
            raise C05ArtifactError("C05 generated member changed")
        member_names.append(ref["path"])
    if member_names != sorted(set(member_names)):
        raise C05ArtifactError("canonical unique C05 member list required")
    archive_path = output / "raw-evidence.tar"
    if (manifest.get("archive_ref") != _output_ref(output, archive_path)
            or result.get("archive_ref") != manifest["archive_ref"]):
        raise C05ArtifactError("C05 archive reference differs")
    if archive_path.read_bytes() != _tar_bytes(output, member_names):
        raise C05ArtifactError("C05 bounded archive does not reproduce")
    ceiling = manifest.get("max_artifact_bytes")
    if (isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 1
            or archive_path.stat().st_size > ceiling
            or sum((output / name).stat().st_size for name in member_names) > ceiling):
        raise C05ArtifactError("C05 prospective artifact byte ceiling violated")
    expected_files = set(member_names) | {"raw-evidence.tar", "raw-manifest.json", "result.json"}
    actual_files = {item.relative_to(output).as_posix() for item in output.rglob("*")
                    if item.is_file() or item.is_symlink()}
    if actual_files != expected_files or any(item.is_symlink() for item in output.rglob("*")):
        raise C05ArtifactError("C05 output contains missing, extra, or symbolic files")
    expected_status = "completed_unqualified" if gate_passed else "incomplete_natural_gate"
    if (candidate["status"] != expected_status
            or candidate.get("natural_gate_passed") is not gate_passed
            or set(candidate.get("role_refs", {})) != set(ROLE_ORDER)):
        raise C05ArtifactError("C05 Natural Gate/role inventory differs")

    role_states, solver_evidence = _construct_sparse_roles(
        bank, gate_passed=gate_passed, graph=graph,
        anchor_landmarks=anchor[landmark_ids],
        b0_landmarks=b0["vertices"][:, landmark_ids],
        timesteps=b0["timesteps"], sampled_surfaces=native["sampled_surfaces"],
        faces=native["faces"],
        localized_radius=_positive("localized_radius", parameters["localized_radius"]),
        temperature=_positive("temperature", parameters["temperature"]),
        unary_weight=_positive("unary_weight", parameters["unary_weight"]),
        spatial_weight=_positive("spatial_weight", parameters["spatial_weight"]),
        max_sweeps=_positive_integer("max_sweeps", parameters["max_sweeps"]),
        face_chunk_size=_positive_integer(
            "face_chunk_size", parameters["face_chunk_size"]))
    lift: np.ndarray | None = None
    if gate_passed:
        try:
            lift = _inverse_distance_lift(anchor, landmark_ids, scales)
        except Exception as error:
            failure = _bounded_failure(error)
            for role in ROLE_ORDER:
                if role_states[role]["status"] == "completed_unqualified":
                    role_states[role] = {"status": "error", **failure}
        if lift is not None:
            for role in ROLE_ORDER:
                state = role_states[role]
                if state["status"] != "completed_unqualified":
                    continue
                try:
                    vertices, clipping = _lift_role(
                        b0["vertices"], state["sparse"], landmark_ids, lift, scales,
                        _positive("displacement_clip_multiplier",
                                  parameters["displacement_clip_multiplier"]))
                except Exception as error:
                    role_states[role] = {"status": "error", **_bounded_failure(error)}
                else:
                    state["vertices"] = vertices
                    state["clipping"] = clipping

    expected_solver = _solver_record(
        evidence=solver_evidence, states=role_states, lift=lift)
    solver_path = output / "solver-certificate.json"
    if (candidate.get("solver_certificate_ref") != _output_ref(output, solver_path)
            or _read_json(solver_path) != expected_solver):
        raise C05ArtifactError("C05 construction/solver certificate differs")

    roles_completed = [role for role in ROLE_ORDER
                       if role_states[role]["status"] == "completed_unqualified"]
    roles_failed = [role for role in ROLE_ORDER if role not in roles_completed]
    if (candidate.get("roles_completed") != roles_completed
            or candidate.get("roles_failed") != roles_failed
            or result.get("roles_completed") != roles_completed
            or result.get("roles_failed") != roles_failed):
        raise C05ArtifactError("C05 fixed role denominator differs")

    expected_members = {"candidate.json", "certificate.json", "solver-certificate.json"}
    expected_members.update(f"roles/{role}/report.json" for role in ROLE_ORDER)
    expected_members.update(f"roles/{role}/sequence.npz" for role in roles_completed)
    if set(member_names) != expected_members:
        raise C05ArtifactError("C05 bounded member inventory differs from role outcomes")

    shared_lift = None
    for role in ROLE_ORDER:
        state = role_states[role]
        row = candidate["role_refs"][role]
        report_path = output / f"roles/{role}/report.json"
        expected_sequence_ref = None
        report_sequence_ref = None
        exact = state["status"] == "completed_unqualified"
        if exact:
            sequence_path = output / f"roles/{role}/sequence.npz"
            expected_sequence_ref = _output_ref(output, sequence_path)
            report_sequence_ref = _sibling_ref(sequence_path)
            arrays = _load_npz(sequence_path, _SEQUENCE_FIELDS)
            for name in _SEQUENCE_FIELDS - {"vertices"}:
                if (arrays[name].dtype != b0[name].dtype
                        or not np.array_equal(arrays[name], b0[name])):
                    raise C05ArtifactError("C05 role changed native identity: " + role)
            if (arrays["vertices"].dtype != b0["vertices"].dtype
                    or arrays["vertices"].shape != b0["vertices"].shape
                    or not np.array_equal(arrays["vertices"], state["vertices"])
                    or not np.isfinite(arrays["vertices"]).all()):
                raise C05ArtifactError("invalid C05 full-sequence export: " + role)
        if (row != {
                "method_id": METHOD_IDS[role],
                "preparation_status": "completed" if exact else "error",
                "sequence_ref": expected_sequence_ref,
                "report_ref": _output_ref(output, report_path)}):
            raise C05ArtifactError("C05 role identity/outcome differs: " + role)
        expected_report = {
            "kind": "c05-spatial-mode-role-report", "version": 1,
            "candidate_id": CANDIDATE_ID, "status": state["status"],
            "uid": uid, "outer_seed": outer_seed,
            "candidate_arm": role, "method_id": METHOD_IDS[role],
            "implementation_ref": inputs["math_implementation"],
            "sequence_ref": report_sequence_ref,
            "source_sequence_ref": inputs["b0_sequence"],
            "source_report_ref": inputs["b0_report"],
            "parity_receipt_ref": inputs["parity_receipt"],
            "certificate_ref": _ref(evidence_root, output / "certificate.json"),
            "solver_certificate_ref": _ref(evidence_root, solver_path),
            "landmark_query_vertex_ids": landmark_query_ids.tolist(),
            "lift_policy": candidate["parameters"]["lift"],
            "clipping": state.get("clipping"),
            "exception_type": state.get("exception_type"),
            "error": state.get("error"),
            "exact_frame_zero": exact, "exact_faces": exact,
            "exact_material_ids": exact, "exact_times": exact,
            **SCOPE,
        }
        if _read_json(report_path) != expected_report:
            raise C05ArtifactError("C05 role report differs: " + role)
        if not exact:
            error_type, error = state["exception_type"], state["error"]
            if not error_type or len(error_type) > 256 or not error or len(error) > 4096:
                raise C05ArtifactError("C05 terminal role error is not bounded: " + role)
        lift_policy = expected_report["lift_policy"]
        shared_lift = lift_policy if shared_lift is None else shared_lift
        if lift_policy != shared_lift:
            raise C05ArtifactError("all C05 roles must share one lift policy")
    return candidate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("materialize", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode-bank-root", type=Path)
    parser.add_argument("--parity-receipt", type=Path)
    parser.add_argument("--b0-sequence", type=Path)
    parser.add_argument("--b0-report", type=Path)
    parser.add_argument("--landmark-count", type=int)
    parser.add_argument("--cluster-radius", type=float)
    parser.add_argument("--natural-gate-min-fraction", type=float)
    parser.add_argument("--localized-radius", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--unary-weight", type=float)
    parser.add_argument("--spatial-weight", type=float)
    parser.add_argument("--max-sweeps", type=int)
    parser.add_argument("--displacement-clip-multiplier", type=float)
    parser.add_argument("--max-artifact-bytes", type=int)
    parser.add_argument("--face-chunk-size", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.operation == "validate":
        candidate = validate_candidate_artifact(args.root, args.output)
        print(json.dumps({"status": candidate["status"], **SCOPE}, sort_keys=True))
        return 0
    required = (
        "mode_bank_root", "parity_receipt", "b0_sequence", "b0_report",
        "landmark_count", "cluster_radius", "natural_gate_min_fraction",
        "localized_radius", "temperature", "unary_weight", "spatial_weight",
        "max_sweeps", "displacement_clip_multiplier", "max_artifact_bytes",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("materialize requires: " + ", ".join(missing))
    result = materialize_candidate(
        args.root, args.mode_bank_root, args.parity_receipt, args.b0_sequence,
        args.b0_report, args.output, landmark_count=args.landmark_count,
        cluster_radius=args.cluster_radius,
        natural_gate_min_fraction=args.natural_gate_min_fraction,
        localized_radius=args.localized_radius, temperature=args.temperature,
        unary_weight=args.unary_weight, spatial_weight=args.spatial_weight,
        max_sweeps=args.max_sweeps,
        displacement_clip_multiplier=args.displacement_clip_multiplier,
        max_artifact_bytes=args.max_artifact_bytes,
        face_chunk_size=args.face_chunk_size)
    print(json.dumps({"status": result["status"], **SCOPE}, sort_keys=True))
    return 0 if result["status"] in ("completed_unqualified",
                                      "incomplete_natural_gate") else 2


if __name__ == "__main__":  # pragma: no cover - Local CLI
    raise SystemExit(main())
