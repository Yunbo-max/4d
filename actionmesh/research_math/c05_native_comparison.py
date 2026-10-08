"""Freeze C05's seven logical native roles without scoring or choosing B*.

This C05-specific boundary consumes ``c05-spatial-mode-candidate`` records:
``outer_seed``, project-root ``input_refs``, output-local ``role_refs``, the
bounded raw manifest, and shared JSON construction/solver certificates.  It
does not reinterpret them as the incompatible C11 per-arm artifact schema.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from research_math import c05_candidate_artifacts as artifact_module
from research_math import spatial_mode_candidate as method_module


CANDIDATE_ID = "4d-math-20261006-c05"
ROLES = ("b0", "b_star", "localized_mean", "temperature_matched_mean",
         "surface_projected_mean", "independent_top1", "joint_spatial_labels")
METHOD_ROLES = ROLES[2:]
CONTROL_ROLES = METHOD_ROLES[:-1]
METHOD_IDS = method_module.METHOD_IDS
CANDIDATE_ROLE = "joint_spatial_labels"
FREEZE_KIND = "c05-native-comparison-freeze"
DECISION_KIND = "c05-b-star-decision"
DECISION_NO_OUTCOMES_FIELD = "selected_without_c05_native_outcomes"
REQUEST_KIND = "c05-native-comparison-request"
ALLOWED_INFERENCE_SEEDS = (42, 314, 2718)
PRIMARY_METRIC = "cd_3d"
GUARDRAIL_METRICS = ("cd_4d", "cd_motion")
SEQUENCE_FIELDS = {"vertices", "faces", "frame_indices", "timesteps",
                   "query_vertex_ids"}
FALSE_SCOPE = {"generated_unexecuted": True, "native_qualified": False,
               "scientific_verdict": "not_computed", "dispatch_ready": False}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return value


def canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Canonical relative POSIX evidence path required")
    pure = PurePosixPath(value)
    if (pure.is_absolute() or pure.as_posix() != value
            or any(part in ("", ".", "..") for part in pure.parts)):
        raise ValueError("Canonical relative POSIX evidence path required")
    return Path(*pure.parts)


def _physical(root: Path, path: Path) -> Path:
    root, unresolved = Path(root).resolve(), Path(path)
    unresolved = unresolved if unresolved.is_absolute() else root / unresolved
    if unresolved.is_symlink() or any(parent.is_symlink()
            for parent in unresolved.parents if parent != root.parent):
        raise ValueError("Physical nonsymlink evidence file required")
    resolved = unresolved.resolve(); resolved.relative_to(root)
    if not resolved.is_file():
        raise ValueError("Physical evidence file required: " + str(path))
    return resolved


def file_ref(root: Path, path: Path) -> dict[str, str]:
    root, path = Path(root).resolve(), _physical(root, path)
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def resolve_ref(root: Path, ref: Mapping[str, Any]) -> Path:
    if (not isinstance(ref, Mapping) or set(ref) != {"path", "sha256"}
            or not isinstance(ref.get("sha256"), str)
            or len(ref["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in ref["sha256"])):
        raise ValueError("Exact path/sha256 reference required")
    path = _physical(Path(root).resolve(), _safe_relative(ref["path"]))
    if digest(path) != ref["sha256"]:
        raise ValueError("Changed pinned file: " + ref["path"])
    return path


def _plain_ref(ref: Mapping[str, Any]) -> dict[str, str]:
    allowed = {"path", "sha256", "bytes", "size_bytes", "content_sha256"}
    if (not isinstance(ref, Mapping) or not {"path", "sha256"}.issubset(ref)
            or not set(ref).issubset(allowed)):
        raise ValueError("Retained exact file reference required")
    _safe_relative(ref["path"])
    sha = ref["sha256"]
    if (not isinstance(sha, str) or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)):
        raise ValueError("Canonical SHA-256 required")
    return {"path": ref["path"], "sha256": sha}


def _resolve_nested_ref(root: Path, document: Path,
                        ref: Mapping[str, Any], *, evidence_root: Path,
                        producer_root: Path) -> Path:
    plain = _plain_ref(ref); relative = _safe_relative(plain["path"])
    namespace = (producer_root if Path(document).resolve().is_relative_to(producer_root)
                 else evidence_root)
    candidates = (namespace / relative, Path(document).parent / relative)
    matches = []
    for candidate in candidates:
        try:
            path = _physical(root, candidate)
        except (ValueError, FileNotFoundError):
            continue
        size = ref.get("bytes", ref.get("size_bytes"))
        if (digest(path) == plain["sha256"] and (size is None or
                (type(size) is int and path.stat().st_size == size))):
            matches.append(path)
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous nested evidence: " + plain["path"])
    return matches[0]


def _iter_refs(value: Any):
    if isinstance(value, Mapping):
        keys = set(value)
        if ({"path", "sha256"}.issubset(keys) and keys.issubset(
                {"path", "sha256", "bytes", "size_bytes", "content_sha256"})):
            yield value
        else:
            for child in value.values():
                yield from _iter_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_refs(child)


def _recursive_closure(root: Path, document: Path, value: Any, *,
                       evidence_root: Path | None = None,
                       producer_root: Path | None = None
                       ) -> list[dict[str, str]]:
    evidence_root = Path(root).resolve() if evidence_root is None else Path(evidence_root).resolve()
    producer_root = evidence_root if producer_root is None else Path(producer_root).resolve()
    pending = [(document, ref) for ref in _iter_refs(value)]
    seen: set[Path] = set(); paths: list[Path] = []
    while pending:
        if len(seen) + len(pending) > 512:
            raise ValueError("C05 recursive evidence closure exceeds 512 files")
        owner, ref = pending.pop(); path = _resolve_nested_ref(
            root, owner, ref, evidence_root=evidence_root,
            producer_root=producer_root)
        if path in seen:
            continue
        seen.add(path); paths.append(path)
        if path.suffix == ".json":
            pending.extend((path, child) for child in _iter_refs(read_json(path)))
    return [file_ref(root, path) for path in paths]


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != SEQUENCE_FIELDS:
            raise ValueError("Exact complete native sequence fields required")
        return {name: archive[name].copy() for name in archive.files}


def _validate_native(arrays: Mapping[str, np.ndarray]) -> None:
    vertices, faces = arrays["vertices"], arrays["faces"]
    if (vertices.dtype != np.float32 or vertices.ndim != 3
            or vertices.shape[0] != 16 or vertices.shape[1] < 3
            or vertices.shape[2] != 3 or not np.isfinite(vertices).all()):
        raise ValueError("Exactly 16 finite float32 full vertex frames required")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= vertices.shape[1]):
        raise ValueError("Valid shared triangle topology required")
    if (not np.issubdtype(arrays["frame_indices"].dtype, np.integer)
            or not np.array_equal(arrays["frame_indices"], np.arange(16))):
        raise ValueError("Original 16-frame order required")
    ids = arrays["query_vertex_ids"]
    if (not np.issubdtype(ids.dtype, np.integer) or ids.shape != (vertices.shape[1],)
            or len(np.unique(ids)) != vertices.shape[1]):
        raise ValueError("Unique original material identities required")
    times = arrays["timesteps"]
    if (times.shape != (16,) or not np.issubdtype(times.dtype, np.number)
            or not np.isfinite(times).all()
            or np.any(np.diff(times.astype(np.float64)) <= 0.0)):
        raise ValueError("One strictly increasing native timestamp per frame required")
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("Finite numeric native array required: " + name)


def _same_native(source: Mapping[str, np.ndarray],
                 arm: Mapping[str, np.ndarray]) -> None:
    _validate_native(source); _validate_native(arm)
    if (set(source) != set(arm) or arm["vertices"].shape != source["vertices"].shape
            or arm["vertices"].dtype != source["vertices"].dtype
            or not np.array_equal(arm["vertices"][0], source["vertices"][0])):
        raise ValueError("C05 role changed native inventory/shape/dtype/anchor")
    for name in SEQUENCE_FIELDS - {"vertices"}:
        if (arm[name].dtype != source[name].dtype
                or not np.array_equal(arm[name], source[name])):
            raise ValueError("C05 role changed topology/identity/time: " + name)


def _content_digest(arrays: Mapping[str, np.ndarray]) -> str:
    value = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        header = json.dumps([name, array.dtype.str, list(array.shape)],
                            separators=(",", ":")).encode()
        value.update(len(header).to_bytes(8, "little")); value.update(header)
        value.update(array.tobytes())
    return value.hexdigest()


def _validate_freeze(freeze: Mapping[str, Any]) -> None:
    required = {"kind", "version", "candidate_id", "uid", "inference_seed",
                "scoring_seed", "primary_metric", "guardrail_metrics", "frozen_at",
                "source_sequence_ref", "source_report_ref", "candidate_artifact_ref",
                "b_star_decision_ref", "roles", "freeze_digest"}
    extra = set(freeze) - required
    if not required.issubset(freeze) or extra not in (set(), {"semantic_review_ref"}):
        raise ValueError("Exact C05 freeze fields required")
    core = {key: value for key, value in freeze.items() if key != "freeze_digest"}
    if (freeze.get("freeze_digest") != canonical_digest(core)
            or freeze.get("kind") != FREEZE_KIND or freeze.get("version") != 1
            or freeze.get("candidate_id") != CANDIDATE_ID
            or freeze.get("scoring_seed") != 44
            or freeze.get("primary_metric") != PRIMARY_METRIC
            or freeze.get("guardrail_metrics") != list(GUARDRAIL_METRICS)):
        raise ValueError("Current exact C05 comparison freeze required")
    try:
        frozen = datetime.fromisoformat(freeze["frozen_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Timezone-aware frozen_at required") from error
    uid = freeze.get("uid")
    if frozen.tzinfo is None:
        raise ValueError("Timezone-aware frozen_at required")
    if not isinstance(uid, str) or not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    if freeze.get("inference_seed") not in ALLOWED_INFERENCE_SEEDS:
        raise ValueError("Frozen C05 outer inference seed is not in G01")
    rows = freeze.get("roles")
    if (not isinstance(rows, list) or [row.get("role") if isinstance(row, dict)
            else None for row in rows] != list(ROLES)):
        raise ValueError("Exactly seven ordered C05 roles must be frozen")


def _decision(root: Path, freeze: Mapping[str, Any]) -> tuple[dict, list[dict]]:
    path = resolve_ref(root, freeze["b_star_decision_ref"]); decision = read_json(path)
    core = {key: value for key, value in decision.items() if key != "decision_digest"}
    if (decision.get("kind") != DECISION_KIND or decision.get("version") != 1
            or decision.get("candidate_id") != CANDIDATE_ID
            or decision.get("uid") != freeze["uid"]
            or decision.get("inference_seed") != freeze["inference_seed"]
            or decision.get(DECISION_NO_OUTCOMES_FIELD) is not True
            or decision.get("decision_digest") != canonical_digest(core)):
        raise ValueError("C05 B* must be selected prospectively without native outcomes")
    try:
        decided = datetime.fromisoformat(decision["decided_at"].replace("Z", "+00:00"))
        frozen = datetime.fromisoformat(freeze["frozen_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Timezone-aware C05 B* decision required") from error
    if decided.tzinfo is None or decided > frozen:
        raise ValueError("C05 B* decision must precede the comparison freeze")
    basis = decision.get("selection_basis_refs")
    if not isinstance(basis, list) or not basis:
        raise ValueError("C05 B* requires prospective selection basis")
    return decision, [file_ref(root, resolve_ref(root, ref)) for ref in basis]


def _local_ref(directory: Path, ref: Mapping[str, Any]) -> Path:
    plain = _plain_ref(ref); path = _physical(directory, _safe_relative(plain["path"]))
    size = ref.get("bytes", ref.get("size_bytes"))
    if (digest(path) != plain["sha256"] or (size is not None and
            (type(size) is not int or path.stat().st_size != size))):
        raise ValueError("Changed C05 artifact member: " + plain["path"])
    return path


def _artifact_closure(root: Path, artifact_path: Path,
                      freeze: Mapping[str, Any], *, evidence_root: Path
                      ) -> tuple[dict, dict, list[dict]]:
    artifact_path = _physical(root, artifact_path)
    if artifact_path.name != "candidate.json":
        raise ValueError("C05 candidate_artifact_ref must name candidate.json")
    evidence_root = Path(evidence_root).resolve()
    evidence_root.relative_to(Path(root).resolve())
    artifact_path.relative_to(evidence_root)
    verified = artifact_module.validate_candidate_artifact(
        root, artifact_path, evidence_root=evidence_root)
    candidate = read_json(artifact_path)
    completed_roles = candidate.get("roles_completed")
    failed_roles = candidate.get("roles_failed")
    role_partition_valid = (
        isinstance(completed_roles, list) and isinstance(failed_roles, list)
        and completed_roles == [role for role in METHOD_ROLES if role in completed_roles]
        and failed_roles == [role for role in METHOD_ROLES if role in failed_roles]
        and set(completed_roles).isdisjoint(failed_roles)
        and set(completed_roles) | set(failed_roles) == set(METHOD_ROLES))
    if (verified != candidate or candidate.get("status") != "completed_unqualified"
            or candidate.get("candidate_id") != CANDIDATE_ID
            or candidate.get("uid") != freeze["uid"]
            or candidate.get("outer_seed") != freeze["inference_seed"]
            or candidate.get("method_ids") != METHOD_IDS
            or set(candidate.get("role_refs", {})) != set(METHOD_ROLES)
            or candidate.get("role_denominator") != len(METHOD_ROLES)
            or not role_partition_valid
            or candidate.get("natural_gate_passed") is not True):
        raise ValueError("Complete same-identity C05 candidate artifact required")
    directory = artifact_path.parent
    mode_path = artifact_module._resolve_ref(
        evidence_root, candidate["input_refs"]["mode_bank"])
    mode_manifest = read_json(mode_path.parent / "raw-manifest.json")
    producer_root = artifact_module._producer_namespace_root(
        evidence_root, mode_path.parent, mode_manifest)
    source_raw_ref = _plain_ref(candidate["input_refs"]["b0_sequence"])
    report_raw_ref = _plain_ref(candidate["input_refs"]["b0_report"])
    source_path = _resolve_nested_ref(
        root, artifact_path, candidate["input_refs"]["b0_sequence"],
        evidence_root=evidence_root, producer_root=producer_root)
    report_path = _resolve_nested_ref(
        root, artifact_path, candidate["input_refs"]["b0_report"],
        evidence_root=evidence_root, producer_root=producer_root)
    source_ref, report_ref = file_ref(root, source_path), file_ref(root, report_path)
    if (source_ref["sha256"] != freeze["source_sequence_ref"]["sha256"]
            or report_ref["sha256"] != freeze["source_report_ref"]["sha256"]):
        raise ValueError("C05 candidate inputs differ from frozen B0")
    general_path = _local_ref(directory, candidate["certificate_ref"])
    solver_path = _local_ref(directory, candidate["solver_certificate_ref"])
    general, solver = read_json(general_path), read_json(solver_path)
    if (general.get("kind") != "c05-spatial-mode-certificate"
            or general.get("status") != "passed" or general.get("uid") != freeze["uid"]
            or general.get("outer_seed") != freeze["inference_seed"]
            or general.get("input_refs") != candidate["input_refs"]
            or general.get("multimodal_landmark_count", 0)
               < general.get("required_multimodal_count", 1)):
        raise ValueError("Passed same-input C05 construction certificate required")
    lift_sha = solver.get("lift_sha256")
    if (solver.get("kind") != "c05-spatial-mode-solver-certificate"
            or solver.get("version") != 1
            or solver.get("candidate_id") != CANDIDATE_ID
            or not isinstance(solver.get("role_statuses"), dict)
            or set(solver["role_statuses"]) != set(METHOD_ROLES)
            or not isinstance(solver.get("role_errors"), dict)
            or set(solver["role_errors"]) != set(METHOD_ROLES)
            or (lift_sha is not None and (not isinstance(lift_sha, str)
                or len(lift_sha) != 64))):
        raise ValueError("Shared C05 solver/lift certificate required")
    manifest = read_json(directory / "raw-manifest.json")
    expected_members = {"candidate.json", "certificate.json", "solver-certificate.json"}
    for role in METHOD_ROLES:
        role_ref = candidate["role_refs"][role]
        if role_ref.get("report_ref", {}).get("path") != f"roles/{role}/report.json":
            raise ValueError("Canonical C05 role report path required: " + role)
        expected_members.add(role_ref["report_ref"]["path"])
        sequence_ref = role_ref.get("sequence_ref")
        if sequence_ref is not None:
            if sequence_ref.get("path") != f"roles/{role}/sequence.npz":
                raise ValueError("Canonical C05 role sequence path required: " + role)
            expected_members.add(sequence_ref["path"])
    member_refs = manifest.get("members")
    if (manifest.get("kind") != "c05-spatial-mode-artifact-manifest"
            or manifest.get("status") != "completed_unqualified"
            or manifest.get("input_refs") != candidate["input_refs"]
            or not isinstance(member_refs, list)
            or {ref.get("path") for ref in member_refs} != expected_members):
        raise ValueError("Complete bounded C05 raw manifest required")
    for ref in member_refs:
        _local_ref(directory, ref)
    source = _arrays(source_path); _validate_native(source)
    roles = {}; general_ref = file_ref(root, general_path); solver_ref = file_ref(root, solver_path)
    for role in METHOD_ROLES:
        row = candidate["role_refs"][role]
        if row.get("method_id") != METHOD_IDS[role]:
            raise ValueError("C05 artifact method identity differs: " + role)
        report_path = _local_ref(directory, row["report_ref"]); report = read_json(report_path)
        completed, preparation_status = _terminal_report(report, freeze, role)
        sequence_ref = row.get("sequence_ref")
        if completed:
            if sequence_ref is None or report.get("sequence_ref") is None:
                raise ValueError("Completed C05 role lacks a sequence: " + role)
            sequence_path = _local_ref(directory, sequence_ref)
            report_sequence_path = _local_ref(report_path.parent, report["sequence_ref"])
        else:
            if sequence_ref is not None or report.get("sequence_ref") is not None:
                raise ValueError("Failed C05 role retained a scoreable sequence: " + role)
            sequence_path = report_sequence_path = None
        if (row.get("preparation_status") != preparation_status
                or completed != (role in completed_roles)
                or (not completed) != (role in failed_roles)):
            raise ValueError("C05 role reference/report status differs: " + role)
        if (report.get("kind") != "c05-spatial-mode-role-report"
                or report.get("candidate_id") != CANDIDATE_ID
                or report.get("uid") != freeze["uid"]
                or report.get("outer_seed") != freeze["inference_seed"]
                or report.get("candidate_arm") != role
                or report.get("method_id") != METHOD_IDS[role]
                or (completed and report_sequence_path != sequence_path)
                or _plain_ref(report.get("implementation_ref", {})) !=
                   _plain_ref(candidate["input_refs"]["math_implementation"])
                or _plain_ref(report.get("source_report_ref", {})) != report_raw_ref
                or _plain_ref(report.get("parity_receipt_ref", {})) !=
                   _plain_ref(candidate["input_refs"]["parity_receipt"])
                or _plain_ref(report.get("certificate_ref", {}))["sha256"] !=
                   general_ref["sha256"]
                or _plain_ref(report.get("solver_certificate_ref", {}))["sha256"] !=
                   solver_ref["sha256"]
                or _plain_ref(report.get("source_sequence_ref", {})) != source_raw_ref
                or (completed and any(report.get(key) is not True for key in
                    ("exact_frame_zero", "exact_faces", "exact_material_ids", "exact_times")))):
            raise ValueError("C05 role/report/shared-certificate mismatch: " + role)
        arrays = None
        if completed:
            arrays = _arrays(sequence_path); _same_native(source, arrays)
        roles[role] = {"role": role, "method_id": METHOD_IDS[role],
            "report_ref": file_ref(root, report_path),
            "sequence_ref": file_ref(root, sequence_path) if completed else None,
            "shared_certificate_ref": general_ref,
            "shared_solver_certificate_ref": solver_ref, "arrays": arrays,
            "preparation_status": preparation_status}
        if not completed:
            roles[role].update(exception_type=report["exception_type"],
                error=report["error"],
                preparation_error=report["exception_type"] + ": " + report["error"])
        expected_solver_status = "completed_unqualified" if completed else "error"
        expected_solver_error = (None if completed else {
            "exception_type": report["exception_type"], "error": report["error"]})
        if (solver["role_statuses"].get(role) != expected_solver_status
                or solver["role_errors"].get(role) != expected_solver_error):
            raise ValueError("C05 role differs from shared solver certificate: " + role)
    if any(value["arrays"] is not None for value in roles.values()) and lift_sha is None:
        raise ValueError("Successful C05 role requires the shared lift hash")
    files = [path for path in directory.rglob("*") if path.is_file()]
    closure = [file_ref(root, path) for path in files]
    closure.extend(_recursive_closure(
        root, artifact_path, candidate["input_refs"],
        evidence_root=evidence_root, producer_root=producer_root))
    return candidate, roles, closure


def _terminal_report(report: Mapping[str, Any], freeze: Mapping[str, Any],
                     role: str) -> tuple[bool, str]:
    seed = report.get("outer_seed", report.get("seed"))
    if report.get("uid") != freeze["uid"] or seed != freeze["inference_seed"]:
        raise ValueError("Role UID/outer-seed mismatch: " + role)
    if report.get("status") in ("completed", "completed_unqualified"):
        return True, "completed"
    if report.get("status") not in ("error", "failed"):
        raise ValueError("Terminal role report required: " + role)
    error_type, error = report.get("exception_type"), report.get("error")
    if (not isinstance(error_type, str) or not error_type.strip() or len(error_type) > 256
            or not isinstance(error, str) or not error.strip() or len(error) > 4096):
        raise ValueError("Bounded terminal role failure required: " + role)
    return False, "error"


def make_request(root: Path, *, freeze_path: Path,
                 artifact_evidence_root: Path | None = None,
                 _verify: bool = True) -> dict[str, Any]:
    root, freeze_path = Path(root).resolve(), _physical(root, freeze_path)
    artifact_evidence_root = (root if artifact_evidence_root is None
                              else Path(artifact_evidence_root).resolve())
    artifact_evidence_root.relative_to(root)
    freeze = read_json(freeze_path); _validate_freeze(freeze)
    source_path = resolve_ref(root, freeze["source_sequence_ref"])
    source_report_path = resolve_ref(root, freeze["source_report_ref"])
    source = _arrays(source_path); _validate_native(source)
    source_report = read_json(source_report_path)
    completed, _ = _terminal_report(source_report, freeze, "b0")
    if (not completed or source_report.get("sha256", {}).get("sequence.npz")
            != freeze["source_sequence_ref"]["sha256"]):
        raise ValueError("Exact completed C05 B0 source pair required")
    decision, basis_refs = _decision(root, freeze)
    artifact_path = resolve_ref(root, freeze["candidate_artifact_ref"])
    candidate, artifact_roles, artifact_refs = _artifact_closure(
        root, artifact_path, freeze, evidence_root=artifact_evidence_root)
    evidence_relative = candidate["input_refs"]["artifact_implementation"]["path"]
    evidence_sentinel = artifact_module._resolve_ref(
        artifact_evidence_root,
        candidate["input_refs"]["artifact_implementation"])
    evidence_sentinel_ref = file_ref(root, evidence_sentinel)
    pinned = [file_ref(root, freeze_path), freeze["source_sequence_ref"],
              freeze["source_report_ref"], freeze["b_star_decision_ref"],
              freeze["candidate_artifact_ref"], evidence_sentinel_ref,
              *basis_refs, *artifact_refs]
    if freeze.get("semantic_review_ref") is not None:
        review_path = resolve_ref(root, freeze["semantic_review_ref"])
        pinned.append(file_ref(root, review_path))
        pinned.extend(_recursive_closure(root, review_path, read_json(review_path)))

    normalized = []; physical = {}; content = {}; explicit_alias = None
    for row in freeze["roles"]:
        role = row["role"]
        if not isinstance(row.get("method_id"), str) or not row["method_id"]:
            raise ValueError("Every C05 role requires a nonempty method_id")
        if role == "b_star" and "alias_of" in row:
            if (set(row) != {"role", "method_id", "alias_of"}
                    or row["alias_of"] not in ("b0", *CONTROL_ROLES)
                    or decision.get("selected_role") != row["alias_of"]
                    or decision.get("selected_method_id") != row["method_id"]):
                raise ValueError("C05 B* alias must match the prospective decision")
            explicit_alias = dict(row); normalized.append(dict(row)); continue
        arrays = None
        if role in METHOD_ROLES:
            if set(row) != {"role", "method_id", "implementation_ref"}:
                raise ValueError("C05 artifact role freeze requires method/source identity")
            if row["method_id"] != METHOD_IDS[role]:
                raise ValueError("C05 method_id differs from candidate artifact: " + role)
            implementation_path = resolve_ref(root, row["implementation_ref"])
            if (digest(implementation_path) != digest(Path(method_module.__file__))
                    or row["implementation_ref"]["sha256"] !=
                       _plain_ref(candidate["input_refs"]["math_implementation"])["sha256"]):
                raise ValueError("C05 method implementation source differs: " + role)
            artifact = artifact_roles[role]; arrays = artifact["arrays"]
            current = {"role": role, "method_id": row["method_id"],
                "report_ref": artifact["report_ref"], "sequence_ref": artifact["sequence_ref"],
                "implementation_ref": row["implementation_ref"],
                "implementation_sha256": row["implementation_ref"]["sha256"],
                "shared_certificate_ref": artifact["shared_certificate_ref"],
                "shared_solver_certificate_ref": artifact["shared_solver_certificate_ref"],
                "case_id": freeze["uid"] + "-" + role if arrays is not None else None,
                "preparation_status": artifact["preparation_status"]}
            if arrays is None:
                current.update(exception_type=artifact["exception_type"],
                    error=artifact["error"],
                    preparation_error=artifact["preparation_error"])
            pinned.extend([row["implementation_ref"], current["report_ref"],
                current["shared_certificate_ref"], current["shared_solver_certificate_ref"]])
            if current["sequence_ref"] is not None:
                pinned.append(current["sequence_ref"])
        else:
            required = {"role", "method_id", "report_ref", "sequence_ref",
                        "implementation_ref"}
            if role == "b0": required.add("generation_identity_ref")
            if set(row) != required:
                raise ValueError("C05 physical baseline role has incomplete refs: " + role)
            report_path = resolve_ref(root, row["report_ref"])
            implementation_path = resolve_ref(root, row["implementation_ref"])
            report = read_json(report_path); completed, status = _terminal_report(report, freeze, role)
            sequence_path = resolve_ref(root, row["sequence_ref"]) if completed else None
            if not completed and row["sequence_ref"] is not None:
                raise ValueError("Failed C05 B* cannot retain a scoreable sequence")
            if role == "b0":
                identity_path = resolve_ref(root, row["generation_identity_ref"])
                identity = read_json(identity_path)
                if (not completed or row["report_ref"] != freeze["source_report_ref"]
                        or row["sequence_ref"] != freeze["source_sequence_ref"]
                        or identity.get("kind") != "native-context-generation-identity"
                        or identity.get("version") != 1 or identity.get("uid") != freeze["uid"]
                        or identity.get("generation", {}).get("seed") != freeze["inference_seed"]
                        or identity.get("native_context_qualified") is not False
                        or identity.get("scientific_effect_qualification") is not False
                        or identity.get("instrument_code_sha256", {}).get(
                            "research_math/native_context_runner.py") != digest(implementation_path)):
                    raise ValueError("C05 B0 generation identity differs")
                pinned.append(row["generation_identity_ref"])
            elif (decision.get("selected_role") != "b_star"
                    or decision.get("selected_method_id") != row["method_id"]
                    or report.get("method_id") != row["method_id"]
                    or report.get("source_sequence_sha256") != freeze["source_sequence_ref"]["sha256"]
                    or report.get("source_report_sha256") != freeze["source_report_ref"]["sha256"]
                    or report.get("implementation_sha256") != digest(implementation_path)):
                raise ValueError("Physical C05 B* differs from prospective decision")
            arrays = _arrays(sequence_path) if completed else None
            if arrays is not None: _same_native(source, arrays)
            current = {"role": role, "method_id": row["method_id"],
                "report_ref": row["report_ref"], "sequence_ref": row["sequence_ref"],
                "implementation_ref": row["implementation_ref"],
                "implementation_sha256": row["implementation_ref"]["sha256"],
                "case_id": freeze["uid"] + "-" + role if completed else None,
                "preparation_status": status}
            if role == "b0": current["generation_identity_ref"] = row["generation_identity_ref"]
            if not completed:
                current.update(exception_type=report["exception_type"], error=report["error"],
                    preparation_error=report["exception_type"] + ": " + report["error"])
            pinned.extend([row["report_ref"], row["implementation_ref"]])
            if completed: pinned.append(row["sequence_ref"])
        normalized.append(current)
        if arrays is not None:
            physical[role] = current; content[role] = _content_digest(arrays)

    canonical = {}; scoring_cases = []
    for role in ROLES:
        item = physical.get(role)
        if item is None: continue
        existing = canonical.get(content[role])
        if existing is None:
            canonical[content[role]] = item; scoring_cases.append(item)
        else:
            row = next(value for value in normalized if value["role"] == role)
            row.update(case_id=existing["case_id"], alias_of=existing["role"],
                       shared_physical_sequence=True)
    if explicit_alias is not None:
        row = next(value for value in normalized if value["role"] == "b_star")
        target = next(value for value in normalized if value["role"] == explicit_alias["alias_of"])
        if explicit_alias["method_id"] != target["method_id"]:
            raise ValueError("Aliased C05 B* method_id must match its target")
        row.update(case_id=target.get("case_id"), preparation_status=target.get("preparation_status"),
                   shared_physical_sequence=True)
        if target.get("preparation_status") == "error":
            row.update(exception_type=target["exception_type"], error=target["error"],
                       preparation_error=target["preparation_error"])

    unique = {}
    for ref in pinned:
        plain = _plain_ref(ref); previous = unique.get(plain["path"])
        if previous is not None and previous != plain:
            raise ValueError("Conflicting C05 closure identity: " + plain["path"])
        unique[plain["path"]] = plain
    request = {"kind": REQUEST_KIND, "version": 1, "candidate_id": CANDIDATE_ID,
        "uid": freeze["uid"], "inference_seed": freeze["inference_seed"],
        "scoring_seed": 44, "primary_metric": PRIMARY_METRIC,
        "guardrail_metrics": list(GUARDRAIL_METRICS), "roles": normalized,
        "role_to_case": {row["role"]: row.get("case_id") for row in normalized},
        "logical_denominator": {"n_roles": len(ROLES),
            "roles": [{"role": row["role"], "preparation_status": row.get("preparation_status"),
                       "case_id": row.get("case_id"), "alias_of": row.get("alias_of")}
                      for row in normalized],
            "failure_policy": "Every frozen role remains; failures are never dropped or zero-imputed."},
        "scoring_cases": scoring_cases, "freeze_ref": file_ref(root, freeze_path),
        "b_star_decision_ref": freeze["b_star_decision_ref"],
        "candidate_artifact_ref": freeze["candidate_artifact_ref"],
        "candidate_evidence_sentinel_ref": evidence_sentinel_ref,
        "candidate_evidence_sentinel_relative": evidence_relative,
        "source_sequence_ref": freeze["source_sequence_ref"],
        "source_report_ref": freeze["source_report_ref"],
        "candidate_outer_seed": candidate["outer_seed"],
        "candidate_role_denominator": candidate["role_denominator"],
        "candidate_roles_completed": candidate["roles_completed"],
        "candidate_roles_failed": candidate["roles_failed"],
        "candidate_input_refs": candidate["input_refs"],
        "candidate_certificate_ref": artifact_roles[CANDIDATE_ROLE]["shared_certificate_ref"],
        "candidate_solver_certificate_ref": artifact_roles[CANDIDATE_ROLE]["shared_solver_certificate_ref"],
        "input_refs": list(unique.values()), **FALSE_SCOPE}
    request["request_digest"] = canonical_digest(request)
    if _verify: verify_request(root, request)
    return request


def verify_request(root: Path, request: Mapping[str, Any]) -> None:
    core = {key: value for key, value in request.items() if key != "request_digest"}
    if request.get("request_digest") != canonical_digest(core):
        raise ValueError("C05 request digest mismatch")
    if (request.get("kind") != REQUEST_KIND or request.get("version") != 1
            or request.get("candidate_id") != CANDIDATE_ID
            or request.get("inference_seed") not in ALLOWED_INFERENCE_SEEDS
            or request.get("scoring_seed") != 44
            or any(request.get(key) != value for key, value in FALSE_SCOPE.items())):
        raise ValueError("Unexecuted exact C05 request scope required")
    if [row.get("role") for row in request.get("roles", [])] != list(ROLES):
        raise ValueError("Exactly seven ordered C05 request roles required")
    if (request.get("logical_denominator", {}).get("n_roles") != len(ROLES)
            or set(request.get("role_to_case", {})) != set(ROLES)):
        raise ValueError("Complete seven-role C05 denominator required")
    refs = request.get("input_refs")
    if not isinstance(refs, list) or len({ref.get("path") for ref in refs}) != len(refs):
        raise ValueError("Unique pinned C05 request closure required")
    pinned = {ref["path"]: ref for ref in refs}
    for ref in refs: resolve_ref(root, ref)
    for key in ("freeze_ref", "b_star_decision_ref", "candidate_artifact_ref",
                "candidate_evidence_sentinel_ref",
                "source_sequence_ref", "source_report_ref", "candidate_certificate_ref",
                "candidate_solver_certificate_ref"):
        if pinned.get(request[key]["path"]) != request[key]:
            raise ValueError("C05 request closure missing: " + key)
    relative = _safe_relative(request.get("candidate_evidence_sentinel_relative"))
    sentinel = resolve_ref(root, request["candidate_evidence_sentinel_ref"])
    evidence_root = sentinel
    for _ in relative.parts:
        evidence_root = evidence_root.parent
    if (evidence_root / relative).resolve() != sentinel:
        raise ValueError("C05 candidate evidence-root binding differs")
    expected = make_request(
        root, freeze_path=resolve_ref(root, request["freeze_ref"]),
        artifact_evidence_root=evidence_root, _verify=False)
    if request != expected:
        raise ValueError("Request differs from current frozen C05 construction")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("request",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv); request = make_request(args.root, freeze_path=args.freeze)
    output = args.output.resolve(); output.relative_to(args.root.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(request, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"request": str(output), "request_digest": request["request_digest"],
        "execution_started": False, "native_qualified": False,
        "dispatch_ready": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
