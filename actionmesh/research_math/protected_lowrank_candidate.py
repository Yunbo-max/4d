"""C15: protected-subspace low-rank repair of a complete native trajectory.

The float64 construction keeps a fixed, explicitly supplied temporal subspace
to a checked numerical tolerance and applies singular-value soft thresholding
only to its orthogonal residual; the float32 native export enforces a frozen
relative coefficient/projector tolerance.  The
first basis vector must be the exact frame-zero vector ``e0``; hence the native
anchor is part of the optimization constraint, not a post-hoc pin.  Two
operation-isolating controls share only the mandatory anchor projector: anchor-
complement SVT with the same lambda, and anchor-complement truncated SVD whose
full exported numeric rank matches the candidate under a frozen relative rule.

This module consumes one retained completed native ``sequence.npz`` and never
loads a model, GT, camera, label, scorer state, or confirmation outcome.  It does
not score or scientifically admit C15.  Run it only as an inner command of the
reviewed research-autopilot harness.  Web-authored source remains
``generated_unexecuted`` until Local acceptance.
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


CANDIDATE_ID = "4d-math-20261006-c15"
CANDIDATE_ARM = "protected_residual_svt"
UNPROTECTED_ARM = "unprotected_svt"
RANK_MATCHED_ARM = "rank_matched_tsvd"
ARMS = (CANDIDATE_ARM, UNPROTECTED_ARM, RANK_MATCHED_ARM)
METHOD_IDS = {
    CANDIDATE_ARM: CANDIDATE_ID,
    UNPROTECTED_ARM: "classical-unprotected-svt-control",
    RANK_MATCHED_ARM: "rank-matched-truncated-svd-control",
}
ALLOWED_BASIS_POLICIES = {
    "analytic_anchored_dct",
    "development_only_frozen",
    "input_only_geometry_predeclared",
}
RANK_POLICY = "candidate_export_numeric_rank"
ERROR_TYPE_LIMIT = 256
ERROR_MESSAGE_LIMIT = 4096
TRACEBACK_LIMIT = 16384


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _positive_integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_failure(error: Exception, traceback_text: str, phase: str) -> dict:
    if phase not in {
            "protected_residual_svt_compute", "unprotected_svt_compute",
            "rank_matched_tsvd_compute", "protected_residual_svt_float32_eligibility",
            "unprotected_svt_float32_eligibility", "rank_matched_tsvd_float32_eligibility"}:
        raise ValueError("Unsupported terminal failure phase")
    return {
        "failure_phase": phase,
        "exception_type": type(error).__name__[:ERROR_TYPE_LIMIT],
        "error": (str(error) or "<empty exception message>")[:ERROR_MESSAGE_LIMIT],
        "traceback": traceback_text[-TRACEBACK_LIMIT:],
    }


def _artifact_member_paths(output: Path, reports: dict[str, dict]) -> list[Path]:
    members = [output / "candidate.json", output / "manifest.json"]
    for arm in ARMS:
        report = reports[arm]
        directory = output / arm
        members.append(directory / "report.json")
        if report["status"] == "completed":
            members.extend((directory / "sequence.npz", directory / "certificate.npz"))
    return sorted(members, key=lambda path: path.relative_to(output).as_posix())


def _write_deterministic_archive(output: Path, reports: dict[str, dict],
                                 max_artifact_bytes: int) -> dict:
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    if max_artifact_bytes < 10240:
        raise ValueError("max_artifact_bytes must allow one tar record")
    if max_artifact_bytes > 1024 * 1024 * 1024:
        raise ValueError("max_artifact_bytes must not exceed the 1 GiB member/closure ceiling")
    members = _artifact_member_paths(output, reports)
    estimated = 1024
    refs = []
    for path in members:
        size = path.stat().st_size
        estimated += 512 + ((size + 511) // 512) * 512
        refs.append({
            "path": path.relative_to(output).as_posix(),
            "sha256": _digest(path), "size_bytes": size,
        })
    estimated = ((estimated + 10239) // 10240) * 10240
    if estimated > max_artifact_bytes:
        raise RuntimeError(
            f"C15 artifact upper bound {estimated} exceeds max_artifact_bytes={max_artifact_bytes}")
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
        raise RuntimeError("Written C15 artifact archive exceeds max_artifact_bytes")
    temporary.replace(archive)
    record = {
        "kind": "c15-terminal-artifact-archive", "version": 1,
        "archive": {"path": "artifact.tar", "sha256": _digest(archive),
                    "size_bytes": archive.stat().st_size},
        "members": refs, "max_artifact_bytes": max_artifact_bytes,
        "terminal_candidate_status": json.loads(
            (output / "candidate.json").read_text())["status"],
    }
    _json(output / "artifact-archive.json", record)
    return record


def _validate_artifact_archive(output: Path, reports: dict[str, dict]) -> dict:
    record_path = output / "artifact-archive.json"
    archive_path = output / "artifact.tar"
    record = json.loads(record_path.read_text())
    expected_members = [{
        "path": path.relative_to(output).as_posix(),
        "sha256": _digest(path), "size_bytes": path.stat().st_size,
    } for path in _artifact_member_paths(output, reports)]
    if (record.get("kind") != "c15-terminal-artifact-archive"
            or record.get("version") != 1
            or record.get("members") != expected_members
            or record.get("terminal_candidate_status")
            != json.loads((output / "candidate.json").read_text()).get("status")
            or not isinstance(record.get("max_artifact_bytes"), int)
            or record["max_artifact_bytes"] < archive_path.stat().st_size
            or record.get("archive") != {
                "path": "artifact.tar", "sha256": _digest(archive_path),
                "size_bytes": archive_path.stat().st_size}):
        raise ValueError("C15 terminal artifact archive record mismatch")
    with tarfile.open(archive_path, mode="r:") as bundle:
        members = bundle.getmembers()
        if [member.name for member in members] != [row["path"] for row in expected_members]:
            raise ValueError("C15 archive member inventory/order mismatch")
        for member, expected in zip(members, expected_members):
            if (not member.isfile() or member.size != expected["size_bytes"]
                    or member.mode != 0o644 or member.uid != 0 or member.gid != 0
                    or member.mtime != 0 or member.uname or member.gname):
                raise ValueError("C15 archive metadata is not canonical")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("C15 archive regular member is unreadable")
            digest = hashlib.sha256()
            remaining = member.size
            while remaining:
                block = stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("C15 archive member is truncated")
                digest.update(block)
                remaining -= len(block)
            if stream.read(1) or digest.hexdigest() != expected["sha256"]:
                raise ValueError("C15 archive member bytes mismatch")
    return {
        "record_sha256": _digest(record_path),
        "archive_sha256": _digest(archive_path),
        "max_artifact_bytes": record["max_artifact_bytes"],
    }


def _positive(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _trajectory(value) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 3
            or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()):
        raise ValueError("Expected finite floating trajectory U[T,D] with T>=2 and D>=3")
    return array


def _economy_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Economy decomposition whose temporal side is at most T by T (T=16 native)."""
    value = _trajectory(matrix).astype(np.float64, copy=False)
    left, singular, right_t = np.linalg.svd(value, full_matrices=False)
    if (left.shape != (value.shape[0], min(value.shape))
            or right_t.shape != (min(value.shape), value.shape[1])
            or not np.isfinite(singular).all()):
        raise ValueError("Invalid economy SVD result")
    return left, singular, right_t


def _numeric_rank(matrix: np.ndarray, tolerance: float) -> int:
    tolerance = _positive("numeric_rank_tolerance", tolerance)
    if tolerance > 1e-3:
        raise ValueError("numeric_rank_tolerance is relative and must be <=1e-3")
    singular = np.linalg.svd(_trajectory(matrix).astype(np.float64, copy=False),
                             compute_uv=False)
    if singular[0] == 0.0:
        return 0
    return int(np.count_nonzero(singular > tolerance * singular[0]))


def validate_temporal_basis(basis, frame_count: int,
                            orthogonality_tolerance: float) -> tuple[np.ndarray, dict]:
    """Require a declared orthonormal basis in exact anchor-first form.

    No QR repair is performed: changing the supplied basis here would silently
    change the method.  Exact ``e0`` structure also makes the frame-zero row of
    the complementary projector identically zero in floating arithmetic.
    """
    tolerance = _positive("orthogonality_tolerance", orthogonality_tolerance)
    if tolerance > 1e-8:
        raise ValueError("orthogonality_tolerance must be <=1e-8 for the reviewed projector")
    value = np.asarray(basis)
    if (value.ndim != 2 or value.shape[0] != frame_count
            or not 2 <= value.shape[1] < frame_count
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()):
        raise ValueError("Basis must be finite floating B[T,K] with 2<=K<T")
    value = value.astype(np.float64, copy=True)
    e0 = np.zeros(frame_count, dtype=np.float64)
    e0[0] = 1.0
    if not np.array_equal(value[:, 0], e0):
        raise ValueError("First basis column must be exact e0")
    if value.shape[1] > 1 and not np.array_equal(value[0, 1:], np.zeros(value.shape[1] - 1)):
        raise ValueError("Every non-anchor basis vector must be exactly zero at frame zero")
    gram = value.T @ value
    gram_error = float(np.linalg.norm(gram - np.eye(value.shape[1]), ord=2))
    gram_condition = float(np.linalg.cond(gram))
    if gram_error > tolerance:
        raise ValueError("Supplied temporal basis is not orthonormal within frozen tolerance")
    if not math.isfinite(gram_condition) or gram_condition > 1.0 / math.sqrt(np.finfo(float).eps):
        raise ValueError("Supplied temporal basis Gram matrix is ill-conditioned")
    projector = value @ np.linalg.solve(gram, value.T)
    if (not np.array_equal(projector[0], e0)
            or not np.array_equal(projector[:, 0], e0)):
        raise ValueError("Basis does not induce an exact frame-zero anchor projector")
    return value, {
        "frame_count": frame_count,
        "basis_rank": int(value.shape[1]),
        "complement_dimension": int(frame_count - value.shape[1]),
        "orthogonality_tolerance": tolerance,
        "gram_spectral_error": gram_error,
        "gram_condition": gram_condition,
        "anchor_column_exact": True,
        "projector_anchor_exact": True,
    }


def _project_to_basis(basis: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Orthogonal span projector ``B(B^T B)^-1 B^T`` without assuming exact Gram I."""
    basis = np.asarray(basis, dtype=np.float64)
    value = _trajectory(matrix).astype(np.float64, copy=False)
    if basis.ndim != 2 or basis.shape[0] != value.shape[0]:
        raise ValueError("Basis/projected trajectory shape mismatch")
    gram = basis.T @ basis
    coefficients = np.linalg.solve(gram, basis.T @ value)
    projected = basis @ coefficients
    if not np.isfinite(projected).all():
        raise ValueError("Nonfinite protected-subspace projection")
    return projected


def build_anchored_dct_basis(frame_count: int, basis_rank: int) -> np.ndarray:
    """Build exact e0 plus orthonormal DCT-II modes on frames 1..T-1.

    The construction is analytic and contains no trajectory, label, scorer, GT,
    or confirmation-dependent quantity.  ``basis_rank`` is therefore the only
    prospective structural choice for this default legal Q builder.
    """
    if (isinstance(frame_count, bool) or not isinstance(frame_count, int)
            or frame_count < 2):
        raise ValueError("frame_count must be an integer >=2")
    if (isinstance(basis_rank, bool) or not isinstance(basis_rank, int)
            or not 2 <= basis_rank < frame_count):
        raise ValueError("basis_rank must be an integer in [2, frame_count)")
    basis = np.zeros((frame_count, basis_rank), dtype=np.float64)
    basis[0, 0] = 1.0
    tail = frame_count - 1
    indices = np.arange(tail, dtype=np.float64)
    for column in range(basis_rank - 1):
        scale = math.sqrt(1.0 / tail) if column == 0 else math.sqrt(2.0 / tail)
        basis[1:, column + 1] = scale * np.cos(
            math.pi * (indices + 0.5) * column / tail)
    return basis


def split_protected_residual(trajectory, basis,
                             orthogonality_tolerance: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Derivation s1: ``U = P U + (I-P)U``, ``P=B(B^TB)^-1B^T``.

    The explicit anchor structure is exact; subspace feasibility is measured and
    retained under the frozen numerical tolerances.
    """
    source = _trajectory(trajectory).astype(np.float64, copy=False)
    temporal_basis, basis_diagnostics = validate_temporal_basis(
        basis, source.shape[0], orthogonality_tolerance)
    protected = _project_to_basis(temporal_basis, source)
    residual = source - protected
    if not np.array_equal(protected[0], source[0]) or np.count_nonzero(residual[0]) != 0:
        raise ValueError("Protected split failed exact anchor identity")
    reconstruction_error = float(np.linalg.norm(protected + residual - source))
    complement_error = float(np.linalg.norm(_project_to_basis(temporal_basis, residual)))
    diagnostics = {
        **basis_diagnostics,
        "split_reconstruction_frobenius": reconstruction_error,
        "residual_protected_overlap_frobenius": complement_error,
        "anchor_exact": True,
    }
    return protected, residual, diagnostics


def _spectral_prox_violation(singular: np.ndarray, shrunk: np.ndarray,
                             lambda_value: float) -> float:
    active = shrunk > 0.0
    violations = []
    if np.any(active):
        violations.append(float(np.max(np.abs(
            singular[active] - shrunk[active] - lambda_value))))
    if np.any(~active):
        violations.append(float(np.max(np.maximum(
            singular[~active] - lambda_value, 0.0))))
    return max(violations, default=0.0)


def solve_residual_svt(trajectory, basis, *, lambda_value: float,
                       orthogonality_tolerance: float) -> tuple[np.ndarray, dict, dict]:
    """Derivations s2/s3: residual SVT plus numerically checked feasible recombination."""
    lambda_value = _positive("lambda_value", lambda_value)
    source = _trajectory(trajectory).astype(np.float64, copy=False)
    temporal_basis, _ = validate_temporal_basis(
        basis, source.shape[0], orthogonality_tolerance)
    protected, observed, split = split_protected_residual(
        source, temporal_basis, orthogonality_tolerance)
    left, singular, right_t = _economy_svd(observed)
    shrunk = np.maximum(singular - lambda_value, 0.0)
    unconstrained_repair = (left * shrunk) @ right_t
    repaired = unconstrained_repair - _project_to_basis(
        temporal_basis, unconstrained_repair)
    output = protected + repaired
    if not np.array_equal(output[0], source[0]):
        raise ValueError("Protected residual SVT changed the exact anchor")
    protected_drift = float(np.linalg.norm(_project_to_basis(
        temporal_basis, output - source)))
    coefficient_drift = float(np.linalg.norm(
        temporal_basis.T @ output - temporal_basis.T @ source))
    complement_error = float(np.linalg.norm(_project_to_basis(temporal_basis, repaired)))
    output_singular = np.linalg.svd(repaired, compute_uv=False)
    data_term = 0.5 * float(np.sum((repaired - observed) ** 2))
    nuclear_term = lambda_value * float(np.sum(output_singular))
    active_rank = int(np.count_nonzero(shrunk > 0.0))
    report = {
        "operator": "protected_residual_nuclear_norm_prox",
        "svd_algorithm": "numpy_economy_svd_on_T_by_3V; native T=16",
        "lambda": lambda_value,
        "active_residual_rank": active_rank,
        "protected_basis_rank": int(temporal_basis.shape[1]),
        "objective": data_term + nuclear_term,
        "data_fidelity_term": data_term,
        "weighted_nuclear_term": nuclear_term,
        "input_residual_nuclear_norm": float(np.sum(singular)),
        "output_residual_nuclear_norm": float(np.sum(output_singular)),
        "preprojection_spectral_soft_threshold_identity_violation": _spectral_prox_violation(
            singular, shrunk, lambda_value),
        "feasible_projection_frobenius": float(np.linalg.norm(
            repaired - unconstrained_repair)),
        "protected_projector_drift_frobenius": protected_drift,
        "protected_coefficient_drift_frobenius": coefficient_drift,
        "repaired_residual_protected_overlap_frobenius": complement_error,
        "anchor_exact": True,
        "split": split,
    }
    certificate = {
        "temporal_basis": temporal_basis,
        "protected_component": protected,
        "observed_residual": observed,
        "repaired_residual": repaired,
        "input_singular_values": singular,
        "thresholded_singular_values": shrunk,
        "output_singular_values": output_singular,
    }
    return output, report, certificate


def _anchor_only_split(trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Split by the mandatory anchor-only projector ``P0=e0 e0^T``."""
    source = _trajectory(trajectory).astype(np.float64, copy=False)
    protected = np.zeros_like(source)
    protected[0] = source[0]
    residual = source - protected
    if not np.array_equal(protected[0], source[0]) or np.count_nonzero(residual[0]) != 0:
        raise ValueError("Anchor-only temporal split failed exact frame-zero identity")
    return protected, residual


def solve_unprotected_svt(trajectory, *, lambda_value: float) -> tuple[np.ndarray, dict, dict]:
    """Anchor-only SVT control with the candidate's same lambda.

    ``P0 U`` is retained and SVT is applied to ``(I-P0)U``.  Thus frame zero is
    a shared optimization constraint while the candidate's additional action
    subspace remains absent from this control.
    """
    lambda_value = _positive("lambda_value", lambda_value)
    source = _trajectory(trajectory).astype(np.float64, copy=False)
    anchor, observed = _anchor_only_split(source)
    left, singular, right_t = _economy_svd(observed)
    shrunk = np.maximum(singular - lambda_value, 0.0)
    unconstrained = (left * shrunk) @ right_t
    repaired = unconstrained.copy()
    repaired[0] = 0.0
    output = anchor + repaired
    if not np.array_equal(output[0], source[0]):
        raise ValueError("Anchor-only SVT control changed frame zero")
    output_singular = np.linalg.svd(repaired, compute_uv=False)
    data_term = 0.5 * float(np.sum((repaired - observed) ** 2))
    nuclear_term = lambda_value * float(np.sum(output_singular))
    report = {
        "operator": "anchor_only_unprotected_action_nuclear_norm_prox",
        "svd_algorithm": "numpy_economy_svd_on_T_by_3V; native T=16",
        "lambda": lambda_value,
        "active_rank": int(np.count_nonzero(shrunk > 0.0)),
        "objective": data_term + nuclear_term,
        "data_fidelity_term": data_term,
        "weighted_nuclear_term": nuclear_term,
        "spectral_prox_kkt_violation": _spectral_prox_violation(
            singular, shrunk, lambda_value),
        "anchor_drift_frobenius": 0.0,
        "anchor_protected": True,
        "additional_action_subspace_protected": False,
        "feasible_projection_frobenius": float(np.linalg.norm(repaired - unconstrained)),
    }
    certificate = {
        "anchor_component": anchor,
        "observed_anchor_complement": observed,
        "input_singular_values": singular,
        "thresholded_singular_values": shrunk,
        "output_singular_values": output_singular,
    }
    return output, report, certificate


def solve_rank_matched_truncated_svd(trajectory, *, target_numeric_rank: int,
                                     numeric_rank_tolerance: float,
                                     residual_rank_policy: str) -> tuple[np.ndarray, dict, dict]:
    """Anchor-only hard-rank action control fixed by the prospective C15 policy."""
    if residual_rank_policy != RANK_POLICY:
        raise ValueError(f"residual_rank_policy must be {RANK_POLICY}")
    for name, value in (("target_numeric_rank", target_numeric_rank),):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    source = _trajectory(trajectory).astype(np.float64, copy=False)
    numeric_rank_tolerance = _positive("numeric_rank_tolerance", numeric_rank_tolerance)
    if numeric_rank_tolerance > 1e-3:
        raise ValueError("numeric_rank_tolerance is relative and must be <=1e-3")
    if target_numeric_rank > source.shape[0]:
        raise ValueError("target_numeric_rank exceeds temporal dimension")
    anchor, observed = _anchor_only_split(source)
    left, singular, right_t = _economy_svd(observed)
    selected = None
    for action_rank in range(source.shape[0]):
        candidate = ((left[:, :action_rank] * singular[:action_rank]) @ right_t[:action_rank]
                     if action_rank else np.zeros_like(observed))
        candidate[0] = 0.0
        proposed = anchor + candidate
        if _numeric_rank(proposed, numeric_rank_tolerance) == target_numeric_rank:
            selected = action_rank, candidate, proposed
            break
    if selected is None:
        raise ValueError("No anchor-only TSVD rank realizes the frozen candidate numeric rank")
    rank, repaired, output = selected
    if not np.array_equal(output[0], source[0]):
        raise ValueError("Anchor-only rank-matched control changed frame zero")
    error = 0.5 * float(np.sum((repaired - observed) ** 2))
    tail_energy = 0.5 * float(np.sum(singular[rank:] ** 2))
    report = {
        "operator": "anchor_only_unprotected_action_rank_matched_truncated_svd",
        "svd_algorithm": "numpy_economy_svd_on_T_by_3V; native T=16",
        "residual_rank_policy": residual_rank_policy,
        "target_candidate_numeric_rank": target_numeric_rank,
        "numeric_rank_relative_tolerance": numeric_rank_tolerance,
        "numeric_rank_threshold_formula": "s_i > relative_tolerance * s_0",
        "matched_action_rank": rank,
        "output_numeric_rank": _numeric_rank(output, numeric_rank_tolerance),
        "objective_half_squared_error": error,
        "eckart_young_tail_energy": tail_energy,
        "eckart_young_certificate_abs_error": abs(error - tail_energy),
        "anchor_drift_frobenius": 0.0,
        "anchor_protected": True,
        "additional_action_subspace_protected": False,
    }
    certificate = {
        "anchor_component": anchor,
        "observed_anchor_complement": observed,
        "input_singular_values": singular,
        "retained_singular_values": singular[:rank],
        "discarded_singular_values": singular[rank:],
        "matched_action_rank": np.asarray([rank], dtype=np.int64),
        "target_candidate_numeric_rank": np.asarray([target_numeric_rank], dtype=np.int64),
    }
    return output, report, certificate


def _load_basis(project_root: Path, basis_path: Path, expected_basis_sha256: str, *, frame_count: int,
                expected_sequence_sha256: str,
                basis_source_policy: str, basis_evidence: Path | None,
                expected_basis_evidence_sha256: str | None,
                orthogonality_tolerance: float) -> tuple[np.ndarray, dict]:
    project_root = Path(project_root).resolve()
    basis_path = Path(basis_path)
    if basis_path.suffix != ".npy" or _digest(basis_path) != expected_basis_sha256:
        raise ValueError("Explicit .npy basis differs from pinned basis identity")
    if basis_source_policy not in ALLOWED_BASIS_POLICIES:
        raise ValueError("Unsupported basis source policy")
    basis = np.load(basis_path, allow_pickle=False)
    basis, diagnostics = validate_temporal_basis(
        basis, frame_count, orthogonality_tolerance)
    if basis_evidence is None or expected_basis_evidence_sha256 is None:
        raise ValueError("Every C15 run requires hash-bound prospective basis evidence")
    basis_evidence = Path(basis_evidence)
    evidence_digest = _digest(basis_evidence)
    if evidence_digest != expected_basis_evidence_sha256:
        raise ValueError("Basis evidence differs from pinned identity")
    evidence = json.loads(basis_evidence.read_text())
    source_refs = evidence.get("source_refs")
    frozen_at = evidence.get("frozen_at")
    try:
        parsed_frozen_at = datetime.fromisoformat(frozen_at)
    except (TypeError, ValueError) as error:
        raise ValueError("Basis evidence requires an ISO-8601 frozen_at") from error
    if (parsed_frozen_at.tzinfo is None
            or evidence.get("kind") != "c15-protection-basis-evidence"
            or evidence.get("version") != "1.0.0"
            or evidence.get("status") != "completed"
            or evidence.get("basis_sha256") != expected_basis_sha256
            or evidence.get("basis_source_policy") != basis_source_policy
            or evidence.get("confirmation_outcomes_used") is not False
            or evidence.get("prospective_freeze") is not True
            or not isinstance(source_refs, list) or not source_refs):
        raise ValueError("Basis evidence does not bind a prospective legal source policy")
    for ref in source_refs:
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str) or not ref["path"]
                or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64):
            raise ValueError("Basis evidence source_refs must be exact path/SHA-256 rows")
        relative_ref = Path(ref["path"])
        if relative_ref.is_absolute() or ".." in relative_ref.parts:
            raise ValueError("Basis evidence source ref must be project-relative")
        cursor = project_root
        for part in relative_ref.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Basis evidence source ref path must not contain symlinks")
        source_path = (project_root / relative_ref).resolve()
        try:
            source_path.relative_to(project_root)
        except ValueError as error:
            raise ValueError("Basis evidence source ref escapes project root") from error
        if not source_path.is_file() or _digest(source_path) != ref["sha256"]:
            raise ValueError("Basis evidence source ref is missing or stale: " + ref["path"])
    if basis_source_policy == "analytic_anchored_dct":
        if (evidence.get("construction_rule")
                != "exact_e0_plus_orthonormal_dct_ii_on_frames_1_to_T_minus_1"
                or evidence.get("frame_count") != frame_count
                or evidence.get("basis_rank") != int(basis.shape[1])
                or evidence.get("input_sequence_sha256") is not None):
            raise ValueError("Analytic DCT evidence does not bind its prospective construction")
        expected_basis = build_anchored_dct_basis(frame_count, int(basis.shape[1]))
        if not np.array_equal(basis, expected_basis):
            raise ValueError("Basis file is not the exact declared anchored DCT construction")
    elif basis_source_policy == "input_only_geometry_predeclared":
        if evidence.get("input_sequence_sha256") != expected_sequence_sha256:
            raise ValueError("Input-only basis evidence must bind this retained prediction")
    elif evidence.get("development_source_refs") != source_refs:
        raise ValueError("Development-only basis evidence must identify its frozen development refs")
    diagnostics.update({
        "basis_sha256": expected_basis_sha256,
        "basis_source_policy": basis_source_policy,
        "basis_evidence_sha256": evidence_digest,
        "confirmation_outcomes_used": False,
        "basis_frozen_at": frozen_at,
        "basis_source_refs": source_refs,
    })
    return basis, diagnostics


def _load_native_case(source_sequence: Path, uid: str, expected_sequence_sha256: str):
    source_sequence = Path(source_sequence)
    if source_sequence.name != "sequence.npz" or _digest(source_sequence) != expected_sequence_sha256:
        raise ValueError("Sequence differs from explicitly pinned input")
    report_path = source_sequence.with_name("report.json")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "completed" or report.get("uid") != uid
            or report.get("sha256", {}).get("sequence.npz") != expected_sequence_sha256):
        raise ValueError("Completed source report for the same pinned UID required")
    seed = report.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Integer source inference seed required")
    with np.load(source_sequence, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    required = {"vertices", "faces", "timesteps", "frame_indices", "query_vertex_ids"}
    if not required.issubset(arrays):
        raise ValueError("Complete native sequence metadata required")
    vertices = arrays["vertices"]
    if (vertices.dtype != np.float32 or vertices.ndim != 3
            or vertices.shape[0] != 16 or vertices.shape[-1] != 3
            or vertices.shape[1] < 3 or not np.isfinite(vertices).all()):
        raise ValueError("C15 requires one complete finite float32 16-frame native mesh")
    faces = arrays["faces"]
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0
            or faces.max() >= vertices.shape[1]):
        raise ValueError("Valid shared triangle topology required")
    if (not np.issubdtype(arrays["frame_indices"].dtype, np.integer)
            or not np.array_equal(arrays["frame_indices"], np.arange(16))):
        raise ValueError("Exactly 16 original frame indices in order required")
    times = arrays["timesteps"]
    if (times.shape != (16,) or not np.issubdtype(times.dtype, np.floating)
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0)):
        raise ValueError("Strictly increasing native timestamps required")
    if (not np.issubdtype(arrays["query_vertex_ids"].dtype, np.integer)
            or not np.array_equal(arrays["query_vertex_ids"], np.arange(vertices.shape[1]))):
        raise ValueError("Original identity vertex mapping required")
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("Non-numeric/nonfinite native array: " + name)
    return arrays, report, report_path


def _prepare_role_output(arm: str, matrix: np.ndarray, arrays: dict,
                         basis: np.ndarray, protected_coefficient_tolerance: float,
                         numeric_rank_tolerance: float) -> tuple[np.ndarray, dict, dict]:
    """Apply deterministic float32 eligibility checks before any artifact I/O."""
    vertices = matrix.reshape(arrays["vertices"].shape).astype(np.float32)
    if not np.isfinite(vertices).all() or vertices.shape != arrays["vertices"].shape:
        raise ValueError("C15 arm changed frame/vertex identity or became nonfinite")
    if not np.array_equal(vertices[0], arrays["vertices"][0]):
        raise ValueError("C15 arm changed the shared exact float32 anchor")
    exported_matrix = vertices.reshape(16, -1).astype(np.float64)
    source_matrix = arrays["vertices"].reshape(16, -1).astype(np.float64)
    coefficient_drift = float(np.linalg.norm(
        basis.T @ (exported_matrix - source_matrix)))
    coefficient_scale = max(1.0, float(np.linalg.norm(basis.T @ source_matrix)))
    coefficient_relative_drift = coefficient_drift / coefficient_scale
    projector_drift = float(np.linalg.norm(_project_to_basis(
        basis, exported_matrix - source_matrix)))
    projector_scale = max(1.0, float(np.linalg.norm(
        _project_to_basis(basis, source_matrix))))
    projector_relative_drift = projector_drift / projector_scale
    protected_relative_drift = max(coefficient_relative_drift, projector_relative_drift)
    if arm == CANDIDATE_ARM and protected_relative_drift > protected_coefficient_tolerance:
        raise ValueError("Float32 candidate exceeds frozen protected-invariance tolerance")
    diagnostics = {
        "protected_coefficient_drift_frobenius": coefficient_drift,
        "protected_coefficient_relative_drift": coefficient_relative_drift,
        "protected_projector_drift_frobenius": projector_drift,
        "protected_projector_relative_drift": projector_relative_drift,
        "protected_invariance_relative_drift": protected_relative_drift,
        "protected_coefficient_tolerance": protected_coefficient_tolerance,
        "numeric_rank": _numeric_rank(exported_matrix, numeric_rank_tolerance),
        "numeric_rank_relative_tolerance": numeric_rank_tolerance,
        "numeric_rank_threshold_formula": "s_i > relative_tolerance * s_0",
    }
    identity = {
        "frames": 16, "vertices": int(vertices.shape[1]),
        "faces_preserved": True, "frame_indices_preserved": True,
        "timestamps_preserved": True, "vertex_ids_preserved": True,
        "float32_vertices": True, "anchor_exact": True,
        "coordinate_clipping": False, "fallback_used": False,
    }
    return vertices, diagnostics, identity


def export_protected_lowrank_candidate(
        source_sequence: Path, basis_path: Path, output: Path, *, uid: str,
        project_root: Path,
        expected_sequence_sha256: str, expected_basis_sha256: str,
        basis_source_policy: str, lambda_value: float,
        residual_rank_policy: str, orthogonality_tolerance: float,
        protected_coefficient_tolerance: float, numeric_rank_tolerance: float,
        max_artifact_bytes: int,
        basis_evidence: Path,
        expected_basis_evidence_sha256: str) -> dict:
    """Export the three C15 artifacts and complete certificates; never score them."""
    started = time.monotonic()
    if not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    lambda_value = _positive("lambda_value", lambda_value)
    protected_coefficient_tolerance = _positive(
        "protected_coefficient_tolerance", protected_coefficient_tolerance)
    if protected_coefficient_tolerance > 1e-4:
        raise ValueError("protected_coefficient_tolerance must be <=1e-4")
    numeric_rank_tolerance = _positive("numeric_rank_tolerance", numeric_rank_tolerance)
    if numeric_rank_tolerance > 1e-3:
        raise ValueError("numeric_rank_tolerance is relative and must be <=1e-3")
    max_artifact_bytes = _positive_integer("max_artifact_bytes", max_artifact_bytes)
    if residual_rank_policy != RANK_POLICY:
        raise ValueError(f"residual_rank_policy must be {RANK_POLICY}")
    arrays, source_report, source_report_path = _load_native_case(
        source_sequence, uid, expected_sequence_sha256)
    basis, basis_diagnostics = _load_basis(
        project_root, basis_path, expected_basis_sha256, frame_count=16,
        expected_sequence_sha256=expected_sequence_sha256,
        basis_source_policy=basis_source_policy, basis_evidence=basis_evidence,
        expected_basis_evidence_sha256=expected_basis_evidence_sha256,
        orthogonality_tolerance=orthogonality_tolerance)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "candidate_id": CANDIDATE_ID,
        "cases": [{
            "case_id": uid + "-" + arm, "uid": uid,
            "case_dir": arm, "arm_role": arm,
        } for arm in ARMS],
        "scope": "single C15 artifact comparison; no B0/B*/Gaussian selection, scorer, or scientific admission",
    }
    _json(output / "manifest.json", manifest)
    for arm in ARMS:
        (output / arm).mkdir()
    trajectory = arrays["vertices"].astype(np.float64).reshape(16, -1)
    computed: dict[str, tuple[np.ndarray, dict, dict]] = {}
    compute_errors: dict[str, tuple[Exception, str, str]] = {}
    try:
        candidate = solve_residual_svt(
            trajectory, basis, lambda_value=lambda_value,
            orthogonality_tolerance=orthogonality_tolerance)
        candidate_export = candidate[0].reshape(arrays["vertices"].shape).astype(np.float32)
        target_rank = _numeric_rank(
            candidate_export.reshape(16, -1).astype(np.float64), numeric_rank_tolerance)
        candidate[1]["export_numeric_rank"] = target_rank
        candidate[1]["numeric_rank_relative_tolerance"] = numeric_rank_tolerance
        candidate[1]["numeric_rank_threshold_formula"] = "s_i > relative_tolerance * s_0"
        computed[CANDIDATE_ARM] = candidate
    except Exception as error:
        compute_errors[CANDIDATE_ARM] = (
            error, traceback.format_exc(), "protected_residual_svt_compute")
    try:
        computed[UNPROTECTED_ARM] = solve_unprotected_svt(
            trajectory, lambda_value=lambda_value)
    except Exception as error:
        compute_errors[UNPROTECTED_ARM] = (
            error, traceback.format_exc(), "unprotected_svt_compute")
    try:
        if CANDIDATE_ARM not in computed:
            raise RuntimeError("Rank-matched control depends on candidate numeric-rank certificate")
        computed[RANK_MATCHED_ARM] = solve_rank_matched_truncated_svd(
            trajectory,
            target_numeric_rank=computed[CANDIDATE_ARM][1]["export_numeric_rank"],
            numeric_rank_tolerance=numeric_rank_tolerance,
            residual_rank_policy=residual_rank_policy)
    except Exception as error:
        compute_errors[RANK_MATCHED_ARM] = (
            error, traceback.format_exc(), "rank_matched_tsvd_compute")
    prepared = {}
    for arm, (matrix, _, _) in computed.items():
        try:
            prepared[arm] = _prepare_role_output(
                arm, matrix, arrays, basis, protected_coefficient_tolerance,
                numeric_rank_tolerance)
        except Exception as error:
            compute_errors[arm] = (
                error, traceback.format_exc(), arm + "_float32_eligibility")
    arm_reports = {}
    for arm in ARMS:
        arm_started = time.monotonic()
        arm_dir = output / arm
        report = {
            "uid": uid, "seed": source_report["seed"],
            "candidate_id": CANDIDATE_ID, "method_id": METHOD_IDS[arm],
            "arm_role": arm,
            "source_sequence_sha256": expected_sequence_sha256,
            "source_report_sha256": _digest(source_report_path),
            "basis_sha256": expected_basis_sha256,
            "basis_evidence_sha256": basis_diagnostics["basis_evidence_sha256"],
            "basis_source_policy": basis_source_policy,
            "residual_rank_policy": residual_rank_policy,
            "implementation_sha256": _digest(Path(__file__)),
            "parameters": {
                "lambda": lambda_value,
                "basis_orthogonality_tolerance": float(orthogonality_tolerance),
                "protected_coefficient_tolerance": protected_coefficient_tolerance,
                "numeric_rank_relative_tolerance": numeric_rank_tolerance,
                "residual_rank_policy": residual_rank_policy,
            },
            "basis_diagnostics": basis_diagnostics,
            "information": "Retained predicted mesh and declared basis only; no GT, camera, labels, scorer state, evaluator alignment, or confirmation outcomes",
            "native_qualified": False, "local_method_verified": False,
            "scientific_verdict": "not_computed", "generated_unexecuted": True,
        }
        status = "error"
        if arm in compute_errors:
            error, error_traceback, phase = compute_errors[arm]
            report.update(_bounded_failure(error, error_traceback, phase))
            report["status"] = status
            report["elapsed_seconds"] = time.monotonic() - arm_started
            report["timing_scope"] = "role materialization/certificate I/O after shared CPU decompositions"
            _json(arm_dir / "report.json", report)
            arm_reports[arm] = report
            continue
        try:
            matrix, diagnostics, certificate = computed[arm]
            vertices, export_diagnostics, output_identity = prepared[arm]
            np.savez_compressed(
                arm_dir / "sequence.npz", **{**arrays, "vertices": vertices})
            np.savez_compressed(arm_dir / "certificate.npz", **certificate)
            report["diagnostics"] = diagnostics
            report["export_diagnostics"] = export_diagnostics
            report["output_identity"] = output_identity
            report["sha256"] = {
                "sequence.npz": _digest(arm_dir / "sequence.npz"),
                "certificate.npz": _digest(arm_dir / "certificate.npz"),
            }
            status = "completed"
        except Exception as error:
            # Artifact I/O/hash failures are infrastructure failures.  They must
            # fail the harness attempt, not become denominator-valid method errors.
            for partial in (arm_dir / "sequence.npz", arm_dir / "certificate.npz"):
                if partial.exists():
                    partial.unlink()
            raise RuntimeError(f"C15 artifact materialization failed for {arm}") from error
        report["status"] = status
        report["elapsed_seconds"] = time.monotonic() - arm_started
        report["timing_scope"] = "role materialization/certificate I/O after shared CPU decompositions"
        _json(arm_dir / "report.json", report)
        arm_reports[arm] = report
    complete = all(row["status"] == "completed" for row in arm_reports.values())
    summary = {
        "status": "completed" if complete else "incomplete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": CANDIDATE_ID, "uid": uid, "seed": source_report["seed"],
        "arms": arm_reports, "native_qualified": False,
        "local_method_verified": False, "candidate_methods_tested": False,
        "scientific_verdict": "not_computed", "generated_unexecuted": True,
        "elapsed_seconds": time.monotonic() - started,
        "timing_scope": "CPU T=16 economy SVD + C15 artifact I/O; excludes generation/scorer/collection",
    }
    _json(output / "candidate.json", summary)
    _write_deterministic_archive(output, arm_reports, max_artifact_bytes)
    return summary


def validate_protected_lowrank_artifact(
        output: Path, source_sequence: Path, basis_path: Path, *, uid: str,
        project_root: Path,
        expected_sequence_sha256: str, expected_basis_sha256: str,
        basis_source_policy: str, lambda_value: float,
        residual_rank_policy: str, orthogonality_tolerance: float,
        protected_coefficient_tolerance: float, numeric_rank_tolerance: float,
        basis_evidence: Path,
        expected_basis_evidence_sha256: str) -> dict:
    """Recompute and validate every role without invoking any scorer.

    This is the canonical artifact-consumption boundary for later C15 comparison
    code.  It verifies content, not scientific admission or native qualification.
    """
    output = Path(output)
    arrays, source_report, _ = _load_native_case(
        source_sequence, uid, expected_sequence_sha256)
    basis, _ = _load_basis(
        project_root, basis_path, expected_basis_sha256, frame_count=16,
        expected_sequence_sha256=expected_sequence_sha256,
        basis_source_policy=basis_source_policy, basis_evidence=basis_evidence,
        expected_basis_evidence_sha256=expected_basis_evidence_sha256,
        orthogonality_tolerance=orthogonality_tolerance)
    protected_coefficient_tolerance = _positive(
        "protected_coefficient_tolerance", protected_coefficient_tolerance)
    if protected_coefficient_tolerance > 1e-4:
        raise ValueError("protected_coefficient_tolerance must be <=1e-4")
    numeric_rank_tolerance = _positive("numeric_rank_tolerance", numeric_rank_tolerance)
    if numeric_rank_tolerance > 1e-3:
        raise ValueError("numeric_rank_tolerance is relative and must be <=1e-3")
    trajectory = arrays["vertices"].astype(np.float64).reshape(16, -1)
    expected: dict[str, tuple[np.ndarray, dict, dict]] = {}
    recompute_errors = {}
    try:
        candidate_expected = solve_residual_svt(
            trajectory, basis, lambda_value=lambda_value,
            orthogonality_tolerance=orthogonality_tolerance)
        candidate_export = candidate_expected[0].reshape(
            arrays["vertices"].shape).astype(np.float32)
        target_rank = _numeric_rank(
            candidate_export.reshape(16, -1).astype(np.float64), numeric_rank_tolerance)
        candidate_expected[1]["export_numeric_rank"] = target_rank
        candidate_expected[1]["numeric_rank_relative_tolerance"] = numeric_rank_tolerance
        candidate_expected[1]["numeric_rank_threshold_formula"] = "s_i > relative_tolerance * s_0"
        expected[CANDIDATE_ARM] = candidate_expected
    except Exception as error:
        recompute_errors[CANDIDATE_ARM] = _bounded_failure(
            error, traceback.format_exc(), "protected_residual_svt_compute")
    try:
        expected[UNPROTECTED_ARM] = solve_unprotected_svt(
            trajectory, lambda_value=lambda_value)
    except Exception as error:
        recompute_errors[UNPROTECTED_ARM] = _bounded_failure(
            error, traceback.format_exc(), "unprotected_svt_compute")
    try:
        if CANDIDATE_ARM not in expected:
            raise RuntimeError("Rank-matched control depends on candidate numeric-rank certificate")
        expected[RANK_MATCHED_ARM] = solve_rank_matched_truncated_svd(
            trajectory,
            target_numeric_rank=expected[CANDIDATE_ARM][1]["export_numeric_rank"],
            numeric_rank_tolerance=numeric_rank_tolerance,
            residual_rank_policy=residual_rank_policy)
    except Exception as error:
        recompute_errors[RANK_MATCHED_ARM] = _bounded_failure(
            error, traceback.format_exc(), "rank_matched_tsvd_compute")
    expected_prepared = {}
    for arm, (matrix, _, _) in expected.items():
        try:
            expected_prepared[arm] = _prepare_role_output(
                arm, matrix, arrays, basis, protected_coefficient_tolerance,
                numeric_rank_tolerance)
        except Exception as error:
            recompute_errors[arm] = _bounded_failure(
                error, traceback.format_exc(), arm + "_float32_eligibility")
    manifest = json.loads((output / "manifest.json").read_text())
    expected_cases = [{
        "case_id": uid + "-" + arm, "uid": uid,
        "case_dir": arm, "arm_role": arm,
    } for arm in ARMS]
    if manifest.get("candidate_id") != CANDIDATE_ID or manifest.get("cases") != expected_cases:
        raise ValueError("C15 artifact manifest differs from the fixed three-role schema")
    implementation_sha256 = _digest(Path(__file__))
    expected_parameters = {
        "lambda": float(lambda_value),
        "basis_orthogonality_tolerance": float(orthogonality_tolerance),
        "protected_coefficient_tolerance": protected_coefficient_tolerance,
        "numeric_rank_relative_tolerance": numeric_rank_tolerance,
        "residual_rank_policy": residual_rank_policy,
    }
    validated = {}
    reports = {}
    for arm in ARMS:
        arm_dir = output / arm
        report = json.loads((arm_dir / "report.json").read_text())
        if (report.get("status") not in ("completed", "error") or report.get("uid") != uid
                or report.get("seed") != source_report["seed"]
                or report.get("candidate_id") != CANDIDATE_ID
                or report.get("method_id") != METHOD_IDS[arm]
                or report.get("arm_role") != arm
                or report.get("source_sequence_sha256") != expected_sequence_sha256
                or report.get("basis_sha256") != expected_basis_sha256
                or report.get("basis_source_policy") != basis_source_policy
                or report.get("residual_rank_policy") != residual_rank_policy
                or report.get("basis_evidence_sha256") != expected_basis_evidence_sha256
                or report.get("implementation_sha256") != implementation_sha256
                or report.get("parameters") != expected_parameters
                or report.get("generated_unexecuted") is not True
                or report.get("native_qualified") is not False
                or report.get("local_method_verified") is not False
                or report.get("scientific_verdict") != "not_computed"
                or not isinstance(report.get("elapsed_seconds"), (int, float))
                or not math.isfinite(report["elapsed_seconds"])
                or report["elapsed_seconds"] < 0.0
                or report.get("timing_scope")
                != "role materialization/certificate I/O after shared CPU decompositions"):
            raise ValueError("C15 role report identity mismatch: " + arm)
        reports[arm] = report
        sequence_path = arm_dir / "sequence.npz"
        certificate_path = arm_dir / "certificate.npz"
        if report["status"] == "error":
            expected_error = recompute_errors.get(arm)
            if (expected_error is None
                    or report.get("failure_phase") != expected_error["failure_phase"]
                    or report.get("exception_type") != expected_error["exception_type"]
                    or report.get("error") != expected_error["error"]
                    or sequence_path.exists() or certificate_path.exists()
                    or report.get("sha256") is not None
                    or not isinstance(report.get("exception_type"), str)
                    or len(report["exception_type"]) > ERROR_TYPE_LIMIT
                    or not isinstance(report.get("error"), str) or not report["error"]
                    or len(report["error"]) > ERROR_MESSAGE_LIMIT
                    or not isinstance(report.get("traceback"), str) or not report["traceback"]
                    or len(report["traceback"]) > TRACEBACK_LIMIT):
                raise ValueError("C15 failed role is not a bounded terminal error: " + arm)
            validated[arm] = {
                "status": "error", "error": report["error"],
                "failure_phase": report["failure_phase"],
                "report_sha256": _digest(arm_dir / "report.json"),
            }
            continue
        if arm not in expected:
            raise ValueError("Completed C15 role cannot be recomputed: " + arm)
        if arm not in expected_prepared:
            raise ValueError("Completed C15 role fails deterministic float32 eligibility: " + arm)
        if report.get("sha256") != {
                "sequence.npz": _digest(sequence_path),
                "certificate.npz": _digest(certificate_path)}:
            raise ValueError("C15 role output hash mismatch: " + arm)
        with np.load(sequence_path, allow_pickle=False) as saved:
            saved_arrays = {name: saved[name].copy() for name in saved.files}
        if set(saved_arrays) != set(arrays):
            raise ValueError("C15 output changed native sequence schema: " + arm)
        for name in arrays:
            expected_array = (expected_prepared[arm][0]
                              if name == "vertices" else arrays[name])
            if not np.array_equal(saved_arrays[name], expected_array):
                raise ValueError(f"C15 output differs from recomputation: {arm}/{name}")
        expected_export = expected_prepared[arm][1]
        if report.get("diagnostics") != expected[arm][1] or report.get("export_diagnostics") != expected_export:
            raise ValueError("C15 report diagnostics differ from recomputation: " + arm)
        if report.get("output_identity") != expected_prepared[arm][2]:
            raise ValueError("C15 output identity certificate mismatch: " + arm)
        if (arm == CANDIDATE_ARM
                and expected_export["protected_invariance_relative_drift"]
                > protected_coefficient_tolerance):
            raise ValueError("C15 completed candidate exceeds protected coefficient tolerance")
        with np.load(certificate_path, allow_pickle=False) as saved:
            if set(saved.files) != set(expected[arm][2]):
                raise ValueError("C15 certificate schema mismatch: " + arm)
            for name, expected_array in expected[arm][2].items():
                if not np.allclose(saved[name], expected_array, rtol=0.0, atol=0.0):
                    raise ValueError(f"C15 certificate differs from recomputation: {arm}/{name}")
        validated[arm] = {"status": "completed", **report["sha256"]}
    candidate = json.loads((output / "candidate.json").read_text())
    expected_terminal = ("completed" if all(
        reports[arm]["status"] == "completed" for arm in ARMS) else "incomplete")
    if (candidate.get("status") != expected_terminal
            or candidate.get("candidate_id") != CANDIDATE_ID
            or candidate.get("uid") != uid or candidate.get("seed") != source_report["seed"]
            or candidate.get("arms") != reports
            or candidate.get("generated_unexecuted") is not True
            or candidate.get("native_qualified") is not False
            or candidate.get("candidate_methods_tested") is not False
            or candidate.get("scientific_verdict") != "not_computed"):
        raise ValueError("C15 terminal candidate summary mismatch")
    archive = _validate_artifact_archive(output, reports)
    return {
        "status": "valid", "terminal_status": expected_terminal,
        "candidate_id": CANDIDATE_ID, "uid": uid,
        "seed": source_report["seed"], "roles": validated,
        "artifact_archive": archive,
        "native_qualified": False, "scientific_admission": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sequence", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--basis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--expected-sequence-sha256", required=True)
    parser.add_argument("--expected-basis-sha256", required=True)
    parser.add_argument("--basis-source-policy", required=True,
                        choices=sorted(ALLOWED_BASIS_POLICIES))
    parser.add_argument("--basis-evidence", required=True, type=Path)
    parser.add_argument("--expected-basis-evidence-sha256", required=True)
    parser.add_argument("--lambda-value", required=True, type=float)
    parser.add_argument("--residual-rank-policy", required=True, choices=[RANK_POLICY])
    parser.add_argument("--basis-orthogonality-tolerance", required=True, type=float)
    parser.add_argument("--protected-coefficient-tolerance", required=True, type=float)
    parser.add_argument("--numeric-rank-tolerance", required=True, type=float)
    parser.add_argument("--max-artifact-bytes", required=True, type=int)
    args = parser.parse_args()
    result = export_protected_lowrank_candidate(
        args.source_sequence, args.basis, args.output, uid=args.uid,
        project_root=args.project_root,
        expected_sequence_sha256=args.expected_sequence_sha256,
        expected_basis_sha256=args.expected_basis_sha256,
        basis_source_policy=args.basis_source_policy,
        lambda_value=args.lambda_value,
        residual_rank_policy=args.residual_rank_policy,
        orthogonality_tolerance=args.basis_orthogonality_tolerance,
        protected_coefficient_tolerance=args.protected_coefficient_tolerance,
        numeric_rank_tolerance=args.numeric_rank_tolerance,
        max_artifact_bytes=args.max_artifact_bytes,
        basis_evidence=args.basis_evidence,
        expected_basis_evidence_sha256=args.expected_basis_evidence_sha256)
    print(json.dumps({
        "status": result["status"], "candidate_id": CANDIDATE_ID,
        "native_qualified": False, "local_method_verified": False,
        "scientific_verdict": "not_computed",
    }))
    # A bounded terminal archive with retained failed arms is a successfully
    # collected artifact, while candidate.json truthfully remains incomplete.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
