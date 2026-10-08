"""C14: anchored group-TV repair in frozen corotational coordinates.

The method fits one proper rigid pose per frame from the predicted mesh alone,
freezes those poses, repairs only the body-frame residual with a nonuniform-time
group total-variation proximal operator, and reconstructs every original vertex
at every original frame.  It uses no GT, camera, scorer transform, labels, or
learned state.  Scientific qualification and official scoring are separate.

Run this module only as an inner command of the reviewed research-autopilot
harness.  Web-authored source is ``generated_unexecuted`` until Local acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np


CANDIDATE_ID = "4d-math-20261006-c14"
ARM = "declared_body_residual_repair"


def _vertices(value) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 3 or array.shape[-1] != 3 or array.shape[0] < 2
            or array.shape[1] < 3):
        raise ValueError("Expected vertices[T,V,3] with T>=2 and V>=3")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("Finite floating-point vertices required")
    return array


def _residuals(value) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 3 or array.shape[-1] != 3 or array.shape[0] < 2
            or array.shape[1] < 1):
        raise ValueError("Expected residuals[T,V,3] with T>=2 and V>=1")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("Finite floating-point residuals required")
    return array


def _positive(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


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


def fit_frozen_rigid_factors(sequence) -> dict[str, np.ndarray]:
    """Uniform-vertex proper Kabsch factors relative to centered frame zero.

    Row convention is ``Y_t ~= X @ R_t + c_t``.  Rank-two support is the
    minimum needed to determine a 3-D proper rotation; ambiguous reflected
    fits are rejected instead of receiving a hidden fallback.
    """
    source = _vertices(sequence).astype(np.float64, copy=False)
    centroids = source.mean(axis=1)
    centered = source - centroids[:, None, :]
    anchor = centered[0].copy()
    rotations, singular_values, fit_rms = [], [], []
    for frame, target in enumerate(centered):
        u, singular, vt = np.linalg.svd(anchor.T @ target)
        if (singular[0] <= np.finfo(np.float64).tiny
                or singular[1] <= 1e-8 * singular[0]):
            raise ValueError(f"Ambiguous rank-deficient body pose at frame {frame}")
        orientation = float(np.linalg.det(u @ vt))
        if orientation < 0.0 and singular[1] - singular[2] <= 1e-8 * singular[0]:
            raise ValueError(f"Ambiguous reflection-corrected body pose at frame {frame}")
        signs = np.ones(3, dtype=np.float64)
        signs[-1] = 1.0 if orientation >= 0.0 else -1.0
        rotation = (u * signs) @ vt
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=1e-10)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=1e-10)):
            raise ValueError(f"Kabsch factor is not SO(3) at frame {frame}")
        rotations.append(rotation)
        singular_values.append(singular)
        fit_rms.append(float(np.sqrt(np.mean((target - anchor @ rotation) ** 2))))
    rotation_rows = np.stack(rotations)
    rotation_rows[0] = np.eye(3)
    body = np.einsum("tvi,tji->tvj", centered, rotation_rows)
    residual = body - anchor
    residual[0] = 0.0
    numerical_zero = (64.0 * np.finfo(np.float64).eps
                      * max(1.0, float(np.max(np.abs(source)))))
    residual[np.abs(residual) <= numerical_zero] = 0.0
    return {
        "anchor": anchor,
        "rotation_rows": rotation_rows,
        "centroids": centroids,
        "singular_values": np.stack(singular_values),
        "fit_rms": np.asarray(fit_rms),
        "body_residual": residual,
    }


def _difference(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Finite one-dimensional timestamps required")
    delta = np.diff(values)
    if np.any(delta <= 0.0):
        raise ValueError("Strictly increasing timestamps required")
    operator = np.zeros((len(values) - 1, len(values)), dtype=np.float64)
    rows = np.arange(len(values) - 1)
    operator[rows, rows] = -1.0 / delta
    operator[rows, rows + 1] = 1.0 / delta
    return operator


def repair_body_residual(residual, times, *, weight: float, rho: float,
                         absolute_tolerance: float, relative_tolerance: float,
                         max_iterations: int) -> tuple[np.ndarray, dict]:
    """Solve anchored nonuniform-time vector group-TV with ADMM.

    ``min_U 0.5 ||U-Z||_F^2 + weight * sum_(t,v) ||(D U)_(t,v)||_2``
    subject to ``U[0] = Z[0]``.  XYZ is one group, so the construction is not
    coordinate-wise soft thresholding and is distinct from Gaussian filtering.
    """
    observed = _residuals(residual).astype(np.float64, copy=False)
    weight = _positive("weight", weight)
    rho = _positive("rho", rho)
    absolute_tolerance = _positive("absolute_tolerance", absolute_tolerance)
    relative_tolerance = _positive("relative_tolerance", relative_tolerance)
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    difference = _difference(np.asarray(times))
    if difference.shape[1] != observed.shape[0]:
        raise ValueError("Timestamp count differs from residual frames")
    if np.count_nonzero(observed) == 0:
        primal_tolerance = (np.sqrt((observed.shape[0] - 1)
                                    * observed.shape[1] * 3)
                            * absolute_tolerance)
        dual_tolerance = (np.sqrt((observed.shape[0] - 1)
                                  * observed.shape[1] * 3)
                          * absolute_tolerance)
        return observed.copy(), {
            "operator": "anchored_nonuniform_group_tv_first_difference",
            "converged": True, "termination_reason": "exact_zero_fixed_point",
            "iterations": 0, "objective": 0.0,
            "primal_residual": 0.0, "dual_residual": 0.0,
            "primal_tolerance": float(primal_tolerance),
            "dual_tolerance": float(dual_tolerance),
            "zero_innovation_groups": int((observed.shape[0] - 1) * observed.shape[1]),
            "weight": weight, "rho": rho,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "max_iterations": max_iterations,
            "time_objective": "discrete sum of ||delta_u/delta_t|| groups; no quadrature delta_t factor",
        }

    anchor = observed[0].copy()
    free_difference = difference[:, 1:]
    anchor_offset = difference[:, :1, None] * anchor[None, :, :]
    system = np.eye(observed.shape[0] - 1) + rho * (free_difference.T @ free_difference)
    condition = float(np.linalg.cond(system))
    if not math.isfinite(condition) or condition > 1.0 / np.sqrt(np.finfo(np.float64).eps):
        raise ValueError("Ill-conditioned anchored group-TV linear system")
    solution = observed.copy()
    innovation = np.einsum("st,tvc->svc", difference, solution)
    dual = np.zeros_like(innovation)
    converged = False
    primal = dual_residual = primal_tolerance = dual_tolerance = math.inf
    for iteration in range(1, max_iterations + 1):
        right = (observed[1:] + rho * np.einsum(
            "st,tvc->svc", free_difference.T,
            innovation - dual - anchor_offset))
        flat = right.reshape(observed.shape[0] - 1, -1)
        solution[1:] = np.linalg.solve(system, flat).reshape(solution[1:].shape)
        solution[0] = anchor
        derivative = np.einsum("st,tvc->svc", difference, solution)
        previous = innovation.copy()
        candidate = derivative + dual
        norms = np.linalg.norm(candidate, axis=2, keepdims=True)
        shrink = np.maximum(0.0, 1.0 - (weight / rho) / np.maximum(norms, np.finfo(np.float64).tiny))
        innovation = shrink * candidate
        dual += derivative - innovation
        primal = float(np.linalg.norm(derivative - innovation))
        dual_projected = rho * np.einsum(
            "st,tvc->svc", free_difference.T, innovation - previous)
        dual_residual = float(np.linalg.norm(dual_projected))
        primal_tolerance = (np.sqrt(innovation.size) * absolute_tolerance
                            + relative_tolerance * max(np.linalg.norm(derivative),
                                                       np.linalg.norm(innovation)))
        stationarity_scale = rho * np.einsum("st,tvc->svc", free_difference.T, dual)
        dual_tolerance = (np.sqrt(solution[1:].size) * absolute_tolerance
                          + relative_tolerance * np.linalg.norm(stationarity_scale))
        if primal <= primal_tolerance and dual_residual <= dual_tolerance:
            converged = True
            break
    derivative = np.einsum("st,tvc->svc", difference, solution)
    objective = (0.5 * float(np.sum((solution - observed) ** 2))
                 + weight * float(np.sum(np.linalg.norm(derivative, axis=2))))
    certificate = {
        "operator": "anchored_nonuniform_group_tv_first_difference",
        "converged": converged,
        "termination_reason": "residual_tolerances" if converged else "iteration_limit",
        "iterations": iteration, "objective": objective,
        "primal_residual": primal, "dual_residual": dual_residual,
        "primal_tolerance": float(primal_tolerance),
        "dual_tolerance": float(dual_tolerance),
        "linear_system_condition": condition,
        "zero_innovation_groups": int(np.count_nonzero(
            np.linalg.norm(innovation, axis=2) <= 10.0 * absolute_tolerance)),
        "weight": weight, "rho": rho,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "max_iterations": max_iterations,
        "time_objective": "discrete sum of ||delta_u/delta_t|| groups; no quadrature delta_t factor",
    }
    if not np.isfinite(solution).all() or not math.isfinite(objective):
        raise ValueError("Nonfinite anchored group-TV solution")
    return solution, certificate


def reconstruct_with_fixed_pose(anchor, residual, rotation_rows, centroids) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=np.float64)
    residual = _vertices(residual).astype(np.float64, copy=False)
    rotations = np.asarray(rotation_rows, dtype=np.float64)
    translations = np.asarray(centroids, dtype=np.float64)
    if anchor.shape != residual.shape[1:] or rotations.shape != (len(residual), 3, 3):
        raise ValueError("Pose/body shapes do not match")
    if translations.shape != (len(residual), 3):
        raise ValueError("Centroid shape does not match")
    gram = np.einsum("tji,tjk->tik", rotations, rotations)
    determinants = np.linalg.det(rotations)
    if (not np.allclose(gram, np.eye(3)[None], atol=1e-10, rtol=1e-10)
            or not np.allclose(determinants, 1.0, atol=1e-10, rtol=1e-10)):
        raise ValueError("Frozen pose factors must be in SO(3)")
    result = (np.einsum("tvi,tij->tvj", anchor + residual, rotations)
              + translations[:, None, :])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reconstructed sequence")
    return result


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
    vertices = _vertices(arrays["vertices"])
    if vertices.dtype != np.float32:
        raise ValueError("Native candidate export requires float32 vertices")
    faces = arrays["faces"]
    if (len(vertices) != 16 or not np.issubdtype(arrays["frame_indices"].dtype, np.integer)
            or not np.array_equal(arrays["frame_indices"], np.arange(16))):
        raise ValueError("Exactly 16 original frames in order required")
    _difference(arrays["timesteps"])
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0
            or faces.max() >= vertices.shape[1]):
        raise ValueError("Valid shared triangle topology required")
    if (not np.issubdtype(arrays["query_vertex_ids"].dtype, np.integer)
            or not np.array_equal(arrays["query_vertex_ids"], np.arange(vertices.shape[1]))):
        raise ValueError("Original identity vertex mapping required")
    for name, array in arrays.items():
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("Non-numeric/nonfinite native array: " + name)
    return arrays, report, report_path


def export_corotational_candidate(source_sequence: Path, output: Path, *, uid: str,
                                   expected_sequence_sha256: str, weight: float,
                                   rho: float, absolute_tolerance: float,
                                   relative_tolerance: float,
                                   max_iterations: int) -> dict:
    """Export C14's full sequence plus pose/solver evidence; never score it."""
    started = time.monotonic()
    if not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe native UID required")
    for name, value in (("weight", weight), ("rho", rho),
                        ("absolute_tolerance", absolute_tolerance),
                        ("relative_tolerance", relative_tolerance)):
        _positive(name, value)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    arrays, source_report, source_report_path = _load_native_case(
        source_sequence, uid, expected_sequence_sha256)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arm_dir = output / ARM
    arm_dir.mkdir()
    manifest = {"candidate_id": CANDIDATE_ID, "cases": [{
        "case_id": uid + "-" + ARM, "uid": uid, "case_dir": ARM,
    }], "scope": "single C14 candidate artifact; no control selection or scorer"}
    _json(output / "manifest.json", manifest)
    report = {
        "uid": uid, "seed": source_report["seed"], "candidate_id": CANDIDATE_ID,
        "method_id": CANDIDATE_ID, "arm_role": ARM,
        "pose_source": "predicted_sequence_geometry_only",
        "pose_fit": "uniform-vertex proper Kabsch to centered frame zero; frozen during repair",
        "repair_operator": "anchored_nonuniform_group_tv_first_difference",
        "source_sequence_sha256": expected_sequence_sha256,
        "source_report_sha256": _digest(source_report_path),
        "implementation_sha256": _digest(Path(__file__)),
        "parameters": {
            "weight": float(weight), "rho": float(rho),
            "absolute_tolerance": float(absolute_tolerance),
            "relative_tolerance": float(relative_tolerance),
            "max_iterations": max_iterations,
            "timestamp_units": "supplied native sequence units; weight scales in those units",
        },
        "information": "Predicted mesh sequence only; no GT, cameras, labels, scorer state, evaluator alignment, or learned weights",
        "native_qualified": False, "local_method_verified": False,
        "scientific_verdict": "not_computed", "generated_unexecuted": True,
    }
    status = "error"
    try:
        factors = fit_frozen_rigid_factors(arrays["vertices"])
        repaired, certificate = repair_body_residual(
            factors["body_residual"], arrays["timesteps"], weight=weight,
            rho=rho, absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations)
        report["solver"] = certificate
        report["pose_fit_diagnostics"] = {
            "max_rms": float(np.max(factors["fit_rms"])),
            "min_second_singular_value": float(np.min(factors["singular_values"][:, 1])),
        }
        if not certificate["converged"]:
            raise RuntimeError("Anchored group-TV solver did not converge under the frozen iteration limit")
        rebuilt = reconstruct_with_fixed_pose(
            factors["anchor"], repaired, factors["rotation_rows"], factors["centroids"])
        rebuilt[0] = arrays["vertices"][0].astype(np.float64)
        rebuilt = rebuilt.astype(arrays["vertices"].dtype)
        if (not np.isfinite(rebuilt).all()
                or rebuilt.shape != arrays["vertices"].shape
                or not np.array_equal(rebuilt[0], arrays["vertices"][0])):
            raise ValueError("C14 changed frame/vertex identity or the exact anchor")
        np.savez_compressed(arm_dir / "sequence.npz", **{**arrays, "vertices": rebuilt})
        np.savez_compressed(
            arm_dir / "certificate.npz",
            anchor=factors["anchor"], rotation_rows=factors["rotation_rows"],
            centroids=factors["centroids"], singular_values=factors["singular_values"],
            fit_rms=factors["fit_rms"], observed_body_residual=factors["body_residual"],
            repaired_body_residual=repaired,
        )
        report["sha256"] = {
            "sequence.npz": _digest(arm_dir / "sequence.npz"),
            "certificate.npz": _digest(arm_dir / "certificate.npz"),
        }
        status = "completed"
    except Exception as error:
        report.update(exception_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    report["status"] = status
    report["elapsed_seconds"] = time.monotonic() - started
    report["timing_scope"] = "CPU pose fit + C14 solve + artifact I/O; excludes generation/scorer/collection"
    _json(arm_dir / "report.json", report)
    summary = {
        "status": "completed" if status == "completed" else "incomplete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": CANDIDATE_ID, "uid": uid, "seed": source_report["seed"],
        "arm": report, "native_qualified": False,
        "local_method_verified": False, "candidate_methods_tested": False,
        "scientific_verdict": "not_computed",
        "elapsed_seconds": time.monotonic() - started,
    }
    _json(output / "candidate.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sequence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--expected-sequence-sha256", required=True)
    parser.add_argument("--weight", required=True, type=float)
    parser.add_argument("--rho", required=True, type=float)
    parser.add_argument("--absolute-tolerance", required=True, type=float)
    parser.add_argument("--relative-tolerance", required=True, type=float)
    parser.add_argument("--max-iterations", required=True, type=int)
    args = parser.parse_args()
    result = export_corotational_candidate(
        args.source_sequence, args.output, uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256,
        weight=args.weight, rho=args.rho,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
        max_iterations=args.max_iterations)
    print(json.dumps({
        "status": result["status"], "candidate_id": CANDIDATE_ID,
        "native_qualified": False, "local_method_verified": False,
        "scientific_verdict": "not_computed",
    }))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
