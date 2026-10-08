"""C08 geometry-specialized endpoint-constrained trajectory bridge.

One receipt-bound 16-frame ActionMesh prediction supplies every scientific
input.  Adjacent predicted meshes define sparse geometry kernels; barycentric
areas at the first and last frame define the endpoint marginals.  A log-domain
Schrodinger/IPF projection changes the complete path law without using latent
attention, ground truth, scorer state, or an external correspondence array.

The four exported roles share the same predicted frames, descriptors, supports
and coordinate lift.  ``endpoint_bridge`` follows the endpoint-scaled kernels;
``local_transition_tracker`` follows the reference kernels;
``geometry_cycle_control`` keeps only mutual adjacent geometry matches; and
``coordinate_smoother`` is the declared no-correspondence simple control.
Every role retains all original vertices, faces and 16 frame IDs.  Paths are
geometry decisions, not claims of true material identity.

Web-authored source is ``generated_unexecuted`` until Local acceptance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import tarfile
import time

import numpy as np

from research_math.area_transport_candidate import (
    _scaled_features,
    _support_hash,
    build_sparse_support,
    geometry_descriptors,
)
from research_math.protected_geometry_candidate import (
    validate_native_arrays,
    vertex_area_weights,
)


CANDIDATE_ID = "4d-math-20261006-c08"
ROLES = (
    "local_transition_tracker",
    "coordinate_smoother",
    "geometry_cycle_control",
    "endpoint_bridge",
)
METHOD_IDS = {
    "local_transition_tracker": "c08-control-local-transition-tracker",
    "coordinate_smoother": "c08-control-coordinate-smoother",
    "geometry_cycle_control": "c08-control-geometry-cycle",
    "endpoint_bridge": CANDIDATE_ID,
}
FEATURE_SOURCE = "predicted_mesh_geometry_only_no_gt_attention_or_external_correspondence"
SUPPORT_POLICY = "adjacent_descriptor_knn_plus_identity_plus_area_and_uniform_feasible_witnesses"
ENDPOINT_POLICY = "first_and_last_predicted_mesh_barycentric_area"
PATH_POLICY = "sparse_whole_path_conditional_mean_from_original_vertex_state"
FRAME_COMPLETION = "all_16_original_frames_with_exact_frame0"
PRODUCER_CODE_PATHS = (
    "actionmesh/research_math/__init__.py",
    "actionmesh/research_math/trajectory_bridge_candidate.py",
    "actionmesh/research_math/area_transport_candidate.py",
    "actionmesh/research_math/protected_geometry_candidate.py",
)
PRODUCER_EXECUTION_CONTRACT = {
    "gpu_count": 0,
    "max_attempts": 1,
    "max_retries_per_trial": 0,
}


class TrajectoryBridgeError(RuntimeError):
    """The frozen geometry-chain or bridge certificate failed."""


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    value = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        value.update(array.dtype.str.encode())
        value.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        value.update(array.tobytes())
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _positive(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name + " must be a finite positive scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(name + " must be a finite positive scalar")
    return value


def _positive_integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be a positive integer")
    value = int(value)
    if value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def producer_environment() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scope": "CPU geometry-only C08 artifacts; no model, GT, scorer or GPU",
        "feature_source": FEATURE_SOURCE,
    }


def _validate_ref(root: Path, ref: dict) -> Path:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str)
            or Path(ref["path"]).is_absolute() or ".." in Path(ref["path"]).parts
            or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64):
        raise ValueError("Exact nonescaping path/hash reference required")
    unresolved = Path(root).resolve() / ref["path"]
    if unresolved.is_symlink() or any(parent.is_symlink() for parent in unresolved.parents):
        raise ValueError("Symlinked C08 reference rejected")
    path = unresolved.resolve()
    path.relative_to(Path(root).resolve())
    if path.is_symlink() or not path.is_file() or digest(path) != ref["sha256"]:
        raise ValueError("Stale referenced source: " + ref["path"])
    return path


def _validate_producer_provenance(root: Path, provenance: dict,
                                  source_refs: dict) -> list[dict]:
    code_refs = provenance.get("code_refs") if isinstance(provenance, dict) else None
    if (set(provenance) != {"kind", "version", "code_refs", "environment",
                            "source_refs", "execution_contract"}
            or provenance.get("kind") != "c08-candidate-producer-provenance"
            or provenance.get("version") != 1
            or [row.get("path") if isinstance(row, dict) else None
                for row in code_refs or []] != list(PRODUCER_CODE_PATHS)
            or provenance.get("environment") != producer_environment()
            or provenance.get("source_refs") != [source_refs["sequence"], source_refs["report"]]
            or provenance.get("execution_contract") != PRODUCER_EXECUTION_CONTRACT):
        raise ValueError("Complete canonical C08 producer closure required")
    for ref in code_refs:
        _validate_ref(root, ref)
    return code_refs


def _row_logsumexp(rows: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    maximum = np.full(size, -np.inf, dtype=np.float64)
    np.maximum.at(maximum, rows, values)
    if not np.isfinite(maximum).all():
        raise TrajectoryBridgeError("Sparse kernel contains an empty row")
    total = np.zeros(size, dtype=np.float64)
    np.add.at(total, rows, np.exp(values - maximum[rows]))
    if np.any(total <= 0.0) or not np.isfinite(total).all():
        raise TrajectoryBridgeError("Sparse log-sum-exp failed")
    return maximum + np.log(total)


def _column_logsumexp(columns: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    return _row_logsumexp(columns, values, size)


def _normalize_rows(rows: np.ndarray, logits: np.ndarray, size: int) -> np.ndarray:
    normalizer = _row_logsumexp(rows, logits, size)
    result = np.exp(logits - normalizer[rows])
    sums = np.zeros(size, dtype=np.float64)
    np.add.at(sums, rows, result)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=5e-13):
        raise TrajectoryBridgeError("Reference kernel row normalization failed")
    return result


def build_geometry_chain(vertices: np.ndarray, faces: np.ndarray, *,
                         epsilon: float, neighbors: int) -> tuple[list[dict], np.ndarray]:
    """Construct 15 sparse row-stochastic adjacent geometry kernels."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if (vertices.dtype != np.dtype(np.float32) or vertices.shape[0] != 16
            or vertices.ndim != 3 or vertices.shape[-1] != 3
            or not np.isfinite(vertices).all()):
        raise ValueError("Exactly 16 finite float32 frames required")
    epsilon = _positive("epsilon", epsilon)
    neighbors = _positive_integer("neighbors", neighbors)
    state_count = vertices.shape[1]
    descriptors, masses = [], []
    expected_ids = np.arange(state_count)
    for frame in vertices:
        descriptor, mass, ids = geometry_descriptors(frame.astype(np.float64), faces)
        if not np.array_equal(ids, expected_ids):
            raise TrajectoryBridgeError(
                "C08 requires every original vertex to retain positive geometric support")
        strict_area = vertex_area_weights(frame.astype(np.float64), faces)
        strict_area /= strict_area.sum()
        if not np.allclose(mass, strict_area, rtol=2e-12, atol=2e-15):
            raise TrajectoryBridgeError("Endpoint/transition area conventions disagree")
        descriptors.append(descriptor); masses.append(strict_area)
    kernels = []
    for index in range(15):
        left, right = _scaled_features(descriptors[index], descriptors[index + 1])
        support = build_sparse_support(
            left, right, masses[index], masses[index + 1], neighbors=neighbors,
            source_ids=expected_ids, target_ids=expected_ids)
        rows = support["rows"].astype(np.int64, copy=False)
        columns = support["columns"].astype(np.int64, copy=False)
        costs = support["costs"].astype(np.float64, copy=False)
        log_probability = -costs / epsilon
        log_probability -= _row_logsumexp(rows, log_probability, state_count)[rows]
        identity = set(zip(rows.tolist(), columns.tolist()))
        if any((state, state) not in identity for state in range(state_count)):
            raise TrajectoryBridgeError("Identity reachability edge missing")
        witness = support["witness"].astype(np.float64, copy=False)
        witness_rows = np.zeros(state_count, dtype=np.float64)
        witness_columns = np.zeros(state_count, dtype=np.float64)
        np.add.at(witness_rows, rows, witness)
        np.add.at(witness_columns, columns, witness)
        if (not np.allclose(witness_rows, masses[index], rtol=1e-13, atol=1e-14)
                or not np.allclose(witness_columns, masses[index + 1],
                                    rtol=1e-13, atol=1e-14)):
            raise TrajectoryBridgeError("Area-feasibility witness failed")
        witness_row_residual = float(np.max(np.abs(witness_rows - masses[index])))
        witness_column_residual = float(
            np.max(np.abs(witness_columns - masses[index + 1])))
        kernels.append({"rows": rows, "columns": columns, "costs": costs,
                        "log_probability": log_probability, "witness": witness,
                        "witness_sha256": _array_digest(rows, columns, witness),
                        "witness_row_residual": witness_row_residual,
                        "witness_column_residual": witness_column_residual,
                        "support_sha256": _support_hash(support)})
    return kernels, np.asarray(masses, dtype=np.float64)


def solve_endpoint_bridge(kernels: list[dict], initial: np.ndarray, final: np.ndarray,
                          *, tolerance: float, max_iterations: int) -> tuple[list[np.ndarray], dict]:
    """Log-domain endpoint scaling without enumerating paths."""
    tolerance = _positive("tolerance", tolerance)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    initial = np.asarray(initial, dtype=np.float64)
    final = np.asarray(final, dtype=np.float64)
    if (initial.ndim != 1 or final.shape != initial.shape
            or np.any(initial <= 0.0) or np.any(final <= 0.0)
            or not np.isfinite(initial).all() or not np.isfinite(final).all()
            or not math.isclose(float(initial.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(float(final.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or len(kernels) != 15):
        raise ValueError("Strict positive normalized C08 endpoints and 15 kernels required")
    size = initial.size
    log_initial, log_final = np.log(initial), np.log(final)
    log_v = np.zeros(size, dtype=np.float64)
    residual = math.inf
    trace = []
    for iteration in range(1, max_iterations + 1):
        backward = [None] * 16
        backward[15] = log_v
        for step in range(14, -1, -1):
            kernel = kernels[step]
            backward[step] = _row_logsumexp(
                kernel["rows"], kernel["log_probability"]
                + backward[step + 1][kernel["columns"]], size)
        log_u = -backward[0]
        forward = log_initial + log_u
        for step in range(15):
            kernel = kernels[step]
            forward = _column_logsumexp(
                kernel["columns"], forward[kernel["rows"]]
                + kernel["log_probability"], size)
        new_log_v = log_final - forward
        new_log_v -= float(np.dot(final, new_log_v))
        residual = float(np.max(np.abs(new_log_v - log_v)))
        log_v = new_log_v
        # Check the actual constrained marginals, not only potential movement.
        check_backward = [None] * 16
        check_backward[15] = log_v
        for step in range(14, -1, -1):
            kernel = kernels[step]
            check_backward[step] = _row_logsumexp(
                kernel["rows"], kernel["log_probability"]
                + check_backward[step + 1][kernel["columns"]], size)
        check_u = -check_backward[0]
        first_log = log_initial + check_u + check_backward[0]
        first_mass = np.exp(first_log - _row_logsumexp(
            np.zeros(size, dtype=np.int64), first_log, 1)[0])
        last_log = log_initial + check_u
        for step in range(15):
            kernel = kernels[step]
            last_log = _column_logsumexp(
                kernel["columns"], last_log[kernel["rows"]]
                + kernel["log_probability"], size)
        last_log += log_v
        last_mass = np.exp(last_log - _row_logsumexp(
            np.zeros(size, dtype=np.int64), last_log, 1)[0])
        absolute = max(float(np.max(np.abs(first_mass - initial))),
                       float(np.max(np.abs(last_mass - final))))
        relative = max(float(np.max(np.abs(first_mass - initial) / initial)),
                       float(np.max(np.abs(last_mass - final) / final)))
        trace.append((iteration, residual, absolute, relative))
        if absolute <= tolerance and relative <= tolerance:
            break
    else:
        raise TrajectoryBridgeError("Endpoint scaling did not converge")
    backward = [None] * 16
    backward[15] = log_v
    for step in range(14, -1, -1):
        kernel = kernels[step]
        backward[step] = _row_logsumexp(
            kernel["rows"], kernel["log_probability"]
            + backward[step + 1][kernel["columns"]], size)
    bridge = []
    max_row_residual = 0.0
    marginal = initial.copy()
    for step, kernel in enumerate(kernels):
        log_probability = (kernel["log_probability"]
            + backward[step + 1][kernel["columns"]]
            - backward[step][kernel["rows"]])
        probability = np.exp(log_probability)
        row_sum = np.zeros(size, dtype=np.float64)
        np.add.at(row_sum, kernel["rows"], probability)
        max_row_residual = max(max_row_residual, float(np.max(np.abs(row_sum - 1.0))))
        next_marginal = np.zeros(size, dtype=np.float64)
        np.add.at(next_marginal, kernel["columns"],
                  marginal[kernel["rows"]] * probability)
        marginal = next_marginal
        bridge.append(probability)
    endpoint_residual = float(np.max(np.abs(marginal - final)))
    if (max_row_residual > max(1e-10, 100.0 * tolerance)
            or endpoint_residual > max(1e-9, 1000.0 * tolerance)):
        raise TrajectoryBridgeError("Endpoint bridge certificate failed")
    return bridge, {"iterations": iteration,
                    "scaling_update_residual": residual,
                    "maximum_absolute_endpoint_residual": trace[-1][2],
                    "maximum_relative_endpoint_residual": trace[-1][3],
                    "residual_trace": np.asarray(trace, dtype=np.float64),
                    "maximum_kernel_row_residual": max_row_residual,
                    "maximum_endpoint_residual": endpoint_residual}


def _sparse_policy_expectation(kernel: dict, probability: np.ndarray,
                               values: np.ndarray) -> np.ndarray:
    """Apply one sparse row-stochastic policy to vector-valued state data."""
    rows, columns = kernel["rows"], kernel["columns"]
    values = np.asarray(values, dtype=np.float64)
    output = np.zeros((int(rows.max()) + 1,) + values.shape[1:], dtype=np.float64)
    np.add.at(output, rows, probability[(slice(None),) + (None,) * (values.ndim - 1)]
              * values[columns])
    return output


def whole_path_conditional_mean_lift(vertices: np.ndarray, kernels: list[dict],
                                     policies: list[np.ndarray]) -> np.ndarray:
    """Bayes squared-coordinate action E[X_s(I_s)|I_0=i], sparsely."""
    output = np.empty_like(vertices)
    output[0] = vertices[0]
    for frame in range(1, 16):
        values = vertices[frame].astype(np.float64)
        for step in range(frame - 1, -1, -1):
            values = _sparse_policy_expectation(kernels[step], policies[step], values)
        output[frame] = values.astype(np.float32)
    if not np.array_equal(output[0], vertices[0]):
        raise TrajectoryBridgeError("Whole-path lift changed the exact anchor")
    return output


def _argmax_mapping(rows: np.ndarray, columns: np.ndarray,
                    values: np.ndarray, size: int) -> np.ndarray:
    """Deterministic row argmax with smallest target-state tie break."""
    result = np.full(size, -1, dtype=np.int64)
    best = np.full(size, -np.inf, dtype=np.float64)
    order = np.lexsort((columns, rows))
    for edge in order:
        row, column, value = int(rows[edge]), int(columns[edge]), float(values[edge])
        if value > best[row]:
            best[row], result[row] = value, column
    if np.any(result < 0):
        raise TrajectoryBridgeError("Path decision encountered an empty row")
    return result


def _whole_paths(mappings: list[np.ndarray], size: int) -> np.ndarray:
    paths = np.empty((16, size), dtype=np.int64)
    paths[0] = np.arange(size)
    for step, mapping in enumerate(mappings):
        paths[step + 1] = mapping[paths[step]]
    return paths


def _cycle_mappings(kernels: list[dict], size: int) -> list[np.ndarray]:
    result = []
    for kernel in kernels:
        geometry_score = -kernel["costs"]
        forward = _argmax_mapping(kernel["rows"], kernel["columns"],
                                  geometry_score, size)
        reverse = _argmax_mapping(kernel["columns"], kernel["rows"],
                                  geometry_score, size)
        mapping = np.arange(size, dtype=np.int64)
        for source, target in enumerate(forward):
            if reverse[target] == source:
                mapping[source] = target
        result.append(mapping)
    return result


def _smooth_coordinates(vertices: np.ndarray, times: np.ndarray,
                        strength: float) -> np.ndarray:
    strength = _positive("smoothing_strength", strength)
    count = len(times)
    matrix = np.eye(count, dtype=np.float64)
    for step, width in enumerate(np.diff(times)):
        weight = strength / (float(width) ** 2)
        matrix[step, step] += weight; matrix[step + 1, step + 1] += weight
        matrix[step, step + 1] -= weight; matrix[step + 1, step] -= weight
    free = np.arange(1, count)
    right = vertices.reshape(count, -1).astype(np.float64)
    solved = np.linalg.solve(matrix[np.ix_(free, free)],
                             right[free] - matrix[free, 0, None] * right[0])
    output = vertices.astype(np.float64).copy(); output[free] = solved.reshape(output[free].shape)
    output[0] = vertices[0]
    return output.astype(np.float32)


def build_role_sequence(vertices: np.ndarray, faces: np.ndarray, times: np.ndarray,
                        role: str, *, epsilon: float, neighbors: int,
                        tolerance: float, max_iterations: int,
                        smoothing_strength: float):
    """Build one role so bridge failure cannot erase independent controls."""
    if role not in ROLES:
        raise ValueError("Unknown C08 role")
    shared = {"feature_source": FEATURE_SOURCE, "support_policy": SUPPORT_POLICY,
        "endpoint_policy": ENDPOINT_POLICY, "path_policy": PATH_POLICY,
        "ground_truth_attention_or_scorer_input": False}
    if role == "coordinate_smoother":
        size = vertices.shape[1]
        output = _smooth_coordinates(vertices, times, smoothing_strength)
        path = np.tile(np.arange(size, dtype=np.int64), (16, 1))
        initial = vertex_area_weights(vertices[0].astype(np.float64), faces)
        final = vertex_area_weights(vertices[-1].astype(np.float64), faces)
        initial /= initial.sum(); final /= final.sum()
        certificate = {"path_state_ids": path,
            "support_sha256": np.asarray([], dtype="S64"),
            "witness_sha256": np.asarray([], dtype="S64"),
            "witness_row_residual": np.asarray([], dtype=np.float64),
            "witness_column_residual": np.asarray([], dtype=np.float64),
            "support_edges": np.asarray([], dtype=np.int64),
            "initial_area": initial, "final_area": final}
        diagnostic = {"frames": 16, "vertices": size,
            "feature_source": FEATURE_SOURCE,
            "path_policy": "same_identity_temporal_coordinate_solve",
            "unique_final_states": size, "surface_vertex_lift": False,
            "convex_hull_coordinate_lift": False, "probabilistic_lift": False,
            "hard_path_is_diagnostic_only": False,
            "true_material_identity_claim": False}
        return output, certificate, diagnostic, shared
    kernels, masses = build_geometry_chain(
        vertices, faces, epsilon=epsilon, neighbors=neighbors)
    size = vertices.shape[1]
    reference_probability = [np.exp(kernel["log_probability"]) for kernel in kernels]
    reference_mappings = [_argmax_mapping(
        kernel["rows"], kernel["columns"], probability, size)
        for kernel, probability in zip(kernels, reference_probability)]
    support_hashes = np.asarray([row["support_sha256"] for row in kernels], dtype="S64")
    witness_hashes = np.asarray([row["witness_sha256"] for row in kernels], dtype="S64")
    support_edges = np.asarray([len(row["rows"]) for row in kernels], dtype=np.int64)
    bridge_diagnostics = None
    if role == "local_transition_tracker":
        output = whole_path_conditional_mean_lift(vertices, kernels, reference_probability)
        path = _whole_paths(reference_mappings, size)
    elif role == "endpoint_bridge":
        bridge_probability, bridge_diagnostics = solve_endpoint_bridge(
            kernels, masses[0], masses[-1], tolerance=tolerance,
            max_iterations=max_iterations)
        bridge_mappings = [_argmax_mapping(
            kernel["rows"], kernel["columns"], probability, size)
            for kernel, probability in zip(kernels, bridge_probability)]
        output = whole_path_conditional_mean_lift(vertices, kernels, bridge_probability)
        path = _whole_paths(bridge_mappings, size)
    else:
        path = _whole_paths(_cycle_mappings(kernels, size), size)
        output = np.empty_like(vertices); output[0] = vertices[0]
        for frame in range(1, 16):
            output[frame] = vertices[frame, path[frame]]
    if (output.dtype != np.dtype(np.float32) or not np.isfinite(output).all()
            or not np.array_equal(output[0], vertices[0])):
        raise TrajectoryBridgeError("C08 export lost float32 identity or anchor")
    certificate = {"path_state_ids": path, "support_sha256": support_hashes,
        "witness_sha256": witness_hashes,
        "witness_row_residual": np.asarray(
            [row["witness_row_residual"] for row in kernels], dtype=np.float64),
        "witness_column_residual": np.asarray(
            [row["witness_column_residual"] for row in kernels], dtype=np.float64),
        "support_edges": support_edges, "initial_area": masses[0],
        "final_area": masses[-1]}
    diagnostic = {"frames": 16, "vertices": size, "feature_source": FEATURE_SOURCE,
        "path_policy": ("same_identity_temporal_coordinate_solve"
                        if role == "coordinate_smoother" else PATH_POLICY),
        "unique_final_states": int(np.unique(path[-1]).size),
        "surface_vertex_lift": role == "geometry_cycle_control",
        "convex_hull_coordinate_lift": role in (
            "local_transition_tracker", "endpoint_bridge"),
        "probabilistic_lift": role in ("local_transition_tracker", "endpoint_bridge"),
        "hard_path_is_diagnostic_only": role in (
            "local_transition_tracker", "endpoint_bridge"),
        "true_material_identity_claim": False}
    if bridge_diagnostics is not None:
        certificate.update({
            "bridge_iterations": np.asarray([bridge_diagnostics["iterations"]], dtype=np.int64),
            "bridge_scaling_update_residual": np.asarray(
                [bridge_diagnostics["scaling_update_residual"]], dtype=np.float64),
            "bridge_maximum_kernel_row_residual": np.asarray(
                [bridge_diagnostics["maximum_kernel_row_residual"]], dtype=np.float64),
            "bridge_maximum_endpoint_residual": np.asarray(
                [bridge_diagnostics["maximum_endpoint_residual"]], dtype=np.float64),
            "bridge_maximum_absolute_endpoint_residual": np.asarray(
                [bridge_diagnostics["maximum_absolute_endpoint_residual"]], dtype=np.float64),
            "bridge_maximum_relative_endpoint_residual": np.asarray(
                [bridge_diagnostics["maximum_relative_endpoint_residual"]], dtype=np.float64),
            "bridge_residual_trace": bridge_diagnostics["residual_trace"]})
        diagnostic.update({key: value for key, value in bridge_diagnostics.items()
                           if key != "residual_trace"})
    return output, certificate, diagnostic, shared


def build_role_sequences(vertices: np.ndarray, faces: np.ndarray, times: np.ndarray, *,
                         epsilon: float, neighbors: int, tolerance: float,
                         max_iterations: int, smoothing_strength: float):
    outputs, certificates, diagnostics, shared = {}, {}, {}, None
    for role in ROLES:
        output, certificate, diagnostic, role_shared = build_role_sequence(
            vertices, faces, times, role, epsilon=epsilon, neighbors=neighbors,
            tolerance=tolerance, max_iterations=max_iterations,
            smoothing_strength=smoothing_strength)
        outputs[role], certificates[role], diagnostics[role] = (
            output, certificate, diagnostic)
        shared = role_shared
    return outputs, certificates, diagnostics, shared


def _load_source(source_case: Path, uid: str, expected_sha256: str):
    source_case = Path(source_case)
    if source_case.is_symlink():
        raise ValueError("Physical C08 source directory required")
    source_case = source_case.resolve()
    sequence, report_path = source_case / "sequence.npz", source_case / "report.json"
    if (sequence.is_symlink() or report_path.is_symlink()
            or not sequence.is_file() or not report_path.is_file()
            or sequence.parent != report_path.parent):
        raise ValueError("Physical sibling C08 source pair required")
    if digest(sequence) != expected_sha256:
        raise ValueError("Native sequence digest changed")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "completed" or report.get("uid") != uid
            or report.get("sha256", {}).get("sequence.npz") != expected_sha256
            or isinstance(report.get("seed"), bool) or not isinstance(report.get("seed"), int)):
        raise ValueError("Completed matching native source pair required")
    with np.load(sequence, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    vertices, faces, times = validate_native_arrays(arrays)
    return arrays, vertices, faces, times, report_path, report["seed"]


def _artifact_members(output: Path, reports: list[dict]) -> list[Path]:
    paths = [output / "candidate.json", output / "manifest.json", output / "common-target.npz"]
    for report in reports:
        directory = output / report["candidate_arm"]
        paths.append(directory / "report.json")
        if report["status"] == "completed":
            paths.extend((directory / "sequence.npz", directory / "certificate.npz"))
    return sorted(paths, key=lambda path: path.relative_to(output).as_posix())


def _write_archive(output: Path, reports: list[dict], maximum: int) -> None:
    maximum = _positive_integer("max_artifact_bytes", maximum)
    rows = [{"path": path.relative_to(output).as_posix(), "sha256": digest(path),
             "size_bytes": path.stat().st_size} for path in _artifact_members(output, reports)]
    temporary, archive = output / "artifact.tar.tmp", output / "artifact.tar"
    with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as bundle:
        for row in rows:
            path = output / row["path"]
            info = tarfile.TarInfo(row["path"]); info.size = row["size_bytes"]
            info.mode = 0o644; info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                bundle.addfile(info, stream)
    if temporary.stat().st_size > maximum:
        temporary.unlink(); raise RuntimeError("C08 artifact exceeds frozen byte ceiling")
    temporary.replace(archive)
    write_json(output / "artifact-archive.json", {
        "kind": "c08-terminal-artifact-archive", "version": 1,
        "archive": {"path": "artifact.tar", "sha256": digest(archive),
                    "size_bytes": archive.stat().st_size},
        "members": rows, "max_artifact_bytes": maximum})


def _validate_archive(output: Path, reports: list[dict]) -> dict:
    archive, record_path = output / "artifact.tar", output / "artifact-archive.json"
    record = json.loads(record_path.read_text())
    rows = [{"path": path.relative_to(output).as_posix(),
             "sha256": digest(path), "size_bytes": path.stat().st_size}
            for path in _artifact_members(output, reports)]
    if (record.get("kind") != "c08-terminal-artifact-archive"
            or record.get("version") != 1 or record.get("members") != rows
            or record.get("archive") != {"path": "artifact.tar",
                "sha256": digest(archive), "size_bytes": archive.stat().st_size}
            or archive.stat().st_size > record.get("max_artifact_bytes", -1)):
        raise ValueError("C08 artifact archive record changed")
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        if [member.name for member in members] != [row["path"] for row in rows]:
            raise ValueError("C08 archive inventory changed")
        for member, row in zip(members, rows):
            stream = bundle.extractfile(member)
            if (not member.isfile() or member.size != row["size_bytes"]
                    or member.mode != 0o644 or member.uid or member.gid or member.mtime
                    or member.uname or member.gname or stream is None
                    or hashlib.sha256(stream.read()).hexdigest() != row["sha256"]):
                raise ValueError("C08 archive member or metadata changed")
    return {"archive": {"path": archive, "sha256": digest(archive)},
            "record": {"path": record_path, "sha256": digest(record_path)}}


def export_trajectory_bridge_candidate(source_case: Path, output: Path, *, uid: str,
                                       expected_sequence_sha256: str,
                                       source_refs: dict, producer_provenance_ref: dict,
                                       producer_provenance_path: Path, epsilon: float,
                                       neighbors: int, tolerance: float,
                                       max_iterations: int, smoothing_strength: float,
                                       coordinate_lower: float, coordinate_upper: float,
                                       bounds_policy: str, max_artifact_bytes: int) -> dict:
    started = time.monotonic()
    if not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe UID required")
    if set(source_refs) != {"sequence", "report"}:
        raise ValueError("Exact source pair required")
    provenance_path = Path(producer_provenance_path).resolve()
    if digest(provenance_path) != producer_provenance_ref.get("sha256"):
        raise ValueError("Producer provenance digest changed")
    relative = Path(producer_provenance_ref.get("path", ""))
    root = provenance_path
    for _ in relative.parts:
        root = root.parent
    if relative.is_absolute() or ".." in relative.parts or (root / relative).resolve() != provenance_path:
        raise ValueError("Producer provenance path alias changed")
    provenance = json.loads(provenance_path.read_text())
    implementation_refs = _validate_producer_provenance(root, provenance, source_refs)
    arrays, vertices, faces, times, report_path, seed = _load_source(
        Path(source_case), uid, expected_sequence_sha256)
    if (source_refs["sequence"]["sha256"] != expected_sequence_sha256
            or source_refs["report"]["sha256"] != digest(report_path)):
        raise ValueError("Source references differ from consumed pair")
    if (not math.isfinite(coordinate_lower) or not math.isfinite(coordinate_upper)
            or coordinate_lower >= coordinate_upper
            or bounds_policy not in ("preserve_and_report", "reject")):
        raise ValueError("Invalid coordinate-bounds policy")
    parameters = {
        "epsilon": _positive("epsilon", epsilon),
        "neighbors": _positive_integer("neighbors", neighbors),
        "tolerance": _positive("tolerance", tolerance),
        "max_iterations": _positive_integer("max_iterations", max_iterations),
        "smoothing_strength": _positive("smoothing_strength", smoothing_strength),
        "coordinate_bounds": [float(coordinate_lower), float(coordinate_upper)],
        "bounds_policy": bounds_policy,
        "feature_source": FEATURE_SOURCE, "support_policy": SUPPORT_POLICY,
        "endpoint_policy": ENDPOINT_POLICY, "path_policy": PATH_POLICY,
    }
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "common-target.npz",
                        source_vertices=vertices, faces=faces, timesteps=times)
    reports = []
    implementation_sha = digest(Path(__file__))
    for role in ROLES:
        arm_started = time.monotonic(); directory = output / role; directory.mkdir()
        report = {
            "candidate_id": CANDIDATE_ID, "candidate_arm": role,
            "method_id": METHOD_IDS[role], "uid": uid, "seed": seed,
            "implementation_sha256": implementation_sha,
            "implementation_refs": implementation_refs,
            "source_sequence_sha256": expected_sequence_sha256,
            "source_report_sha256": digest(report_path),
            "producer_provenance_ref": producer_provenance_ref,
            "parameters": parameters, "generated_unexecuted": False,
            "native_qualified": False, "scientific_admission": False,
            "local_method_verified": False, "frame_completion": FRAME_COMPLETION,
            "scientific_verdict": "not_computed",
        }
        try:
            result, certificate, diagnostic, _ = build_role_sequence(
                vertices, faces, times, role, epsilon=parameters["epsilon"],
                neighbors=parameters["neighbors"], tolerance=parameters["tolerance"],
                max_iterations=parameters["max_iterations"],
                smoothing_strength=parameters["smoothing_strength"])
            bounds = {"below": int(np.count_nonzero(result < coordinate_lower)),
                      "above": int(np.count_nonzero(result > coordinate_upper)),
                      "lower": coordinate_lower, "upper": coordinate_upper,
                      "policy": bounds_policy}
            if bounds_policy == "reject" and (bounds["below"] or bounds["above"]):
                raise ValueError("C08 output violates frozen coordinate bounds")
            np.savez_compressed(directory / "sequence.npz", **{**arrays, "vertices": result})
            np.savez_compressed(directory / "certificate.npz", **certificate)
            report.update(status="completed", diagnostics=diagnostic, bounds=bounds,
                          sha256={"sequence.npz": digest(directory / "sequence.npz"),
                                  "certificate.npz": digest(directory / "certificate.npz")})
        except Exception as error:
            for name in ("sequence.npz", "certificate.npz"):
                (directory / name).unlink(missing_ok=True)
            report.update(status="error", exception_type=type(error).__name__,
                          error=str(error)[:4096])
        report["elapsed_seconds"] = time.monotonic() - arm_started
        write_json(directory / "report.json", report); reports.append(report)
    completed = [row["candidate_arm"] for row in reports if row["status"] == "completed"]
    common_ref = {"path": "common-target.npz", "sha256": digest(output / "common-target.npz")}
    record = {
        "kind": "c08-geometry-trajectory-bridge-candidate", "version": 1,
        "candidate_id": CANDIDATE_ID, "uid": uid, "seed": seed,
        "roles": list(ROLES), "arms": reports,
        "status": "completed" if len(completed) == len(ROLES) else "incomplete",
        "source_sequence_sha256": expected_sequence_sha256,
        "source_report_sha256": digest(report_path),
        "source_refs": source_refs, "producer_provenance_ref": producer_provenance_ref,
        "implementation_refs": implementation_refs, "implementation_sha256": implementation_sha,
        "common_target": common_ref,
        "source_delivery_status": "generated_unexecuted_at_authoring",
        "native_qualified": False, "scientific_admission": False,
        "local_method_verified": False, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "timing_scope": "CPU geometry chain/bridge and artifact I/O only",
    }
    write_json(output / "candidate.json", record)
    write_json(output / "manifest.json", {
        "expected_roles": list(ROLES),
        "cases": [{"case_id": uid + "-" + role, "uid": uid,
                   "case_dir": role, "arm_role": role} for role in completed],
        "common_target": common_ref,
        "scope": "C08 geometry-specialized generated_unexecuted artifacts; no B0/B*/scoring/admission",
    })
    _write_archive(output, reports, max_artifact_bytes)
    return record


def validate_candidate_artifact(root: Path, candidate_path: Path) -> dict:
    """Replay every role and verify the retained terminal artifact exactly."""
    root, candidate_path = Path(root).resolve(), Path(candidate_path).resolve()
    candidate_path.relative_to(root)
    record = json.loads(candidate_path.read_text())
    if (record.get("kind") != "c08-geometry-trajectory-bridge-candidate"
            or record.get("version") != 1 or record.get("candidate_id") != CANDIDATE_ID
            or record.get("roles") != list(ROLES)
            or len(record.get("arms", [])) != len(ROLES)
            or record.get("implementation_sha256") != digest(Path(__file__))
            or record.get("native_qualified") is not False
            or record.get("scientific_admission") is not False
            or record.get("local_method_verified") is not False):
        raise ValueError("Current generated-unexecuted C08 candidate required")
    if record.get("source_delivery_status") != "generated_unexecuted_at_authoring":
        raise ValueError("C08 source-delivery scope changed")
    source_paths = {key: _validate_ref(root, ref) for key, ref in record["source_refs"].items()}
    provenance_path = _validate_ref(root, record["producer_provenance_ref"])
    implementation_refs = _validate_producer_provenance(
        root, json.loads(provenance_path.read_text()), record["source_refs"])
    if record.get("implementation_refs") != implementation_refs:
        raise ValueError("C08 producer closure changed")
    arrays, vertices, faces, times, report_path, seed = _load_source(
        source_paths["sequence"].parent, record["uid"], record["source_refs"]["sequence"]["sha256"])
    if (source_paths["report"] != report_path.resolve() or seed != record["seed"]
            or record.get("source_sequence_sha256")
            != record["source_refs"]["sequence"]["sha256"]
            or record.get("source_report_sha256")
            != record["source_refs"]["report"]["sha256"]):
        raise ValueError("C08 source aliases changed")
    parameters = record["arms"][0]["parameters"]
    if (not isinstance(parameters, dict)
            or set(parameters) != {"epsilon", "neighbors", "tolerance",
                "max_iterations", "smoothing_strength", "coordinate_bounds",
                "bounds_policy", "feature_source", "support_policy",
                "endpoint_policy", "path_policy"}
            or parameters["feature_source"] != FEATURE_SOURCE
            or parameters["support_policy"] != SUPPORT_POLICY
            or parameters["endpoint_policy"] != ENDPOINT_POLICY
            or parameters["path_policy"] != PATH_POLICY
            or parameters["bounds_policy"] not in ("preserve_and_report", "reject")
            or not isinstance(parameters["coordinate_bounds"], list)
            or len(parameters["coordinate_bounds"]) != 2):
        raise ValueError("Canonical C08 parameter schema required")
    lower, upper = parameters["coordinate_bounds"]
    if (isinstance(lower, (bool, np.bool_)) or isinstance(upper, (bool, np.bool_))
            or not np.isscalar(lower) or not np.isscalar(upper)
            or not math.isfinite(float(lower)) or not math.isfinite(float(upper))
            or float(lower) >= float(upper)):
        raise ValueError("Canonical C08 coordinate bounds required")
    normalized_parameters = dict(parameters)
    normalized_parameters.update(
        epsilon=_positive("epsilon", parameters["epsilon"]),
        neighbors=_positive_integer("neighbors", parameters["neighbors"]),
        tolerance=_positive("tolerance", parameters["tolerance"]),
        max_iterations=_positive_integer(
            "max_iterations", parameters["max_iterations"]),
        smoothing_strength=_positive(
            "smoothing_strength", parameters["smoothing_strength"]),
        coordinate_bounds=[float(lower), float(upper)])
    if normalized_parameters != parameters:
        raise ValueError("Canonical typed C08 parameters required")
    common_ref = record.get("common_target")
    common_path = candidate_path.parent / "common-target.npz"
    if (common_ref != {"path": "common-target.npz", "sha256": digest(common_path)}
            or common_path.is_symlink()):
        raise ValueError("C08 common target identity changed")
    with np.load(common_path, allow_pickle=False) as saved:
        common = {name: saved[name].copy() for name in saved.files}
    if (set(common) != {"source_vertices", "faces", "timesteps"}
            or not np.array_equal(common["source_vertices"], vertices)
            or not np.array_equal(common["faces"], faces)
            or not np.array_equal(common["timesteps"], times)):
        raise ValueError("C08 common target contents changed")
    completed = []
    for role, saved_report in zip(ROLES, record["arms"]):
        directory = candidate_path.parent / role
        report = json.loads((directory / "report.json").read_text())
        if report != saved_report or report.get("candidate_arm") != role:
            raise ValueError("C08 terminal report changed")
        if (report.get("candidate_id") != CANDIDATE_ID
                or report.get("method_id") != METHOD_IDS[role]
                or report.get("uid") != record["uid"]
                or report.get("seed") != record["seed"]
                or report.get("implementation_sha256") != digest(Path(__file__))
                or report.get("implementation_refs") != implementation_refs
                or report.get("source_sequence_sha256")
                != record["source_sequence_sha256"]
                or report.get("source_report_sha256") != record["source_report_sha256"]
                or report.get("producer_provenance_ref")
                != record["producer_provenance_ref"]
                or report.get("parameters") != parameters
                or report.get("generated_unexecuted") is not False
                or report.get("native_qualified") is not False
                or report.get("scientific_admission") is not False
                or report.get("local_method_verified") is not False
                or report.get("scientific_verdict") != "not_computed"
                or report.get("frame_completion") != FRAME_COMPLETION):
            raise ValueError("C08 terminal report identity changed")
        try:
            output, certificate, diagnostic, _ = build_role_sequence(
                vertices, faces, times, role, epsilon=parameters["epsilon"],
                neighbors=parameters["neighbors"], tolerance=parameters["tolerance"],
                max_iterations=parameters["max_iterations"],
                smoothing_strength=parameters["smoothing_strength"])
            replay_error = None
        except Exception as error:
            output = certificate = diagnostic = None
            replay_error = error
        if report.get("status") == "error":
            if (replay_error is None
                    or report.get("exception_type") != type(replay_error).__name__
                    or report.get("error") != str(replay_error)[:4096]
                    or (directory / "sequence.npz").exists()
                    or (directory / "certificate.npz").exists()):
                raise ValueError("C08 retained failure does not replay exactly")
            continue
        if report.get("status") != "completed" or replay_error is not None:
            raise ValueError("Invalid C08 terminal role state")
        sequence, certificate_path = directory / "sequence.npz", directory / "certificate.npz"
        if report.get("sha256") != {"sequence.npz": digest(sequence),
                                     "certificate.npz": digest(certificate_path)}:
            raise ValueError("C08 role hashes changed")
        with np.load(sequence, allow_pickle=False) as saved:
            observed = {name: saved[name].copy() for name in saved.files}
        validate_native_arrays(observed)
        if set(observed) != set(arrays) or not np.array_equal(observed["vertices"], output):
            raise ValueError("C08 sequence differs from replay")
        for name in set(arrays) - {"vertices"}:
            if not np.array_equal(observed[name], arrays[name]):
                raise ValueError("C08 changed native identity array")
        with np.load(certificate_path, allow_pickle=False) as saved:
            observed_certificate = {name: saved[name].copy() for name in saved.files}
        if set(observed_certificate) != set(certificate) or any(
                not np.array_equal(observed_certificate[name], certificate[name])
                for name in observed_certificate):
            raise ValueError("C08 certificate differs from replay")
        if report.get("diagnostics") != diagnostic:
            raise ValueError("C08 diagnostics changed")
        expected_bounds = {"below": int(np.count_nonzero(output < lower)),
                           "above": int(np.count_nonzero(output > upper)),
                           "lower": lower, "upper": upper,
                           "policy": parameters["bounds_policy"]}
        if (report.get("bounds") != expected_bounds
                or (parameters["bounds_policy"] == "reject"
                    and (expected_bounds["below"] or expected_bounds["above"]))):
            raise ValueError("C08 bounds evidence changed")
        completed.append(role)
    expected_status = "completed" if len(completed) == len(ROLES) else "incomplete"
    if record.get("status") != expected_status:
        raise ValueError("C08 aggregate status disagrees with terminal roles")
    expected_manifest = {"expected_roles": list(ROLES),
        "cases": [{"case_id": record["uid"] + "-" + role,
                   "uid": record["uid"], "case_dir": role, "arm_role": role}
                  for role in completed],
        "common_target": common_ref,
        "scope": "C08 geometry-specialized generated_unexecuted artifacts; no B0/B*/scoring/admission"}
    if json.loads((candidate_path.parent / "manifest.json").read_text()) != expected_manifest:
        raise ValueError("C08 manifest changed")
    archive = _validate_archive(candidate_path.parent, record["arms"])
    refs = {name: {"path": value["path"].resolve().relative_to(root).as_posix(),
                   "sha256": value["sha256"]} for name, value in archive.items()}
    return {**record, "completed_roles": completed, "artifact_archive": refs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--expected-sequence-sha256", required=True)
    parser.add_argument("--source-sequence-ref", required=True)
    parser.add_argument("--source-report-ref", required=True)
    parser.add_argument("--producer-provenance", type=Path, required=True)
    parser.add_argument("--producer-provenance-ref", required=True)
    parser.add_argument("--expected-producer-provenance-sha256", required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--neighbors", type=int, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--max-iterations", type=int, required=True)
    parser.add_argument("--smoothing-strength", type=float, required=True)
    parser.add_argument("--coordinate-bounds", nargs=2, type=float, required=True)
    parser.add_argument("--bounds-policy", choices=("preserve_and_report", "reject"), required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    args = parser.parse_args()
    source = args.source_sequence.resolve()
    result = export_trajectory_bridge_candidate(
        source.parent, args.output, uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256,
        source_refs={"sequence": {"path": args.source_sequence_ref,
                                  "sha256": args.expected_sequence_sha256},
                     "report": {"path": args.source_report_ref,
                                "sha256": digest(source.with_name("report.json"))}},
        producer_provenance_ref={"path": args.producer_provenance_ref,
                                 "sha256": args.expected_producer_provenance_sha256},
        producer_provenance_path=args.producer_provenance,
        epsilon=args.epsilon, neighbors=args.neighbors, tolerance=args.tolerance,
        max_iterations=args.max_iterations, smoothing_strength=args.smoothing_strength,
        coordinate_lower=args.coordinate_bounds[0], coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy, max_artifact_bytes=args.max_artifact_bytes)
    print(json.dumps({"status": result["status"], "candidate_id": CANDIDATE_ID,
                      "generated_unexecuted": False, "native_qualified": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
