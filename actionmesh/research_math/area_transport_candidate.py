"""C06 geometry-only area-marginal transport for complete native 4D meshes.

The source is one receipt-bound ActionMesh prediction.  Per-frame local mesh
geometry supplies rotation-invariant descriptors, barycentric areas and a sparse
support.  A deterministic feasible mass witness is included in that support, so
the two balanced Sinkhorn arms never depend on latent attention or an external
correspondence array.  Row softmax, uniform-mass transport and area-marginal
transport share the same descriptor cost, support, entropy and barycentric lift.

Transported barycentres need not lie on the target surface.  This generated
source establishes no natural ambiguity, native benefit or scientific admission.
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

from research_math.protected_geometry_candidate import (
    validate_native_arrays as _validate_native_arrays,
)


CANDIDATE_ID = "4d-math-20261006-c06"
ROLES = ("row_softmax", "vertex_density_transport", "area_marginal_transport")
METHOD_IDS = {
    "row_softmax": "c06-control-row-softmax",
    "vertex_density_transport": "c06-control-vertex-density-transport",
    "area_marginal_transport": CANDIDATE_ID,
}
FEATURE_SOURCE = "predicted_mesh_geometry_only_no_gt_or_model_attention"
FRAME_COMPLETION = "all_16_original_frames_with_exact_frame0"
PRODUCER_CODE_PATHS = (
    "actionmesh/research_math/__init__.py",
    "actionmesh/research_math/area_transport_candidate.py",
    "actionmesh/research_math/protected_geometry_candidate.py",
    "actionmesh/research_ten/__init__.py",
    "actionmesh/research_ten/m01_elasticity.py",
)
PRODUCER_EXECUTION_CONTRACT = {
    "gpu_count": 0,
    "max_attempts": 1,
    "max_retries_per_trial": 0,
}


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


def validate_native_arrays(arrays: dict):
    return _validate_native_arrays(arrays)


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy")}


def producer_environment() -> dict[str, str]:
    return {
        "python_executable": sys.executable,
        **runtime_versions(),
        "scope": "CPU geometry-only C06 artifacts; no model, GT, scorer or GPU",
        "feature_source": FEATURE_SOURCE,
    }


def _validate_producer_provenance(root: Path, provenance: dict,
                                  source_refs: dict) -> list[dict]:
    if not isinstance(provenance, dict):
        raise ValueError("C06 producer provenance object required")
    root = Path(root).resolve()
    implementation_refs = provenance.get("code_refs")
    if (set(provenance) != {"kind", "version", "code_refs", "environment",
                            "source_refs", "execution_contract"}
            or provenance.get("kind") != "c06-candidate-producer-provenance"
            or provenance.get("version") != 1
            or not isinstance(implementation_refs, list)
            or [ref.get("path") if isinstance(ref, dict) else None
                for ref in implementation_refs] != list(PRODUCER_CODE_PATHS)
            or provenance.get("environment") != producer_environment()
            or provenance.get("source_refs")
            != [source_refs["sequence"], source_refs["report"]]
            or provenance.get("execution_contract")
            != PRODUCER_EXECUTION_CONTRACT):
        raise ValueError("Complete canonical C06 producer closure required")
    for ref, expected_path in zip(implementation_refs, PRODUCER_CODE_PATHS):
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or ref.get("path") != expected_path
                or not isinstance(ref.get("sha256"), str)
                or len(ref["sha256"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in ref["sha256"])):
            raise ValueError("Exact C06 implementation ref required")
        path = (root / expected_path).resolve(); path.relative_to(root)
        if path.is_symlink() or not path.is_file() or digest(path) != ref["sha256"]:
            raise ValueError("Stale C06 implementation ref")
    return implementation_refs


def _positive(name: str, value) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name + " must be a finite positive scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(name + " must be a finite positive scalar")
    return result


def _positive_integer(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(name + " must be a positive integer")
    return result


def _edges(faces: np.ndarray) -> np.ndarray:
    raw = np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2]), axis=0)
    raw.sort(axis=1)
    return np.unique(raw, axis=0)


def geometry_descriptors(vertices: np.ndarray, faces: np.ndarray):
    """Return descriptors, positive masses and retained original vertex IDs.

    Degenerate faces and vertices with no retained incident area are removed
    only from transport. Their original IDs remain in the exported sequence and
    use the same-ID native prediction as a declared fallback.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all() or len(vertices) < 4):
        raise ValueError("Finite [V,3] mesh vertices required")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(vertices))):
        raise ValueError("Valid shared triangle topology required")
    faces = faces.astype(np.int64, copy=False)
    triangles = vertices[faces]
    doubled = np.linalg.norm(np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0]), axis=1)
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    scale_squared = float(np.mean(np.sum(centered * centered, axis=1)))
    if not math.isfinite(scale_squared) or scale_squared <= 0.0:
        raise ValueError("Positive finite rotation-invariant mesh scale required")
    face_tolerance = 100.0 * np.finfo(np.float64).eps * scale_squared
    retained_faces = faces[np.isfinite(doubled) & (doubled > face_tolerance)]
    retained_doubled = doubled[np.isfinite(doubled) & (doubled > face_tolerance)]
    if not len(retained_faces):
        raise ValueError("No positive-area transport face remains")
    area = np.zeros(len(vertices), dtype=np.float64)
    for corner in range(3):
        np.add.at(area, retained_faces[:, corner], retained_doubled / 6.0)
    active = np.zeros(len(vertices), dtype=bool)
    active[retained_faces.reshape(-1)] = True
    active_ids = np.flatnonzero(active).astype(np.int64)
    if not len(active_ids):
        raise ValueError("No positive-area transport vertex remains")
    edges = _edges(retained_faces)
    length = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    scale = math.sqrt(float(area[active_ids].sum()))
    if not np.isfinite(length).all() or np.any(length <= 0.0) or scale <= 0.0:
        raise ValueError("Positive finite mesh scale and edge lengths required")
    degree = np.zeros(len(vertices), dtype=np.float64)
    first = np.zeros(len(vertices), dtype=np.float64)
    second = np.zeros(len(vertices), dtype=np.float64)
    for endpoint in (0, 1):
        np.add.at(degree, edges[:, endpoint], 1.0)
        np.add.at(first, edges[:, endpoint], length)
        np.add.at(second, edges[:, endpoint], length * length)
    if np.any(degree[active_ids] <= 0):
        raise ValueError("Every retained vertex needs a nonempty one-ring")
    mean = np.zeros(len(vertices), dtype=np.float64)
    deviation = np.zeros(len(vertices), dtype=np.float64)
    mean[active_ids] = first[active_ids] / degree[active_ids]
    deviation[active_ids] = np.sqrt(np.maximum(
        0.0, second[active_ids] / degree[active_ids] - mean[active_ids] ** 2))
    triangles = vertices[retained_faces]
    angle_sum = np.zeros(len(vertices), dtype=np.float64)
    for corner in range(3):
        left = triangles[:, (corner + 1) % 3] - triangles[:, corner]
        right = triangles[:, (corner + 2) % 3] - triangles[:, corner]
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        if np.any(denominator <= 0.0):
            raise ValueError("Degenerate triangle encountered")
        cosine = np.clip(np.einsum("ij,ij->i", left, right) / denominator, -1.0, 1.0)
        np.add.at(angle_sum, retained_faces[:, corner], np.arccos(cosine))
    edge_occurrences: dict[tuple[int, int], int] = {}
    for left, right in np.concatenate((
            retained_faces[:, :2], retained_faces[:, 1:],
            retained_faces[:, ::2]), axis=0):
        key = tuple(sorted((int(left), int(right))))
        edge_occurrences[key] = edge_occurrences.get(key, 0) + 1
    boundary = np.zeros(len(vertices), dtype=bool)
    for (left, right), count in edge_occurrences.items():
        if count == 1:
            boundary[left] = boundary[right] = True
    reference_angle = np.where(boundary, np.pi, 2.0 * np.pi)
    normalized_area = area / area[active_ids].sum()
    safe_mean = mean.copy()
    safe_mean[degree <= 0] = scale
    descriptor = np.column_stack((
        np.log(np.maximum(normalized_area, np.finfo(np.float64).tiny)),
        np.log(safe_mean / scale), deviation / scale,
        degree / max(1.0, float(degree.max())),
        (reference_angle - angle_sum) / reference_angle,
    ))[active_ids]
    if not np.isfinite(descriptor).all() or np.any(area[active_ids] <= 0.0):
        raise ValueError("Nonfinite geometry descriptor")
    mass = area[active_ids] / area[active_ids].sum()
    return descriptor, mass, active_ids


def _scaled_features(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate((source, target), axis=0)
    center = joined.mean(axis=0)
    scale = np.sqrt(np.mean((joined - center) ** 2, axis=0))
    scale = np.maximum(scale, 128.0 * np.finfo(np.float64).eps)
    return (source - center) / scale, (target - center) / scale


def _greedy_witness(source: np.ndarray, target: np.ndarray):
    """Northwest-corner feasible plan; vertex order is only a feasibility key."""
    original_source = source.copy(); original_target = target.copy()
    source = source.copy(); target = target.copy()
    rows, columns, mass = [], [], []
    i = j = 0
    while i < len(source) and j < len(target):
        source_value, target_value = source[i], target[j]
        value = min(source_value, target_value)
        if value > 0.0:
            rows.append(i); columns.append(j); mass.append(value)
        source[i] -= value; target[j] -= value
        if source_value <= target_value:
            i += 1
        if target_value <= source_value:
            j += 1
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    mass = np.asarray(mass, dtype=np.float64)
    observed_source = np.zeros(len(original_source), dtype=np.float64)
    observed_target = np.zeros(len(original_target), dtype=np.float64)
    np.add.at(observed_source, rows, mass)
    np.add.at(observed_target, columns, mass)
    # Independently normalized floating marginals can differ by a final ulp.
    # Certify the actual mass residuals, not exact simultaneous index exhaustion.
    if (not np.allclose(observed_source, original_source,
                               atol=1e-15, rtol=1e-14)
            or not np.allclose(observed_target, original_target,
                               atol=1e-15, rtol=1e-14)):
        raise RuntimeError("Could not construct exact sparse marginal witness")
    return rows, columns, mass


def _strict_witness_on_support(edge_keys: set[tuple[int, int]],
                               source: np.ndarray,
                               target: np.ndarray) -> dict[tuple[int, int], float]:
    """Put positive dust on every edge, then close residual marginals."""
    n, m = len(source), len(target)
    row_degree = np.zeros(n, dtype=np.int64)
    column_degree = np.zeros(m, dtype=np.int64)
    for row, column in edge_keys:
        row_degree[row] += 1; column_degree[column] += 1
    plan: dict[tuple[int, int], float] = {}
    dust_fraction = 1e-6
    for row, column in sorted(edge_keys):
        value = dust_fraction * min(
            source[row] / row_degree[row],
            target[column] / column_degree[column])
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError("Could not assign positive support dust")
        plan[(row, column)] = float(value)
    residual_source = source.copy(); residual_target = target.copy()
    for (row, column), value in plan.items():
        residual_source[row] -= value; residual_target[column] -= value
    if (np.any(residual_source <= 0.0) or np.any(residual_target <= 0.0)
            or not np.isclose(residual_source.sum(), residual_target.sum(),
                              atol=1e-15, rtol=1e-14)):
        raise RuntimeError("Positive support dust exhausted a marginal")
    rows, columns, mass = _greedy_witness(residual_source, residual_target)
    for row, column, value in zip(rows, columns, mass):
        key = (int(row), int(column))
        plan[key] = plan.get(key, 0.0) + float(value)
    observed_source = np.zeros(n, dtype=np.float64)
    observed_target = np.zeros(m, dtype=np.float64)
    for (row, column), value in plan.items():
        observed_source[row] += value; observed_target[column] += value
    if (set(edge_keys) - set(plan)
            or not all(value > 0.0 and math.isfinite(value)
                       for value in plan.values())
            or not np.allclose(observed_source, source, atol=1e-14, rtol=1e-13)
            or not np.allclose(observed_target, target, atol=1e-14, rtol=1e-13)):
        raise RuntimeError("Strict support witness does not close marginals")
    return plan


def build_sparse_support(source_features: np.ndarray, target_features: np.ndarray,
                         source_mass: np.ndarray, target_mass: np.ndarray,
                         *, neighbors: int, source_ids: np.ndarray | None = None,
                         target_ids: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Union feature-kNN, identity and an explicit feasible area witness."""
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)
    source_mass = np.asarray(source_mass, dtype=np.float64)
    target_mass = np.asarray(target_mass, dtype=np.float64)
    if (source_features.ndim != 2 or target_features.ndim != 2
            or target_features.shape[1:] != source_features.shape[1:]
            or source_mass.shape != (len(source_features),)
            or target_mass.shape != (len(target_features),)
            or not all(np.isfinite(x).all() for x in (
                source_features, target_features, source_mass, target_mass))
            or np.any(source_mass <= 0.0) or np.any(target_mass <= 0.0)):
        raise ValueError("Positive masses and matching finite feature banks required")
    if not np.isclose(source_mass.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Source mass must sum to one")
    if not np.isclose(target_mass.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Target mass must sum to one")
    neighbors = _positive_integer("neighbors", neighbors)
    if neighbors > len(target_features):
        raise ValueError("neighbors exceeds retained target vertex count")
    source_ids = (np.arange(len(source_features), dtype=np.int64)
                  if source_ids is None else np.asarray(source_ids))
    target_ids = (np.arange(len(target_features), dtype=np.int64)
                  if target_ids is None else np.asarray(target_ids))
    if (source_ids.shape != (len(source_features),)
            or target_ids.shape != (len(target_features),)
            or not np.issubdtype(source_ids.dtype, np.integer)
            or not np.issubdtype(target_ids.dtype, np.integer)
            or len(set(source_ids.tolist())) != len(source_ids)
            or len(set(target_ids.tolist())) != len(target_ids)):
        raise ValueError("Distinct retained original vertex IDs required")
    # Retrieve the kth radius, then include every boundary tie and select by
    # exact squared descriptor cost followed by original target vertex ID.
    from scipy.spatial import cKDTree
    tree = cKDTree(target_features)
    distances, _ = tree.query(
        source_features, k=neighbors, workers=1)
    if neighbors == 1:
        distances = distances[:, None]
    edges: dict[tuple[int, int], tuple[float, float]] = {}
    knn_edges: set[tuple[int, int]] = set()
    target_column_by_id = {
        int(original_id): column
        for column, original_id in enumerate(target_ids.tolist())
    }
    for row in range(len(source_features)):
        radius = np.nextafter(float(distances[row, -1]), math.inf)
        tied = np.asarray(tree.query_ball_point(
            source_features[row], radius, workers=1), dtype=np.int64)
        if len(tied) < neighbors:
            raise RuntimeError("kNN boundary query returned too few candidates")
        exact_cost = np.sum(
            (target_features[tied] - source_features[row]) ** 2, axis=1)
        order = np.lexsort((target_ids[tied], exact_cost))[:neighbors]
        selected = tied[order]
        for column, cost in zip(selected, exact_cost[order]):
            key = (row, int(column))
            edges[key] = (float(cost), 0.0)
            knn_edges.add(key)
        column = target_column_by_id.get(int(source_ids[row]))
        if column is not None:
            identity_cost = float(np.sum(
                (source_features[row] - target_features[column]) ** 2))
            edges.setdefault((row, column), (identity_cost, 0.0))
    uniform_source = np.full(len(source_mass), 1.0 / len(source_mass))
    uniform_target = np.full(len(target_mass), 1.0 / len(target_mass))
    # Close the support monotonically until both frozen marginal systems have
    # a strictly positive feasible plan on every retained edge. This is the
    # total-support condition needed by diagonal Sinkhorn scaling; a merely
    # feasible NW plan is not sufficient for triangular sparse supports.
    # Each nonterminal round adds at least one previously absent bipartite
    # edge, so this complete-graph deficit is a finite guaranteed bound.
    maximum_closure_rounds = (
        len(source_mass) * len(target_mass) - len(edges) + 1)
    for _ in range(maximum_closure_rounds):
        keys = set(edges)
        area_plan = _strict_witness_on_support(keys, source_mass, target_mass)
        uniform_plan = _strict_witness_on_support(
            keys, uniform_source, uniform_target)
        added = (set(area_plan) | set(uniform_plan)) - keys
        if not added:
            break
        for row, column in added:
            cost = float(np.sum(
                (source_features[row] - target_features[column]) ** 2))
            edges[(row, column)] = (cost, 0.0)
    else:
        raise RuntimeError("Strict shared support closure did not stabilize")
    ordered_edges = sorted(edges)
    rows = np.asarray([item[0] for item in ordered_edges], dtype=np.int64)
    columns = np.asarray([item[1] for item in ordered_edges], dtype=np.int64)
    costs = np.asarray([edges[item][0] for item in ordered_edges], dtype=np.float64)
    witness = np.asarray([area_plan[item] for item in ordered_edges], dtype=np.float64)
    uniform_witness = np.asarray(
        [uniform_plan[item] for item in ordered_edges], dtype=np.float64)
    if not np.isfinite(costs).all() or np.any(costs < 0.0):
        raise ValueError("Finite nonnegative descriptor costs required")
    if np.any(witness <= 0.0) or np.any(uniform_witness <= 0.0):
        raise RuntimeError("Shared support lacks strict positive witnesses")
    ordered_knn = sorted(knn_edges)
    return {
        "rows": rows, "columns": columns, "costs": costs,
        "witness": witness, "uniform_witness": uniform_witness,
        "knn_rows": np.asarray([item[0] for item in ordered_knn], dtype=np.int64),
        "knn_columns": np.asarray(
            [item[1] for item in ordered_knn], dtype=np.int64),
    }


def _group_logsumexp(indices: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    result = np.full(size, -np.inf, dtype=np.float64)
    np.logaddexp.at(result, indices, values)
    return result


def _bounded_dual_newton(potential, evaluate):
    """One bounded Newton-CG step on the unchanged entropic dual objective.

    evaluate returns value, gradient, Hessian-vector product and a positive
    diagonal scale. Gauge/ridge terms regularize only the search direction;
    acceptance always uses the original objective and marginal residuals.
    """
    from scipy.sparse.linalg import LinearOperator, cg

    value, gradient, hessian_vector, scale = evaluate(potential)
    size = len(potential)
    ridge = 1e-12 * max(float(np.max(scale)), np.finfo(np.float64).tiny)
    def product(vector):
        return hessian_vector(vector) + ridge * vector + np.mean(vector)
    operator = LinearOperator((size, size), matvec=product, dtype=np.float64)
    direction, _ = cg(operator, -gradient, rtol=1e-7, atol=0.,
                      maxiter=min(size, 100))
    direction -= np.mean(direction)
    slope = float(np.dot(gradient, direction))
    if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0.:
        return potential
    # Bound potential movement even for almost disconnected sparse kernels.
    maximum = float(np.max(np.abs(direction)))
    if maximum > 20.:
        direction *= 20. / maximum
        slope = float(np.dot(gradient, direction))
    error = float(np.max(np.abs(gradient)))
    for backtrack in range(20):
        step = 0.5 ** backtrack
        trial = potential + step * direction
        trial -= np.mean(trial)
        trial_value, trial_gradient, _, _ = evaluate(trial)
        rounding = 32 * np.finfo(np.float64).eps * max(1., abs(value))
        if (np.isfinite(trial_value) and
                (trial_value <= value + 1e-4 * step * slope or
                 abs(trial_value - value) <= rounding and
                 np.max(np.abs(trial_gradient)) < error)):
            return trial
    return potential


def sparse_sinkhorn(rows: np.ndarray, columns: np.ndarray, costs: np.ndarray,
                    source_mass: np.ndarray, target_mass: np.ndarray, *,
                    epsilon: float, tolerance: float,
                    max_iterations: int) -> tuple[np.ndarray, dict]:
    epsilon = _positive("epsilon", epsilon)
    tolerance = _positive("tolerance", tolerance)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    rows, columns = np.asarray(rows), np.asarray(columns)
    costs = np.asarray(costs, dtype=np.float64)
    source_mass, target_mass = (np.asarray(source_mass, dtype=np.float64),
                                np.asarray(target_mass, dtype=np.float64))
    n, m = len(source_mass), len(target_mass)
    if (rows.shape != columns.shape or rows.shape != costs.shape or rows.ndim != 1
            or not np.issubdtype(rows.dtype, np.integer)
            or not np.issubdtype(columns.dtype, np.integer)
            or source_mass.shape != (n,) or target_mass.shape != (m,)
            or n < 1 or not np.isfinite(costs).all() or np.any(costs < 0.0)
            or not np.isfinite(source_mass).all()
            or not np.isfinite(target_mass).all()
            or np.any(rows < 0) or np.any(rows >= n)
            or np.any(columns < 0) or np.any(columns >= m)
            or np.any(source_mass <= 0.0) or np.any(target_mass <= 0.0)
            or not np.isclose(source_mass.sum(), 1.0, atol=1e-12, rtol=0)
            or not np.isclose(target_mass.sum(), 1.0, atol=1e-12, rtol=0)
            or len(set(zip(rows.tolist(), columns.tolist()))) != len(rows)
            or set(rows.tolist()) != set(range(n))
            or set(columns.tolist()) != set(range(m))):
        raise ValueError("Valid sparse transport arrays required")
    log_kernel = -costs / epsilon
    log_source, log_target = np.log(source_mass), np.log(target_mass)
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(m, dtype=np.float64)
    def dual_state(potential):
        norm = _group_logsumexp(rows, log_kernel + potential[columns], n)
        weights = np.exp(log_source[rows] + log_kernel + potential[columns] - norm[rows])
        column_mass = np.bincount(columns, weights=weights, minlength=m)
        gradient = column_mass - target_mass
        def hessian_vector(vector):
            row_product = np.bincount(rows, weights=weights * vector[columns], minlength=n)
            return (column_mass * vector - np.bincount(
                columns, weights=weights * (row_product / source_mass)[rows], minlength=m))
        value = float(np.dot(source_mass, norm) - np.dot(target_mass, potential))
        return value, gradient, hessian_vector, column_mass

    residual = relative_residual = math.inf
    for iteration in range(1, max_iterations + 1):
        row_norm = _group_logsumexp(rows, log_kernel + log_v[columns], n)
        if not np.isfinite(row_norm).all():
            raise RuntimeError("Sparse support has an empty source row")
        log_u = log_source - row_norm
        column_norm = _group_logsumexp(columns, log_kernel + log_u[rows], m)
        if not np.isfinite(column_norm).all():
            raise RuntimeError("Sparse support has an empty target column")
        log_v = log_target - column_norm
        if iteration % 10 == 0:
            log_v = _bounded_dual_newton(log_v, dual_state)
            log_u = log_source - _group_logsumexp(rows, log_kernel + log_v[columns], n)
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            plan = np.exp(log_u[rows] + log_kernel + log_v[columns])
            row_sum = np.zeros(n); column_sum = np.zeros(m)
            np.add.at(row_sum, rows, plan); np.add.at(column_sum, columns, plan)
            residual = max(float(np.max(np.abs(row_sum - source_mass))),
                           float(np.max(np.abs(column_sum - target_mass))))
            relative_residual = max(
                float(np.max(np.abs(row_sum - source_mass) / source_mass)),
                float(np.max(np.abs(column_sum - target_mass) / target_mass)))
            if (np.isfinite(plan).all() and residual <= tolerance
                    and relative_residual <= tolerance):
                break
    if (not np.isfinite(plan).all() or not math.isfinite(relative_residual)
            or residual > tolerance or relative_residual > tolerance):
        raise RuntimeError("Sparse Sinkhorn did not close both frozen marginals: "
                           f"iteration={iteration}, absolute={residual:.17g}, "
                           f"relative={relative_residual:.17g}, "
                           f"log_kernel_range=[{log_kernel.min():.17g},{log_kernel.max():.17g}]")
    objective = float(np.dot(plan, costs) + epsilon * np.dot(
        plan, np.log(np.maximum(plan, np.finfo(np.float64).tiny)) - 1.0))
    row_sum = np.zeros(n); column_sum = np.zeros(m)
    np.add.at(row_sum, rows, plan); np.add.at(column_sum, columns, plan)
    row_residual = float(np.max(np.abs(row_sum - source_mass)))
    column_residual = float(np.max(np.abs(column_sum - target_mass)))
    return plan, {"iterations": iteration,
                  "maximum_marginal_residual": max(row_residual, column_residual),
                  "maximum_row_residual": row_residual,
                  "maximum_column_residual": column_residual,
                  "maximum_relative_row_residual": float(np.max(
                      np.abs(row_sum - source_mass) / source_mass)),
                  "maximum_relative_column_residual": float(np.max(
                      np.abs(column_sum - target_mass) / target_mass)),
                  "column_marginal": "enforced",
                  "objective": objective, "support_edges": len(rows)}


def _row_softmax(rows: np.ndarray, costs: np.ndarray, n: int,
                 epsilon: float) -> np.ndarray:
    log_weight = -costs / epsilon
    normalizer = _group_logsumexp(rows, log_weight, n)
    if not np.isfinite(normalizer).all():
        raise RuntimeError("Sparse support has an empty row")
    return np.exp(log_weight - normalizer[rows])


def _lift(rows: np.ndarray, columns: np.ndarray, plan: np.ndarray,
          row_mass: np.ndarray, target_vertices: np.ndarray) -> np.ndarray:
    output = np.zeros((len(row_mass), 3), dtype=np.float64)
    observed_mass = np.zeros(len(row_mass), dtype=np.float64)
    if not np.isfinite(plan).all() or np.any(plan < 0.0):
        raise RuntimeError("Finite nonnegative transport weights required")
    np.add.at(observed_mass, rows, plan)
    if np.any(observed_mass <= 0.0) or not np.isfinite(observed_mass).all():
        raise RuntimeError("Every transported barycentre needs positive mass")
    np.add.at(output, rows, plan[:, None] * target_vertices[columns])
    # Exact OT has observed_mass == row_mass. Normalize the numerical iterate
    # by its actual mass to preserve barycentres and translation equivariance.
    output /= observed_mass[:, None]
    if not np.isfinite(output).all():
        raise RuntimeError("Nonfinite transported barycentre")
    return output


def _support_hash(support: dict[str, np.ndarray]) -> str:
    value = hashlib.sha256()
    for name in ("rows", "columns", "costs"):
        array = np.ascontiguousarray(support[name])
        value.update(name.encode())
        value.update(array.dtype.str.encode())
        value.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        value.update(array.tobytes())
    return value.hexdigest()


def build_role_sequence(vertices: np.ndarray, faces: np.ndarray, role: str, *,
                        epsilon: float, neighbors: int, tolerance: float,
                        max_iterations: int) -> tuple[np.ndarray, dict, dict]:
    if role not in ROLES:
        raise ValueError("Unknown C06 role: " + role)
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if (vertices.dtype != np.dtype(np.float32) or vertices.ndim != 3
            or vertices.shape[0] != 16 or vertices.shape[-1] != 3
            or not np.isfinite(vertices).all()):
        raise ValueError("Exactly sixteen float32 native frames required")
    epsilon = _positive("epsilon", epsilon)
    neighbors = _positive_integer("neighbors", neighbors)
    tolerance = _positive("tolerance", tolerance)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    anchor = vertices[0].astype(np.float64)
    source_descriptor, source_area, source_ids = geometry_descriptors(anchor, faces)
    uniform_source = np.full(
        len(source_ids), 1.0 / len(source_ids), dtype=np.float64)
    output = vertices.astype(np.float64).copy()
    row_residuals, column_residuals = [], []
    relative_row_residuals, relative_column_residuals = [], []
    iterations, objectives, support_sizes, support_hashes = [], [], [], []
    active_source_masks, active_target_masks = [], []
    for frame in range(1, 16):
        target = vertices[frame].astype(np.float64)
        target_descriptor, target_area, target_ids = geometry_descriptors(target, faces)
        uniform_target = np.full(
            len(target_ids), 1.0 / len(target_ids), dtype=np.float64)
        source_feature, target_feature = _scaled_features(
            source_descriptor, target_descriptor)
        support = build_sparse_support(
            source_feature, target_feature, source_area, target_area,
            neighbors=neighbors, source_ids=source_ids, target_ids=target_ids)
        if role == "row_softmax":
            plan = _row_softmax(support["rows"], support["costs"],
                                len(source_ids), epsilon)
            row_mass = np.ones(len(source_ids), dtype=np.float64)
            observed_row = np.zeros(len(source_ids), dtype=np.float64)
            np.add.at(observed_row, support["rows"], plan)
            diagnostics = {
                "iterations": 0,
                "maximum_row_residual": float(np.max(np.abs(observed_row - 1.0))),
                "maximum_column_residual": None,
                "maximum_relative_row_residual": float(np.max(
                    np.abs(observed_row - 1.0))),
                "maximum_relative_column_residual": None,
                "column_marginal": "not_enforced",
                "transport_cost": float(np.dot(plan, support["costs"])),
                "entropic_objective": None,
                "support_edges": len(plan),
            }
        else:
            row_mass = (uniform_source if role == "vertex_density_transport"
                        else source_area)
            column_mass = (uniform_target if role == "vertex_density_transport"
                           else target_area)
            # Support construction has already certified a strictly positive
            # feasible witness for both this uniform system and the area system.
            plan, diagnostics = sparse_sinkhorn(
                support["rows"], support["columns"], support["costs"],
                row_mass, column_mass, epsilon=epsilon,
                tolerance=tolerance, max_iterations=max_iterations)
            diagnostics["transport_cost"] = float(np.dot(plan, support["costs"]))
            diagnostics["entropic_objective"] = diagnostics.pop("objective")
        output[frame, source_ids] = _lift(
            support["rows"], support["columns"], plan, row_mass,
            target[target_ids])
        # Source vertices removed from transport retain the same-ID native
        # prediction already present in output[frame].
        row_residuals.append(diagnostics["maximum_row_residual"])
        column_residuals.append(-1.0 if diagnostics["maximum_column_residual"] is None
                                else diagnostics["maximum_column_residual"])
        relative_row_residuals.append(diagnostics["maximum_relative_row_residual"])
        relative_column_residuals.append(
            -1.0 if diagnostics["maximum_relative_column_residual"] is None
            else diagnostics["maximum_relative_column_residual"])
        iterations.append(diagnostics["iterations"])
        objectives.append(-1.0 if diagnostics["entropic_objective"] is None
                          else diagnostics["entropic_objective"])
        support_sizes.append(diagnostics["support_edges"])
        support_hashes.append(_support_hash(support))
        source_mask = np.zeros(len(anchor), dtype=np.uint8)
        target_mask = np.zeros(len(anchor), dtype=np.uint8)
        source_mask[source_ids] = 1; target_mask[target_ids] = 1
        active_source_masks.append(source_mask); active_target_masks.append(target_mask)
    exported = output.astype(np.float32)
    exported[0] = vertices[0]
    if not np.array_equal(exported[0], vertices[0]) or not np.isfinite(exported).all():
        raise RuntimeError("C06 float32 export lost anchor or finiteness")
    certificate = {
        "maximum_row_residual": np.asarray(row_residuals, dtype=np.float64),
        "maximum_column_residual_or_minus_one": np.asarray(
            column_residuals, dtype=np.float64),
        "column_marginal_enforced": np.full(
            15, role != "row_softmax", dtype=np.uint8),
        "maximum_relative_row_residual": np.asarray(
            relative_row_residuals, dtype=np.float64),
        "maximum_relative_column_residual_or_minus_one": np.asarray(
            relative_column_residuals, dtype=np.float64),
        "iterations": np.asarray(iterations, dtype=np.int64),
        "entropic_objective_or_minus_one": np.asarray(objectives, dtype=np.float64),
        "support_edges": np.asarray(support_sizes, dtype=np.int64),
        "support_sha256": np.asarray(support_hashes, dtype="S64"),
        "active_source_mask": np.asarray(active_source_masks, dtype=np.uint8),
        "active_target_mask": np.asarray(active_target_masks, dtype=np.uint8),
    }
    report = {"frames": 15, "feature_source": FEATURE_SOURCE,
              "maximum_row_residual": max(row_residuals, default=0.0),
              "maximum_column_residual": (None if role == "row_softmax" else
                  max(column_residuals, default=0.0)),
              "maximum_relative_row_residual": max(
                  relative_row_residuals, default=0.0),
              "maximum_relative_column_residual": (
                  None if role == "row_softmax" else
                  max(relative_column_residuals, default=0.0)),
              "column_marginal": ("not_enforced" if role == "row_softmax"
                                    else "enforced"),
              "maximum_iterations": max(iterations, default=0),
              "maximum_support_edges": max(support_sizes, default=0),
              "transport_active_source_vertices": int(len(source_ids)),
              "native_fallback_source_vertices": int(len(anchor) - len(source_ids)),
              "convex_hull_only": True,
              "surface_membership_guaranteed": False}
    return exported, certificate, report


def build_transport_sequences(vertices: np.ndarray, faces: np.ndarray, *,
                              epsilon: float, neighbors: int, tolerance: float,
                              max_iterations: int):
    outputs, reports, certificates = {}, {}, {}
    for role in ROLES:
        output, certificate, report = build_role_sequence(
            vertices, faces, role, epsilon=epsilon, neighbors=neighbors,
            tolerance=tolerance, max_iterations=max_iterations)
        outputs[role], reports[role], certificates[role] = output, report, certificate
    shared = {"feature_source": FEATURE_SOURCE,
              "support_policy": "feature_knn_plus_identity_plus_area_and_uniform_feasible_witnesses",
              "lift": "per-source transported target-coordinate barycentre",
              "ground_truth_or_scorer_input": False}
    return outputs, reports, certificates, shared


def _load_source(source_case: Path, uid: str, expected_sequence_sha256: str):
    sequence = Path(source_case) / "sequence.npz"
    report_path = sequence.with_name("report.json")
    if sequence.is_symlink() or report_path.is_symlink():
        raise ValueError("Physical receipt-bound source files required")
    if digest(sequence) != expected_sequence_sha256:
        raise ValueError("Source sequence differs from pinned digest")
    report = json.loads(report_path.read_text())
    if (report.get("status") != "completed" or report.get("uid") != uid
            or report.get("sha256", {}).get("sequence.npz") != expected_sequence_sha256
            or isinstance(report.get("seed"), bool)
            or not isinstance(report.get("seed"), int)):
        raise ValueError("Completed same-UID source report and integer seed required")
    with np.load(sequence, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    vertices, faces, times = validate_native_arrays(arrays)
    return arrays, vertices, faces, times, report_path, report["seed"]


def _artifact_members(output: Path, reports: list[dict]) -> list[Path]:
    paths = [output / "candidate.json", output / "manifest.json",
             output / "common-target.npz"]
    for report in reports:
        directory = output / report["candidate_arm"]
        paths.append(directory / "report.json")
        if report["status"] == "completed":
            paths.extend((directory / "sequence.npz", directory / "certificate.npz"))
    return sorted(paths, key=lambda path: path.relative_to(output).as_posix())


def _write_archive(output: Path, reports: list[dict], maximum: int) -> None:
    maximum = _positive_integer("max_artifact_bytes", maximum)
    rows = [{"path": path.relative_to(output).as_posix(),
             "sha256": digest(path), "size_bytes": path.stat().st_size}
            for path in _artifact_members(output, reports)]
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
        temporary.unlink()
        raise RuntimeError("C06 artifact exceeds frozen byte ceiling")
    temporary.replace(archive)
    write_json(output / "artifact-archive.json", {
        "kind": "c06-terminal-artifact-archive", "version": 1,
        "archive": {"path": "artifact.tar", "sha256": digest(archive),
                    "size_bytes": archive.stat().st_size},
        "members": rows, "max_artifact_bytes": maximum})


def _validate_archive(output: Path, reports: list[dict]) -> dict:
    archive, record_path = output / "artifact.tar", output / "artifact-archive.json"
    record = json.loads(record_path.read_text())
    rows = [{"path": path.relative_to(output).as_posix(),
             "sha256": digest(path), "size_bytes": path.stat().st_size}
            for path in _artifact_members(output, reports)]
    if (record.get("kind") != "c06-terminal-artifact-archive"
            or record.get("version") != 1 or record.get("members") != rows
            or record.get("archive") != {"path": "artifact.tar",
                "sha256": digest(archive), "size_bytes": archive.stat().st_size}
            or archive.stat().st_size > record.get("max_artifact_bytes", -1)):
        raise ValueError("C06 artifact archive record changed")
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        if [member.name for member in members] != [row["path"] for row in rows]:
            raise ValueError("C06 archive inventory changed")
        for member, row in zip(members, rows):
            stream = bundle.extractfile(member)
            if (not member.isfile() or member.size != row["size_bytes"]
                    or member.mode != 0o644 or member.uid or member.gid or member.mtime
                    or member.uname or member.gname or stream is None
                    or hashlib.sha256(stream.read()).hexdigest() != row["sha256"]):
                raise ValueError("C06 archive member or metadata changed")
    return {"archive": {"path": archive, "sha256": digest(archive)},
            "record": {"path": record_path, "sha256": digest(record_path)}}


def export_area_transport_candidate(source_case: Path, output: Path, *, uid: str,
                                    expected_sequence_sha256: str,
                                    source_refs: dict, producer_provenance_ref: dict,
                                    producer_provenance_path: Path, epsilon: float,
                                    neighbors: int, tolerance: float,
                                    max_iterations: int,
                                    coordinate_lower: float,
                                    coordinate_upper: float,
                                    bounds_policy: str,
                                    max_artifact_bytes: int) -> dict:
    started = time.monotonic()
    if not uid or Path(uid).name != uid or uid in (".", ".."):
        raise ValueError("Filesystem-safe UID required")
    if (set(source_refs) != {"sequence", "report"}
            or any(set(ref) != {"path", "sha256"} for ref in source_refs.values())):
        raise ValueError("Exact source sequence/report references required")
    if (not isinstance(producer_provenance_ref, dict)
            or set(producer_provenance_ref) != {"path", "sha256"}
            or not isinstance(producer_provenance_ref.get("path"), str)
            or Path(producer_provenance_ref["path"]).is_absolute()
            or ".." in Path(producer_provenance_ref["path"]).parts
            or not isinstance(producer_provenance_ref.get("sha256"), str)
            or len(producer_provenance_ref["sha256"]) != 64
            or digest(producer_provenance_path)
            != producer_provenance_ref["sha256"]):
        raise ValueError("Exact producer provenance reference required")
    producer_provenance = json.loads(Path(producer_provenance_path).read_text())
    provenance_relative = Path(producer_provenance_ref["path"])
    provenance_root = Path(producer_provenance_path).resolve()
    for _ in provenance_relative.parts:
        provenance_root = provenance_root.parent
    if ((provenance_root / provenance_relative).resolve()
            != Path(producer_provenance_path).resolve()):
        raise ValueError("Producer provenance path/ref alias changed")
    implementation_refs = _validate_producer_provenance(
        provenance_root, producer_provenance, source_refs)
    epsilon = _positive("epsilon", epsilon)
    neighbors = _positive_integer("neighbors", neighbors)
    tolerance = _positive("tolerance", tolerance)
    max_iterations = _positive_integer("max_iterations", max_iterations)
    if (not math.isfinite(coordinate_lower) or not math.isfinite(coordinate_upper)
            or coordinate_lower >= coordinate_upper):
        raise ValueError("Finite ordered coordinate bounds required")
    if bounds_policy not in ("preserve_and_report", "reject"):
        raise ValueError("Invalid coordinate bounds policy")
    arrays, vertices, faces, times, report_path, seed = _load_source(
        source_case, uid, expected_sequence_sha256)
    if (source_refs["sequence"]["sha256"] != expected_sequence_sha256
            or source_refs["report"]["sha256"] != digest(report_path)):
        raise ValueError("Source references differ from consumed pair")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    parameters = {"epsilon": epsilon, "neighbors": neighbors,
                  "tolerance": tolerance, "max_iterations": max_iterations,
                  "coordinate_bounds": [coordinate_lower, coordinate_upper],
                  "bounds_policy": bounds_policy,
                  "feature_source": FEATURE_SOURCE,
                  "support_policy": "feature_knn_plus_identity_plus_area_and_uniform_feasible_witnesses",
                  "lift": "per-source transported target-coordinate barycentre"}
    np.savez_compressed(output / "common-target.npz",
                        source_vertices=vertices, faces=faces, timesteps=times)
    reports = []
    implementation_sha = digest(Path(__file__))
    for role in ROLES:
        arm_started = time.monotonic(); directory = output / role; directory.mkdir()
        report = {"candidate_id": CANDIDATE_ID, "candidate_arm": role,
                  "method_id": METHOD_IDS[role], "uid": uid, "seed": seed,
                  "implementation_sha256": implementation_sha,
                  "source_sequence_sha256": expected_sequence_sha256,
                  "source_report_sha256": digest(report_path),
                  "producer_provenance_ref": producer_provenance_ref,
                  "implementation_refs": implementation_refs,
                  "parameters": parameters, "generated_unexecuted": False,
                  "native_qualified": False, "scientific_admission": False,
                  "local_method_verified": False,
                  "frame_completion": FRAME_COMPLETION,
                  "scientific_verdict": "not_computed"}
        try:
            result, certificate, diagnostics = build_role_sequence(
                vertices, faces, role, epsilon=epsilon, neighbors=neighbors,
                tolerance=tolerance, max_iterations=max_iterations)
            bounds = {"below": int(np.count_nonzero(result < coordinate_lower)),
                      "above": int(np.count_nonzero(result > coordinate_upper)),
                      "lower": coordinate_lower, "upper": coordinate_upper,
                      "policy": bounds_policy}
            if bounds_policy == "reject" and (bounds["below"] or bounds["above"]):
                raise ValueError("C06 output violates frozen coordinate bounds")
            np.savez_compressed(directory / "sequence.npz",
                                **{**arrays, "vertices": result})
            np.savez_compressed(directory / "certificate.npz", **certificate)
            report.update(status="completed", diagnostics=diagnostics,
                          bounds=bounds, sha256={
                              "sequence.npz": digest(directory / "sequence.npz"),
                              "certificate.npz": digest(directory / "certificate.npz")})
        except Exception as error:
            for name in ("sequence.npz", "certificate.npz"):
                (directory / name).unlink(missing_ok=True)
            report.update(status="error", exception_type=type(error).__name__,
                          error=str(error)[:4096])
        report["elapsed_seconds"] = time.monotonic() - arm_started
        write_json(directory / "report.json", report); reports.append(report)
    common_ref = {"path": "common-target.npz",
                  "sha256": digest(output / "common-target.npz")}
    completed = [row["candidate_arm"] for row in reports if row["status"] == "completed"]
    record = {"kind": "c06-area-transport-candidate", "version": 1,
              "candidate_id": CANDIDATE_ID, "uid": uid, "seed": seed,
              "roles": list(ROLES), "arms": reports,
              "status": "completed" if len(completed) == len(ROLES) else "incomplete",
              "source_refs": source_refs,
              "source_sequence_sha256": expected_sequence_sha256,
              "source_report_sha256": digest(report_path),
              "producer_provenance_ref": producer_provenance_ref,
              "implementation_refs": implementation_refs,
              "implementation_sha256": implementation_sha,
              "common_target": common_ref,
              "source_delivery_status": "generated_unexecuted_at_authoring",
              "native_qualified": False, "scientific_admission": False,
              "local_method_verified": False,
              "recorded_at": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": time.monotonic() - started,
              "timing_scope": "CPU geometry transport and artifact I/O only"}
    write_json(output / "candidate.json", record)
    write_json(output / "manifest.json", {
        "expected_roles": list(ROLES),
        "cases": [{"case_id": uid + "-" + role, "uid": uid,
                   "case_dir": role, "arm_role": role} for role in completed],
        "common_target": common_ref,
        "scope": "C06 geometry-only generated_unexecuted artifacts; no B0/B*/scoring/admission"})
    _write_archive(output, reports, max_artifact_bytes)
    return record


def validate_candidate_artifact(root: Path, candidate_path: Path) -> dict:
    root, candidate_path = Path(root).resolve(), Path(candidate_path).resolve()
    candidate_path.relative_to(root)
    record = json.loads(candidate_path.read_text())
    if (record.get("kind") != "c06-area-transport-candidate"
            or record.get("version") != 1 or record.get("candidate_id") != CANDIDATE_ID
            or record.get("roles") != list(ROLES)
            or len(record.get("arms", [])) != len(ROLES)
            or record.get("implementation_sha256") != digest(Path(__file__))
            or record.get("native_qualified") is not False
            or record.get("scientific_admission") is not False
            or record.get("local_method_verified") is not False):
        raise ValueError("Current generated-unexecuted C06 candidate required")
    if (record.get("source_delivery_status")
            != "generated_unexecuted_at_authoring"):
        raise ValueError("C06 source delivery status changed")
    directory = candidate_path.parent
    source_refs = record.get("source_refs")
    if not isinstance(source_refs, dict) or set(source_refs) != {"sequence", "report"}:
        raise ValueError("C06 source closure required")
    resolved = {}
    for key, ref in source_refs.items():
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref.get("path"), str)
                or not isinstance(ref.get("sha256"), str)
                or len(ref["sha256"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in ref["sha256"])
                or Path(ref["path"]).is_absolute()
                or ".." in Path(ref["path"]).parts):
            raise ValueError("Exact nonescaping C06 source ref required")
        path = (root / ref["path"]).resolve(); path.relative_to(root)
        if path.is_symlink() or not path.is_file() or digest(path) != ref["sha256"]:
            raise ValueError("Stale C06 source ref")
        resolved[key] = path
    provenance_ref = record.get("producer_provenance_ref")
    if (not isinstance(provenance_ref, dict)
            or set(provenance_ref) != {"path", "sha256"}
            or not isinstance(provenance_ref.get("path"), str)
            or Path(provenance_ref["path"]).is_absolute()
            or ".." in Path(provenance_ref["path"]).parts
            or not isinstance(provenance_ref.get("sha256"), str)
            or len(provenance_ref["sha256"]) != 64):
        raise ValueError("C06 producer provenance ref required")
    provenance_path = (root / provenance_ref["path"]).resolve()
    provenance_path.relative_to(root)
    if (provenance_path.is_symlink() or not provenance_path.is_file()
            or digest(provenance_path) != provenance_ref["sha256"]):
        raise ValueError("Stale C06 producer provenance ref")
    provenance = json.loads(provenance_path.read_text())
    implementation_refs = _validate_producer_provenance(
        root, provenance, source_refs)
    if record.get("implementation_refs") != implementation_refs:
        raise ValueError("C06 producer closure changed")
    arrays, vertices, faces, times, report_path, seed = _load_source(
        resolved["sequence"].parent, record["uid"], source_refs["sequence"]["sha256"])
    if (resolved["report"] != report_path.resolve() or record.get("seed") != seed
            or record.get("source_sequence_sha256") != source_refs["sequence"]["sha256"]
            or record.get("source_report_sha256") != source_refs["report"]["sha256"]):
        raise ValueError("C06 source aliases changed")
    common = directory / "common-target.npz"
    common_ref = {"path": "common-target.npz", "sha256": digest(common)}
    if record.get("common_target") != common_ref:
        raise ValueError("C06 common source snapshot changed")
    with np.load(common, allow_pickle=False) as saved:
        if (set(saved.files) != {"source_vertices", "faces", "timesteps"}
                or not np.array_equal(saved["source_vertices"], vertices)
                or not np.array_equal(saved["faces"], faces)
                or not np.array_equal(saved["timesteps"], times)):
            raise ValueError("C06 common source snapshot differs from B0")
    completed = []
    parameters = record["arms"][0].get("parameters")
    expected_parameter_keys = {
        "epsilon", "neighbors", "tolerance", "max_iterations",
        "coordinate_bounds", "bounds_policy", "feature_source",
        "support_policy", "lift",
    }
    if (not isinstance(parameters, dict)
            or set(parameters) != expected_parameter_keys
            or _positive("epsilon", parameters["epsilon"]) != parameters["epsilon"]
            or _positive_integer("neighbors", parameters["neighbors"])
            != parameters["neighbors"]
            or _positive("tolerance", parameters["tolerance"])
            != parameters["tolerance"]
            or _positive_integer("max_iterations", parameters["max_iterations"])
            != parameters["max_iterations"]
            or not isinstance(parameters["coordinate_bounds"], list)
            or len(parameters["coordinate_bounds"]) != 2
            or not all(math.isfinite(value)
                       for value in parameters["coordinate_bounds"])
            or parameters["coordinate_bounds"][0]
            >= parameters["coordinate_bounds"][1]
            or parameters["bounds_policy"] not in ("preserve_and_report", "reject")
            or parameters["feature_source"] != FEATURE_SOURCE
            or parameters["support_policy"]
            != "feature_knn_plus_identity_plus_area_and_uniform_feasible_witnesses"
            or parameters["lift"]
            != "per-source transported target-coordinate barycentre"):
        raise ValueError("C06 frozen parameters required")
    for role, saved_report in zip(ROLES, record["arms"]):
        path = directory / role / "report.json"; report = json.loads(path.read_text())
        if (report != saved_report or report.get("candidate_arm") != role
                or report.get("candidate_id") != CANDIDATE_ID
                or report.get("method_id") != METHOD_IDS[role]
                or report.get("uid") != record["uid"]
                or report.get("seed") != record["seed"]
                or report.get("implementation_sha256") != digest(Path(__file__))
                or report.get("source_sequence_sha256")
                != source_refs["sequence"]["sha256"]
                or report.get("source_report_sha256")
                != source_refs["report"]["sha256"]
                or report.get("producer_provenance_ref") != provenance_ref
                or report.get("implementation_refs") != implementation_refs
                or report.get("parameters") != parameters
                or report.get("generated_unexecuted") is not False
                or report.get("native_qualified") is not False
                or report.get("scientific_admission") is not False
                or report.get("local_method_verified") is not False
                or report.get("frame_completion") != FRAME_COMPLETION
                or report.get("scientific_verdict") != "not_computed"):
            raise ValueError("C06 terminal report changed")
        replay_error = None
        try:
            expected, certificate, diagnostics = build_role_sequence(
                vertices, faces, role, epsilon=parameters["epsilon"],
                neighbors=parameters["neighbors"], tolerance=parameters["tolerance"],
                max_iterations=parameters["max_iterations"])
            bounds = {"below": int(np.count_nonzero(
                          expected < parameters["coordinate_bounds"][0])),
                      "above": int(np.count_nonzero(
                          expected > parameters["coordinate_bounds"][1])),
                      "lower": parameters["coordinate_bounds"][0],
                      "upper": parameters["coordinate_bounds"][1],
                      "policy": parameters["bounds_policy"]}
            if parameters["bounds_policy"] == "reject" and (bounds["below"] or bounds["above"]):
                raise ValueError("C06 output violates frozen coordinate bounds")
        except Exception as error:
            replay_error = error
        if report.get("status") == "completed":
            if replay_error is not None:
                raise ValueError("Completed C06 arm fails replay") from replay_error
            sequence, certificate_path = (directory / role / "sequence.npz",
                                          directory / role / "certificate.npz")
            if report.get("sha256") != {"sequence.npz": digest(sequence),
                                         "certificate.npz": digest(certificate_path)}:
                raise ValueError("C06 arm hashes changed")
            with np.load(sequence, allow_pickle=False) as saved:
                arm = {name: saved[name].copy() for name in saved.files}
            validate_native_arrays(arm)
            if set(arm) != set(arrays) or not np.array_equal(arm["vertices"], expected):
                raise ValueError("C06 sequence differs from replay")
            for name in set(arrays) - {"vertices"}:
                if not np.array_equal(arm[name], arrays[name]):
                    raise ValueError("C06 changed native identity array")
            with np.load(certificate_path, allow_pickle=False) as saved:
                observed = {name: saved[name].copy() for name in saved.files}
            if set(observed) != set(certificate) or any(
                    not np.array_equal(observed[name], certificate[name])
                    for name in observed):
                raise ValueError("C06 certificate differs from replay")
            if report.get("diagnostics") != diagnostics or report.get("bounds") != bounds:
                raise ValueError("C06 diagnostics changed")
            completed.append(role)
        elif report.get("status") == "error":
            if (replay_error is None
                    or report.get("exception_type") != type(replay_error).__name__
                    or report.get("error") != str(replay_error)[:4096]
                    or (directory / role / "sequence.npz").exists()
                    or (directory / role / "certificate.npz").exists()):
                raise ValueError("C06 failure evidence differs from replay")
        else:
            raise ValueError("C06 role must terminate completed or error")
    status = "completed" if len(completed) == len(ROLES) else "incomplete"
    if record.get("status") != status:
        raise ValueError("C06 aggregate status changed")
    manifest = {"expected_roles": list(ROLES),
                "cases": [{"case_id": record["uid"] + "-" + role,
                           "uid": record["uid"], "case_dir": role,
                           "arm_role": role} for role in completed],
                "common_target": common_ref,
                "scope": "C06 geometry-only generated_unexecuted artifacts; no B0/B*/scoring/admission"}
    if json.loads((directory / "manifest.json").read_text()) != manifest:
        raise ValueError("C06 manifest changed")
    archive = _validate_archive(directory, record["arms"])
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
    parser.add_argument("--coordinate-bounds", nargs=2, type=float, required=True)
    parser.add_argument("--bounds-policy", choices=("preserve_and_report", "reject"),
                        required=True)
    parser.add_argument("--max-artifact-bytes", type=int, required=True)
    args = parser.parse_args()
    source = args.source_sequence.resolve()
    if source.name != "sequence.npz":
        parser.error("--source-sequence must name sequence.npz")
    result = export_area_transport_candidate(
        source.parent, args.output, uid=args.uid,
        expected_sequence_sha256=args.expected_sequence_sha256,
        source_refs={"sequence": {"path": args.source_sequence_ref,
                                  "sha256": args.expected_sequence_sha256},
                     "report": {"path": args.source_report_ref,
                                "sha256": digest(source.with_name("report.json"))}},
        producer_provenance_ref={
            "path": args.producer_provenance_ref,
            "sha256": args.expected_producer_provenance_sha256},
        producer_provenance_path=args.producer_provenance,
        epsilon=args.epsilon, neighbors=args.neighbors,
        tolerance=args.tolerance, max_iterations=args.max_iterations,
        coordinate_lower=args.coordinate_bounds[0],
        coordinate_upper=args.coordinate_bounds[1],
        bounds_policy=args.bounds_policy,
        max_artifact_bytes=args.max_artifact_bytes)
    print(json.dumps({"status": result["status"], "candidate_id": CANDIDATE_ID,
                      "native_qualified": False, "scientific_admission": False}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
