"""Observation-supported local elasticity with a matrix-free ARAP solver.

The observation input is a measured 2-D track and confidence in the *same*
camera convention used by ``project``. Ground truth and evaluation region
labels are not inputs. A local rigid counterfactual is fitted independently
in each one-ring neighborhood; stiffness falls only when the original
nonrigid prediction explains observations better than that counterfactual.
This is a numerical hypothesis implementation, not a calibrated tracker or
proof that 2-D evidence uniquely identifies 3-D strain.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ElasticWeights:
    edge_weights: np.ndarray
    support: np.ndarray
    strain: np.ndarray
    raw_reprojection_error: np.ndarray
    rigid_reprojection_error: np.ndarray


@dataclass
class ArapResult:
    trajectories: np.ndarray
    energies: list[float]
    cg_iterations: list[int]
    converged: bool


def _geometry(rest, trajectories, edges):
    p = np.asarray(rest, dtype=float)
    q = np.asarray(trajectories, dtype=float)
    e0 = np.asarray(edges)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError("rest must be a nonempty (V,3) array")
    if q.ndim != 3 or q.shape[1:] != p.shape or len(q) == 0:
        raise ValueError("trajectories must have shape (T,V,3)")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("geometry must be finite")
    if e0.size == 0:
        e0 = np.empty((0, 2), dtype=int)
    if e0.ndim != 2 or e0.shape[1] != 2 or e0.dtype.kind not in "iu":
        raise ValueError("edges must be an integer (E,2) array")
    e = e0.astype(int)
    if (e < 0).any() or (e >= len(p)).any() or (e[:, 0] == e[:, 1]).any():
        raise ValueError("invalid edge indices or self edges")
    if len(e) and len(np.unique(np.sort(e, axis=1), axis=0)) != len(e):
        raise ValueError("each undirected edge must occur exactly once")
    if len(e) and (np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1) <= 1e-12).any():
        raise ValueError("rest edges must have nonzero length")
    return p, q, e


def mesh_edges(faces, n_vertices: int) -> np.ndarray:
    """Extract unique undirected edges from an integer triangle array."""
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu":
        raise ValueError("faces must be an integer (F,3) array")
    if (f < 0).any() or (f >= n_vertices).any():
        raise ValueError("face index outside mesh")
    e = np.concatenate([f[:, :2], f[:, 1:], f[:, [2, 0]]])
    if (e[:, 0] == e[:, 1]).any():
        raise ValueError("degenerate triangle indices")
    return np.unique(np.sort(e, axis=1), axis=0)


def _rotation(covariance):
    u, _, vh = np.linalg.svd(covariance)
    d = np.ones(covariance.shape[:-2] + (3,))
    d[..., -1] = np.where(np.linalg.det(vh.swapaxes(-1, -2) @ u.swapaxes(-1, -2)) < 0, -1, 1)
    return (vh.swapaxes(-1, -2) * d[..., None, :]) @ u.swapaxes(-1, -2)


def observation_supported_weights(
    rest, trajectories, edges, observed_uv,
    project: Callable[[np.ndarray, int], np.ndarray], confidence, *,
    stiffness: float = 10., min_stiffness: float = .1,
    evidence_floor: float = 1., strain_scale: float = .05,
) -> ElasticWeights:
    """Estimate (T,E) stiffness from observed reprojection improvement.

    ``project(vertices, frame_index)`` returns (V,2), in observed_uv units.
    Confidence is (T,V), in [0,1]; zero-confidence observations may be NaN.
    ``evidence_floor`` is the 2-D noise floor in those same units, and
    ``strain_scale`` is a dimensionless relative neighborhood deformation.
    Neighborhoods without three noncollinear rest points cannot establish
    strain support and retain full stiffness. Memory is O(T(V+E)).
    """
    p, q, e = _geometry(rest, trajectories, edges)
    uv = np.asarray(observed_uv, float)
    conf = np.asarray(confidence, float)
    if uv.shape != q.shape[:2] + (2,) or conf.shape != q.shape[:2]:
        raise ValueError("observed_uv/confidence shapes must be (T,V,2)/(T,V)")
    if not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any():
        raise ValueError("confidence must be finite in [0,1]")
    if not np.isfinite(uv[conf > 0]).all():
        raise ValueError("positive-confidence observations must be finite")
    if not (np.isfinite([stiffness, min_stiffness, evidence_floor, strain_scale]).all()
            and stiffness >= min_stiffness >= 0 and evidence_floor > 0 and strain_scale > 0):
        raise ValueError("invalid stiffness or evidence scales")
    neighbors = [{i} for i in range(len(p))]
    for a, b in e:
        neighbors[a].add(int(b))
        neighbors[b].add(int(a))
    rigid = q.copy()
    strain = np.zeros(q.shape[:2])
    for i, ids_set in enumerate(neighbors):
        ids = np.array(sorted(ids_set))
        x = p[ids] - p[ids].mean(0)
        if len(ids) < 3 or np.linalg.matrix_rank(x) < 2:
            continue
        y_center = q[:, ids].mean(1)
        y = q[:, ids] - y_center[:, None]
        cov = np.einsum("ni,tnj->tij", x, y)
        rotations = _rotation(cov)
        rigid[:, i] = np.einsum("tij,j->ti", rotations, p[i] - p[ids].mean(0)) + y_center
        fitted = np.einsum("tij,nj->tni", rotations, x)
        strain[:, i] = np.sqrt(np.mean(np.sum((y - fitted) ** 2, axis=-1), axis=1)
                                / np.mean(np.sum(x ** 2, axis=1)))
    raw_error = np.zeros(q.shape[:2])
    rigid_error = np.zeros_like(raw_error)
    safe_uv = np.where(conf[..., None] > 0, uv, 0.)
    for t in range(len(q)):
        raw_uv = np.asarray(project(q[t], t), float)
        rigid_uv = np.asarray(project(rigid[t], t), float)
        if raw_uv.shape != (len(p), 2) or rigid_uv.shape != raw_uv.shape:
            raise ValueError("project must return (V,2)")
        if not np.isfinite(raw_uv[conf[t] > 0]).all() or not np.isfinite(rigid_uv[conf[t] > 0]).all():
            raise ValueError("visible projections must be finite")
        raw_error[t] = np.where(conf[t] > 0, np.linalg.norm(raw_uv - safe_uv[t], axis=1), 0.)
        rigid_error[t] = np.where(conf[t] > 0, np.linalg.norm(rigid_uv - safe_uv[t], axis=1), 0.)
    improvement = np.clip((rigid_error ** 2 - raw_error ** 2)
                          / (rigid_error ** 2 + evidence_floor ** 2), 0., 1.)
    support = conf * improvement * strain / (strain + strain_scale)
    edge_support = .5 * (support[:, e[:, 0]] + support[:, e[:, 1]])
    weights = stiffness - (stiffness - min_stiffness) * edge_support
    return ElasticWeights(weights, support, strain, raw_error, rigid_error)


def _edge_weights(weights, frames, edges):
    try:
        w = np.broadcast_to(np.asarray(weights, float), (frames, edges)).copy()
    except ValueError as exc:
        raise ValueError("edge_weights must broadcast to (T,E)") from exc
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("edge_weights must be finite and nonnegative")
    return w


def _cg(operator, rhs, x, diagonal, tol, maxiter):
    residual = rhs - operator(x)
    z = residual / diagonal
    direction = z.copy()
    rz = float(np.sum(residual * z))
    threshold = tol * max(1., float(np.linalg.norm(rhs)))
    if np.linalg.norm(residual) <= threshold:
        return x, 0, True
    for k in range(maxiter):
        ad = operator(direction)
        denom = float(np.sum(direction * ad))
        if denom <= 0:
            return x, k, False
        alpha = rz / denom
        x = x + alpha * direction
        residual = residual - alpha * ad
        if np.linalg.norm(residual) <= threshold:
            return x, k + 1, True
        z = residual / diagonal
        next_rz = float(np.sum(residual * z))
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    return x, maxiter, False


def arap_fit(rest, trajectories, edges, edge_weights, *, data_weight: float = 1.,
             iterations: int = 20, anchor_frame: int = 0, times=None,
             temporal_weight: float = 0., fixed_vertices=None,
             tol: float = 1e-8, cg_maxiter: int = 500) -> ArapResult:
    """Local/global ARAP with exact anchor/pins and preconditioned CG.

    Minimize a target-position data term and directed ARAP edge residuals.
    An optional temporal term smooths *correction velocity*, using actual
    positive time intervals; thus a sequence of rigid targets remains a
    zero-cost solution. Each local step computes batched 3x3 SVDs; each
    global step uses edge accumulation, never a dense V-by-V matrix.
    fixed_vertices is an integer list pinned to input in every frame.
    ``converged`` means both outer change and all CG solves met tolerance.
    It is not a global-optimum or collision guarantee.
    """
    p, target, e = _geometry(rest, trajectories, edges)
    t_count, n, _ = target.shape
    w = _edge_weights(edge_weights, t_count, len(e))
    if not isinstance(anchor_frame, (int, np.integer)) or not 0 <= anchor_frame < t_count:
        raise ValueError("anchor_frame outside trajectory")
    if (not isinstance(iterations, (int, np.integer)) or iterations < 1
            or not isinstance(cg_maxiter, (int, np.integer)) or cg_maxiter < 1):
        raise ValueError("iteration limits must be positive integers")
    if not np.isfinite([data_weight, temporal_weight, tol]).all() or data_weight <= 0 or temporal_weight < 0 or tol <= 0:
        raise ValueError("invalid solver weights/tolerance")
    time = np.arange(t_count, dtype=float) if times is None else np.asarray(times, float)
    if time.shape != (t_count,) or not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError("times must be finite and strictly increasing")
    temporal = temporal_weight / np.diff(time) ** 2
    free = np.ones((t_count, n, 1), dtype=bool)
    free[anchor_frame] = False
    if fixed_vertices is not None:
        ids = np.asarray(fixed_vertices)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids < 0).any() or (ids >= n).any():
            raise ValueError("fixed_vertices must be valid integer IDs")
        free[:, ids] = False
    a, b = e.T
    rest_edges = p[a] - p[b]

    def temporal_laplacian(x):
        y = np.zeros_like(x)
        delta = temporal[:, None, None] * (x[1:] - x[:-1])
        y[1:] += delta
        y[:-1] -= delta
        return y

    def operator(x):
        y = data_weight * x + temporal_laplacian(x)
        wd = w[..., None] * (x[:, a] - x[:, b])
        for t in range(t_count):
            np.add.at(y[t], a, wd[t])
            np.add.at(y[t], b, -wd[t])
        return y

    diag = np.full((t_count, n), data_weight)
    for t in range(t_count):
        np.add.at(diag[t], a, w[t])
        np.add.at(diag[t], b, w[t])
    diag[1:] += temporal[:, None]
    diag[:-1] += temporal[:, None]
    diag = np.where(free, diag[..., None], 1.)
    pinned = np.where(free, 0., target)
    pinned_product = operator(pinned)
    q = target.copy()
    energies, cg_iterations = [], []
    all_cg = True
    outer_converged = False
    for _ in range(iterations):
        cov = np.zeros((t_count, n, 3, 3))
        outer = w[..., None, None] * np.einsum("ei,tej->teij", rest_edges, q[:, a] - q[:, b])
        for t in range(t_count):
            np.add.at(cov[t], a, outer[t])
            np.add.at(cov[t], b, outer[t])
        rotations = _rotation(cov)
        desired_a = np.einsum("teij,ej->tei", rotations[:, a], rest_edges)
        desired_b = np.einsum("teij,ej->tei", rotations[:, b], rest_edges)
        desired = .5 * (desired_a + desired_b)
        rhs = data_weight * target + temporal_laplacian(target)
        for t in range(t_count):
            np.add.at(rhs[t], a, w[t, :, None] * desired[t])
            np.add.at(rhs[t], b, -w[t, :, None] * desired[t])
        rhs = np.where(free, rhs - pinned_product, 0.)
        new_free, steps, ok = _cg(lambda x: np.where(free, operator(np.where(free, x, 0.)), x),
                                    rhs, np.where(free, q, 0.), diag, tol, cg_maxiter)
        new_q = np.where(free, new_free, target)
        delta = new_q - target
        edge_delta = new_q[:, a] - new_q[:, b]
        energy = data_weight * np.sum(delta ** 2)
        energy += .5 * np.sum(w[..., None] * ((edge_delta - desired_a) ** 2 + (edge_delta - desired_b) ** 2))
        energy += np.sum(temporal[:, None, None] * np.diff(delta, axis=0) ** 2)
        energies.append(float(energy))
        cg_iterations.append(steps)
        all_cg &= ok
        change = np.linalg.norm(new_q - q)
        q = new_q
        if change <= tol * max(1., float(np.linalg.norm(q))):
            outer_converged = True
            break
    return ArapResult(q, energies, cg_iterations, bool(all_cg and outer_converged))


def stiffness_controls(rest, trajectories, edges, observed_weights, *,
                       uniform_grid=(.001, 1., 100.), seed: int = 0) -> dict[str, np.ndarray]:
    """Return executable uniform-grid, mean-matched, motion and shuffled controls.

    Motion weights use the same min/max stiffness per frame as the candidate.
    Shuffling retains each frame's entire stiffness distribution. Nothing is
    selected using test error; uniform-grid selection requires separate dev data.
    """
    p, q, e = _geometry(rest, trajectories, edges)
    w = _edge_weights(observed_weights, len(q), len(e))
    if not len(e):
        raise ValueError("controls require at least one edge")
    values = np.asarray(uniform_grid, float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("uniform_grid must contain nonnegative finite values")
    controls = {f"uniform_{value:g}": np.full_like(w, value) for value in values}
    controls["uniform_mean"] = np.broadcast_to(w.mean(1, keepdims=True), w.shape).copy()
    motion = np.linalg.norm(q - q[0], axis=2)
    motion = .5 * (motion[:, e[:, 0]] + motion[:, e[:, 1]])
    motion /= np.maximum(motion.max(axis=1, keepdims=True), 1e-12)
    controls["motion"] = w.max(1, keepdims=True) - np.ptp(w, axis=1, keepdims=True) * motion
    rng = np.random.default_rng(seed)
    controls["shuffled"] = np.stack([rng.permutation(row) for row in w])
    return controls


def demo() -> dict:
    """Constructed observation control; truth is used only for reporting."""
    import time
    start = time.perf_counter()
    tetra = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rest = np.concatenate([tetra, tetra + [3, 0, 0]])
    edges = np.array([(i + k, j + k) for k in (0, 4) for i in range(4) for j in range(i + 1, 4)])
    pred = rest.copy()
    for k in (0, 4):
        center = rest[k:k + 4].mean(0)
        pred[k:k + 4] = center + 1.4 * (rest[k:k + 4] - center)
    tracks = np.stack([rest, pred])
    observed = tracks[..., :2].copy()
    observed[1, 4:] = rest[4:, :2]
    estimate = observation_supported_weights(rest, tracks, edges, observed,
        lambda x, t: x[:, :2], np.ones((2, 8)), stiffness=10., min_stiffness=.001,
        evidence_floor=.001, strain_scale=.001)
    controls = stiffness_controls(rest, tracks, edges, estimate.edge_weights,
                                  uniform_grid=(.001, 1., 100.))
    controls["observed"] = estimate.edge_weights
    truth = pred.copy()
    truth[4:] = rest[4:]
    scores = {}
    for name, weights in controls.items():
        result = arap_fit(rest, tracks, edges, weights, iterations=20)
        scores[name] = {"constructed_error": float(np.linalg.norm(result.trajectories[1] - truth)),
                        "anchor_max_error": float(np.max(np.abs(result.trajectories[0] - rest))),
                        "converged": result.converged}
    return {"method": 1, "validation": "constructed numerical controls only",
            "scores": scores, "elapsed_seconds": time.perf_counter() - start,
            "missing_for_natural_validation": ["measured image tracks/confidence", "camera calibration", "natural failure census"]}
