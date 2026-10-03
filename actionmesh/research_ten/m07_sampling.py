"""H7: fixed-budget sparse trajectory querying and spatial reconstruction.

Only queried trajectories are visible to the allocator. Every probe becomes a
final control point and consumes one unique point query. The shared interpolator
is Euclidean inverse-distance displacement interpolation; it is deliberately
not presented as the unimplemented Fast4DMesh geodesic rigid-skinning pipeline.
"""
from time import perf_counter
import numpy as np


def _points(reference_points):
    points = np.asarray(reference_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("reference_points must be finite nonempty (N,3)")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("coincident reference vertices are ambiguous for this spatial interpolator")
    return points


def _chunk_size(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("distance_chunk_size must be a positive integer")
    return int(value)


def _distance_block(target, controls):
    # Avoid the additional (B,M,3) broadcast temporary. Axis order is fixed so
    # changing the target chunk partition does not alter arithmetic or ties.
    squared = (target[:, None, 0] - controls[None, :, 0]) ** 2
    squared += (target[:, None, 1] - controls[None, :, 1]) ** 2
    squared += (target[:, None, 2] - controls[None, :, 2]) ** 2
    return np.sqrt(squared, out=squared)


def _nearest_distances(target, controls, chunk_size):
    result = np.empty(len(target))
    for start in range(0, len(target), chunk_size):
        stop = min(start + chunk_size, len(target))
        result[start:stop] = _distance_block(target[start:stop], controls).min(axis=1)
    return result


def _weights(target, controls, neighbors=4, chunk_size=256):
    k = min(neighbors, len(controls))
    nearest = np.empty((len(target), k), dtype=int)
    selected = np.empty((len(target), k))
    for start in range(0, len(target), chunk_size):
        stop = min(start + chunk_size, len(target))
        distances = _distance_block(target[start:stop], controls)
        nearest[start:stop] = np.argsort(distances, axis=1, kind="stable")[:, :k]
        selected[start:stop] = np.take_along_axis(distances, nearest[start:stop], axis=1)
    # Scale by each row's smallest nonzero distance, avoiding huge inverse powers.
    nonzero = np.maximum(selected, np.finfo(float).tiny)
    ratios = nonzero[:, :1] / nonzero
    weights = ratios ** 2
    exact = selected[:, 0] == 0
    weights[exact] = 0
    weights[exact, 0] = 1
    weights /= weights.sum(axis=1, keepdims=True)
    return nearest, weights


def interpolate_trajectories(reference_points, sample_indices, sampled_trajectories, *,
                             distance_chunk_size=256):
    """Interpolate (T,M,3) absolute queried tracks to all N reference vertices.

    Displacements from each control's reference position are interpolated using
    its four closest controls, inverse squared Euclidean distance, and exact
    interpolation at queried vertices. A single control reproduces translation.
    This does not guarantee rigid rotations, topology preservation or geodesic
    separation of nearby body parts.
    Distance workspaces contain at most distance_chunk_size target rows; no
    full (N,M,3) tensor or full (N,M) distance matrix is materialized.
    """
    points = _points(reference_points)
    chunk_size = _chunk_size(distance_chunk_size)
    indices = np.asarray(sample_indices)
    tracks = np.asarray(sampled_trajectories, dtype=float)
    if indices.ndim != 1 or len(indices) < 1 or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(points)) or len(np.unique(indices)) != len(indices):
        raise ValueError("sample_indices must be distinct valid integer indices")
    if tracks.ndim != 3 or tracks.shape[1:] != (len(indices), 3) or tracks.shape[0] < 1 or not np.isfinite(tracks).all():
        raise ValueError("sampled_trajectories must be finite (T,M,3)")
    nearest, weights = _weights(points, points[indices], chunk_size=chunk_size)
    displacements = tracks - points[indices][None]
    out = points[None] + np.einsum("nk,tnkd->tnd", weights, displacements[:, nearest])
    out[:, indices] = tracks
    return out


def _fps_next(points, selected, chunk_size):
    if not selected:
        return int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    distance = _nearest_distances(points, points[np.asarray(selected)], chunk_size)
    distance[selected] = -np.inf
    return int(np.argmax(distance))


def _observed_scores(points, selected, tracks, strategy, chunk_size):
    if strategy == "motion":
        return np.linalg.norm(tracks - tracks[0:1], axis=2).max(axis=0)
    if len(selected) < 2:
        return np.zeros(len(selected))
    residual = np.zeros(len(selected))
    reference_controls = points[np.asarray(selected)]
    displacement = tracks - reference_controls[None]
    for i in range(len(selected)):
        others = np.delete(np.arange(len(selected)), i)
        nearest, weights = _weights(reference_controls[i:i + 1], reference_controls[others], chunk_size=chunk_size)
        prediction = np.einsum("nk,tnkd->tnd", weights, displacement[:, others][:, nearest])[:, 0]
        residual[i] = np.sqrt(np.mean(np.square(prediction - displacement[:, i])))
    return residual


def allocate_controls(reference_points, query, budget, *, strategy="residual",
                      curvature=None, initial_count=None, distance_chunk_size=256):
    """Spend exactly ``budget`` unique vertex queries, including all probes.

    ``query(indices: ndarray[int,M]) -> ndarray[float,T,M,3]`` is the only motion
    observation interface. It must return finite absolute trajectories in the
    same coordinate frame and fixed time order on every call. No full-trajectory
    array is an argument. The callback may wrap an actual decoder or a prerecorded
    sparse-query service; caller must account for its true model-level cost.

    All strategies start with the same FPS probes (default min(budget,
    max(3,budget//3))). Remaining queries are sequential; every result is used
    in final interpolation. ``fps`` uses geometric coverage. ``curvature``
    accepts a static finite nonnegative (N,) score. ``motion`` interpolates
    observed motion amplitude; ``residual`` interpolates leave-one-out error
    computed only on already queried controls. The adaptive score is nearest
    queried-point distance times (0.1 + normalized observed score), retaining
    a coverage incentive. This heuristic cannot detect motion in an entirely
    unobserved region with no predictive sparse evidence.
    Distance computation is chunked into at most distance_chunk_size rows,
    using O(distance_chunk_size * budget) workspace rather than O(N * budget).
    """
    points = _points(reference_points)
    chunk_size = _chunk_size(distance_chunk_size)
    n = len(points)
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or not 1 <= budget <= n:
        raise ValueError("budget must be an integer between 1 and N")
    if not callable(query):
        raise ValueError("query must be a callable sparse trajectory provider")
    if strategy not in ("fps", "curvature", "motion", "residual"):
        raise ValueError("strategy must be fps, curvature, motion or residual")
    if initial_count is None:
        initial_count = min(int(budget), max(3, int(budget) // 3))
    if isinstance(initial_count, bool) or not isinstance(initial_count, (int, np.integer)) or not 1 <= initial_count <= budget:
        raise ValueError("initial_count must be an integer between 1 and budget")
    curve = None if curvature is None else np.asarray(curvature, dtype=float)
    if curve is not None and (curve.shape != (n,) or not np.isfinite(curve).all() or np.any(curve < 0)):
        raise ValueError("curvature must be finite nonnegative shape (N,)")
    if strategy == "curvature" and curve is None:
        raise ValueError("curvature strategy requires static curvature observations")
    started = perf_counter()
    selected, history = [], []
    for _ in range(initial_count):
        selected.append(_fps_next(points, selected, chunk_size))
    query_started = perf_counter()
    tracks = np.asarray(query(np.asarray(selected, dtype=int)), dtype=float)
    query_seconds = perf_counter() - query_started
    if tracks.ndim != 3 or tracks.shape[1:] != (len(selected), 3) or tracks.shape[0] < 1 or not np.isfinite(tracks).all():
        raise ValueError("query must return finite trajectories of shape (T,number_requested,3)")
    tracks = tracks.copy()
    calls = 1
    history.append({"query_count": len(selected), "new_indices": selected.copy(), "phase": "probe", "observed_residual_max": 0.})
    while len(selected) < budget:
        distance = _nearest_distances(points, points[np.asarray(selected)], chunk_size)
        residual_max = 0.
        if strategy == "fps":
            acquisition = distance
        else:
            if strategy == "curvature":
                signal = curve.copy()
            else:
                scores = _observed_scores(points, selected, tracks, strategy, chunk_size)
                if strategy == "residual":
                    residual_max = float(scores.max())
                nearest, weights = _weights(points, points[np.asarray(selected)], chunk_size=chunk_size)
                signal = np.sum(weights * scores[nearest], axis=1)
            maximum = float(signal.max())
            signal = signal / maximum if maximum > 1e-15 else np.zeros(n)
            acquisition = distance * (.1 + signal)
        acquisition[selected] = -np.inf
        index = int(np.argmax(acquisition))
        query_started = perf_counter()
        observed = np.asarray(query(np.array([index], dtype=int)), dtype=float)
        query_seconds += perf_counter() - query_started
        calls += 1
        if observed.shape != (tracks.shape[0], 1, 3) or not np.isfinite(observed).all():
            raise ValueError("subsequent query changed time count, returned nonfinite data or wrong shape")
        tracks = np.concatenate((tracks, observed), axis=1)
        selected.append(index)
        history.append({"query_count": len(selected), "new_indices": [index], "phase": strategy,
                        "observed_residual_max": residual_max})
    interpolation_started = perf_counter()
    dense = interpolate_trajectories(points, np.asarray(selected), tracks,
                                     distance_chunk_size=chunk_size)
    interpolation_seconds = perf_counter() - interpolation_started
    elapsed = perf_counter() - started
    return {"sample_indices": np.asarray(selected, dtype=int), "sampled_trajectories": tracks,
            "trajectories": dense, "strategy": strategy, "query_count": len(selected),
            "probe_count": int(initial_count), "adaptive_count": len(selected) - int(initial_count),
            "query_calls": calls, "history": history,
            "timing_seconds": {"total": elapsed, "query": query_seconds,
                               "interpolation": interpolation_seconds,
                               "allocation": max(0., elapsed - query_seconds - interpolation_seconds)},
            "interpolator": "euclidean_inverse_distance_displacement_k4"}


def demo():
    """Quality-cost curves on a constructed localized deformation.

    Dense truth is held by a test oracle and evaluation only. The allocator can
    retrieve trajectories solely by paying for indices through its callback.
    """
    coordinate = np.linspace(0, 1, 81)
    points = np.column_stack((coordinate, .05 * np.sin(np.pi * coordinate), np.zeros(len(coordinate))))
    truth = np.broadcast_to(points, (9, len(points), 3)).copy()
    truth[:, :, 2] += np.sin(np.linspace(0, np.pi, 9))[:, None] * np.exp(-((coordinate - .5) / .10) ** 2)
    curvature = np.abs(np.sin(np.pi * coordinate))
    region = np.abs(coordinate - .5) < .16
    rows = []
    for budget in (5, 9, 15):
        for strategy in ("fps", "curvature", "motion", "residual"):
            calls = []
            def oracle(indices):
                calls.extend(indices.tolist())
                return truth[:, indices].copy()
            result = allocate_controls(points, oracle, budget, strategy=strategy, curvature=curvature)
            rows.append({"strategy": strategy, "budget": budget, "unique_queries": len(set(calls)),
                         "total_queried_points": len(calls), "probe_queries": result["probe_count"],
                         "trajectory_rmse": float(np.sqrt(np.mean((result["trajectories"] - truth) ** 2))),
                         "local_region_rmse": float(np.sqrt(np.mean((result["trajectories"][:, region] - truth[:, region]) ** 2))),
                         "timing_seconds": result["timing_seconds"]})
    return {"method": 7, "evidence": "constructed_control_only", "quality_cost_curve": rows,
            "limitations": ["Spatial inverse-distance interpolation is not Fast4DMesh geodesic rigid skinning.",
                            "Only unique vertex queries are counted; real decoder batching and wall time must be measured separately.",
                            "All full trajectories in this demo belong to the evaluation oracle, never to allocation inputs.",
                            "No natural failure census, mesh CD metrics or GPU memory benchmark is claimed."]}
