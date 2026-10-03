"""H4: offline, two-sided occlusion repair with exact visible observations.

Visibility and correspondences are supplied observations, not inferred here.
Only closed visible -> hidden -> visible intervals are changed. A fixed-boundary
quadratic solve combines acceleration regularization with displacement evidence
from adjacent points visible throughout the relevant observations.
"""
import numpy as np


def _inputs(trajectories, visible, surface_edges, timestamps):
    x = np.asarray(trajectories, dtype=float)
    vis = np.asarray(visible)
    if x.ndim != 3 or x.shape[2] != 3 or min(x.shape[:2]) < 1 or not np.isfinite(x).all():
        raise ValueError("trajectories must be finite and nonempty (T,N,3)")
    if vis.shape != x.shape[:2] or vis.dtype != bool:
        raise ValueError("visible must be Boolean shape (T,N)")
    t, n = vis.shape
    times = np.arange(t, dtype=float) if timestamps is None else np.asarray(timestamps, dtype=float)
    if times.shape != (t,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    edges = np.empty((0, 2), int) if surface_edges is None else np.asarray(surface_edges)
    if edges.size == 0:
        edges = np.empty((0, 2), int)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("surface_edges must be integer (E,2)")
    if np.any(edges < 0) or np.any(edges >= n) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("invalid surface edge indices")
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return x, vis, edges, times


def _gaps(visible):
    """Yield point, hidden start/end (inclusive), and availability of both anchors."""
    for point in range(visible.shape[1]):
        hidden = ~visible[:, point]
        starts = np.flatnonzero(hidden & ~np.r_[False, hidden[:-1]])
        ends = np.flatnonzero(hidden & ~np.r_[hidden[1:], False])
        for start, end in zip(starts, ends):
            yield point, int(start), int(end), start > 0 and end < len(visible) - 1


def _report(out, repaired, records, visible, solver):
    return {"trajectories": out, "repaired_mask": repaired,
            "repaired_count": int(repaired.sum()),
            "abstained_count": int((~visible & ~repaired).sum()),
            "intervals": records, "solver": solver}


def linear_interpolation(trajectories, visible, *, timestamps=None):
    """Same-evidence linear baseline. Open-ended occlusions are unchanged."""
    x, vis, _, times = _inputs(trajectories, visible, None, timestamps)
    out, repaired, records = x.copy(), np.zeros_like(vis), []
    for p, start, end, closed in _gaps(vis):
        record = {"point": p, "start": start, "end": end, "status": "repaired" if closed else "missing_endpoint"}
        records.append(record)
        if closed:
            a, b = start - 1, end + 1
            fraction = (times[start:end + 1] - times[a]) / (times[b] - times[a])
            out[start:end + 1, p] = (1 - fraction[:, None]) * x[a, p] + fraction[:, None] * x[b, p]
            repaired[start:end + 1, p] = True
    return _report(out, repaired, records, vis, "linear_endpoints")


def repair_occlusions(trajectories, visible, *, surface_edges=None,
                      timestamps=None, temporal_weight=1.0, surface_weight=1.0,
                      max_gap_frames=256):
    """Repair finite (T,N,3) tracks using Boolean (T,N) visibility.

    Each closed gap minimizes temporal_weight times squared changes in interval
    velocity plus surface_weight times squared neighbor-displacement residuals.
    Neighbor targets use only observed positions: an adjacent point must be
    visible at the hidden frame and at both gap endpoints. Its relative offset
    to the target point is linearly interpolated from the two visible endpoints.
    Unknown target coordinates and originally predicted hidden positions never
    enter the right-hand side. Visible positions are exact Dirichlet constraints.

    With surface_weight=0 this is the plain temporal baseline. This reference
    method is not an image-based bidirectional tracker or a rotation-aware
    surface deformation model. Open gaps and underconstrained systems abstain.
    The dense reference solve is limited to max_gap_frames hidden samples per
    interval; longer gaps abstain with gap_budget_exceeded before allocation.
    """
    if not all(np.isfinite(w) and w >= 0 for w in (temporal_weight, surface_weight)):
        raise ValueError("weights must be finite and nonnegative")
    if temporal_weight == 0 and surface_weight == 0:
        raise ValueError("at least one regularizer must be positive")
    if isinstance(max_gap_frames, (bool, np.bool_)) or not isinstance(max_gap_frames, (int, np.integer)) or max_gap_frames < 1:
        raise ValueError("max_gap_frames must be a positive integer")
    x, vis, edges, times = _inputs(trajectories, visible, surface_edges, timestamps)
    out, repaired, records = x.copy(), np.zeros_like(vis), []
    adjacency = [set() for _ in range(x.shape[1])]
    for a, b in edges:
        adjacency[a].add(int(b))
        adjacency[b].add(int(a))
    for p, start, end, closed in _gaps(vis):
        record = {"point": p, "start": start, "end": end, "status": "missing_endpoint"}
        records.append(record)
        if not closed:
            continue
        a, b, count = start - 1, end + 1, end - start + 1
        if count > max_gap_frames:
            record["status"] = "gap_budget_exceeded"
            continue
        matrix, rhs, neighbor_constraints = [], [], 0
        if temporal_weight > 0:
            weight = np.sqrt(temporal_weight)
            for t in range(a + 1, b):
                left_dt, right_dt = times[t] - times[t - 1], times[t + 1] - times[t]
                coefficients = (1 / left_dt, -1 / left_dt - 1 / right_dt, 1 / right_dt)
                row, target = np.zeros(count), np.zeros(3)
                for frame, coeff in zip((t - 1, t, t + 1), coefficients):
                    if start <= frame <= end:
                        row[frame - start] = coeff
                    else:
                        target -= coeff * x[frame, p]
                matrix.append(weight * row)
                rhs.append(weight * target)
        if surface_weight > 0:
            weight = np.sqrt(surface_weight)
            for q in sorted(adjacency[p]):
                if not (vis[a, q] and vis[b, q]):
                    continue
                offset_a, offset_b = x[a, p] - x[a, q], x[b, p] - x[b, q]
                for t in range(start, end + 1):
                    if not vis[t, q]:
                        continue
                    fraction = (times[t] - times[a]) / (times[b] - times[a])
                    target = x[t, q] + (1 - fraction) * offset_a + fraction * offset_b
                    row = np.zeros(count)
                    row[t - start] = weight
                    matrix.append(row)
                    rhs.append(weight * target)
                    neighbor_constraints += 1
        record["neighbor_constraints"] = neighbor_constraints
        if not matrix:
            record["status"] = "underconstrained"
            continue
        solution, _, rank, _ = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)
        if rank < count or not np.isfinite(solution).all():
            record["status"] = "underconstrained"
            continue
        out[start:end + 1, p] = solution
        repaired[start:end + 1, p] = True
        record["status"] = "repaired"
    return _report(out, repaired, records, vis, "visible_neighbor_constrained_least_squares")


def demo():
    """Constructed occlusion with independently visible moving neighbor."""
    truth = np.zeros((7, 2, 3))
    truth[:, :, 0] = np.arange(7)[:, None]
    truth[:, :, 1] = np.array([0, 1, 2, 3, 2, 1, 0])[:, None]
    truth[:, 1, 1] += 1
    raw = truth.copy()
    raw[1:-1, 0] += np.array([3, -3, 1])
    visible = np.ones((7, 2), bool)
    visible[1:-1, 0] = False
    runs = {"raw": raw, "linear": linear_interpolation(raw, visible)["trajectories"],
            "plain_temporal": repair_occlusions(raw, visible, surface_weight=0.)["trajectories"],
            "surface_repair": repair_occlusions(raw, visible, surface_edges=[[0, 1]], temporal_weight=.1, surface_weight=1.)["trajectories"]}
    one_sided = visible.copy()
    one_sided[-1, 0] = False
    abstain = repair_occlusions(raw, one_sided, surface_edges=[[0, 1]])
    return {"method": 4, "evidence": "constructed_control_only",
            "metrics": {key: {"occluded_rmse": float(np.sqrt(np.mean((x[~visible] - truth[~visible]) ** 2))),
                              "visible_max_change": float(np.max(np.abs(x[visible] - raw[visible]))),
                              "reappearance_step_error": float(np.linalg.norm((x[-1, 0] - x[-2, 0]) - (truth[-1, 0] - truth[-2, 0])))}
                        for key, x in runs.items()},
            "one_sided_repaired_count": abstain["repaired_count"],
            "limitations": ["Visibility and reliable point correspondences must be supplied.",
                            "Visible-neighbor offset transport is not a full surface deformation solver.",
                            "No image-based bidirectional tracker or natural-occlusion benchmark has been implemented."]}
