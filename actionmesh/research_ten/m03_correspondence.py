"""H3: exact temporal selection on small, explicit candidate neighborhoods.

This is a candidate *selector*, not a matcher. Inputs must contain independently
obtained feasible target matches; a deterministic decoder trajectory is not a
top-k candidate set. Connected surface neighborhoods are solved jointly with
second-order dynamic programming. The exponential neighborhood state count is
capped explicitly instead of silently replacing the solver with a heuristic.
"""
from itertools import product
import numpy as np


def _inputs(candidates, unary_cost, valid, reference_points, surface_edges,
            timestamps, point_ids, anchor_indices, candidate_ids):
    c = np.asarray(candidates, dtype=float)
    u = np.asarray(unary_cost, dtype=float)
    v = np.asarray(valid)
    ref = np.asarray(reference_points, dtype=float)
    if c.ndim != 4 or c.shape[-1] != 3 or min(c.shape[:3]) < 1:
        raise ValueError("candidates must have nonempty shape (T,N,K,3)")
    t, n, _ = c.shape[:3]
    if u.shape != c.shape[:3] or v.shape != u.shape or v.dtype != bool:
        raise ValueError("unary_cost and Boolean valid must have shape (T,N,K)")
    if ref.shape != (n, 3) or not all(np.isfinite(x).all() for x in (c, u, ref)):
        raise ValueError("candidates, unary costs and reference points must be finite")
    if not v.any(axis=2).all():
        raise ValueError("a point/frame has no valid candidate; selection must abstain")
    times = np.arange(t, dtype=float) if timestamps is None else np.asarray(timestamps, dtype=float)
    if times.shape != (t,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    edges = np.empty((0, 2), int) if surface_edges is None else np.asarray(surface_edges)
    if edges.size == 0:
        edges = np.empty((0, 2), int)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
        raise ValueError("surface_edges must be integer shape (E,2)")
    if np.any(edges < 0) or np.any(edges >= n) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("surface edge indices are invalid")
    if len(edges) and np.any(np.linalg.norm(ref[edges[:, 0]] - ref[edges[:, 1]], axis=1) <= 1e-12):
        raise ValueError("surface edges must have nonzero reference length")
    ids = np.arange(n) if point_ids is None else np.asarray(point_ids)
    if ids.shape != (n,) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != n:
        raise ValueError("point_ids must be unique integers of shape (N,)")
    anchors = None if anchor_indices is None else np.asarray(anchor_indices)
    if anchors is not None:
        if anchors.shape != (n,) or anchors.dtype.kind not in "iu" or np.any(anchors < 0) or np.any(anchors >= c.shape[2]):
            raise ValueError("anchor_indices must identify one candidate at the first frame per point")
        if not v[0, np.arange(n), anchors].all():
            raise ValueError("first-frame anchor selects an invalid candidate")
        v = v.copy()
        v[0] = False
        v[0, np.arange(n), anchors] = True
    match_ids = None if candidate_ids is None else np.asarray(candidate_ids)
    if match_ids is not None and (match_ids.shape != u.shape or match_ids.dtype.kind not in "iu" or np.any(match_ids[v] < 0)):
        raise ValueError("candidate_ids must have integer shape (T,N,K), nonnegative for valid matches")
    return c, u, v, ref, edges, times, ids, match_ids


def _components(n, edges):
    adjacency = [set() for _ in range(n)]
    for a, b in edges:
        adjacency[a].add(int(b))
        adjacency[b].add(int(a))
    unseen = set(range(n))
    components = []
    while unseen:
        pending, found = [min(unseen)], []
        while pending:
            p = pending.pop()
            if p in unseen:
                unseen.remove(p)
                found.append(p)
                pending.extend(sorted(adjacency[p] & unseen, reverse=True))
        components.append(np.array(sorted(found), int))
    return components


def _solve_component(c, u, valid, points, edges, ref, times, tw, sw, max_states, order):
    states, positions, data = [], [], []
    local = {int(p): i for i, p in enumerate(points)}
    component_edges = [(local[int(a)], local[int(b)], np.linalg.norm(ref[a] - ref[b]))
                       for a, b in edges if int(a) in local]
    for t in range(len(c)):
        choices = [np.flatnonzero(valid[t, p]) for p in points]
        count = 1
        for options in choices:
            count *= len(options)
        if count > max_states:
            raise ValueError(f"surface component requires {count} joint states; cap is {max_states}; split neighborhoods explicitly")
        state = np.array(list(product(*choices)), int)
        x = c[t, points[None, :], state]
        value = u[t, points[None, :], state].sum(axis=1)
        for a, b, distance in component_edges:
            value = value + sw * (np.linalg.norm(x[:, a] - x[:, b], axis=1) - distance) ** 2
        states.append(state)
        positions.append(x)
        data.append(value)
    length = len(c)
    if length == 1:
        idx = int(np.argmin(data[0]))
        return [states[0][idx]], float(data[0][idx]), np.count_nonzero(np.isclose(data[0], data[0][idx], rtol=1e-10, atol=1e-12)) > 1
    if order == 1:
        values, paths, multiplicity = data[0], [], np.ones(len(states[0]), int)
        for t in range(1, length):
            velocity = (positions[t][None] - positions[t - 1][:, None]) / (times[t] - times[t - 1])
            total = values[:, None] + tw * np.square(velocity).sum(axis=(2, 3))
            arg = total.argmin(axis=0)
            best = total[arg, np.arange(len(arg))]
            multiplicity = np.minimum(2, (np.isclose(total, best[None], rtol=1e-10, atol=1e-12) * multiplicity[:, None]).sum(axis=0))
            values = best + data[t]
            paths.append(arg)
        idx = int(values.argmin())
        ambiguous = np.sum(multiplicity[np.isclose(values, values[idx], rtol=1e-10, atol=1e-12)]) > 1
        chosen = [idx]
        for back in reversed(paths):
            idx = int(back[idx])
            chosen.append(idx)
        chosen.reverse()
        return [states[t][s] for t, s in enumerate(chosen)], float(values.min()), bool(ambiguous)
    values = data[0][:, None] + data[1][None, :]
    multiplicity = np.ones_like(values, int)
    backs = {}
    for t in range(2, length):
        previous_velocity = (positions[t - 1][None] - positions[t - 2][:, None]) / (times[t - 1] - times[t - 2])
        next_velocity = (positions[t][None] - positions[t - 1][:, None]) / (times[t] - times[t - 1])
        new = np.empty((len(states[t - 1]), len(states[t])))
        new_multiplicity = np.empty_like(new, int)
        back = np.empty_like(new, int)
        for j in range(len(states[t - 1])):
            acceleration = next_velocity[j][None] - previous_velocity[:, j][:, None]
            totals = values[:, j, None] + tw * np.square(acceleration).sum(axis=(2, 3))
            back[j] = totals.argmin(axis=0)
            minimum = totals[back[j], np.arange(len(states[t]))]
            new[j] = minimum + data[t]
            new_multiplicity[j] = np.minimum(2, (np.isclose(totals, minimum[None], rtol=1e-10, atol=1e-12) * multiplicity[:, j, None]).sum(axis=0))
        values, multiplicity, backs[t] = new, new_multiplicity, back
    first, second = np.unravel_index(values.argmin(), values.shape)
    minimum = float(values[first, second])
    ambiguous = np.sum(multiplicity[np.isclose(values, minimum, rtol=1e-10, atol=1e-12)]) > 1
    chosen = [0] * length
    chosen[-2], chosen[-1] = int(first), int(second)
    for t in range(length - 1, 1, -1):
        chosen[t - 2] = int(backs[t][chosen[t - 1], chosen[t]])
    return [states[t][s] for t, s in enumerate(chosen)], minimum, bool(ambiguous)


def select_correspondences(candidates, unary_cost, valid, reference_points, *,
                           surface_edges=None, timestamps=None, point_ids=None,
                           anchor_indices=None, candidate_ids=None,
                           temporal_weight=1.0, surface_weight=1.0,
                           max_joint_states=64):
    """Select explicit matches without using evaluation labels.

    Shapes: candidates (T,N,K,3), unary_cost/valid (T,N,K), reference_points
    (N,3), surface_edges (E,2). Lower unary costs mean better visible/matcher
    evidence; invisible frames should have uninformative costs supplied by the
    caller. Invalid slots remain finite but are excluded by the Boolean mask.

    Minimize unary + surface_weight * squared rest-edge-length error +
    temporal_weight * squared change in interval velocity. Exact DP operates
    independently on each connected surface component, with O(T*S^3) work and
    O(T*S^2) memory. No global surface rotation/rigidity or one-to-one matching
    guarantee is implied. Candidate IDs are carried through for diagnostics,
    never used as privileged identity labels; anchors are trusted first-frame
    matches. A tied optimum is returned deterministically and flagged ambiguous.
    """
    if not all(np.isfinite(w) and w >= 0 for w in (temporal_weight, surface_weight)):
        raise ValueError("weights must be finite and nonnegative")
    if isinstance(max_joint_states, bool) or not isinstance(max_joint_states, (int, np.integer)) or max_joint_states < 1:
        raise ValueError("max_joint_states must be a positive integer")
    c, u, v, ref, edges, times, ids, match_ids = _inputs(
        candidates, unary_cost, valid, reference_points, surface_edges, timestamps,
        point_ids, anchor_indices, candidate_ids)
    selected = np.empty(c.shape[:2], int)
    objective, ambiguous = 0., False
    for component in _components(c.shape[1], edges):
        path, cost, tie = _solve_component(c, u, v, component, edges, ref, times,
                                          temporal_weight, surface_weight, max_joint_states, 2)
        selected[:, component] = np.asarray(path)
        objective += cost
        ambiguous |= tie
    rows, cols = np.indices(selected.shape)
    return {"trajectories": c[rows, cols, selected].copy(), "candidate_indices": selected,
            "point_ids": ids.copy(), "selected_candidate_ids": None if match_ids is None else match_ids[rows, cols, selected].copy(),
            "objective": objective, "ambiguous": bool(ambiguous),
            "status": "ambiguous" if ambiguous else "selected", "solver": "exact_component_second_order_dp"}


def top1(candidates, unary_cost, valid):
    """Framewise minimum unary cost on the same explicit valid candidates."""
    c = np.asarray(candidates, float)
    n = c.shape[1] if c.ndim == 4 else 0
    c, u, v, *_ = _inputs(c, unary_cost, valid, np.zeros((n, 3)), None, None, None, None, None)
    indices = np.where(v, u, np.inf).argmin(axis=2)
    rows, cols = np.indices(indices.shape)
    return c[rows, cols, indices].copy()


def shortest_motion(candidates, unary_cost, valid, *, timestamps=None,
                    anchor_indices=None, temporal_weight=1.0):
    """First-order minimum-velocity baseline, with the same unary evidence."""
    if not np.isfinite(temporal_weight) or temporal_weight < 0:
        raise ValueError("temporal_weight must be finite and nonnegative")
    c = np.asarray(candidates, float)
    n = c.shape[1] if c.ndim == 4 else 0
    c, u, v, ref, edges, times, *_ = _inputs(c, unary_cost, valid, np.zeros((n, 3)), None, timestamps, None, anchor_indices, None)
    selected = np.empty(c.shape[:2], int)
    for p in range(n):
        path, _, _ = _solve_component(c, u, v, np.array([p]), edges, ref, times, temporal_weight, 0., c.shape[2], 1)
        selected[:, p] = np.asarray(path)[:, 0]
    rows, cols = np.indices(selected.shape)
    return c[rows, cols, selected].copy()


def demo():
    """Constructed crossing control, not evidence of natural matcher quality."""
    c = np.zeros((7, 1, 2, 3))
    c[:, 0, 0, 0] = np.arange(-3, 4)
    c[:, 0, 1, 0] = np.arange(3, -4, -1)
    u = np.zeros((7, 1, 2))
    u[4:, 0, 0] = .1
    valid = np.ones_like(u, bool)
    ids = np.broadcast_to([10, 20], u.shape)
    selected = select_correspondences(c, u, valid, [[-3, 0, 0]], anchor_indices=[0], candidate_ids=ids)
    permutation = np.array([0, 4, 2, 6, 1, 5, 3])
    shuffled = select_correspondences(c[permutation], u[permutation], valid[permutation], [[-3, 0, 0]], anchor_indices=[0])
    inverse = np.argsort(permutation)
    methods = {"top1": top1(c, u, valid), "shortest_motion": shortest_motion(c, u, valid, anchor_indices=[0]),
               "temporal_joint": selected["trajectories"], "time_shuffled": shuffled["trajectories"][inverse]}
    truth = c[:, :, 0]
    return {"method": 3, "evidence": "constructed_control_only",
            "metrics": {key: {"trajectory_rmse": float(np.sqrt(np.mean((x - truth) ** 2))),
                              "wrong_identity_frames_excluding_crossing": int(np.sum((np.abs(x[:, 0, 0] - truth[:, 0, 0]) > 1e-8) & (truth[:, 0, 0] != 0)))}
                        for key, x in methods.items()},
            "candidate_entries_per_method": int(valid.size), "candidate_acquisition": "not_implemented",
            "limitations": ["Caller must supply independently obtained feasible candidates and visible evidence.",
                            "Small connected neighborhoods only; exact joint state space grows exponentially.",
                            "Ties are ambiguous, and candidate coverage is not guaranteed."]}
