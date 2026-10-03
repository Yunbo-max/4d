"""Bounded normal contact repair of explicit sampled linear constraints.

Dykstra projections find the minimum Euclidean displacement satisfying the
supplied halfspaces, fixed vertices, and per-vertex displacement balls, when
the intersection is feasible and iterations converge. This is a standard
convex projection construction, not IPC, and neither the supplied-constraint
solver nor the vertex-triangle candidate detector guarantees continuous
collision freedom, all self-intersections, or absence of flipped triangles.
"""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np


@dataclass(frozen=True)
class ContactConstraint:
    """Require normal dot sum(coefficients[i] * vertices[indices[i]]) >= min_gap.

    Coefficients must sum to zero. A point/triangle contact uses (1,-b0,-b1,-b2).
    The caller must establish nonadjacency, the safe signed side, and appropriate
    clearance; mere unsigned proximity cannot establish penetration direction.
    ``normal`` is normalized during validation, so min_gap has world units.
    """
    indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    normal: tuple[float, float, float]
    min_gap: float = 0.


@dataclass
class ContactRepair:
    vertices: np.ndarray
    displacement: np.ndarray
    initial_violation: float
    final_violation: float
    iterations: int
    converged: bool
    guarantee: str = "supplied sampled linear constraints only; no CCD or IPC guarantee"


@dataclass
class ContactDetection:
    contacts: list[ContactConstraint]
    tested_pairs: int
    skipped_initially_ambiguous: int
    truncated: bool
    guarantee: str = "nearby nonadjacent vertex-triangle samples only; edge-edge and swept collisions not covered"


def _vertices(vertices):
    x = np.asarray(vertices, float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("vertices must be finite nonempty (V,3)")
    return x


def _constraints(contacts, n):
    parsed = []
    for c in contacts:
        if not isinstance(c, ContactConstraint):
            raise ValueError("contacts must contain ContactConstraint objects")
        ids = np.asarray(c.indices)
        coeff = np.asarray(c.coefficients, float)
        normal = np.asarray(c.normal, float)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 2
                or len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= n).any()):
            raise ValueError("contact indices must be unique valid integer IDs")
        if (coeff.shape != ids.shape or not np.isfinite(coeff).all()
                or abs(coeff.sum()) > 1e-9 or not (coeff > 0).any() or not (coeff < 0).any()):
            raise ValueError("contact coefficients must be finite, nontrivial, and sum to zero")
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 1e-12:
            raise ValueError("contact normal must be a finite nonzero 3-vector")
        if not np.isfinite(c.min_gap) or c.min_gap < 0:
            raise ValueError("min_gap must be finite nonnegative")
        parsed.append((ids.astype(int), coeff, normal / np.linalg.norm(normal), float(c.min_gap)))
    return parsed


def _fixed(fixed_vertices, n):
    fixed = np.zeros(n, bool)
    if fixed_vertices is not None:
        ids = np.asarray(fixed_vertices)
        if ids.size == 0:
            return fixed
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids < 0).any() or (ids >= n).any():
            raise ValueError("fixed_vertices must be integer IDs")
        fixed[ids] = True
    return fixed


def _violation(vertices, parsed):
    return max((max(0., gap - float(np.dot(coeff @ vertices[ids], normal)))
                for ids, coeff, normal, gap in parsed), default=0.)


def repair_contacts(vertices, contacts, *, max_displacement: float = .01,
                    fixed_vertices=None, max_iterations: int = 1000,
                    tolerance: float = 1e-8) -> ContactRepair:
    """Minimum-change bounded contact-normal repair using Dykstra projections.

    max_displacement is a scalar or (V,) finite nonnegative world-unit bound.
    Displacements stay in the span of each vertex's incident contact normals;
    for a single normal the entire tangential component is exactly preserved.
    With different incident normals there is no common contact tangent plane.
    An infeasible bound/pin configuration returns converged=False and the
    actual residual; it never claims an unresolved contact was repaired.
    Memory O(V + number_of_contacts); per-iteration work is linear in supplied
    contact support plus V. Normals/barycentric weights are frozen for this solve.
    """
    x = _vertices(vertices)
    parsed = _constraints(contacts, len(x))
    fixed = _fixed(fixed_vertices, len(x))
    try:
        bounds = np.broadcast_to(np.asarray(max_displacement, float), (len(x),)).copy()
    except ValueError as exc:
        raise ValueError("max_displacement must be scalar or (V,)") from exc
    if not np.isfinite(bounds).all() or (bounds < 0).any():
        raise ValueError("displacement bounds must be finite nonnegative")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive iteration limit and tolerance required")
    bounds[fixed] = 0.
    initial = _violation(x, parsed)
    displacement = np.zeros_like(x)
    if not parsed or initial <= tolerance:
        return ContactRepair(x.copy(), displacement, initial, initial, 0, True)
    # Scalar duals suffice because each halfspace correction is collinear
    # with its constant gradient. Ball duals need one vector per vertex.
    duals = np.zeros(len(parsed))
    ball_dual = np.zeros_like(x)
    gradients, rhs = [], []
    for ids, coeff, normal, gap in parsed:
        gradient = coeff[:, None] * normal
        gradient[fixed[ids]] = 0.
        gradients.append(gradient)
        rhs.append(gap - float(np.dot(coeff @ x[ids], normal)))
    converged = False
    for iteration in range(1, max_iterations + 1):
        previous = displacement.copy()
        for k, ((ids, _, _, _), gradient) in enumerate(zip(parsed, gradients)):
            norm2 = float(np.sum(gradient ** 2))
            if norm2 <= 1e-30:
                continue
            y = displacement[ids] + duals[k] * gradient
            alpha = max(0., (rhs[k] - float(np.sum(gradient * y))) / norm2)
            displacement[ids] = y + alpha * gradient
            duals[k] = -alpha
        y = displacement + ball_dual
        norms = np.linalg.norm(y, axis=1)
        ratio = np.minimum(1., bounds / np.maximum(norms, 1e-30))
        displacement = y * ratio[:, None]
        ball_dual = y - displacement
        displacement[fixed] = 0.
        violation = _violation(x + displacement, parsed)
        change = float(np.max(np.linalg.norm(displacement - previous, axis=1)))
        if change <= tolerance and violation <= tolerance:
            converged = True
            break
    repaired = x + displacement
    repaired[fixed] = x[fixed]
    return ContactRepair(repaired, displacement, initial, _violation(repaired, parsed), iteration, converged)


def project_contact_constraints(vertices, contacts, *, fixed_vertices=None,
                                max_iterations: int = 1000,
                                tolerance: float = 1e-8) -> ContactRepair:
    """Plain cyclic halfspace projection control, without displacement bounds.

    This standard normal contact control seeks feasibility; it has no Dykstra
    duals and does not in general find the minimum-change feasible point.
    It is not an IPC baseline or a full nonlinear collision response system.
    """
    x = _vertices(vertices)
    parsed = _constraints(contacts, len(x))
    fixed = _fixed(fixed_vertices, len(x))
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("positive iteration limit and tolerance required")
    current = x.copy()
    initial = _violation(current, parsed)
    if initial <= tolerance:
        return ContactRepair(current, current - x, initial, initial, 0, True)
    for iteration in range(1, max_iterations + 1):
        for ids, coeff, normal, gap in parsed:
            gradient = coeff[:, None] * normal
            gradient[fixed[ids]] = 0.
            norm2 = float(np.sum(gradient ** 2))
            violation = gap - float(np.dot(coeff @ current[ids], normal))
            if norm2 > 1e-30 and violation > 0:
                current[ids] += violation / norm2 * gradient
        remaining = _violation(current, parsed)
        if remaining <= tolerance:
            break
    return ContactRepair(current, current - x, initial, remaining, iteration, remaining <= tolerance)


def repair_contact_trajectory(trajectories, contacts_by_frame, *, anchor_frame: int = 0,
                              max_displacement: float = .01, fixed_vertices=None,
                              max_iterations: int = 1000, tolerance: float = 1e-8):
    """Repair frames independently and retain the exact input anchor frame.

    contacts_by_frame contains one contact list per frame. Reports include
    unresolved constraints at the pinned anchor instead of hiding them.
    This does not add temporal smoothing or continuous collision detection.
    """
    q = np.asarray(trajectories, float)
    if q.ndim != 3 or q.shape[2] != 3 or len(q) == 0 or not np.isfinite(q).all():
        raise ValueError("trajectories must be finite (T,V,3)")
    if len(contacts_by_frame) != len(q):
        raise ValueError("one contact list is required per frame")
    if not isinstance(anchor_frame, (int, np.integer)) or not 0 <= anchor_frame < len(q):
        raise ValueError("anchor_frame outside trajectory")
    reports = [repair_contacts(frame, contacts_by_frame[t], max_displacement=max_displacement,
                 fixed_vertices=np.arange(q.shape[1]) if t == anchor_frame else fixed_vertices,
                 max_iterations=max_iterations, tolerance=tolerance) for t, frame in enumerate(q)]
    return np.stack([r.vertices for r in reports]), reports


def detect_vertex_triangle_contacts(reference, vertices, faces, *, clearance: float = .001,
                                    search_radius: float = .05, max_candidates: int = 10000,
                                    exclude_topological_neighbors: bool = True) -> ContactDetection:
    """Generate near-field contacts from oriented nonadjacent vertex/face samples.

    The reference must be a known safe pose with the same correspondence and
    faces; its signed side determines the allowed side of each current face.
    Current closest plane projections must lie inside the triangle. A spatial
    hash supplies the broad phase; large face boxes fall back to a linear AABB
    scan to avoid pathological hash-cell expansion. Deep penetration beyond
    search_radius, edge-edge intersections, initially coplanar contacts and
    frame-interior crossings are not exhaustively detected. Re-detect after a
    large repair because barycentric coordinates/normals were linearized.
    """
    ref, x = _vertices(reference), _vertices(vertices)
    if ref.shape != x.shape:
        raise ValueError("reference and current geometry must correspond")
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu" or (f < 0).any() or (f >= len(x)).any():
        raise ValueError("faces must be valid integer (F,3)")
    if not np.isfinite([clearance, search_radius]).all() or clearance < 0 or search_radius <= 0 or clearance > search_radius:
        raise ValueError("require 0 <= clearance <= positive search_radius")
    if not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if not isinstance(exclude_topological_neighbors, (bool, np.bool_)):
        raise ValueError("exclude_topological_neighbors must be boolean")
    adjacency = [set() for _ in range(len(x))]
    for triangle in f:
        for i in triangle:
            adjacency[i].update(map(int, triangle))
    cell_size = search_radius
    # Do not cast out-of-range float coordinates to int64: NumPy can silently
    # wrap them. If the grid is numerically unsafe, use bounded AABB scans.
    cell_limit = 2 ** 50
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_vertices = x / cell_size
    hash_safe = np.isfinite(scaled_vertices).all() and np.all(np.abs(scaled_vertices) < cell_limit)
    hashed = {}
    if hash_safe:
        for i, cell in enumerate(np.floor(scaled_vertices).astype(np.int64)):
            hashed.setdefault(tuple(map(int, cell)), []).append(i)
    contacts = []
    tested, ambiguous = 0, 0
    for triangle in f:
        a, b, c = x[triangle]
        ab, ac = b - a, c - a
        normal = np.cross(ab, ac)
        ref_normal = np.cross(ref[triangle[1]] - ref[triangle[0]], ref[triangle[2]] - ref[triangle[0]])
        norm, ref_norm = np.linalg.norm(normal), np.linalg.norm(ref_normal)
        if min(norm, ref_norm) <= 1e-12:
            continue
        normal /= norm
        ref_normal /= ref_norm
        low = x[triangle].min(0) - search_radius
        high = x[triangle].max(0) + search_radius
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_box = np.array([low, high]) / cell_size
        box_safe = hash_safe and np.isfinite(scaled_box).all() and np.all(np.abs(scaled_box) < cell_limit)
        if box_safe:
            low_cell, high_cell = [tuple(map(int, row)) for row in np.floor(scaled_box)]
            # Python integers avoid a negative overflow that could bypass the
            # work cap and enter a practically unbounded Cartesian product.
            count = math.prod(hi - lo + 1 for lo, hi in zip(low_cell, high_cell))
        else:
            count = math.inf
        if count > max(4096, len(hashed) * 8):
            candidates = np.flatnonzero(((x >= low) & (x <= high)).all(axis=1)).tolist()
        else:
            candidates = []
            for cell in product(*(range(lo, hi + 1) for lo, hi in zip(low_cell, high_cell))):
                candidates.extend(hashed.get(cell, ()))
        excluded = set(map(int, triangle))
        if exclude_topological_neighbors:
            for i in triangle:
                excluded.update(adjacency[i])
        d00, d01, d11 = np.dot(ab, ab), np.dot(ab, ac), np.dot(ac, ac)
        denominator = d00 * d11 - d01 * d01
        if denominator <= 1e-24:
            continue
        for vertex in candidates:
            if vertex in excluded:
                continue
            tested += 1
            delta = x[vertex] - a
            signed = float(np.dot(delta, normal))
            if abs(signed) > search_radius:
                continue
            d20, d21 = np.dot(delta, ab), np.dot(delta, ac)
            b1 = (d11 * d20 - d01 * d21) / denominator
            b2 = (d00 * d21 - d01 * d20) / denominator
            bary = np.array([1. - b1 - b2, b1, b2])
            if bary.min() < -1e-10:
                continue
            previous_side = float(np.dot(ref[vertex] - ref[triangle[0]], ref_normal))
            if abs(previous_side) <= 1e-10:
                ambiguous += 1
                continue
            normal_oriented = normal * np.sign(previous_side)
            if signed * np.sign(previous_side) >= clearance:
                continue
            bary = np.maximum(bary, 0.)
            bary /= bary.sum()
            contacts.append(ContactConstraint((int(vertex), *map(int, triangle)),
                (1., *(-bary).tolist()), tuple(normal_oriented.tolist()), float(clearance)))
            if len(contacts) >= max_candidates:
                return ContactDetection(contacts, tested, ambiguous, True)
    return ContactDetection(contacts, tested, ambiguous, False)


def demo() -> dict:
    """A single sampled contact can tie the standard control; report that tie."""
    import time
    from .m01_elasticity import arap_fit
    start = time.perf_counter()
    reference = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [.25, .25, .1]])
    current = reference.copy()
    current[3] += [.08, -.03, -.12]
    detection = detect_vertex_triangle_contacts(reference, current, np.array([[0, 1, 2]]),
                                                 clearance=.01, search_radius=.15)
    repaired = repair_contacts(current, detection.contacts, max_displacement=.1, fixed_vertices=[0, 1, 2])
    standard = project_contact_constraints(current, detection.contacts, fixed_vertices=[0, 1, 2])
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    rigid = arap_fit(reference, np.stack([reference, current]), edges, 100.).trajectories[1]
    parsed = _constraints(detection.contacts, len(current))
    scores = {}
    for name, value in (("original", current), ("bounded_normal", repaired.vertices),
                        ("standard_contact_projection", standard.vertices), ("uniform_arap_disconnected_weak_control", rigid)):
        scores[name] = {"sampled_violation": _violation(value, parsed),
                        "tangential_change": float(np.linalg.norm((value - current)[:, :2])),
                        "max_displacement": float(np.max(np.linalg.norm(value - current, axis=1)))}
    return {"method": 8, "validation": "constructed sampled-contact controls only",
            "detected_contacts": len(detection.contacts), "converged": repaired.converged,
            "scores": scores, "elapsed_seconds": time.perf_counter() - start,
            "limitations": [repaired.guarantee, detection.guarantee,
                            "ARAP demo has an isolated contact probe and is a weak disconnected control, not a qualified collision baseline",
                            "IPC baseline not implemented", "natural failure census required"]}
