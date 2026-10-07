"""Constructed one-endpoint contact controls; no 4D, physics or novelty claim.

The additive k-best event alignment is separate from shared-trajectory repair.
Unknown/unobserved native episodes use a zero-cost null operation, not deletion.
All positive observations are enforced equally by the proposed wrapper and the
direct continuous comparator: the discrete layer cannot manufacture an advantage.

Continuous variables are endpoint positions X[T,D]. The objective is
  ||X-X0||^2 + smoothness * ||diff(X-X0)||^2
subject to exact observed contact positions and per-frame L2 displacement balls.
Explicit noncontact clearances are handled by alternating projections, a local
heuristic: failed iterations are "unresolved" unless a simple certificate exists.
Modes are typed alignment labels, not material/friction/force models.
"""
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class ContactEvent:
    partner: str
    start: int
    end: int
    mode: str = "touch"
    actor: str = "actor"


@dataclass(frozen=True)
class ContactEvidence:
    partner: str | None
    start: int
    end: int
    state: str = "contact"
    mode: str = "touch"
    actor: str = "actor"


@dataclass(frozen=True)
class EditOperation:
    kind: str
    native_index: int | None
    evidence_index: int | None


@dataclass(frozen=True)
class EventAlignment:
    cost: float
    operations: tuple[EditOperation, ...]


def _validate_intervals(items, horizon=None):
    for item in items:
        if (not isinstance(item.start, (int, np.integer))
                or not isinstance(item.end, (int, np.integer))
                or item.start < 0 or item.end < item.start
                or (horizon is not None and item.end >= horizon)):
            raise ValueError("invalid inclusive frame interval")
        if not item.actor or not item.mode:
            raise ValueError("actor and mode must be nonempty")
        if isinstance(item, ContactEvidence):
            if item.state not in ("contact", "noncontact", "unknown"):
                raise ValueError("unknown evidence state")
            if item.state != "unknown" and not item.partner:
                raise ValueError("known evidence needs a partner")
        elif not item.partner:
            raise ValueError("events need a partner")
    actors = {x.actor for x in items}
    if len(actors) > 1:
        raise ValueError("prototype supports one endpoint actor only")


def align_events(native_events, evidence, max_candidates=8, time_weight=0.0):
    """K-best additive ordered edit alignments, NOT a joint geometric optimum.

    Known contact evidence is sorted by (start,end,input index). Overlapping
    positives are still allowed and their feasibility is checked continuously.
    This routine does not claim a general concurrent-event/partial-order solver.
    Missing annotation and explicit unknown have identical null semantics.
    """
    native_events, evidence = list(native_events), list(evidence)
    _validate_intervals(native_events + evidence)
    if (not isinstance(max_candidates, (int, np.integer)) or max_candidates < 1
            or not np.isfinite(time_weight) or time_weight < 0):
        raise ValueError("invalid candidate cap or time weight")
    if any(native_events[k].start > native_events[k+1].start
           for k in range(len(native_events)-1)):
        raise ValueError("native events must be ordered by onset")
    positives = sorted([(i, e) for i, e in enumerate(evidence) if e.state == "contact"],
                       key=lambda pair: (pair[1].start, pair[1].end, pair[0]))
    m, n = len(native_events), len(positives)
    states = [[[] for _ in range(n+1)] for _ in range(m+1)]
    states[0][0] = [EventAlignment(0.0, ())]

    def push(i, j, candidate):
        states[i][j].append(candidate)
        states[i][j].sort(key=lambda a: (a.cost, tuple(
            (o.kind, -1 if o.native_index is None else o.native_index,
             -1 if o.evidence_index is None else o.evidence_index)
            for o in a.operations)))
        del states[i][j][max_candidates:]

    for i in range(m+1):
        for j in range(n+1):
            for current in tuple(states[i][j]):
                if i < m:
                    event = native_events[i]
                    # A positive different partner constrains this single
                    # endpoint, but absent/occluded evidence says nothing.
                    known = any(e.state != "unknown"
                                and e.start <= event.end and event.start <= e.end
                                and (e.state == "contact" or e.partner == event.partner)
                                for e in evidence)
                    op = EditOperation("delete" if known else "null", i, None)
                    push(i+1, j, EventAlignment(current.cost + float(known),
                                                current.operations + (op,)))
                if j < n:
                    ei, observation = positives[j]
                    push(i, j+1, EventAlignment(current.cost+1.0,
                                                current.operations+(EditOperation("insert", None, ei),)))
                if i < m and j < n:
                    ei, observation = positives[j]
                    event = native_events[i]
                    same = (event.actor, event.partner, event.mode) == (
                        observation.actor, observation.partner, observation.mode)
                    shift = abs(event.start-observation.start)+abs(event.end-observation.end)
                    op = EditOperation("match" if same else "replace", i, ei)
                    push(i+1, j+1, EventAlignment(current.cost + float(not same)
                                                + time_weight*shift, current.operations+(op,)))
    return states[m][n]


def _inputs(native, targets, evidence, max_displacement):
    native = np.asarray(native, dtype=np.float64)
    if (native.ndim != 2 or len(native) < 2 or native.shape[1] < 1
            or not np.isfinite(native).all()):
        raise ValueError("native must be finite [T,D], T>=2")
    if not np.isfinite(max_displacement) or max_displacement < 0:
        raise ValueError("max_displacement must be finite and nonnegative")
    evidence = list(evidence)
    _validate_intervals(evidence, len(native))
    expanded = {}
    for partner, target in targets.items():
        array = np.asarray(target, dtype=np.float64)
        if array.shape == (native.shape[1],):
            array = np.broadcast_to(array, native.shape)
        if array.shape != native.shape or not np.isfinite(array).all():
            raise ValueError("targets must be finite [D] or [T,D]")
        expanded[partner] = array
    for e in evidence:
        if e.state != "unknown" and e.partner not in expanded:
            raise ValueError("missing target for known evidence")
    return native, expanded, evidence


def _audit(trajectory, native, targets, evidence, max_displacement, clearance, tolerance):
    contact_errors, violations = [], []
    for e in evidence:
        if e.state == "unknown":
            continue
        span = slice(e.start, e.end+1)
        distance = np.linalg.norm(trajectory[span]-targets[e.partner][span], axis=1)
        if e.state == "contact":
            contact_errors.extend(distance.tolist())
        else:
            violations.extend(np.maximum(clearance-distance, 0).tolist())
    max_move = float(np.linalg.norm(trajectory-native, axis=1).max())
    contact = max(contact_errors, default=0.0)
    negative = max(violations, default=0.0)
    return {
        "feasible": bool(contact <= tolerance and negative <= tolerance
                         and max_move <= max_displacement+tolerance),
        "max_contact_error": float(contact),
        "max_noncontact_violation": float(negative),
        "max_displacement": max_move,
    }


def repair_trajectory(native, targets, evidence, max_displacement,
                      smoothness=0.2, clearance=0.05, tolerance=1e-8,
                      max_iterations=2000):
    """Direct same-information endpoint solver; all bounds are per-frame L2.

    Positive constraints alone form a convex quadratic problem. Noncontacts
    make this projection routine heuristic. Certified failure is reported only
    for conflicting exact targets, target outside the budget, or a negative
    constraint whose entire displacement ball lies inside the forbidden ball.
    """
    started = time.perf_counter()
    native, targets, evidence = _inputs(native, targets, evidence, max_displacement)
    if (not np.isfinite([smoothness, clearance, tolerance]).all()
            or smoothness < 0 or clearance < 0 or tolerance <= 0
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("invalid optimization parameters")
    fixed = np.zeros(len(native), dtype=bool)
    positions = native.copy()
    certificate = None
    for e in evidence:
        if e.state != "contact":
            continue
        for frame in range(e.start, e.end+1):
            target = targets[e.partner][frame]
            if fixed[frame] and np.linalg.norm(positions[frame]-target) > tolerance:
                certificate = "incompatible simultaneous exact contacts"
            if np.linalg.norm(target-native[frame]) > max_displacement+tolerance:
                certificate = "observed contact outside displacement ball"
            fixed[frame], positions[frame] = True, target
    for e in evidence:
        if e.state != "noncontact":
            continue
        span = np.arange(e.start, e.end+1)
        target = targets[e.partner][span]
        if np.any(np.linalg.norm(native[span]-target, axis=1)+max_displacement
                  < clearance-tolerance):
            certificate = "displacement ball contained in forbidden contact ball"
        relevant = span[fixed[span]]
        if (len(relevant) and np.any(np.linalg.norm(
                positions[relevant]-targets[e.partner][relevant], axis=1)
                < clearance-tolerance)):
            certificate = "exact positive contact conflicts with explicit noncontact"
    if certificate is not None:
        trajectory = native.copy()
        audit = _audit(trajectory, native, targets, evidence, max_displacement, clearance, tolerance)
        audit.update(feasible=False, status="certified_infeasible", certificate=certificate,
                     trajectory=trajectory, iterations=0, converged=False,
                     elapsed_seconds=time.perf_counter()-started)
        return audit

    displacement = np.zeros_like(native)
    displacement[fixed] = positions[fixed]-native[fixed]
    step = 0.9/(2.0+8.0*smoothness)
    converged = False
    for iteration in range(1, max_iterations+1):
        gradient = 2.0*displacement
        differences = np.diff(displacement, axis=0)
        gradient[:-1] -= 2.0*smoothness*differences
        gradient[1:] += 2.0*smoothness*differences
        candidate = native + displacement - step*gradient
        for e in evidence:
            if e.state != "noncontact":
                continue
            for frame in range(e.start, e.end+1):
                if fixed[frame]:
                    continue
                delta = candidate[frame]-targets[e.partner][frame]
                distance = np.linalg.norm(delta)
                if distance < clearance:
                    if distance <= np.finfo(float).eps:
                        # Predetermined axis, not a tuned direction or random search.
                        delta = np.zeros(native.shape[1])
                        delta[0] = 1.0
                        distance = 1.0
                    candidate[frame] = targets[e.partner][frame]+clearance*delta/distance
        proposal = candidate-native
        lengths = np.linalg.norm(proposal, axis=1)
        proposal *= np.minimum(1.0, max_displacement/np.maximum(lengths, 1e-300))[:, None]
        proposal[fixed] = positions[fixed]-native[fixed]
        update = float(np.max(np.abs(proposal-displacement)))
        displacement = proposal
        if update < tolerance*0.05:
            converged = True
            break
    trajectory = native+displacement
    audit = _audit(trajectory, native, targets, evidence, max_displacement, clearance, tolerance)
    audit.update(
        trajectory=trajectory, status="feasible" if audit["feasible"] else "unresolved",
        certificate=None, iterations=iteration, converged=converged,
        objective=float(np.sum(displacement**2)+smoothness*np.sum(np.diff(displacement, axis=0)**2)),
        elapsed_seconds=time.perf_counter()-started,
        solver_scope="convex positive constraints; heuristic explicit noncontact projections")
    return audit


def retime_trajectory(native, sample_times):
    """Sample a piecewise-linear native trajectory using a strict monotone clock."""
    native = np.asarray(native, dtype=np.float64)
    times = np.asarray(sample_times, dtype=np.float64)
    if (native.ndim != 2 or len(native) < 2 or not np.isfinite(native).all()
            or times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > len(native)-1):
        raise ValueError("strictly increasing finite in-range sample times required")
    return np.stack([np.interp(times, np.arange(len(native)), native[:, d])
                     for d in range(native.shape[1])], axis=1)


def retime_event_times(events, sample_times):
    """Map source event boundaries through inverse clock; requires full coverage."""
    events = list(events)
    _validate_intervals(events)
    times = np.asarray(sample_times, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("strict monotone times required")
    if any(e.start < times[0] or e.end > times[-1] for e in events):
        raise ValueError("clock must cover every event")
    return [(float(np.interp(e.start, times, np.arange(len(times)))),
             float(np.interp(e.end, times, np.arange(len(times))))) for e in events]


def retiming_baseline(native, targets, evidence, max_displacement,
                      grid_size=257, clearance=0.05, tolerance=1e-8):
    """Lexicographic (evidence error, displacement) warp on a finite time grid.

    Endpoints are fixed and interpolation is linear. Grid failure is not a
    continuous impossibility certificate. No post-retiming spatial repair.
    Displacement breaks exact evidence-cost ties only; it cannot trade away
    contact accuracy. This is a finite-grid optimum, not a continuous optimum.
    """
    native, targets, evidence = _inputs(native, targets, evidence, max_displacement)
    if (not isinstance(grid_size, (int, np.integer)) or grid_size < len(native)
            or not np.isfinite([clearance, tolerance]).all()
            or clearance < 0 or tolerance <= 0):
        raise ValueError("invalid retiming grid or tolerance")
    times = np.linspace(0, len(native)-1, grid_size)
    candidates = retime_trajectory(native, times)
    displacement_cost = np.sum((native[:, None, :]-candidates[None, :, :])**2, axis=2)
    cost = np.zeros_like(displacement_cost)
    moves = np.linalg.norm(native[:, None, :]-candidates[None, :, :], axis=2)
    cost[moves > max_displacement+tolerance] = np.inf
    for e in evidence:
        if e.state == "unknown":
            continue
        for frame in range(e.start, e.end+1):
            distances = np.linalg.norm(candidates-targets[e.partner][frame], axis=1)
            cost[frame] += (distances**2 if e.state == "contact"
                            else np.maximum(clearance-distances, 0)**2)
    cost[0, 1:], cost[-1, :-1] = np.inf, np.inf
    scores = np.full_like(cost, np.inf)
    displacement_scores = np.full_like(cost, np.inf)
    previous = np.full(cost.shape, -1, dtype=int)
    scores[0] = cost[0]
    displacement_scores[0] = displacement_cost[0]
    for frame in range(1, len(native)):
        best_pair, best_index = (np.inf, np.inf), -1
        for j in range(grid_size):
            if j and np.isfinite(scores[frame-1, j-1]):
                previous_pair = (scores[frame-1, j-1], displacement_scores[frame-1, j-1])
                if previous_pair < best_pair:
                    best_pair, best_index = previous_pair, j-1
            if best_index >= 0:
                scores[frame, j] = best_pair[0]+cost[frame, j]
                displacement_scores[frame, j] = best_pair[1]+displacement_cost[frame, j]
                previous[frame, j] = best_index
    if not np.isfinite(scores[-1, -1]):
        return {"feasible": False, "status": "grid_no_admissible_path",
                "trajectory": native.copy(), "sample_times": np.arange(len(native), dtype=float),
                "continuous_impossibility_claim": False}
    indices = [grid_size-1]
    for frame in range(len(native)-1, 0, -1):
        indices.append(int(previous[frame, indices[-1]]))
    clock = times[np.array(indices[::-1])]
    trajectory = retime_trajectory(native, clock)
    audit = _audit(trajectory, native, targets, evidence, max_displacement, clearance, tolerance)
    audit.update(trajectory=trajectory, sample_times=clock,
                 status="feasible" if audit["feasible"] else "grid_residual",
                 grid_size=grid_size, continuous_impossibility_claim=False)
    return audit


def solve_contact_events(native, targets, native_events, evidence,
                         max_displacement, max_candidates=8, **kwargs):
    """Bounded event hypotheses + shared direct endpoint repair.

    Exact known endpoints do not depend on which additive alignment explains
    them. Reuse the identical continuous solution across candidates rather than
    rerunning it or claiming improvement over the direct baseline.
    """
    evidence = list(evidence)
    native_events = list(native_events)
    _validate_intervals(native_events, len(native))
    alignments = align_events(native_events, evidence, max_candidates=max_candidates)
    result = repair_trajectory(native, targets, evidence, max_displacement, **kwargs)
    result.update(
        alignment=alignments[0].operations, alignment_cost=alignments[0].cost,
        candidate_count=len(alignments), continuous_solves=1,
        candidate_costs=[a.cost for a in alignments],
        global_joint_optimality_claim=False,
        continuous_comparator="same_information_direct_solver",
        discrete_layer_benefit_claim=False)
    return result


def run_demo():
    """Deterministic constructed controls; includes a strong-baseline tie."""
    cases = {
        "missing_event": (np.zeros((7, 1)), {"A": np.array([1.])}, [],
                          [ContactEvidence("A", 3, 3)], 1.0),
        "wrong_partner": (np.zeros((7, 1)),
                          {"A": np.array([0.]), "B": np.array([1.])},
                          [ContactEvent("A", 3, 3)], [ContactEvidence("B", 3, 3)], 1.0),
        "wrong_order": (np.linspace(-1, 1, 7)[:, None],
                        {"A": np.array([-1.]), "B": np.array([1.])},
                        [ContactEvent("A", 0, 0), ContactEvent("B", 6, 6)],
                        [ContactEvidence("B", 1, 1), ContactEvidence("A", 5, 5)], 2.0),
        "timing_only": (np.array([0., 0., 1., 1., 0.])[:, None],
                        {"A": np.array([1.])}, [ContactEvent("A", 2, 3)],
                        [ContactEvidence("A", 1, 1)], 1.0),
        "budget_impossible": (np.zeros((7, 1)), {"A": np.array([1.])}, [],
                              [ContactEvidence("A", 3, 3)], 0.5),
        "unknown": (np.zeros((7, 1)), {"A": np.array([0.])},
                    [ContactEvent("A", 1, 5)], [ContactEvidence(None, 0, 6, "unknown")], 0.0),
    }
    controls, all_same = {}, True
    for label, (native, targets, events, evidence, budget) in cases.items():
        repair = solve_contact_events(native, targets, events, evidence, budget)
        direct = repair_trajectory(native, targets, evidence, budget)
        retimed = retiming_baseline(native, targets, evidence, budget)
        same = bool(np.array_equal(repair["trajectory"], direct["trajectory"]))
        all_same &= same
        controls[label] = {
            "repair_feasible": repair["feasible"], "repair_status": repair["status"],
            "retiming_feasible": retimed["feasible"], "retiming_status": retimed["status"],
            "direct_solver_equal": same, "max_displacement": repair["max_displacement"],
            "max_contact_error": repair["max_contact_error"],
            "operations": [op.kind for op in repair["alignment"]],
            "candidate_count": repair["candidate_count"],
            "native": native.tolist(), "repaired": repair["trajectory"].tolist(),
            "retimed": retimed["trajectory"].tolist(),
        }
    return {
        "schema_version": 1, "evidence_kind": "constructed_toy_controls",
        "controls": controls, "direct_solver_matches_all_repairs": bool(all_same),
        "natural_dataset_evidence": False, "global_joint_optimality_claim": False,
        "gpu_used": False, "units": "arbitrary endpoint coordinates and frames",
        "limitations": [
            "One endpoint actor; target points, not mesh surfaces or contact forces.",
            "Modes are labels; no friction, support stability or physical simulator.",
            "Event alignment is additive and separate from shared trajectory feasibility.",
            "Exact contact evidence makes direct continuous optimization equally capable.",
            "Finite-grid retiming failure alone is not a continuous impossibility proof.",
            "No natural failure prevalence, cross-model result or novelty evidence.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite existing results")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = run_demo()
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "controls": len(report["controls"]),
                      "evidence_kind": report["evidence_kind"]}))


if __name__ == "__main__":
    main()
