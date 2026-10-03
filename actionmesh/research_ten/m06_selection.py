"""H6: action-gated selection of a video before the expensive 4D backend.

All candidates must belong to the SAME asset, prompt and camera convention.
Action/text/appearance scores are observable scorer outputs in [0,1], obtained
with one frozen scorer shared by all policies. This module does not invent a
semantic scorer or generate Wan videos. ``track_features`` supplies inexpensive
2D proportion/coverage proxies from an actual tracker's outputs. Perspective
and articulation can change those proxies: they are hypotheses, not calibrated
3D feasibility probabilities. No ground-truth or downstream 4D quality enters
the selection interface. Diagnostic full-backend runs are accounted separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _scores(value, count, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {count} finite scores")
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} scores must lie in [0,1]")
    return array.copy()


@dataclass(frozen=True)
class CandidateFeatures:
    ids: tuple[str, ...]
    action: np.ndarray
    text: np.ndarray
    appearance: np.ndarray
    proportion: np.ndarray
    trackability: np.ndarray

    def __post_init__(self):
        ids = tuple(self.ids)
        if not ids or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError("nonempty string candidate IDs are required")
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        object.__setattr__(self, "ids", ids)
        for name in ("action", "text", "appearance", "proportion", "trackability"):
            object.__setattr__(self, name, _scores(getattr(self, name), len(ids), name))


def select_video(features: CandidateFeatures, *, action_threshold: float = .5,
                 policy: str = "feasibility", weights=(1., 1., 1.),
                 valid=None, seed: int = 42) -> dict:
    """Select one candidate, abstaining if the shared action/validity gate fails.

    ``weights`` are fixed on development inputs before evaluating new assets.
    Geometric-mean aggregation penalizes a candidate with a zero component.
    Other policies use the exact same eligible set. Ties use input order.
    """
    if not isinstance(features, CandidateFeatures):
        raise TypeError("features must be CandidateFeatures")
    if not np.isfinite(action_threshold) or not 0 <= action_threshold <= 1:
        raise ValueError("action_threshold must lie in [0,1]")
    if policy not in ("feasibility", "text", "appearance", "random"):
        raise ValueError("policy must be feasibility, text, appearance, or random")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("selection seed must be a nonnegative integer")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (3,) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("three finite nonnegative weights with positive sum required")
    w = w / w.max()
    w = w / w.sum()
    n = len(features.ids)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    else:
        valid = np.asarray(valid)
        if valid.shape != (n,) or valid.dtype != bool:
            raise ValueError("valid must be a Boolean mask with one item per video")
    eligible = np.flatnonzero(valid & (features.action >= action_threshold))
    values = np.stack([features.appearance, features.proportion, features.trackability], axis=1)
    # Handle zero weights/zero components without log(0)*0 or artificial epsilon.
    active = w > 0
    proxy = np.prod(values[:, active] ** w[active], axis=1)
    chosen = None
    if len(eligible):
        if policy == "random":
            chosen = int(np.random.default_rng(seed).choice(eligible))
        else:
            ranking = {"feasibility": proxy, "text": features.text,
                       "appearance": features.appearance}[policy]
            chosen = int(eligible[np.argmax(ranking[eligible])])
    return {"status": "selected" if chosen is not None else "no_eligible_candidate",
            "selected_index": chosen,
            "selected_id": features.ids[chosen] if chosen is not None else None,
            "eligible_indices": eligible.tolist(), "policy": policy,
            "action_threshold": float(action_threshold), "weights": w.tolist(),
            "feasibility_scores": proxy.tolist(),
            "backend_calls_used_by_selector": 0}


def track_features(tracks, visible, reference_xy, *, min_pair_count: int = 3,
                   min_frame_fraction: float = .5, max_pairs: int = 4096) -> dict:
    """Compute score proxies without free full-backend queries.

    Inputs: tracks [C,T,P,2], Boolean visibility [C,T,P], and reference projected
    asset keypoints [P,2] with identical point IDs. Invisible coordinates may be
    NaN; visible ones must be finite. A deterministic capped set of point pairs
    prevents quadratic storage. Every usable frame has at least min_pair_count
    visible nondegenerate pairs. Global image translation/uniform scale cancel;
    perspective and articulation do not. A lack of evidence yields valid=False.
    """
    xy = np.asarray(tracks, dtype=np.float64)
    mask = np.asarray(visible)
    reference = np.asarray(reference_xy, dtype=np.float64)
    if xy.ndim != 4 or xy.shape[-1] != 2 or min(xy.shape[:3]) < 1:
        raise ValueError("tracks must be nonempty [C,T,P,2]")
    c, t, p, _ = xy.shape
    if t < 2 or p < 3:
        raise ValueError("at least two frames and three point IDs required")
    if mask.shape != xy.shape[:-1] or mask.dtype != bool:
        raise ValueError("visible must be Boolean [C,T,P]")
    if reference.shape != (p, 2) or not np.isfinite(reference).all():
        raise ValueError("reference_xy must be finite [P,2]")
    if not np.isfinite(xy[mask]).all():
        raise ValueError("visible tracks must be finite")
    if not isinstance(min_pair_count, int) or min_pair_count < 1:
        raise ValueError("min_pair_count must be positive integer")
    if not isinstance(max_pairs, int) or max_pairs < min_pair_count:
        raise ValueError("max_pairs must cover min_pair_count")
    if not np.isfinite(min_frame_fraction) or not 0 < min_frame_fraction <= 1:
        raise ValueError("min_frame_fraction must lie in (0,1]")
    # Uniformly sample unordered-pair indices via bounded scanning. O(P²) in
    # integer operations only for <=90 points, otherwise deterministic sparse
    # pairs generated without allocating all P² combinations.
    total_pairs = p*(p-1)//2
    if total_pairs <= max_pairs:
        left, right = np.triu_indices(p, 1)
    else:
        pairs = set()
        rng = np.random.default_rng(0)
        while len(pairs) < max_pairs:
            a, b = rng.integers(p, size=2)
            if a != b:
                pairs.add((int(min(a, b)), int(max(a, b))))
        left, right = np.array(sorted(pairs)).T
    ref_length = np.linalg.norm(reference[left] - reference[right], axis=-1)
    nondegenerate = ref_length > np.finfo(float).eps * max(1., np.max(ref_length))
    left, right, ref_length = left[nondegenerate], right[nondegenerate], ref_length[nondegenerate]
    proportions = np.zeros(c)
    frame_counts = np.zeros(c, dtype=int)
    coverage = np.mean(mask[:, 1:] & mask[:, :-1], axis=(1, 2))
    for candidate in range(c):
        errors = []
        for frame in range(t):
            usable = mask[candidate, frame, left] & mask[candidate, frame, right]
            if np.count_nonzero(usable) < min_pair_count:
                continue
            lengths = np.linalg.norm(xy[candidate, frame, left[usable]] -
                                     xy[candidate, frame, right[usable]], axis=-1)
            ratios = lengths / ref_length[usable]
            scale = float(np.median(ratios))
            if scale <= 0 or not np.isfinite(scale) or np.any(ratios <= 0):
                continue
            errors.append(float(np.mean(np.abs(np.log(ratios / scale)))))
        frame_counts[candidate] = len(errors)
        if errors:
            proportions[candidate] = float(np.exp(-np.mean(errors)))
    valid = frame_counts >= int(np.ceil(t * min_frame_fraction))
    return {"proportion": proportions, "trackability": coverage,
            "valid": valid, "usable_frame_counts": frame_counts,
            "pair_count": int(len(left)),
            "scope": "2D proxy; not calibrated 3D feasibility or semantic action score"}


def run_pipeline(generate, score, reconstruct, *, seeds=(42, 43, 44, 45),
                 action_threshold=.5, policy="feasibility", weights=(1., 1., 1.),
                 selection_seed=0) -> dict:
    """Actually call supplied video generator/scorer and ONE selected backend.

    ``generate(seed)`` returns a video path or caller-owned video object;
    ``score(video)`` returns action,text,appearance,proportion,trackability and
    optionally valid=False; ``reconstruct(video)`` returns a caller-owned 4D
    result. None sees diagnostic GT. Generation, scoring and selected backend
    wall times/call counts are separate. The caller supplies frozen existing
    Wan/ActionMesh integrations and must keep task/prompt/camera fixed. This
    function does not estimate unexecuted cost from a cached quality table.
    Exceptions propagate without silently replacing a failed candidate.
    """
    import time
    seeds = tuple(seeds)
    if not seeds or any(not isinstance(s, (int, np.integer)) or isinstance(s, bool) or s < 0 for s in seeds):
        raise ValueError("seeds must be nonempty nonnegative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("candidate seeds must be distinct")
    if not all(callable(f) for f in (generate, score, reconstruct)):
        raise TypeError("generate, score and reconstruct must be callable")
    # Validate configuration before any expensive or externally visible call.
    empty = CandidateFeatures(("validation",), [0.], [0.], [0.], [0.], [0.])
    select_video(empty, action_threshold=action_threshold, policy=policy,
                 weights=weights, seed=selection_seed)
    videos, rows, times = [], [], []
    names = ("action", "text", "appearance", "proportion", "trackability")
    validity = []
    for seed in seeds:
        start = time.perf_counter()
        video = generate(int(seed))
        generation_seconds = time.perf_counter()-start
        start = time.perf_counter()
        row = score(video)
        scoring_seconds = time.perf_counter()-start
        if not isinstance(row, dict) or not all(name in row for name in names):
            raise ValueError("score callback must return all five observable features")
        valid = row.get("valid", True)
        if not isinstance(valid, (bool, np.bool_)):
            raise ValueError("score validity must be Boolean")
        # Validate as received, rather than discovering invalid scorer output
        # only after generating the rest of an expensive candidate batch.
        CandidateFeatures((str(seed),), **{name: [row[name]] for name in names})
        videos.append(video); rows.append(row); validity.append(bool(valid))
        times.append({"seed": int(seed), "generation_seconds": generation_seconds,
                      "scoring_seconds": scoring_seconds})
    features = CandidateFeatures(tuple(str(s) for s in seeds),
                                 **{name: [row[name] for row in rows] for name in names})
    result = select_video(features, action_threshold=action_threshold, policy=policy,
                          weights=weights, valid=validity, seed=selection_seed)
    output = None
    backend_seconds = 0.
    if result["selected_index"] is not None:
        start = time.perf_counter()
        output = reconstruct(videos[result["selected_index"]])
        backend_seconds = time.perf_counter()-start
    return {"selection": result, "output": output,
            "candidate_times": times, "backend_seconds": backend_seconds,
            "counts": {"video_generations": len(videos), "video_scores": len(rows),
                       "backend_calls": int(result["selected_index"] is not None)},
            "scope": "Only selected candidate reconstructed; exhaustive downstream diagnostics excluded"}


def demo() -> dict:
    features = CandidateFeatures(("static", "distorted", "usable"),
        [.1, .9, .8], [.2, .95, .7], [1., .2, .8], [1., .2, .9], [1., .8, .9])
    policies = {name: select_video(features, policy=name, action_threshold=.5)
                for name in ("feasibility", "text", "appearance", "random")}
    return {"evidence_type": "constructed_control", "policies": policies,
            "claim": "Common action gate prevents static candidate selection; no natural video ranking tested"}
