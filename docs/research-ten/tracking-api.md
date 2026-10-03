# Candidate selection, occlusion repair and budgeted sampling APIs

Run from the `actionmesh` directory (`cd actionmesh` from the repository root). These NumPy reference
implementations do not load a matching network, estimate visibility, or implement
Fast4DMesh. All three `demo()` functions return JSON-safe constructed controls,
including unfavorable controls; they are software checks rather than natural-data
validation. Numerical results use floating-point 3D coordinates in one shared
coordinate convention. Inputs are never modified.

## H3: explicit top-k correspondence selection

```python
select_correspondences(candidates, unary_cost, valid, reference_points, *,
                       surface_edges=None, timestamps=None, point_ids=None,
                       anchor_indices=None, candidate_ids=None,
                       temporal_weight=1.0, surface_weight=1.0,
                       max_joint_states=64)
top1(candidates, unary_cost, valid)
shortest_motion(candidates, unary_cost, valid, *, timestamps=None,
                anchor_indices=None, temporal_weight=1.0)
demo()
```

`candidates.npz` contract:

| Field | Shape / dtype | Meaning |
| --- | --- | --- |
| `candidates` | `(T,N,K,3)` float | Explicit independently acquired feasible target matches, absolute coordinates. |
| `unary_cost` | `(T,N,K)` float | Lower is stronger matcher/visible evidence. No GT labels. |
| `valid` | `(T,N,K)` bool | Excludes invalid slots. Every point/frame must have a valid slot. |
| `reference_points` | `(N,3)` float | Same physical source points, in stable order. |
| `surface_edges` | `(E,2)` integer, optional | Point-index edges with nonzero reference length. |
| `timestamps` | `(T,)` float, optional | Strictly increasing; defaults to frame indices. |
| `point_ids` | `(N,)` unique integer, optional | Stable source identities, defaults to point indices. |
| `anchor_indices` | `(N,)` integer, optional | Trusted candidate slot at the first frame. |
| `candidate_ids` | `(T,N,K)` integer, optional | Matcher-provided target IDs carried into diagnostic output. |

All coordinate/cost entries must be finite, including masked-out slots; use the
mask rather than NaN sentinels. Candidate slots may be reordered between frames.
Candidate IDs are not used as privileged correctness labels or an identity
penalty. Stable source IDs and first-frame anchors preserve the query identity;
temporal and neighborhood evidence select its target trajectory. The algorithm
cannot recover a correct match absent from the candidate set.

```python
import numpy as np
from research_ten.m03_correspondence import select_correspondences

with np.load("candidates.npz", allow_pickle=False) as data:
    required = {key: data[key] for key in
                ("candidates", "unary_cost", "valid", "reference_points")}
    optional = {key: data[key] for key in
                ("surface_edges", "timestamps", "point_ids", "anchor_indices", "candidate_ids")
                if key in data}
result = select_correspondences(**required, **optional)
np.savez_compressed("selected_tracks.npz",
                    trajectories=result["trajectories"],
                    candidate_indices=result["candidate_indices"],
                    point_ids=result["point_ids"])
print(result["objective"], result["status"])
```

The solver minimizes unary costs, squared changes in interval velocity, and
squared deviations from source surface-edge lengths. Each connected surface
component is solved exactly by second-order dynamic programming. If it has `S`
joint candidate assignments, time is `O(T*S^3)` and stored backpointers cost
`O(T*S^2)`. Each frame must respect `max_joint_states`; oversized components raise
`ValueError` rather than silently changing the optimization. This is intended for
small ambiguous part neighborhoods. Surface edges impose distance evidence, not
a full rigidity model or a global one-to-one matching constraint.

Output is a dictionary containing `trajectories (T,N,3)`, `candidate_indices
(T,N)`, `point_ids (N,)`, optional `selected_candidate_ids (T,N)`, scalar
`objective`, `ambiguous`, `status`, and `solver`. Equal-cost solutions are selected
deterministically and flagged ambiguous, including ties between identical
candidate slots. Confidence calibration and real candidate acquisition are not
implemented. `top1` and `shortest_motion` return arrays `(T,N,3)`; the demo also
permutes candidate/evidence time order and restores it for evaluation.

## H4: two-sided occlusion repair

```python
repair_occlusions(trajectories, visible, *, surface_edges=None,
                  timestamps=None, temporal_weight=1.0, surface_weight=1.0,
                  max_gap_frames=256)
linear_interpolation(trajectories, visible, *, timestamps=None)
demo()
```

`occlusion.npz` requires finite `trajectories (T,N,3)` and Boolean `visible
(T,N)`. Optional `surface_edges (E,2)` index stable corresponding surface points;
optional `timestamps (T,)` are strictly increasing. Visibility must mean a
trusted observation, not merely “the decoder produced a position.” A confidence
threshold or visibility estimator is the caller's responsibility. Hidden input
positions remain finite so intervals that abstain can preserve the original
prediction, but they are never used as solve targets.

```python
import numpy as np
from research_ten.m04_occlusion import repair_occlusions

with np.load("occlusion.npz", allow_pickle=False) as data:
    result = repair_occlusions(
        data["trajectories"], data["visible"],
        surface_edges=data["surface_edges"] if "surface_edges" in data else None,
        timestamps=data["timestamps"] if "timestamps" in data else None)
np.savez_compressed("repaired_tracks.npz", trajectories=result["trajectories"],
                    repaired_mask=result["repaired_mask"])
print(result["intervals"])
```

Every maximal hidden interval with visible samples at both ends is eligible.
Visible samples remain exact. A neighboring point contributes only when visible
at the hidden frame and at both interval endpoints. Its observed trajectory plus
an interpolated endpoint offset supplies a target; a temporal acceleration term
regularizes the gap. This simple neighbor transport can fail under large local
rotations or nonlocal adjacency. No hidden GT, occluded neighbor prediction, or
new correspondence search enters the solve.

Missing endpoints and rank-deficient systems are recorded as `missing_endpoint`
and `underconstrained`; those samples remain unchanged. Output includes
`trajectories`, `repaired_mask`, repaired/abstained counts, and interval status
records. `surface_weight=0` supplies the plain temporal baseline; both weights
zero raise `ValueError`. `linear_interpolation` supplies a same-evidence linear
baseline. Neither control is a full image-based bidirectional tracker.

The dense least-squares reference implementation accepts at most
`max_gap_frames=256` hidden samples in one closed gap. A longer interval abstains
with `gap_budget_exceeded` before creating the solve matrix; all input samples,
including its visible endpoints, remain unchanged. The positive integer cap is
explicitly configurable. Raising it increases quadratic matrix storage and
factorization cost; a sparse/banded long-gap solver is not implemented. The
linear interpolation baseline does not require a dense solve and has no such cap.

## H7: fixed-budget sparse querying

```python
allocate_controls(reference_points, query, budget, *, strategy="residual",
                  curvature=None, initial_count=None, distance_chunk_size=256)
interpolate_trajectories(reference_points, sample_indices, sampled_trajectories, *,
                         distance_chunk_size=256)
demo()
```

`surface.npz` contains finite `reference_points (N,3)` and optionally finite
nonnegative `curvature (N,)`. Coincident reference vertices are rejected because
the spatial interpolator cannot distinguish their identities. A decoder callback
must implement:

```python
query(indices: np.ndarray) -> np.ndarray  # input (M,) integer; output (T,M,3)
```

It receives **only requested vertex indices** and returns finite absolute
trajectories in exactly that order. Every call must use the same frames and
coordinate normalization. The first call requests FPS probes, followed by one
new vertex per call. All probes remain final controls; exactly `budget` distinct
vertices are queried, and no callback call requests previously queried indices.
`budget` is between 1 and `N`, and the default probe count is
`min(budget, max(3, budget//3))`. `initial_count` changes this count explicitly.

Executable callback-contract example (an analytic software fixture, not a
trained decoder or natural benchmark):

```python
import numpy as np
from research_ten.m07_sampling import allocate_controls

u = np.linspace(0., 1., 33)
points = np.column_stack((u, np.zeros((len(u), 2))))
time = np.linspace(0., 1., 5)
def query(indices):
    # Replace this body with the actual sparse decoder call in real experiments.
    # Only requested points are evaluated here; no full trajectory prequery.
    tracks = np.broadcast_to(points[indices], (len(time), len(indices), 3)).copy()
    tracks[:, :, 1] += np.sin(np.pi * time)[:, None] * np.exp(
        -((points[indices, 0] - .5) / .1) ** 2)[None]
    return tracks

result = allocate_controls(points, query, budget=9, strategy="residual")
np.savez_compressed("sparse_animation.npz",
                    trajectories=result["trajectories"],
                    sample_indices=result["sample_indices"],
                    sampled_trajectories=result["sampled_trajectories"])
assert result["query_count"] == 9
```

All four strategies use the same probe policy and interpolator. `fps` continues
farthest-point coverage; `curvature` uses only static supplied curvature;
`motion` extrapolates the motion magnitude observed at queried controls;
`residual` extrapolates leave-one-out interpolation errors computed only on
queried tracks. This score is multiplied by distance to existing controls, with
a fixed coverage floor. It cannot discover arbitrary unobserved motion without
spatial predictive evidence. A complete trajectory cache may serve as a *test
oracle* and evaluation reference; its acquisition cost cannot be omitted from a
claimed real sparse run.

Output includes all sampled controls, interpolated `trajectories (T,N,3)`,
`query_count`, `probe_count`, `adaptive_count`, `query_calls`, acquisition history,
and measured query/interpolation/allocation times. A “point query” includes all
requested times. Actual decoder batching, per-frame costs, peak GPU memory and
model forward passes must be logged by the real provider; point counts alone do
not establish a runtime speedup.

All FPS, acquisition and interpolation distance calculations process at most
`distance_chunk_size` target points at a time (positive integer, default 256).
No full `N × M × 3` displacement tensor or unbounded `N × M` distance matrix is
allocated for `M` controls. Distance workspace is `O(distance_chunk_size × M)`;
the selected four neighbors and weights cost `O(N)`. The returned dense
animation still necessarily costs `O(T × N)`. Chunk partition changes are tested
to preserve queried indices and reconstructed coordinates exactly.

The common interpolator uses four nearest Euclidean controls and inverse squared
distance on displacements. Queried vertices are exact, and a single queried
global translation propagates exactly. Rigid rotations, mesh geometry quality,
geodesic separation and collision prevention are not guaranteed. This is a
transparent numerical sparse baseline, **not a faithful Fast4DMesh baseline**.

## Local verification

```sh
python3 -m unittest discover -s research_ten/tests
python3 -c 'import json; from research_ten.m07_sampling import demo; print(json.dumps(demo(), allow_nan=False))'
```

The owned tests cover temporal crossing identity, candidate-slot reordering,
exact DP versus brute-force enumeration, invalid masks, ambiguous optima and
state caps; visible-anchor preservation, absent-endpoint abstention and
irregular-time interpolation and long-gap budget abstention; paid unique queries,
sparse-only callbacks, sample exactness, chunk equivalence and invalid decoder
outputs. Constructed quality curves retain
ties and regressions rather than requiring adaptive sampling to win every budget.
