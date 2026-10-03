# Geometry candidate APIs: methods 1, 5 and 8

These NumPy implementations operate on corresponding vertices in a shared world coordinate convention. They do not train or call the ActionMesh model. `demo()` in each module returns constructed numerical-control evidence, not natural-data research validation. Run examples from `actionmesh/` so `research_ten` is importable.

## 1. Observation-supported elasticity

```python
observation_supported_weights(
    rest, trajectories, edges, observed_uv, project, confidence, *,
    stiffness=10., min_stiffness=.1, evidence_floor=1., strain_scale=.05,
) -> ElasticWeights

arap_fit(
    rest, trajectories, edges, edge_weights, *, data_weight=1.,
    iterations=20, anchor_frame=0, times=None, temporal_weight=0.,
    fixed_vertices=None, tol=1e-8, cg_maxiter=500,
) -> ArapResult

mesh_edges(faces, n_vertices) -> ndarray
stiffness_controls(rest, trajectories, edges, observed_weights, *,
                   uniform_grid=(.001, 1., 100.), seed=0) -> dict
```

Actual input NPZ contract: `rest[V,3]`, `trajectories[T,V,3]`, integer `faces[F,3]`, actual `times[T]`, measured `observed_uv[T,V,2]`, `confidence[T,V]` in `[0,1]`, and world-to-image homogeneous projection matrices `projection[T,3,4]`. Vertex IDs in the measurements must correspond to the mesh; the API does not manufacture a tracker. Invisible measurements may be NaN only at confidence zero. Coordinates of visible points must project finitely, and `evidence_floor` uses the same pixel/normalized units as the measurements.

```python
import numpy as np
from research_ten.m01_elasticity import (
    observation_supported_weights, arap_fit, mesh_edges, stiffness_controls,
)

d = np.load("observed_mesh_tracks.npz", allow_pickle=False)
rest, tracks = d["rest"], d["trajectories"]
edges = mesh_edges(d["faces"], len(rest))
projection = d["projection"]

def project(vertices, frame):
    homogeneous = np.column_stack([vertices, np.ones(len(vertices))])
    image = homogeneous @ projection[frame].T
    with np.errstate(divide="ignore", invalid="ignore"):
        return image[:, :2] / image[:, 2:3]

weights = observation_supported_weights(
    rest, tracks, edges, d["observed_uv"], project, d["confidence"],
    evidence_floor=1.,  # measured 1-pixel noise floor for this example
)
result = arap_fit(rest, tracks, edges, weights.edge_weights, times=d["times"],
                  anchor_frame=0, temporal_weight=.1)
np.savez("elasticity_result.npz", trajectories=result.trajectories,
         support=weights.support, stiffness=weights.edge_weights,
         converged=result.converged)

# Evaluate each control separately with identical observations and solve settings.
controls = stiffness_controls(rest, tracks, edges, weights.edge_weights)
control_results = {
    name: arap_fit(rest, tracks, edges, w, times=d["times"], temporal_weight=.1)
    for name, w in controls.items()
}
```

The support is confidence times improvement in squared reprojection error over a best one-ring rigid counterfactual, attenuated when inferred strain is small. Degenerate one-rings retain full stiffness. Projection residuals are exposed in `ElasticWeights`; no ground-truth geometry or evaluation region label sets weights. Rotation/camera errors can still imitate strain, so this is evidence gating, not strain identifiability.

The local/global solver uses batched 3×3 SVD and matrix-free preconditioned CG, with O(T(V+E)) storage. The anchor and optional integer vertex pins remain exact. Optional temporal regularization smooths correction velocity using actual time gaps, preserving a rigid input sequence. `ArapResult.energies`, `cg_iterations`, and `converged` expose solver status. `converged=False` must not be hidden when an iteration cap is reached. Uniform grid, mean-matched uniform, motion-amplitude, and shuffled-stiffness controls are implemented; any tuning requires separate development data. The final constructed demo uses three preset uniform weights (.001, 1, 100) and 20 iterations for every solve; its shuffled control remains explicitly unconverged. An intermediate diagnostic with six weights and 100 iterations is preserved separately in `geometry-demos-diagnostic-expanded.json`; it is not a natural screening protocol or evidence of a qualified strongest ARAP baseline. No collision/foldover guarantee is provided.

## 5. Stable-region scale decomposition

```python
fit_similarity(source, target, weights=None, *, robust=True,
               max_iterations=30, hypotheses=48, seed=0) -> SimilarityFit

stabilize_scale(rest, trajectories, stable_mask, *, anchor_frame=0,
                confidence=None, camera_convention="fixed_metric_world",
                robust=True) -> ScaleDecomposition

scale_controls(rest, trajectories, *, anchor_frame=0) -> dict
```

`stable_mask[V]` must actually have boolean dtype and identify at least three noncollinear, verified size-stable vertices. Unknown masks, fractional masks, insufficient confidence and degenerate fits raise `ValueError`. A body/torso is not assumed rigid by default. The accepted camera conventions are `fixed_metric_world` and `camera_compensated_world`; the latter means compensation already happened before this API. Monocular scale ambiguity is not solved here.

```python
import numpy as np
from research_ten.m05_scale import stabilize_scale, scale_controls

d = np.load("stable_region_tracks.npz", allow_pickle=False)
result = stabilize_scale(
    d["rest"], d["trajectories"], d["stable_mask"],
    confidence=d["confidence"], camera_convention="camera_compensated_world",
)
np.savez("scale_result.npz", trajectories=result.corrected,
         scales=result.scales, rotations=result.rotations,
         translations=result.translations, centers=result.centers,
         local_residuals=result.local_residuals, fit_weights=result.fit_weights)
controls = scale_controls(d["rest"], d["trajectories"])
```

Weighted similarity fitting uses sampled-triple robust initialization and Tukey IRLS. A majority of valid stable correspondences is assumed. Correction removes scale relative to the anchor about the fitted stable-region center, retaining its true world path and pose; scale-normalized local articulation is retained. The anchor is exact. The output exposes fit weights and canonical local residuals. This does not separately repair arbitrary limb-specific scale errors. Intentional inflation or topology change is outside the stable-region assumption. Controls are unmodified input, bbox-diagonal stabilization, and global least-squares Sim(3); they intentionally illustrate articulation/size confounding.

## 8. Bounded normal contact repair

```python
ContactConstraint(indices, coefficients, normal, min_gap=0.)
repair_contacts(vertices, contacts, *, max_displacement=.01,
                fixed_vertices=None, max_iterations=1000, tolerance=1e-8)
project_contact_constraints(vertices, contacts, *, fixed_vertices=None,
                            max_iterations=1000, tolerance=1e-8)
repair_contact_trajectory(trajectories, contacts_by_frame, *, anchor_frame=0,
                          max_displacement=.01, fixed_vertices=None,
                          max_iterations=1000, tolerance=1e-8)
detect_vertex_triangle_contacts(reference, vertices, faces, *, clearance=.001,
                                search_radius=.05, max_candidates=10000,
                                exclude_topological_neighbors=True)
```

Each contact requires `normal · Σ coefficients[i]*vertices[indices[i]] >= min_gap`. Normals are normalized; gaps and bounds are world distances. Coefficients sum to zero, with both signs present. For a point/triangle contact use IDs `(point, a, b, c)` and coefficients `(1, -bary_a, -bary_b, -bary_c)`. The caller must supply nonadjacent contacts and a justified safe side. Signed contacts can come from an external collision detector; do not convert unsigned nearest distances into assumed penetration signs.

The following contract stores equal-width constraints from an external detector as numeric NPZ arrays: `contact_frame[K]`, `contact_indices[K,4]`, `contact_coefficients[K,4]`, `contact_normals[K,3]`, and `contact_gap[K]`, alongside `trajectories[T,V,3]`.

```python
import numpy as np
from research_ten.m08_contact import ContactConstraint, repair_contact_trajectory

d = np.load("signed_contacts.npz", allow_pickle=False)
contacts = [[] for _ in range(len(d["trajectories"]))]
for frame, ids, coeffs, normal, gap in zip(
    d["contact_frame"], d["contact_indices"], d["contact_coefficients"],
    d["contact_normals"], d["contact_gap"],
):
    contacts[int(frame)].append(ContactConstraint(
        tuple(ids), tuple(coeffs), tuple(normal), float(gap)))
repaired, reports = repair_contact_trajectory(
    d["trajectories"], contacts, anchor_frame=0, max_displacement=.01)
np.savez("contact_result.npz", trajectories=repaired,
         final_violation=np.array([r.final_violation for r in reports]),
         converged=np.array([r.converged for r in reports]))
```

Alternatively, `detect_vertex_triangle_contacts(safe_reference, current_frame, faces, ...)` generates near-field candidates using a spatial hash, reference-side orientation, barycentric plane projection, and topological-neighbor exclusion. Use only a known safe, corresponding reference. It reports truncation and skips ambiguous initially coplanar contacts and degenerate faces. Integer-safe cell-count caps and coordinate-range checks switch large or unsafe hash boxes to finite linear AABB scans. It can miss deep penetration outside the search radius, edge/edge collisions and other intersections. Normals and barycentric weights are frozen during each repair; re-detection is needed after larger changes.

Dykstra projections minimize squared vertex displacement subject to supplied halfspaces, exact pins and per-vertex displacement balls. A single incident normal preserves all tangent components; with several normals movement lies in their span. Noncontact vertices are unchanged. Infeasible bounds or iteration limits return unresolved residuals with `converged=False`, including the pinned anchor. `project_contact_constraints` is the executable standard cyclic halfspace-projection baseline; it has no displacement bound and need not minimize displacement. The demo shows the single-contact tie with standard projection. Its ARAP fixture has an isolated contact probe, is explicitly labeled a weak disconnected control, and cannot establish superiority over a qualified full-mesh ARAP baseline.

**Scope:** This is sampled linear constraint repair, not IPC, a nonlinear global contact solver, all-intersection detection, or continuous collision detection. No frame-interior collision guarantee or natural-data superiority is claimed. A full IPC comparison, natural self-intersection census, edge-health checks, motion metrics and real runtime evaluation remain necessary for the research hypothesis.
