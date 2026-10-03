# CD-M controls and anchor-error diagnostics

Status: protocol/design audit only, 2026-10-02. No new metric run, model run, GPU operation, SSH, or code change. Existing metric fixtures are not repeated.

## What the actual evaluator measures

Primary evidence is the checked-in official source:

- `actionmesh/repo/actionbench/chamfer.py`: `compute_motion_chamfer_score` establishes two nearest-neighbor maps at frame zero, retains those maps, and averages unsquared Euclidean position errors over time in both directions. It does not subtract initial position error or estimate material identity from video.
- `actionmesh/repo/actionbench/sample_mesh.py`: `sample_synchronized_points` chooses faces and barycentric coordinates on the first mesh and uses them throughout the fixed-topology sequence. Fixed indices ensure an index-defined trajectory, not that the trajectory is physically correct.
- `actionmesh/repo/actionbench/icp.py`: `gradient_icp` fits rotation, translation, and anisotropic scale using 24 starts and 200 iterations.
- `actionmesh/repo/actionbench/benchmark.py`: one frame-zero transform is used for CD4D and CD-M; CD3D instead fits each frame separately. Surface CD uses independently sampled clouds, whereas CD-M uses synchronized material samples.
- `actionmesh/research_census_eval.py`: preserves those definitions and budgets. Its already-passed analytic swap fixture proves formula sensitivity, not that the real data's first-frame matches are correct.

Conclusion: official CD-M can detect a change in point trajectories while the per-frame surface stays fixed. It is **not a motion-only metric**. Anchor geometry, incorrect initial nearest matches, shape error, missed components, symmetric parts, and alignment can all affect it. Lower CD-M by itself does not establish better material motion independent of surface projection. A per-frame projection can lower position error while damaging identity; conversely correct motion can retain a large static anchor offset.

The current mannequin and bat scores are baseline measurements only; they do not yet quantify this ambiguity.

## One production negative control: same surfaces, scrambled trajectories

Use the actual cached native `sequence.npz`, the saved shared-anchor matrix, and the official synchronized sampler at 100,000 points with seed 44. Call the aligned clouds `P[t,i]` and GT `G[t,j]`. Freeze alignment, samples, and GT; do not refit ICP.

Keep frame zero exactly unchanged. For each later frame independently permute the **sample rows**, not mesh vertices or faces. A vertex permutation with unchanged faces generally changes the mesh and is not a valid shape-preserving control.

To preserve even the official 10,000-query Monte Carlo surface CD exactly, obtain the predicted query index set `S = RandomState(44).permutation(100000)[:10000]`. For each frame, permute rows within S and within its complement separately. Thus the full point set and the selected query-point set are both unchanged. GT and its seed-45 query set remain unchanged. Surface CD in each frame is invariant up to floating summation order, while trajectories are scrambled. Use a recorded independent control seed, e.g. 20261002, without selecting a seed based on the result.

Report:

1. Exact bijection/block-membership checks, frame-zero identity, and hash of the permutations.
2. Original and scrambled official CD-M, plus per-frame contributions.
3. Original and scrambled surface CD on these identical synchronized point sets. This is a *matched-sample auxiliary surface score*, not the original independently sampled published CD4D scalar.
4. Maximum absolute per-frame surface-score difference, with a small fixed numerical tolerance such as 1e-10 in double-precision KDTree distance accumulation.

This control is a point-trajectory metric intervention. It is not a realizable fixed-topology mesh baseline. A CD-M increase is expected for a meaningful baseline, but it is not guaranteed if the baseline trajectories/initial matches are already poor. Do not discard assets where scrambling fails to worsen CD-M; these are important failures of the proposed interpretation.

## One production positive/identity control: harmless relabeling

Apply one common query-block-preserving row permutation to **every frame including frame zero**. This relabels the same complete trajectories, without changing motion or shapes. Recompute first-frame nearest-neighbor maps from the relabeled frame zero. Official CD-M and the matched-sample surface scores should remain invariant.

This differs from reusing a fixed map after changing frame-zero order, which would intentionally evaluate the wrong point identities. Distance ties can change which equal-distance trajectory is selected; count/report any frame-zero nearest-neighbor ties or correspondence-map differences rather than conceal that ambiguity. No need to rerun the already-passed small `G versus G` and swap fixtures.

## An actual-data oracle reference for anchor/matching ambiguity

Define nearest-neighbor maps once using the aligned native anchor:

```
nu(i) = nearest GT index to P[0,i]
mu(j) = nearest prediction index to G[0,j]
e(i)  = P[0,i] - G[0,nu(i)]
```

Construct the **GT-motion-transfer oracle** only for evaluation:

```
O[t,i] = P[0,i] + G[t,nu(i)] - G[0,nu(i)]
       = G[t,nu(i)] + e(i)
```

The anchor is unchanged exactly. Each anchor sample receives the displacement of its nearest GT material sample. This uses future GT and must never supply a guidance signal, select hyperparameters, train a model, or appear as an ordinary training-free baseline. Save it under an explicit `oracle_gt_motion_transfer` label.

Its prediction-to-GT directional CD-M equals the mean anchor residual `mean_i ||e(i)||` at every frame. This is an analytic audit invariant. Its other direction contains

```
e(mu(j)) + G[t,nu(mu(j))] - G[t,j]
```

and can be large when the two nearest-neighbor maps are not inverse, especially around close or symmetric moving parts. Record the cycle-assignment mismatch rate, the initial spatial cycle distance, and the future cycle-motion separation. A large oracle residual demonstrates that the metric's anchor/matching convention itself imposes an appreciable burden.

**This oracle score is not a mathematical error floor or optimal ceiling.** A different trajectory may compromise between several GT assignments and outperform this particular transfer. Report it as a reference, not as a bound, and never define “percent of possible improvement” from it.

## A genuine conservative lower bound, if a floor is needed

For the official equal sampling counts `P=Q=N=100000`, let `J_i = {j : mu(j)=i}` and define, for each future frame,

```
L[t] = (1/N) sum_i max_{j in J_i} ||G[t,nu(i)] - G[t,j]||,
```

where an empty maximum is zero. The sum of distances from any predicted position to `G[t,nu(i)]` and to all `G[t,j]` in its group is at least the largest distance from `G[t,nu(i)]` to a group member, by the triangle inequality. Therefore `L[t]` is a lower bound on that frame's bidirectional CD-M **for fixed first-frame assignments**, even allowing arbitrary, disconnected future point locations. It can be loose and need not be attainable by a mesh.

Because the anchor is fixed, its actual `C[0]` cannot be improved. A valid sequence lower bound is

```
LB = (C[0] + sum_{t=1..15} L[t]) / 16.
```

Compute group maxima with `maximum.at` on the 100k GT-to-prediction indices; no iterative solver or GPU is needed. Verify `CDM(native) >= LB` and `CDM(oracle) >= LB` within numerical tolerance. The bound applies only while the anchor, alignment, and maps stay fixed. Recomputing them for a changed anchor changes the bound and defeats a clean intervention comparison.

This lower bound measures the incompatibility of nearest-neighbor assignments, not all error caused by geometry. Do not call `CDM - LB` a pure material-motion error.

## Supporting matched-sample diagnostics

For every frame, compute both the fixed-assignment CD-M contribution `C[t]` and the full-query nearest-surface distance `S[t]` on the **same** aligned synchronized prediction/GT point sets. With all 100k queries in both directions, `C[t] >= S[t]`: fixed first-frame matches cannot beat nearest matches chosen freely in the current frame. Report the nonnegative gap as “cost of retaining first-frame assignments,” not as a uniquely identified correspondence error.

Do not subtract the existing published CD4D number from CD-M and interpret the difference causally: those scores use different prediction sampling and query conventions. Do not normalize each future frame or refit its scale. Optional displacement-relative error can subtract initial offsets, but it still inherits incorrect initial nearest maps and must be labeled an auxiliary metric, not official CD-M.

## Minimum controls before a steering claim about motion

1. Same input, noise seed, frozen weights, unchanged anchor/topology, shared saved alignment, fixed sampling faces/barycentric coordinates, and identical compute budget accounting. Baseline and candidate use the same evaluation inputs and maps whenever the anchor is fixed.
2. Both production controls above pass or their failures are explicitly retained. The oracle and lower bound expose how much first-frame matching ambiguity remains for each asset.
3. Improvement in official CD-M is accompanied by the matched-sample shape score, full-query assignment gap, and all per-frame values. The already-running Stage-I/Stage-II shared-alignment GT diagnostic is needed to distinguish gross shape degradation from a possible material-trajectory issue.
4. Compare against a direct output-space surface/track fitting baseline using exactly the same non-GT observation energy and evaluation budget. If the apparent gain is explained by surface projection, claim geometric fitting only. An internal steering claim needs evidence beyond that baseline.
5. Verify held-out visible material tracks or independent rendered-view/edit persistence using evidence not consumed by the guidance objective. This is necessary to support physical/material motion rather than merely a lower evaluator score. Geometry-only symmetric/untextured regions cannot identify tangential material motion from one view without additional evidence.
6. Preserve all eight asset outcomes and the frozen seed42 denominator. The two seed43 repeats show only local seed sensitivity; no population significance, broad generalization, or cross-model claim follows.

These controls qualify evaluation. They do not by themselves demonstrate a new method or warrant a positive paper result.

## Cost and outputs

After a saved alignment is available, this is a CPU-only cached-output diagnostic: official CPU material sampling, two first-frame KDTree maps, 16 frames of indexed vector arithmetic, and optional per-frame KDTree surface scores. The raw 16x100kx3 float32 prediction and GT arrays are about 19.2MB each; oracle and scrambled arrays can be generated one frame at a time. Actual runtime is unmeasured. It should not load a model or repeat generation/ICP.

Persist source hashes (sequence, GT, shared matrix, official metric/sampling sources), sample IDs/barycentric coordinates, control permutation seeds/hashes, both initial maps, per-frame tables, oracle invariants, the lower bound, and the explicit future-GT/oracle label. Keep these artifacts separate from official baseline result JSONs.
