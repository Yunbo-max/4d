# Round 2 geometry candidate: edit-direction identifiability

Date: 2026-10-02. Status: proposed developmental probe; no implementation or GPU execution in this subtask. Scope: frozen output geometry from the four existing ActionMesh trajectories, within 22 GiB. This does not establish a new inference-time latent steering method or a natural benchmark improvement.

## Decision and reason

Test **whether the requested edit direction, combined with local geometric support, predicts when local transport is unsafe better than a scalar fit-residual gate**. Retain local/global Kabsch as baseline machinery. Do not describe rotation covariance, spectral regularization, or deformation transfer as new.

The previous probe supplies a concrete failure: local transport beat global on constructed articulation but was 12.84 times worse under noisy global motion; the fixed gate split 6 wins and 6 losses over asset/seed cells. Its scalar residual and rank scores cannot distinguish two different edit directions on the identical patch. A thin patch can determine its long-axis motion accurately while leaving twist about that axis poorly determined. An edit along the axis barely depends on the uncertain twist; an equally large transverse edit does. This yields a paired prediction unavailable to a direction-blind gate.

The most important caveat is prevalence. Existing full, corresponding meshes may contain little relevant degeneracy. If the effect requires artificially squeezing every patch into a line, it is a diagnostic example, not an important demonstrated problem in these assets. Occlusion should not be claimed: all current vertices are available, and dropping vertices would be constructed partial observation.

## Specific mechanism

Use the existing 512 sampled points, 32 fit neighbors and eight disjoint diagnostic neighbors. Fit reference-to-frame local and global proper rotations exactly as in `research_edit_probe.py`. Use column-vector notation below; implementation must respect the existing row-vector convention.

For centered reference patch points r_j, form the 3-by-3 rotation information matrix

`J = sum_j (||r_j||^2 I - r_j r_j^T)`.

If the patch scatter eigenvalues are s1 >= s2 >= s3, J's eigenvalues are `s2+s3, s1+s3, s1+s2`. A planar rank-two patch still identifies a rigid rotation; do not mark all surfaces degenerate just because s3 is small. A near-line patch has one weak rotation mode.

Let `J = Q diag(lambda_k) Q^T`. Estimate a coordinate noise scale sigma from the held-out point-to-point residual under the fit rotation, including the fit translation. In constructed noisy controls, report both the practical residual estimate and an explicitly labeled known-noise oracle score. Use a declared natural-data floor `sigma_floor = 1e-3 * patch_RMS_radius`, with sensitivity at 1e-4 and 1e-2; this floor is an engineering tolerance, not estimated generative uncertainty.

Set angular variances `c_k = min(sigma^2 / lambda_k, (pi/2)^2)` with null eigenvalues assigned `(pi/2)^2`, and `C = Q diag(c_k) Q^T`. Zero eigenvalues must not receive zero variance through a pseudoinverse. The finite cap avoids claiming a small-angle approximation is calibrated for arbitrarily large uncertainty.

For initial edit d, compute

`u(d)^2 = trace([d]_x C [d]_x^T) / ||d||^2`.

For d=0, retain zero. This is an **action sensitivity proxy**, not a calibrated posterior probability. It follows by linearizing the action of rotational error on d. For a line with axis q, the large uncertainty term is proportional to `||q cross d||^2`, which vanishes for an axial edit. A fit residual, rank ratio, or total `trace(C)` does not have this property.

The minimum intervention is a direction-dependent local/global gate: choose local transport if `u(d) <= tau`, global otherwise. Select tau only on the other assets using a fixed grid `[0.02, 0.05, 0.1, 0.2, 0.5]`, then evaluate the held-out asset. The parameter fit is a small decision calibration, not neural weight training. Also report an untuned tau=0.1. The method has no guarantee that global transport is good; the test must include regimes where global is biased.

Do not add energy penalties or latent gradients before this causal prediction survives. If later moved into an energy, its independent target would need to be material edit adherence, not reduction of its own uncertainty score.

## Strong alternatives and ablations

1. Always local, always global, existing fixed residual gate, and world offset.
2. Recalibrated scalar residual gate with the same leave-one-asset-out procedure and equal threshold-grid size. Also use a geometry-aware but direction-blind score `sqrt(trace(C))`; beating only the old fixed threshold is insufficient.
3. **Bootstrap action spread:** 16 resamples of the same 32 neighbors, refit rotations, transport d, and use the relative offset spread in the identical gate/calibration. This uses the requested edit direction without an analytic information model. If equal or better, prefer this simple method; the analytic score may still have a speed advantage, but that alone is not a 4D scientific contribution.
4. **Global-prior rotation regularization:** fit `argmin_R sum ||R r_j-y_j||^2 + beta ||R-R_global||_F^2` by one augmented Kabsch/SVD, with beta calibrated on other assets. This is a stronger, standard geometry-conditioned alternative than scalar gating. Normalize the first term by patch support energy before applying a dimensionless beta grid. It may solve the problem without selecting edit directions explicitly.
5. Decisive ablation: shuffle paired edit directions in u while keeping each method's actual edit and all observations fixed. If performance persists, the claimed direction mechanism is unsupported.

All methods see identical points, noise, local/global fits, edit magnitude, and calibration folds. Do not compare a tuned directional gate to an untuned weak scalar gate. Record gate coverage, not just final EPE; a win from always reverting to global under noise is already explained by the baseline.

## Bounded test, maximum ten GPU minutes

Reuse kangaroo, octopus, mushroom, panda and source vertex IDs. Suggested wall-time allocation: 1 minute geometry census; 5 minutes batched controlled fits; 2 minutes natural-track diagnostics; 2 minutes export. No generator forward pass or gradient is necessary; CPU is also plausible. Use float64 for information eigendecomposition and small SVDs if affordable. Stop at the budget rather than reducing the declared evaluation after observing results.

**First, census unchanged geometry.** For every existing reference fit patch, save `lambda_min(J)/trace(J)`, radius, and paired u for the three patch PCA-axis edit directions at the original edit magnitude. Report distribution over *all* sampled patches and separately the existing edited region. Give each asset its own summary. Predeclare a weak-support stratum `lambda_min(J)/trace(J) < 0.01`; retain and report assets with zero eligible patches. No GT is needed for this census, and it cannot demonstrate natural failure frequency.

**Second, paired directional controlled test on unchanged patches.** Apply the previous analytical global rotation and two-region articulation generators with seeds 0/1/2 and point noise `{0, 0.003*L, 0.01*L}`. Use exactly the same observed sequence twice, with equal-norm edits parallel and perpendicular to each patch's dominant reference axis. Keep the same region weights. The targets are d transported by the injected, known material rotation. Run all methods above. Restrict primary mechanistic reporting to the declared support strata, and also report every-patch averages so selective focus remains visible.

For each asset and regime, measure edit-offset EPE/L and EPE/initial-edit-norm (excluding zero edits), local-versus-global harm decision regret, fallback coverage, and the paired difference between axial and transverse error. Prefer continuous error calibration plots/correlations alongside harm AUROC; local uncertainty alone need not predict harm against a biased global estimator. Report the global prior's error explicitly. Aggregate first within each asset, then across the four assets; three seeds are repeated perturbations, not three independent objects.

**Third, analytical stress test if census is weak.** On copies of the reference patches only, compress the two minor PCA axes by factors `{1, 0.1, 0.01}`, then apply known rigid rotations and observation noise. Label every modified example as constructed geometry. This tests the expected approach to twist ambiguity. It cannot rescue a missing natural prevalence result. Include an isotropic patch and a planar patch as positive controls against indiscriminate rank gating. Include exact-line zero-noise behavior as an algebraic unit control; do not count it as performance evidence.

**Natural trajectories:** only report intervention size, edit amplitude, unedited-vertex invariance, local/global disagreement and perturbation sensitivity. Any self-held-out rigidity measure remains diagnostic, not motion correctness. No GT exists for a natural material edit on these generated tracks. Observations injected into a pre-existing generated track are still an engineered corruption, even though the underlying shape came from ActionMesh.

## Predeclared outcomes

- **Mechanism survives:** on unchanged weak-support patches, the directional score ranks the axial/transverse error gap correctly on at least three of four assets that have eligible patches; leave-one-asset-out EPE improves at least 10% versus the best scalar gate, without more than 5% loss on clean articulation. Report the absolute EPE and denominators too. With fewer than three eligible assets, this outcome is unavailable.
- **Replace by simpler alternative:** bootstrap spread or global-prior Procrustes matches or beats the directional method within 5% EPE. Keep that alternative as a baseline; do not market analytic covariance as the innovation.
- **Kill current action policy:** it does not beat scalar alternatives, improves only by globally reverting in noisy-global controls, breaks clean local articulation, or direction shuffling preserves the gain.
- **Importance unresolved:** gains occur only on squeezed synthetic patches, or eligible unchanged patches are too scarce. Keep the counterexample as a limitation/test fixture; do not advance a natural 4D claim.
- None of these is formal Gate A PASS: the saved project still lacks a natural failure denominator, independent downstream endpoint, qualified second backbone and complete collision evidence.

## Primary literature, bounded to two papers

Search date/cutoff: 2026-10-02. Queries: `Markley attitude determination vector observations singular value decomposition covariance NASA 1988`; `Sumner Popovic deformation transfer triangle meshes 2004 paper`; and a targeted NASA search for the exact Markley title. This was a mechanism collision check, not systematic novelty clearance.

1. **Markley (1988), Attitude Determination Using Vector Observations and the Singular Value Decomposition.** [Scanned primary paper](https://www.malcolmdshuster.com/FC_Markley_1988_J_SVD_JAS_MDSscan.pdf), [author-uploaded abstract](https://www.researchgate.net/publication/243753921_Attitude_Determination_Using_Vector_Observations_and_the_Singular_Value_Decomposition). Read depth: abstract plus verified primary PDF availability; scan extraction had no text and screenshot fetch failed. The abstract explicitly describes rotation-estimate covariance and its directional eigensystem. This is sufficient to reject novelty for the covariance component, but exact-equation equivalence has not been audited. The J and action linearization above are the present derivation, not a quotation or claimed implementation of Markley's equations.
2. **Sumner and Popovic (2004), Deformation Transfer for Triangle Meshes.** [Primary paper PDF](https://www.cs.toronto.edu/~jacobson/seminar/sumner-and-popovic-2004.pdf), [authors' project page](https://people.csail.mit.edu/sumner/research/deftransfer/). Read depth: Sections 3–4, equations 1–10 and surrounding text, pp. 2–4 of the preprint. It transfers local affine transformations with consistency constraints, and explicitly introduces a perpendicular fourth point because three triangle vertices do not determine the out-of-plane affine extension. This strongly overlaps the transport setting. Its affine ambiguity must not be confused with rigid Kabsch: three noncollinear points do determine a proper rigid rotation.

An incidental citing-paper abstract on the Markley page mentioned direction-of-interest attitude weighting in 2026. It was not opened as a third paper under this budget and is not relied on scientifically. This further discourages any novelty claim around query-dependent rotation uncertainty; full collision work would need to inspect it before promotion.

## Handoff

Recommendation: run this as a cheap explanatory probe of the failed residual gate, with analytic directional covariance treated as a candidate *baseline*. The potentially useful result is a measured editing failure tied to directional identifiability. A covariance formula, regularizer, or synthetic line example alone is not sufficient for a publishable 4D method.
