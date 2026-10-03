# Round 2 proposal: counterfactual utility for a specified geometric edit

2026-10-02. Proposal only; no code or GPU work performed. M-entry, exploratory development. The existing research-state remains `repair_evidence`, with Natural Gate 0, natural failure denominator, second generator and downstream accuracy unresolved. This note is neither a formal Idea Atom approval nor Gate A or a novelty decision.

## Verdict

**Credible as a cheap discriminating pilot, weak as a new-method claim.** Test whether the geometry actually supports the requested edit direction, rather than assigning a generic reliability score to the motion fit. The mechanism is distinguishable from the prior fixed residual gate, but is analytically close to direction-weighted registration validation. It must beat ordinary held-out rigid-fit selection. A gain over only the old threshold is insufficient.

## Existing evidence and scope

Read the completed `4d/results/three-ideas-20261002/README.md`, `4d/actionmesh/research_edit_probe.py`, and `4d/actionmesh/research_moment_energy.py` in full, including the omitted middle of the edit probe through a targeted second read.

The local/global transport tradeoff is observed in constructed controls: local improves articulated edits, but noisy globally rigid motion strongly favors global. The prior gate is direction-blind and had six wins and six losses across asset/seed cells. The native energy implementation instead verifies descent only in a frozen-velocity endpoint extrapolation, not intervention utility after model reevaluation. This proposal addresses the former tradeoff with output geometry and does not claim native latent steering.

Parent question retained, provisionally: when should an existing generated motion field transport a specified edit locally rather than globally, so that the edit follows material motion without amplifying tracking noise? No new natural prevalence claim is introduced.

## Minimal mechanism

Reuse the same proper local and global Kabsch rotations, obtaining two candidate edit fields

`d_t^L(i) = d_0(i) R_t^L(i)` and `d_t^G(i) = d_0(i) R_t^G`.

For a query vertex `i` and a held-out neighboring vertex `j`, let `e_t(i,j)=v_t(j)-v_t(i)`. Virtually insert the particular edit at either sign. The antisymmetric change in squared spoke length is exact:

`h(e,d) = (||e+d||² - ||e-d||²)/4 = e·d`.

For a locally rigid material neighborhood, the appropriate edit transports this response: `h(e_t,d_t)=h(e_0,d_0)`. Score each proposed action, rather than the unedited field, using

`J_a(i,t) = mean_j [(e_t(i,j)·d_t^a(i) - e_0(i,j)·d_0(i))²] / (||d_0(i)||² L²)`.

Here `L` is the fixed full first-frame bounding-box diagonal; unedited points are skipped. Select local iff `J_L < J_G`; ties choose global. Preserve the anchor exactly. No trained parameters, fitted gate threshold, denoiser backpropagation, or model calls are needed. The virtual probes have an analytic implementation, so there is no finite-difference step to tune.

The antithetic construction isolates the edit-induced response: base edge length and the quadratic edit magnitude cancel. The candidates preserve the same edit magnitude, so a zero edit cannot win by suppressing the requested change.

**Honest equivalence:** with `r_a=e_t-e_0 R_a`, the mismatch is `r_a·d_t^a`. This is a directional projection of a registration residual, not a magical uncertainty estimator. The actual research question is whether that intervention-specific projection reduces edit decision regret better than full residual validation.

Under the ideal control's true local rotation, let `delta_d=d_t^a-d_t^true`. The same mismatch equals `e_t·delta_d`, so its mean square is `delta_d C_t delta_d^T`, where `C_t=mean_j e_t^T e_t` is the held-out spoke second-moment matrix. Thus it is an observable, anisotropically weighted surrogate for actual edit error, with a concrete failure mode when `C_t` is poorly conditioned. This derivation motivates the pilot and does not claim the equality survives noisy or nonrigid observations.

## Causal prediction and falsifier

Motion fit errors orthogonal to the requested displacement can inflate a generic residual while having little effect on that edit. Conversely, a modest residual aligned with the edit can make its transport harmful. Consequently, changing only the edit direction on an identical source trajectory should change the correct local/global choice and the proposed utility ranking, while leaving a generic rigid-fit gate unchanged.

Prediction: among point/frame/direction cases where local and global genuinely differ in analytic edit EPE, the sign of `J_G-J_L` predicts the sign of `EPE_G-EPE_L`. The final selected field should reduce both EPE and selection regret against a direction-blind held-out selector, after aggregating each asset equally.

Kill the necessity claim if full 3D held-out selection matches or outperforms this rule; if direction changes do not create different useful decisions; or if it merely optimizes its own spoke score without improving independently known target offsets. Treat a natural-track spoke improvement alone as no evidence of correctness.

## Strong simple baselines and decisive ablation

1. Always global and always local, regenerated under the same fit support.
2. Old fit-residual threshold at 0.005, retained as a historical baseline.
3. **Main baseline:** choose local/global using ordinary held-out 3D error `mean_j ||e_0(i,j)R_a-e_t(i,j)||²/L²`, with exactly the same support and candidate rotations. No threshold tuning is needed.
4. A fixed 50:50 blend of the two edit vectors, explicitly reporting its edit amplitude loss; this checks whether discrete selection is unnecessary.
5. **Direction ablation:** use the mean directional score for the three Cartesian unit directions for every requested edit. Their sum is proportional to full 3D residual error. This provides an algebraic implementation check and tests the contribution of using the requested direction.

Report the unattainable pointwise EPE oracle only as a regret lower bound, never an executable baseline. Compare against the better fixed candidate at the aggregate level as well. Rotations and validation support must match for all selectors.

## Proposed pilot, no more than 10 GPU minutes

Use the existing kangaroo, octopus, mushroom and panda trajectories; no downloads, new models, training, or denoising. All are developmental assets already inspected. Use seeds 0/1/2 and preserve their IDs; they are repeated sampling conditions, not independent assets.

Sample 1,024 corresponding vertices per asset/seed and randomly split identities into 512 fit/query vertices, 256 utility-only support vertices and 256 report-only support vertices. Fit global on the 512 fit vertices; fit each query's local rotation on its 32 nearest fit vertices excluding itself. Use eight nearest utility-only vertices for the decision score and eight report-only neighbors for natural-track diagnostics. **Do not** reuse the old all-points global fit when claiming held-out validation: that would leak utility support into global fitting. Rerun all baselines on this split rather than comparing numbers directly to the prior 512-point experiment.

Keep the existing soft upper-z edit region and amplitude 0.03 times fixed scale. Run seven predefined unit directions: x, y, z, normalized x+y, x+z, y+z, and the previous normalized (1,0.3,0.2). Use three existing controls: global rigid, two-region articulation, and globally rigid with noise sigma 0.005 times scale. The redundant frame-order `fast` control does not count as another regime. Reuse each observed control trajectory for all directions so direction is the only changed factor.

Primary outcome: true edit-offset EPE divided by fixed scale. Secondary outcomes: regret relative to the two-candidate pointwise oracle; accuracy/AUC of signed utility versus actual signed treatment benefit; direction-dependent decision changes; harmful local-selection rate in noisy motion; anchor, finiteness and edit-amplitude checks. Report per asset, per regime and per direction before any equal-asset macro average. Exclude only numerical oracle ties with absolute EPE difference <=1e-6, matching the previous probe; disclose the excluded denominator.

For natural trajectories, report held-out edge-effect drift, candidate disagreement, decision frequency and temporal switching only. There is no natural edit truth in these files. Selection might flicker even when pointwise EPE improves; record this as a cost rather than silently adding smoothing after seeing results.

Freeze exploratory continuation rules before execution:

- **Promising:** >=10% macro EPE reduction against the main held-out baseline over articulation plus noisy controls, benefit in at least three of four asset means, and no >10% regression in the noisy regime. Also require improvement over the better fixed candidate's overall macro EPE. Global-rigid is a sanity case, excluded from percentage aggregation near zero.
- **Reject mechanism necessity:** no EPE gain over the main baseline, or direction-agnostic ablation is equally good within 1% macro EPE.
- **Inconclusive/developmental:** smaller, mixed or unsupported gains; no natural-task or novelty claim follows.

Estimate: under one GPU minute for tensor/SVD work, conservatively cap at ten minutes wall-clock with one sequential CUDA process and peak GPU memory logging. The prior smaller probes took seconds; this is an estimate, not a new measurement. Abort rather than change global environment if CUDA setup fails. Expected working tensors are tiny relative to 22 GiB; enforce the actual resource bound from observed memory, not this estimate. CPU-only execution is also feasible but would not satisfy a request explicitly requiring a GPU run.

## Main risks

The response-preservation assumption inherits local rigidity. Real expansion, shear, sliding contacts or incorrect correspondences can break it. Near-planar spokes poorly identify edit components normal to the surface and can admit ambiguities; the axis sweep may expose this but does not establish universal identifiability. Euclidean neighborhoods can cross articulations; using geodesic neighborhoods may fix both methods equally and must not be introduced after results solely to rescue the candidate. Averaging only eight utility spokes may add enough variance that ordinary 3D validation wins. The constructed rotations cannot establish prevalence or downstream value in natural editing.

## Bounded primary-source check

Date/cutoff: 2026-10-02. Provider: web search/open. Exact discovery queries:

- `mesh deformation transfer directional differential coordinates deformation gradients Sumner Popovic 2004`
- `deformation transfer adaptive local global rigidity validation cross validation edit transport`

Only two paper sources were opened for this check; broad search snippets were not treated as evidence. No exhaustive coverage or formal collision verdict is claimed. Tool transcript retains the returned pages; this note is not a normalized full collision evidence snapshot.

1. [Sumner and Popovic, Deformation Transfer for Triangle Meshes (2004)](https://people.csail.mit.edu/sumner/research/deftransfer/), [author-hosted PDF](https://people.csail.mit.edu/sumner/research/deftransfer/Sumner2004DTF.pdf). Author abstract and PDF opened; D1 triage. Already transfers source transformations and solves for consistent target deformation. Transformation transport is established prior work, so it cannot be the candidate claim.
2. [Fast 4D Mesh Generation by Spatio-Temporal Attention Chains, arXiv:2605.19786v1](https://arxiv.org/html/2605.19786v1). D2 targeted read of Sections 4.1–4.3 and Appendix D.2. Section 4.2 and D.2 use local rigid motion from nearby landmarks, geodesic weights and weighted Procrustes. These components overlap the transport machinery directly. Confidence-based filtering/smoothing also exists. The inspected sections do not establish the specific edit-direction-conditioned, held-out action comparison here; that observation is only a bounded closest-work hypothesis, not evidence of novelty.

No third paper was pursued. If this inexpensive pilot survives its strong baseline, the next decision is natural-task evidence and a focused functional collision audit, not immediate method branding.
