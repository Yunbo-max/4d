# Independent audit of the first 24 normal-task cells

2026-10-02. CPU arithmetic review of existing outputs only: no GPU invocation, model calls, new experiments, or method changes. This review covers four ActionBench assets × three predeclared seeds × clean/noisy observations. It does not include the still-pending 360 generated-asset cells.

Audited input: `/Users/yunbo/Documents/Codex/4d/results/round2-ideas-20261002/raw/research-round2-20261002/pilot/results.json`. Exact digest, per-asset checks and reproduced values are in [round2-result-audit.json](round2-result-audit.json).

## Arithmetic and provenance

- All 12 sampled arrays exactly equal the corresponding rows of the four full original `surfaces.npy` arrays. All recorded material-point indices match both the cached indices and `default_rng(seed).choice(100000,1024,replace=False)` for seeds 17/29/41.
- All four sampled NPZ SHA256 values match the hashes recorded by the GPU run. Fixed scale recomputed from the full original first-frame bounding box matches the cached scale exactly.
- All four saved clean seed-17 source trajectories exactly match the first 512 sampled query identities.
- Independently recomputed upper-z quantile weights, normalized provided normals and the target `position + .03 * fixed_scale * initial_weight * future_normal`. Maximum target discrepancy is **2.80e-8 of fixed scale**, consistent with float32 saved coordinates versus independent float64 arithmetic. Initial edit discrepancy is at most **6.53e-9 of scale**.
- Reproduced mean EPE from saved coordinates for eight methods on each of the four saved clean seed-17 trajectories: **32 method/cell values**, maximum absolute discrepancy **1.27e-9**. Each comparison excludes frame zero and includes exactly the 179 edited query points. This checks saved-output arithmetic independently of the original metrics helper.
- No noisy per-point trajectory files or seed-29/41 trajectory files were saved in this attempt. Their EPEs were therefore independently reaggregated from JSON, not independently recomputed from coordinates. This audit makes no stronger reproduction claim for those 20 cells.

The attempt's overall `failed` status comes from four generated-asset `FileNotFoundError` entries with the erroneous extra `outputs/` path component. It contains all 24 normal-task cells. Those transport-path failures neither invalidate the reviewed normal arithmetic nor count as negative method outcomes. The corrected rerun must remain separately identified.

## Utility versus the strong held-out baseline

Macro values below average seeds within each asset and then assets equally; the balanced design makes this numerically equal to the 12-cell mean for each regime. EPE is divided by the fixed object scale.

| Regime | Always local | Full 3D held-out selector | Directional utility | Utility EPE increase vs held-out |
|---|---:|---:|---:|---:|
| Clean | 0.000635767 | 0.000639736 | 0.000650016 | **+1.607%** |
| Noisy | 0.000945565 | 0.000954714 | 0.001041648 | **+9.106%** |

Clean utility wins in 6/12 asset-seed cells and 2/4 asset means. Noisy utility wins in **0/12 cells** and **0/4 asset means**. These are repeated sampling/noise conditions on four objects, not 12 independent objects. Always-local also beats utility in both aggregate regimes.

The independent Cartesian-axis ablation and full held-out selector have exactly the same reported EPE in all 24 cells. Their maximum algebraic score-identity discrepancy is 8.57e-8, from float32 rotations/arithmetic. Coverage can differ for near-zero score ties even where the candidate offsets are effectively equal; the equal EPEs are the relevant ablation result here.

## What explains the observed delta

Measured behavior: utility invokes global fallback more often than the full held-out selector—**10.282% versus 6.629%** of edited frame/points for clean data, and **28.650% versus 23.296%** for noisy data. Global transport is much worse than local on these articulated normal-following tasks, so extra incorrect fallback is a plausible contributor to the utility penalty. The saved aggregate records do not isolate which individual fallback events account for it, so this is an interpretation, not a proven causal decomposition.

Utility does slightly lower the separate report-spoke response drift: clean RMS 0.00342123 versus 0.00351900; noisy 0.00818829 versus 0.00821525. That favorable geometric proxy accompanies worse independently defined normal-offset EPE. It is direct evidence against treating response preservation alone as edit correctness.

The mechanism assumes a rigid relation between anchor and future spokes. Provided-normal relief edits on nonrigid surfaces need not obey that relation exactly; curvature, anisotropic local support and noisy spokes can make the one-direction measurement less informative than full 3D validation. These are plausible explanations only. This audit did not fit a post hoc threshold or launch an ablation to select one explanation.

## Claims permitted by this subset

The module ran, targets and saved EPE arithmetic are consistent, and the proposed utility **does not improve the strong baseline on these aggregate normal-task results**. It therefore has no supported positive efficacy claim from this subset. A result such as lower geometric response drift, better performance than the old fixed threshold, or a favorable individual asset does not establish the intended advantage.

The other primary q50 selectors also lose to always-local in both normal-task aggregates: directional covariance EPE is 0.00250598/0.00289485 and temporal action EPE is 0.00240290/0.00267619 for clean/noisy. There are narrower favorable comparisons among ranking controls, but they do not establish superiority over the strongest simple fixed transport. All 24 cells have zero edited weak-support queries, so they cannot validate the geometry method's proposed weak-identifiability mechanism.

Do not declare final synthetic pilot pass/fail from these 24 cells: the frozen primary generated-asset regimes are in the separate pending run. Nothing here establishes natural generator failure prevalence, cross-model performance, native diffusion steering, novelty, or publication readiness.
