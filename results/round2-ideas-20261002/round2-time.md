# Round 2 temporal candidate: edit-direction temporal path disagreement

Date: 2026-10-02. Proposal only; no new code or GPU execution in this scout. This uses the existing four mesh assets and the previous local/global transport implementations. It is a developmental falsification candidate, not a novelty finding or formal Gate A.

## Decision and actual difference

Test whether **disagreement between independently fitted temporal paths, after applying the requested edit direction**, identifies when local transport should fall back to global transport. The previous gate used a scalar fit residual at the destination; it won on only 6/12 asset×seed cells. Here the observable is the change in the *specific action* under different temporal explanations. This is a selective intervention policy, not temporal smoothing or another displacement penalty.

The known parent problem is preserving an initial local geometric edit through frozen-model animation without corrupting the source motion. Current evidence is restricted to four ActionMesh outputs and constructed diagnostic targets. There is no established natural failure denominator or edit-quality GT, so Natural Gate 0 remains unresolved.

## Mechanism precise enough to implement

For each edited vertex i, retain its fixed material-neighbor IDs, and fit proper Kabsch rotations independently for every required frame pair. Let L_ab be the local rotation from frame a to b, and G_ab the global rotation. Use the row-vector convention already used by `research_edit_probe.py`.

For destination t and intermediate frames m≠0,t, compare:

    local_direct = d_i @ L_0t
    local_via_m  = d_i @ L_0m @ L_mt
    global_direct = d_i @ G_0t
    global_via_m  = d_i @ G_0m @ G_mt

Use all m for T=16 (or a fixed evenly distributed subset), including future frames as an offline editor. Define s_L as the median over m of norm(local_direct−local_via_m)/norm(d_i), and s_G analogously. Rank action risk by s_L−s_G. Among edited vertices at each t, fall back from direct local to direct global on the highest-risk 25%, 50%, or 75%. Keep the zero frame exact. These are fixed coverage budgets, not fitted uncertainty thresholds; report every budget, with 50% as the preregistered primary contrast. If all scores are within a numerical tolerance, retain local rather than arbitrarily fallback.

Do not compute L_ab L_ba when the inverse is explicitly set to L_ab transpose. Do not build each relative rotation from already synchronized absolute rotations. Both make cycle closure tautological. Here every edge is independently fitted from the observed points. The three-edge discrepancy can be nonzero under nonrigidity/noise even with exact vertex correspondence.

Do not optimize mesh temporal derivatives or average trajectories. An edit along an unaffected axis can have small action disagreement even when the full rotation matrices disagree; this is the distinguishing prediction.

## Minimum falsification

Reuse kangaroo, octopus, mushroom, panda paths from `research_edit_probe.py`. Sample 512 corresponding vertices, 32 fit neighbors and 8 disjoint heldout neighbors, fixed first-frame full-mesh diagonal as scale. Seeds 17,29,41; same assets remain developmental regardless of new seeds. Use existing d0 support but test three independent edit directions and a diagonal, retaining amplitude 0.03×scale.

Construct diagnostic sequences on real initial geometry with exact transported-edit targets:

1. Noncommuting two-region articulation: region-specific Rz(theta(t)) Rx(phi(t)), opposite angle signs by region; theta and phi have different phase, with regions centered separately. This replaces the old single-axis, commuting control. Edit GT is the known region rotation applied to d0. Report boundary and interior separately.
2. The same articulation plus independent observation noise on non-anchor frames. Keep the clean target and noisy source separate. This tests the actual bias/variance tradeoff rather than only global noisy rotation where global is almost guaranteed to win.
3. Global rotation with the same noise: necessary simple-baseline win case. The policy must not hurt the global baseline while claiming robustness.

Positive control: noiseless global rotation, where all methods must recover the analytic edit to floating-point tolerance. Also include one limiting/model-mismatch control: anisotropic stretch with d0 transported by the true affine map. Rigid paths can agree yet be wrong for this task. A low cycle score must never be interpreted as a calibrated confidence or correctness guarantee.

Natural four-asset trajectories: compute action-disagreement distributions, chosen fallback fractions, heldout edge-effect drift, and raw animations. These are diagnostics only. Do not call a reduction in disagreement, strain, or acceleration “quality improvement.”

## Strong baselines and discriminating controls

- Direct global, direct local, previous fixed residual gate, and no transport/world offset.
- Fit-residual **ranking** at exactly the same fallback coverage, so a gain cannot come from a better threshold or more abstention.
- Direction-blind matrix-cycle risk: median Frobenius norm of L_0t−L_0m L_mt, subtract corresponding global risk; same coverage. This isolates the edit-direction contribution.
- Spatial split/bootstrap action disagreement using the same number of Kabsch fits, without extra temporal paths. This tests whether temporal information is needed at all.
- Oracle local/global choice by true diagnostic edit error, explicitly labeled oracle; verifies that measurable headroom exists.

Primary error: mean edited-vertex offset EPE/scale against known analytic target, separately per asset×seed×regime×direction. Primary comparison: the proposed risk ranking versus the strongest executable non-oracle ranking baseline at 50% fallback. Show 25/50/75% risk-coverage curves and always-local/always-global endpoints. Aggregate first within asset; four assets are the independent descriptive units. No vertex/frame pseudo-replication or population significance.

Pre-result decision: KEEP FOR FURTHER DEVELOPMENT only if the proposed policy beats both fit-residual ranking and direction-blind cycle risk by ≥5% mean EPE on mixed articulated/noisy controls, improves at least 3/4 assets, and does not raise the global-noisy EPE by >5% over the strongest baseline. If bootstrap performs equally well (within 1% EPE), abandon the temporal-mechanism claim. If oracle headroom is <5%, mark the probe uninformative rather than claiming failure of all temporal methods. If no action-direction benefit appears, retain generic path disagreement only as an old-tool baseline and do not rename it as a new method.

## Budget and integration

No denoiser invocation, new weights, training, or latent backpropagation. At T=16, 4 assets, 512 points, and 32 neighbors, the pairwise Kabsch workload is small. Evaluate frame pairs sequentially or in bounded batches. A conservative bound is 10 minutes GPU, 22 GiB peak total process memory; expected memory is far below this, but measure rather than claim it. Existing full meshes stay CPU-side; only sampled tracks go to GPU. Cached AM latents are unnecessary for this minimum test. If it passes, a later native steering hook remains a separate engineering and scientific obligation.

## Strongest counterargument

Path consistency can mistake coherent bias for truth. Global fits can be consistent yet ignore articulation; local fits can be inconsistent yet still beat global. The proposed score measures action sensitivity to temporal fitting choices, not actual improvement. Direction-blind cycle filtering and spatial bootstrap may explain all gains. The experiment is designed to kill those unsupported interpretations cheaply.

## Narrow literature collision record

Search date/cutoff 2026-10-02. Exact queries: `"deformation transfer" "cycle" temporal consistency editing`; `"rotation averaging" "cycle consistency" deformation graph`; `Sumner Popovic deformation transfer triangle meshes 2004 pdf`. Searches returned additional works, but only the following two primary sources are used as evidence; this is a narrow collision screen, not systematic coverage.

1. Li and Ling, *On the Robustness of Multi-View Rotation Averaging* (2021), https://arxiv.org/html/2102.05454v1 (abstract and full-text accessible; relevant locator: introduction and cycle-consistency/IRLS initialization). It already uses cycle consistency to reduce influence of unreliable pairwise rotations. Generic temporal rotation-cycle gating therefore has a major functional collision. Applying it to mesh frames does not establish novelty. Any retained investigation must test a concrete edit-direction-dependent decision beyond that primitive.
2. Sumner and Popović, *Deformation Transfer for Triangle Meshes* (SIGGRAPH 2004), author paper https://homes.cs.washington.edu/~jovan/papers/sumner-2004-dtt.pdf and project https://people.csail.mit.edu/sumner/research/deftransfer/ (project abstract and returned paper excerpt read). Transformation transfer and constraint-based consistency are established. Local Kabsch transport itself is not the new contribution.

Status: technically executable candidate; anticipated collision is substantial; no novelty claim. Even a successful controlled probe does not establish a worthwhile natural task or a general 4D research contribution.
