# Three exploratory training-free 4D probes — protocol v1

Frozen before observing new results, 2026-10-02. User explicitly requested three ideas and tests using research-autopilot. This authorizes bounded exploratory implementation and GPU runs. This is **not a formal Gate A**: natural failure prevalence, a frozen parent problem, qualified downstream endpoints and cross-backbone validation remain missing. The earlier research ledger stays in evidence repair. No novelty or ICML readiness is inferred from a successful run.

## Scope and budget

- Existing allocated GPU: one RTX 2080 Ti, 22528 MiB, serial execution. No training, new model downloads, paid provisioning or global environment changes.
- Existing ActionMesh weights and official kangaroo input. Standard 16-frame context, FP16 low-RAM, 100 Stage-0 steps and 30 Stage-I steps, seed 42.
- Share identical preparation and the first 20 of 30 native sampling steps. Branch only at step index 20. Save hashes, boundary latent, full schedule, all failed attempts, logs, wall time and peak memory. This saves repeated computation without changing any branch's 30-step trajectory.
- Five suffixes: unmodified, selective CFG cap, global CFG cap, channel-moment energy, scalar-RMS energy. Decode all 16 frames with the same anchor and frozen Stage-II weights; no fresh video generation.
- At most 45 minutes total GPU execution for this packet, 12 minutes per native command, 3 minutes for edit probes, <1 GiB new disk. Stop on OOM or finite-check failure; at most one implementation repair per distinct failure. Infeasible jobs remain failures, not silently smaller official settings.
- Four pre-existing assets for output-space edit probes: kangaroo, octopus, mushroom, panda. These are four subjects from **one** 4D backbone, not independent generator replications. Seeds 0, 1, 2 for constructed observation noise; no tuning on outcome.

## Idea 1: Bound anomalous conditional forcing late in sampling

Hypothesis: image-conditioned velocity increments that are much larger in a few frames can cause a disproportionate geometric change. Capping only those increments may preserve motion better than reducing guidance everywhere.

Mechanism: native conditional increment `delta = v(image+anchor) - v(anchor)`; each frame's RMS over tokens/channels is capped at the within-clip 75th percentile of unobserved frames in the final 10 steps. Direction and anchor remain unchanged. This is a norm constraint, **not calibrated epistemic uncertainty**.

Unique prediction: at the same aggregate conditional-update norm, selective capping changes geometry/motion less than global scaling while reducing extreme edge/extent diagnostics. Strong simple alternative: one global scale, computed from the same branch predictions to match aggregate norm at each step.

Falsifier: intervention is numerically inactive, removes >10% of motion without independent task improvement, or offers no advantage over global scaling. A single natural clip with no GT cannot decide quality; runtime/mechanism can pass while scientific outcome remains inconclusive.

Closest-work risks: ordinary CFG rescaling, clipping/trust-region methods, USteer-style inference control. Mere transfer to 4D is insufficient novelty. Sources: [USteer](https://arxiv.org/html/2609.38962v1), [ActionMesh](https://github.com/facebookresearch/actionmesh).

## Idea 2: Energy with an uncertainty band over permutation-invariant shape statistics

Hypothesis: constrain only excessive drift of latent channel variances from the anchor, leaving a dead zone for conditional disagreement, to avoid forcing valid motion to the anchor.

Mechanism: for the actual additive scheduler, extrapolate `xhat = z + s * stopgrad(v)`, where `s=t/1000`. This is a constant-velocity endpoint approximation. Compare token-centered per-channel variance of xhat to the anchor; dead-zone width is detached conditioning-branch disagreement plus a fixed numerical/relative floor. Apply a normalized energy descent correction to velocity, capped at 10% of its framewise RMS, in the last 10 steps. Backpropagate only through the small xhat energy, **not the denoiser or its weights**. Token-permutation invariance avoids assuming token indices are tracked material points.

Unique prediction: channel-wise constraints preserve useful anisotropic latent variation better than a scalar RMS constraint at the same correction budget. Strong simple alternative: scalar centered RMS energy with the same schedule and norm cap.

Falsifier: non-finite/zero gradients, no local energy descent, geometry becomes static/collapses, or the scalar alternative gives the same/better outcome. Lower latent energy alone is never evidence of better 4D geometry. The uncertainty band is not calibrated.

Closest-work risks: FreeDoM/Universal Guidance, moment matching and robust dead-zone penalties. Sources: [FreeDoM](https://arxiv.org/html/2303.09833v1), [Universal Guidance](https://arxiv.org/html/2302.07121v1), [USplat4D](https://arxiv.org/html/2510.12768v3).

## Idea 3: Transport local edits with the source motion

Hypothesis: a shape edit defined in material/local coordinates remains attached to a moving part better than applying the same world-space offset to every frame, and local transport can outperform a single global rigid transform for articulated motion.

Mechanism: fit local rigid transforms on tracked source vertices, transport a fixed edit offset, and leave unedited vertices untouched. This is **output-space geometry processing**, not a native latent or activation intervention. Foundation-model weights remain frozen.

Tests: analytical global rotation and spatially separated local rotations supply an explicit target; noisy observations test sensitivity. Compare world-fixed offsets, global Kabsch transport, local Kabsch transport and unchanged output. Evaluate edit offset error against analytical target and held-out-neighbor consistency; report untouched drift only as a construction sanity check. Real generated sequences have no edit ground truth and yield diagnostics only.

Falsifier: local fit is no better than global fit in articulated controls, substantially more noise-sensitive, or unstable on real tracks. A rigid-transform positive control is not evidence of natural task importance. Strong closest-work collision with deformation transfer / local rigidity / Fast4DMesh is expected; any future claim needs an added inference-control contribution.

Sources: [Fast4DMesh](https://arxiv.org/html/2605.19786v1), [Self-Guidance](https://arxiv.org/html/2306.00986v2), [Dynamic-eDiTor](https://arxiv.org/html/2512.00677v1).

## Measurement and decisions

Native results: finite vertices, fixed topology, exact preserved anchor, active intervention count, gradient/correction magnitude, shape extent, edge-length strain, displacement and first/second frame differences. Normalize by the **fixed first-frame bounding-box diagonal**, use model frame-step units, no per-frame ICP. Compare each branch to the shared-prefix unmodified branch using same vertex indices. These measures diagnose changes, not reconstruction accuracy.

Edit controls: known target offset RMSE, held-out-neighbor error and noise robustness; all asset/seed cells retained. Real-sequence preservation metrics are not GT motion scores.

Report each as IMPLEMENTATION VERIFIED / FAILED and SCIENTIFIC REVISE / INCONCLUSIVE / REJECT EXPLANATION. No formal PASS, statistical population claim, benchmark victory or cross-model transfer claim from this packet. Developmental observations cannot later be relabeled a prospective confirmatory split.
