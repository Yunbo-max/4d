# Bounded ARAP solver-qualification child plan

Status: read-only recommendation; **not implemented or launched**. Keep v1/v2 artifacts and their unqualified labels intact. This is additional optimization work for the same declared simple baseline, not a new method or evidence of efficacy. Earlier GT outcomes were already disclosed by the parent; none is used to choose settings below, and no GT files were read in this review.

## What failed

Root reports all 25 fixed-rotation global solves for each nonanchor frame converged and satisfied global/local descent. There are no new degenerate faces. Trust violation divided by anchor diagonal is zero except bat frame15 at `6.2619e-9`, within the existing `1e-6` float32 export tolerance.

The only failed condition is final outer relative energy change:

| Frozen asset | Frame8 | Frame15 | Whole ARAP-arm elapsed |
|---|---:|---:|---:|
| mannequin | 0.05335718725 | 0.06591082209 | 145.9488 s |
| bat | 0.01604727771 | 0.02225141471 | 167.8242 s |

The threshold is `1e-3`. These are 1.6%–6.6% energy changes per outer block, so the fifth block still materially changes the declared objective. Inner QP convergence is not convergence of the outer alternation over geometry, closest points and rotations. Equally, a small outer objective change is only the existing numerical qualification proxy; it is not a proof of a global nonconvex optimum or correct material correspondence.

Source: `research_census_surface_baseline_v2.py:226–289`. Each outer block computes closest points, performs five local/global solves, refreshes closest points and rotations, and measures `(abs(E_after-E_before)/max(abs(E_before),1e-12))`. The existing all-solve and all-outer descent checks must remain active.

## Recommended one child, frozen before execution

Run **ARAP only**, on both already selected assets, frames0/8/15. Copy/verify the held anchor; optimize frames8 and15. Reuse the identical source native sequence, Stage-I triangle targets, exact-query cache, topology and coordinate system. Do not rerun Stage0, StageI, StageII, identity, direct closest-surface or normal-only fitting.

Use a **maximum of 20 outer blocks per optimized frame**, with the same five local/global solves per block. Begin checking the existing convergence predicate after block5 and stop at its first passing block; these current cases will require additional blocks. If it never passes, return the block20 result as `outer_cap_reached`, numerically unqualified. This is a fixed fourfold iteration allowance, chosen from the incomplete optimization and observed runtime, not a search over caps against reconstruction scores. There is no automatic second increase to 40/80/etc.

Freeze a **900-second wall limit per asset**, with a **420-second per-nonanchor-frame limit**, checked inside every projected-gradient iteration and triangle-query batch. Remaining allowance covers source validation, anchor and serialization. Use an external guard slightly beyond the script limit only to terminate a stuck dependency call, not to extend the numerical budget. Both assets receive the same caps. The first-five replay is charged to these budgets. An approximately fourfold ARAP cost is 584–671 seconds per asset, plus setup; this is an estimate, not a promised runtime. Execute CPU-only with the same numeric library versions and thread environment. Record actual CPU time and RSS; GPU allocation is unnecessary.

The only changed numeric budget is outer cap5→20. Preserve exactly:

- Objective, coefficients, uniform graph and same-frame native reference X.
- Trust radius `0.02*anchor_bbox_diagonal` and original trust-ball centers X.
- Five local/global solves per outer block; maximum200 projected steps per solve; normalized projected-step tolerance `1e-7`; the existing Lipschitz step.
- Outer relative-change threshold `1e-3`, global/local/outer descent checks, and finite-output checks.
- Existing float32 export, `1e-6` trust-violation tolerance, zero-new-degenerate-face requirement, exact anchor and faces.

Do not relax a tolerance, increase the inner cap, change weights, enlarge the trust radius, switch correspondence rules, or choose a checkpoint because of GT. If an inner solve fails its existing qualification, retain that failure; an outer-only extension cannot silently reclassify it. Keep final geometry qualification separate from solver convergence. Reaching an energy stop and then failing geometry checks is an unqualified result, not a reason to search later checkpoints for a preferable score.

## Existing artifacts cannot support exact continuation

`research_census_case.py:323` explicitly stores source vertices as float32. V2 optimizes float64 Y but casts it back to the source dtype at `research_census_surface_baseline_v2.py:458`. Its saved displacement is computed **after** this cast (`:466`). Saved rotations and closest points do not reconstruct the lost float64 Y. Thus promoting saved vertices to float64 is a quantized warm start, not the exact state after outer5. It can change nearest-face ties and later nonconvex trajectories.

For the clean child, replay **only ARAP** from original X through five blocks, under the unchanged v2 settings. Require the resulting float32 geometry to equal the v2 ARAP frame bytes, and check the existing solver log/energy values for deterministic agreement before continuation. If the replay differs, stop and report a lineage/reproducibility mismatch; do not present its remaining iterations as an exact continuation. Save the uncast float64 Y at the outer5 boundary and each subsequent completed block. Continue the same loop through the frozen convergence/cap rule.

At an outer boundary, Y plus original X/faces/target hashes, completed block count and accumulated solver logs suffice to reconstruct the next block: closest points and local rotations are recomputed at its start. Save float64 checkpoints before export, along with package/source hashes and thread environment. Avoid promising exact resume from an interrupted inner solve unless its full inner state is separately saved. Future resumes from a validated float64 outer checkpoint need not replay earlier blocks.

This replay costs roughly one additional original ARAP pass per asset but preserves its initialization and trajectory. A quantized warm-start experiment is possible, but it must be named a separate child with that limitation; it is not the recommended baseline qualification path.

## Stop rules and scientific interpretation

Record one explicit terminal reason per frame: `existing_criteria_passed`, `outer_cap_reached`, `frame_time_budget`, `asset_time_budget`, `inner_solve_unqualified`, `descent_failure`, `geometry_invalid`, or `replay_mismatch`. Save the last completed outer checkpoint on budget termination. Preserve failures in the two-asset denominator. Root's independent evaluator may later score valid geometry, but it must carry these qualifiers and never choose the stop/checkpoint.

If all frames satisfy the unchanged criteria, describe it as **the declared ARAP baseline with numerical qualification under this budget**, not an optimal or universally strongest ARAP solution. Compare future methods to that completed child and retain the five-block cost/result as a budget comparison.

If either asset remains unqualified, stop this qualification campaign after the one bounded child. Report the achieved objective decrease, final relative change, projected residuals, geometry validity and consumed work. Label it a budget-limited ARAP baseline. We can still report comparisons at matched explicit budgets, but cannot claim a proposed method defeats a sufficiently solved strong ARAP baseline, or that ARAP's remaining failure establishes the need for a new steering method. Three tested frames still do not establish full-sequence CD-M or temporal correspondence quality.
