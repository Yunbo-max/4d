# Resume the full-population baseline

The user has repeatedly authorized completing the baseline qualification and all
15 native comparisons. The completed one-asset FP16 low-RAM unit took 1330.47 s.
Its scorer uses CUDA forward and the declared upstream CPU KNN backward policy.
This execution continues that explicit current-release profile; it does not
claim equivalence to the unpublished leaderboard hardware/dtype configuration.

Ruling: use a separately named population-unit contract to admit each exact
canonical population index. Preserve the original first-UID calibration check.
Changing the old calibration rule would silently change the earlier experiment;
the new contract keeps that boundary explicit.

Freeze all 128 released UIDs in lexicographic order. Index 0 already has a
completed three-arm unit (`complete-lowram-r7`) and is retained as development
evidence. Start the next window at index 1, with nine consecutive units and a
2700 s timeout per unit. The window is bounded by 28800 s, reserving 1800 s for
collection. This is an engineering baseline campaign; no candidate is admitted
and no pass threshold is inferred from one sample. All failed/timed-out units
remain in the denominator and require separate diagnosis before retry.

Reuse generation, export, native/world/body controls, and unchanged official
metric budgets. Each unit must revalidate its source, model and input hashes;
no GT or score is used to choose a UID or alter generation. Preserve all raw
attempt evidence. Do not repeat successful calibration or scorer parity.

Implementation: extend unit-manifest validation for the new explicit contract;
add a plan-only population window builder composing the existing manifest and
complete-unit builders. Test that indexed selection is exact, the original
calibration rule remains strict, and invalid/overbudget windows are rejected.
Run all engineering acceptance through the installed Linux harness before GPU
dispatch. No global environment changes or data deletion are required.

The final population aggregate, protocol admission, trusted replay and natural
failure/strong-control analysis remain necessary. A completed window cannot be
reported as 128/128 or as any of the 15 methods.

Upstream `de608b1` was merged during preparation. Both the source-complete-unit
admission (`admit-complete-r9b`) and pricing (`price-full128-r9`) now passed.
The frozen 5/4 price reports 1664 s per unit and capacity 16; the execution
window deliberately uses the previously declared conservative 2700 s cap and
nine units. This operational headroom costs throughput, does not change any
generation/evaluation parameters, and does not confer scientific qualification.
The first admission attempt exposed a controller-absolute output path; builders
now write inside the harness attempt. Original run evidence was not edited.
