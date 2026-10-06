# Ordinary mesh-control preparation — 2026-10-06

Completed scope: two ordinary baseline adapters and a CPU harness-plan builder.
This is baseline qualification preparation, permitted before Natural Gate 0;
it is not selected-candidate implementation or a frozen candidate experiment.
The C14 specification already identifies classical pose-factored smoothing as
a required strong simple comparator. No originality claim is attached to it.

## Algorithms and information

All three arms consume exactly the same completed predicted 16-frame sequence.
The native arm is an exact byte copy. World Gaussian smooths displacements from
the anchor. Body Gaussian estimates a proper rigid least-squares fit from the
anchor to each frame with uniform vertex weights, smooths the remaining local
residuals, and reconstructs with the original fitted pose. Neither fit uses GT,
camera calibration, scorer ICP, input labels, model updates or external weights.

For row-vector coordinates, let X = Y_0 - mean(Y_0), and fit
Y_t approximately X R_t + c_t. Compute U_t = (Y_t-c_t) R_t^T - X,
smooth U across time, then reconstruct (X + smooth(U)_t) R_t + c_t.
The fit is restricted to SO(3), without scale. Frame zero is restored exactly.
The kernel is exp(-(t-s)^2/(2 sigma^2)), truncated at 4 sigma and normalized
within the original finite window; no periodic wrap, padding, resampling or
frame interpolation. Both smoothing arms use the same sigma.

An exact rigid sequence has zero body residual, so its motion is preserved by
construction up to floating-point error. This property does not establish
preservation of articulation or native quality. Mesh-driven pose fitting can
absorb articulation, while world smoothing can erase true rapid motion.
Collinear/low-rank covariance and reflection-corrected fits with a nonunique
smallest singular direction are explicit failed arms. Relative fit tolerance
is 1e-8. No silent fallback or failed-case exclusion is allowed.

## Software evidence

The existing 47 research_math checks plus 19 new algebra, admission and I/O
checks pass: **66 total**. Checks ran inside the installed research-autopilot
outer harness. Failing pre-implementation and review-regression logs are retained.
An additional CPU-only staging integration used a clearly labelled engineering
fixture, exported all nine declared files, and executed no native scorer.
See [engineering evidence](../../results/baseline-controls-20261006/engineering-evidence.json).
Constructed software fixtures have no benchmark status or scientific scores.

## Local preparation steps

Use an isolated checkout containing this commit and the full installed
research-autopilot skill. Preserve any existing dirty GPU worktree; stage a copy
of one previously completed source sequence and its original report under this
checkout's inputs directory. Verify the copied bytes against the original
generator receipt. Reusing these exposed assets is development only.
The source sequence must contain vertices, shared faces, original frame_indices,
timesteps 0..15, and query_vertex_ids; source/report hashes must agree.

In the existing native inference Conda environment, generate an artifact-only
plan. Replace the three path values with the actual checkout, installed skill
and staged source; do not guess a cache path or inference environment.

```bash
project_dir=/absolute/path/to/4d-checkout
skill_dir=/absolute/path/to/installed/research-autopilot
source_sequence="$project_dir/inputs/development-case/sequence.npz"
python "$project_dir/actionmesh/prepare_mesh_controls.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --source-sequence "$source_sequence" \
  --run-id ordinary-controls-development-001 \
  --plan-dir "$project_dir/plans/ordinary-controls-development-001" \
  --sigma 1 --wall-seconds 600
```

Sigma=1 frame is an explicit initial baseline setting, not a tuned or qualified
optimum. Freeze the development sweep/selection rule before inspecting control
scores if further bandwidths are needed. The preparation command prints the
exact plan digest without starting execution. Review its actual pinned inputs,
interpreter and finite CPU envelope, then use that digest:

```bash
python "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/ordinary-controls-development-001/harness.json" \
  --root "$project_dir" --execute --approved-plan-digest PRINTED_DIGEST
```

The run receives CPU-only resources, a 2 GiB RAM reservation and no GPU.
Single-use plan/output/run IDs prevent overwrites. On a shared host, use the
existing harness owner; a driver-lock conflict must be reconciled, not bypassed
by another pool. Collect the attempt's workspace/actionmesh/control-output
directory plus the complete receipt, logs and environment.json. It contains
native/world_gaussian/body_gaussian mesh sequences and reports, body poses,
a manifest retaining all three arms, and controls.json. Failed arms cause a
nonzero exit and remain in the manifest. Adapter timing excludes generation,
official scoring and collection; it cannot size the eight-hour queue.

## Native qualification still required

The manifest is compatible with research_census_eval.py's existing full mesh
interface. Its declared count of three is three arms of one asset, not three
independent assets. Pair results by UID/inference seed/arm; do not use the
evaluator's all-arms UID average as a treatment contrast or family statistic.

Before scientific dispatch, acquire and pin the real source/input/GT hashes,
runtime, development cohort and full qualification protocol. Dispatch the
existing scorer as an inner job of the reviewed outer harness: preserve all
16 frames, 100,000 surface points, 10,000 ICP points, 24 initial rotations,
200 ICP iterations, native anisotropic alignment, scoring seed 44 and all
three official metrics. Do not run the analytic metric-self-test as native
qualification or score these engineering fixtures. Retain every arm, failure,
original scorer output, log, telemetry and replay receipt.

The current run has no GPU connection or complete original native artifacts;
no official three-arm score or end-to-end resource measurement was produced.
Pending: live full native replay, strongest-comparator qualification, natural
failure analysis, Natural Gate 0/IPCG admission, numerical candidate criteria,
full per-candidate protocols and a measured eight-hour queue. Developmental
qualification can proceed once its real inputs and native execution are available;
candidate investment still requires the current scientific gates.

Return the full original evidence using the existing evidence exporter, plus
the new three-arm preparation receipt/artifacts and live official scoring output.
Start with one complete multi-arm timing/resource unit on the actual 2080 Ti;
reserve 1,800 of 28,800 seconds for collection. Never substitute historical
two-arm timing or these software-fixture timings for the new unit's cost.
