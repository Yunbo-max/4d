# 4D research execution rules

These rules apply to the entire repository. The Web research agent authors and reviews committed artifacts; Local Codex controls the user's authenticated checkout and the separate Linux GPU host. The GPU host runs ordinary Conda processes and does not require a remote Codex or GPT session.

## Read first

Current authoring handoff: `rounds/20261008-native-context/WEB_HANDOFF.md`.
GPU STOP remains in force. This source delivery does not authorize any new GPU
run, enable an existing campaign, or install a running supervisor. Read the
[asset acquisition section](LOCAL_AGENT_RUNBOOK.md#download-datasets-and-models)
and the new handoff before accepting the changed software. Keep source authored,
Local software acceptance, native instrumentation replay and scientific method
evidence as separate states.
The paired instrument's plan-only entry is `actionmesh/prepare_native_context.py`.
Its complete raw archive, manifest and result must all be receipt-bound. On a
failure/timeout retain the actual partial attempt tree; never reconstruct success
or retry to fill a missing bundle. See the current handoff's continuation section
and the maintained `longgoal-20261007/CANDIDATE_INPUT_AUDIT.json` delivery inventory.
The C13-required quadratic strong control has a separate CPU-only plan entry at
`actionmesh/prepare_quadratic_acceleration_control.py`. It is not the C13
group-trend method and must not be reported as candidate admission or a native
result. Its current source and tests are `generated_unexecuted`; use the common
software-acceptance plan before its single-use real-sequence harness plan.
Completed paired-context evidence is consumed only through the CPU-only
`actionmesh/prepare_native_context_consumption.py` plan. It must pin and rehash
the result, manifest and tar as three explicit staged paths, bind the expected
producer UID/GPU/generation identity/source-time mode, enforce frozen archive, metadata and expansion
ceilings, validate every regular archive member, and extract from an immutable
private snapshot into a new single-use workspace. A consumed bundle remains unqualified transport
evidence and must not bypass Gate 0, candidate admission or GPU STOP.

Before setup, acceptance, execution, repair, collection, or delivery, read at the exact delivered commit:

1. `LOCAL_AGENT_RUNBOOK.md`
2. `rounds/20261006-baseline-qualification/WEB_HANDOFF.md`
3. `docs/research-math-20261006/CURRENT.json`
4. `docs/research-math-20261006/STEPWISE_PROGRESS.md`
5. `rounds/20261006-baseline-qualification/windows/20261006T2316Z-baseline-qualification/REVIEW_PACKET.md`
6. `docs/research-math-20261006/ACTIONBENCH_SCORER_PARITY.md`
7. `docs/research-math-20261006/actionbench-qualification-source-audit.json`
8. `docs/research-math-20261006/actionbench-full128-reproduction-contract.json`
9. `docs/research-math-20261006/actionbench-full128-reproduction-source-review.json`
10. `docs/research-math-20261006/actionbench-full128-generation-source-audit.json`
11. `docs/research-math-20261006/actionbench-full128-snapshot-contract.json`
12. `docs/research-math-20261006/actionbench-full128-dataset-semantics-contract.json`
13. `docs/research-math-20261006/actionbench-current-release-unit-contract.json`
14. `docs/research-math-20261006/actionbench-complete-unit-admission-contract.json`
15. `docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json`
16. `docs/research-math-20261006/actionbench-full128-active-batch-reconciliation-contract.json`

Do not infer current status from older receipts. Mathematical construction, a generated request, a static receipt, and a software test are not native scientific qualification.

## Safety and provenance

- Preserve dirty and running worktrees. Do not force-push, reset, or overwrite another update.
- Use a fresh checkout/worktree at the exact delivered commit. Record `git rev-parse HEAD`, `git status --short`, and any applied patch.
- The target is one RTX 2080 Ti, native/Conda, without Docker. Inspect the actual device and environment before launch; historical telemetry is not current telemetry.
- Keep controller Git/SSH/file-transfer commands separate from executable project tasks.
- Run every test, experiment, evaluation, or other scientific workload through the installed `research-autopilot` `scripts/run_harness.py`, using the committed plan builder and its printed approved-plan digest. A plan builder may be invoked directly only to emit and validate the plan; it must not perform the workload itself.
- Do not dispatch a candidate merely because it is mathematically selected. Candidate execution requires baseline/native-scoring qualification, Natural Gate 0, IPCG, code/design verification, a simple baseline, and an ablation.
- Do not convert missing results into zeroes or successes. Record `generated_unexecuted`, `insufficient_evidence`, or the observed failure.
- Keep ActionMesh generation seed `42` distinct from the pinned official evaluator sampling seed `44`. The conditional full-128 reproduction contract is not a one-UID threshold, a candidate protocol, or a dispatch-ready queue.
- The current public ActionMesh/Hugging Face revisions are pin candidates for a new current-release reproduction, not proof of the unpublished leaderboard generation environment. Do not call them an exact published-run replay. The public entrypoint defaults to seed `44`; the target generation row requires an explicit `--seed 42`.
- A successful full-128 snapshot or dataset-semantics admission is engineering input evidence only. It does not prove tracked correspondences, model loading, inference, scorer qualification, runtime compatibility, or a candidate effect.

## Runtime and evidence contract

- The hard window is 28,800 seconds with 1,800 seconds reserved for collection. Measure a complete current multi-arm unit before budgeting; do not reuse old two-arm or sparse-diagnostic timing.
- Source-inspect the harness, plan, and execution code before approval. Confirm exact commands, write targets, timeout behavior, GPU count, device policy, and evidence outputs.
- Retain the plan, attempts, stdout/stderr, status snapshots, runtime/device samples, output hashes, native evaluator sources, requests, reports, and raw per-pass score outputs.
- Before using a completed calibration unit for queue pricing, run the committed
  CPU-only complete-unit admission plan. It must rehash the exact 117 receipt
  outputs, the runner's 121-file pre-result inventory (116 declared outputs plus
  five exact retained evaluator caches), canonical harness/native
  records, all three score rows, and resource telemetry. A completed runner
  summary or receipt alone is insufficient.
- Queue pricing is a separate CPU-only admitted pass. It applies the frozen 5/4
  headroom to the admitted complete-unit elapsed time and partitions all 128 UIDs
  without selection. Its output is not an executable queue and must retain
  `queue_approved=false`, `queue_generated=false` and `dispatch_ready=false`.
- Before compiling a Full128 window, promote a hash-bound active-batch
  reconciliation at the canonical path. This compiler revision accepts exactly
  the complete retained `population-gpu-current-r9` handoff: the canonical
  harness digest must recompute, and every referenced native plan must resolve
  with its own SHA-256 and valid plan digest. Any intersecting completed,
  running, pending, or failed UID blocks plan generation and may not be retried
  or omitted. A later retained run requires a reviewed compiler revision.
- Bind each native score to the sample UID, protocol/device manifest, evaluator/source hashes, mesh/GT/report hashes, and pass manifest. Re-run the committed integrity validator before accepting a score.
- Debug from the earliest causal source or configuration and the attempt log. Do not patch only the final symptom.

## Return and delivery

- Fill the actual window ID and result paths in the round review packet; never invent future attempt paths.
- Push reviewable summaries, receipts, and source-safe logs to `main` with expected-head/concurrent-update protection. Do not commit private credentials or large raw assets.
- Read back the final commit and report the exact commit plus result locator. Until that readback exists, the scientific state remains unchanged.

