# 4D research execution rules

These rules apply to the entire repository. The Web research agent authors and reviews committed artifacts; Local Codex controls the user's authenticated checkout and the separate Linux GPU host. The GPU host runs ordinary Conda processes and does not require a remote Codex or GPT session.

## Read first

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

## Runtime and evidence contract

- The hard window is 28,800 seconds with 1,800 seconds reserved for collection. Measure a complete current multi-arm unit before budgeting; do not reuse old two-arm or sparse-diagnostic timing.
- Source-inspect the harness, plan, and execution code before approval. Confirm exact commands, write targets, timeout behavior, GPU count, device policy, and evidence outputs.
- Retain the plan, attempts, stdout/stderr, status snapshots, runtime/device samples, output hashes, native evaluator sources, requests, reports, and raw per-pass score outputs.
- Bind each native score to the sample UID, protocol/device manifest, evaluator/source hashes, mesh/GT/report hashes, and pass manifest. Re-run the committed integrity validator before accepting a score.
- Debug from the earliest causal source or configuration and the attempt log. Do not patch only the final symptom.

## Return and delivery

- Fill the actual window ID and result paths in the round review packet; never invent future attempt paths.
- Push reviewable summaries, receipts, and source-safe logs to `main` with expected-head/concurrent-update protection. Do not commit private credentials or large raw assets.
- Read back the final commit and report the exact commit plus result locator. Until that readback exists, the scientific state remains unchanged.
