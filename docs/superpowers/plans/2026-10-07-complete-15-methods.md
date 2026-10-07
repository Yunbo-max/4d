# Complete the 15-method research run

> **For agentic workers:** Use `superpowers:executing-plans` to execute this plan in the current session. The user has explicitly authorized implementation, execution, debugging, supervision, and result delivery; no new plan approval is required.

**Goal:** Finish evaluator qualification and execute all 15 selected methods with their declared comparators and ablations, preserving every outcome.

**Architecture:** Reuse the committed ActionMesh implementations, official ActionBench scorer, and research-autopilot harness. Repair the earliest failing component and version any scientific configuration change before collecting new results. A persistent task ledger separates implementation, engineering checks, evaluator qualification, and measured method outcomes.

**Tech Stack:** Python, NumPy, PyTorch/PyTorch3D, Conda-backed inference environment, SSH, Git, research-autopilot.

**Spec:** `AGENTS.md`, `LOCAL_AGENT_RUNBOOK.md`, `docs/research-math-20261006/revisions/20261006-mechanism-boundaries/selection.json`, and the user's 2026-10-07 instruction to finish both outstanding groups and debug while running.

## Global constraints

- One authorized RTX 2080 Ti at the existing SSH target. Inspect live allocation before dispatch.
- Official evaluation: 16 frames, 100000 surface points, 10000 ICP points, 24 initial rotations, 200 ICP iterations, scorer seed 44. Generation seed 42 is distinct.
- Native/Conda, no Docker. Reuse current cached assets; pin revisions and hashes for additional downloads and mirrors.
- Each execution window is at most 28800 seconds, reserving 1800 for collection; the long goal continues across windows.
- Every new scientific workload and engineering check runs through the committed builder and `run_harness.py` with a reviewed exact digest.
- Keep existing failures. Do not relax exact replay tolerance to convert observed failures into passes.
- No gate, parity sidecar, candidate result, or efficacy claim may be fabricated. One development UID does not establish population efficacy.
- Preserve raw GT, existing outputs, and shared Git metadata. Only remove redundant or confirmed archived artifacts.

## Review focus

- Repeated GPU computations can differ while source/data hashes agree: isolate sampling, ICP and CUDA reduction boundaries before altering execution.
- Official GLB loading may change mesh ordering: compare actual loaded geometry/topology, not only export checks.
- Harness test workspaces contain copied Python files: generated `/runs/` must not be mistaken for uncommitted project source.
- Sparse worktrees must include root rules and ignore files; their parent Git metadata must remain available.
- A complete source/model snapshot, a software pass, and a successful scorer invocation are different milestones.

## Tasks

### 1. Restore current execution and preserve evidence

- [x] Fetch and pull delivered `8d2f1e77fe2bd5c80bddd6f21b0ee080d1778eec`, preserving user files and previous six-pass results.
- [x] Create isolated local and remote source worktrees, with root files included.
- [x] Run current software acceptance through the harness: 179 tests passed, run `software-8d2f1e7-001`.
- [ ] Commit the `/runs/` generated-workspace ignore rule; regenerate and inspect the parity plan.
- [ ] Run independent official/faithful three-arm parity; save complete failed or passed output.

### 2. Diagnose and verify score reproducibility

Files: `actionmesh/research_math/actionbench_parity.py`, `actionmesh/official_actionbench_adapter.py`, `actionmesh/research_census_eval.py`, and their existing tests.

- [ ] Compare exact input/source/runtime/GLB hashes and all actual process commands from the completed parity run.
- [ ] Identify the earliest differing operation using isolated engineering instrumentation; record full tensors by hashes where sufficient.
- [ ] Before a fix, retain a failing regression showing the causal mismatch. Do not manufacture benchmark examples.
- [ ] Implement only the evidenced correction or explicitly version a required execution-policy change; retain unchanged official metric definitions and budgets.
- [ ] Repeat affected software checks and independent native parity through reviewed harness plans.
- [ ] Promote the full passed bundle and run `prepare_actionbench_parity_finalization.py` through its CPU harness.

### 3. Complete qualified baseline and data/runtime prerequisites

- [ ] Inventory all five roots specified by `actionbench-full128-snapshot-contract.json`, including live disk usage and immutable revision metadata.
- [ ] Acquire missing released bytes within the storage budget, checking mirrors against pinned source identities.
- [ ] Run the committed snapshot admission; preserve exact missing-file failures and repair them.
- [ ] Freeze a current-public-release generation configuration supported by the actual hardware; do not mislabel it as the unpublished leaderboard environment.
- [ ] Measure a complete generation/export/official-score unit; author the remaining source-backed native protocol and replay integration.
- [ ] Execute full baseline reproduction and formal qualification, retaining negative reproduction outcomes.

### 4. Complete each of the 15 selected methods

Execution order and exact IDs remain the committed selection. Each ledger row must receive all of the following; no count increments from unit tests alone:

- [ ] Read that row's current math card, native interface, information-access constraints, comparator and falsifier.
- [ ] Verify Natural Gate 0/IPCG and implementation/design admission from actual evidence; complete missing evidence instead of substituting flags.
- [ ] Implement the declared operator, simplest comparator and decisive ablation with matching inputs and budgets.
- [ ] Run meaningful regression/algebra checks through a CPU harness.
- [ ] Freeze sample inventory, seeds, tuning/confirmation split, native metrics and prospective decision rules before inspecting new scores.
- [ ] Run complete native comparisons, preserve every failure, verify output bindings, and assign a scoped evidence-based outcome.
- [ ] Record costs and adjust the next window's queue from measured complete units.

Rows: C02, C01, C10, C13, C14, C04, C03, C20, C11, C12, C15, C05, C08, C06, C07.

### 5. Supervise and deliver

- [ ] Check resource state and actual attempt logs during active work; report progress approximately hourly.
- [ ] Reconcile failed/interrupted attempts before retrying, and retain different attempts under different IDs.
- [ ] Persist progress in `docs/research-math-20261006/longgoal-20261007/STATUS.json` and per-window review packets.
- [ ] Push safe code, manifests, raw metric/log evidence and summaries with expected-head protection; read back the commit and synchronize the user's local checkout.
- [ ] Mark the long goal complete only when both evaluator obligations and all 15 requested native method experiments have actual final evidence. Do not count a blocked/unimplemented row as executed.
