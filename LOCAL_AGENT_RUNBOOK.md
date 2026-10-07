# Local Agent Runbook — 4D baseline/native qualification

## Scope and current truth

This runbook is the Local Codex entry point for round `20261006-baseline-qualification`. It covers acceptance of the committed control/scoring software, recovery and verification of one original development asset, construction of the three-arm baseline unit, native-scoring replay, integrity validation, and evidence return.

Latest accepted setup evidence is
`rounds/20261006-baseline-qualification/windows/20261006T2316Z-baseline-qualification/REVIEW_PACKET.md`.
It records 127/127 software checks, a captured native runtime, restored original
development asset, and completed native/world-Gaussian/body-Gaussian prediction
arms plus a validated scoring request. **No official score or GPU scorer replay
ran.** Do not repeat accepted setup merely because older sections below describe
its original execution order; first validate the returned hashes against the
current checkout and continue from the first open prerequisite.

There are 20 mathematical constructions and 15 conditionally selected candidates,
but **0/15 candidate implementations have a complete validation design and 0/15
have native results**. C02 has a corrected development operator and software
tests only; it is not admitted here as a new-method arm.

Read these files at the delivered commit before acting:

- `AGENTS.md`
- `rounds/20261006-baseline-qualification/WEB_HANDOFF.md`
- `docs/research-math-20261006/CURRENT.json`
- `docs/research-math-20261006/STEPWISE_PROGRESS.md`
- `docs/research-math-20261006/BASELINE_CONTROLS.md`
- `docs/research-math-20261006/BASELINE_SCORING.md`
- `docs/research-math-20261006/SCORING_OUTPUT_INTEGRITY.md`
- `docs/research-math-20261006/NATIVE_RUNTIME_CAPTURE.md`
- `docs/research-math-20261006/ACTIONBENCH_SCORER_PARITY.md`
- `docs/research-math-20261006/revisions/20261006-mechanism-boundaries/selection.json`

## Roles and unresolved host facts

| Item | Authoritative status |
|---|---|
| Controller | Local Codex on the user's computer, with the authenticated Git checkout and configured SSH access |
| Compute | Separate Linux GPU host; ordinary Conda processes; no remote Codex/GPT required |
| Repository checkout | Resolve an absolute fresh path locally; do not reuse a dirty or running tree |
| GPU-host project path | Last returned as `/root/rivermind-data/actionmesh-repro`; verify the current clean checkout and exact delivered commit before use |
| SSH alias/endpoint | Unknown. Reuse the user's already configured authorized target; do not guess or extract credentials |
| Installed skill directory | Unknown. Locate the complete current installed `research-autopilot` package and record its absolute path |
| Conda env/interpreter | Last returned as `/root/rivermind-data/actionmesh-repro/inference-env/bin/python`, a Conda-backed venv; revalidate captured package/dependency hashes; no Docker |
| GPU | Last observed RTX 2080 Ti, UUID `GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740`, driver 580.119.02, 22,528 MiB total; recheck identity/free VRAM/processes before launch |
| ActionBench revision | `2796071cbe6248422fcbeab3101fa9f9886cb7b9` |
| Original sample and GT paths | Returned UID `000-048_45e57349f062416aaf11f2c31587da16`; GT remains host-only at SHA-256 `25881f0d7a9f41578f77ba6be70f810ddcbcf4236a6834713ea197ea2916823e` |
| Time budget | 28,800 s hard window, 1,800 s collection reserve; current complete three-arm unit time is unmeasured |

The packet's host/device/free-space values are observations from that completed
setup window, not current telemetry. Recheck them before GPU work. If the
development asset, GT, evaluator source, or compatible environment no longer
matches the returned hashes, stop before scoring and return the exact mismatch.

## Ordered execution

Use shell variables only after resolving their literal values. Never substitute a guessed path.

### 1. Establish the delivered revision

In the authenticated local checkout, fetch `origin/main`, read the delivery receipt, and set `delivered_sha` to its exact commit. Create a fresh worktree or clone at that commit. Record:

```bash
git -C "$project_dir" rev-parse HEAD
git -C "$project_dir" status --short
```

The first command must equal `delivered_sha`; the second must be empty before execution. Read every file listed above from this checkout. If `main` advanced, do not silently substitute its head.

### 2. Resolve the installed harness and execution host

On the user's computer, identify the existing SSH target. On the GPU host, locate the complete installed `research-autopilot` directory, the project checkout, the compatible Conda interpreter, and the ActionBench checkout at the required revision. Record absolute paths and SHA-256 hashes of the harness and relevant source files.

Before approving any plan, source-inspect:

- `$skill_dir/scripts/run_harness.py`
- the committed plan builder named below
- every executable command in the generated `harness.json`

Confirm GPU count, timeout, device visibility, output directories, stop behavior, and that all writes stay inside the intended project/attempt roots.

The committed `prepare_*.py` builders below may be invoked directly only to emit/check their plans. Every test, control construction, native evaluation, and integrity workload represented by those plans must be launched by `run_harness.py`; a builder that performs its own scientific workload is not acceptable.

### 3. Run the parser-only acceptance suite through the harness

On the GPU host, from a clean delivered checkout:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_control_scoring_checks.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --run-id control-scoring-integrity-software-001 \
  --plan-dir "$project_dir/plans/control-scoring-integrity-software-001"
```

Review the generated plan and capture the printed digest. Then execute only with that exact digest:

```bash
"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/control-scoring-integrity-software-001/harness.json" \
  --root "$project_dir" \
  --execute \
  --approved-plan-digest "$approved_plan_digest"
```

This is a CPU/software acceptance task (`gpu_count: 0`). The current source discovers the repository's `research_math` unit tests; record the observed test count and outcome instead of copying historical counts. A pass proves parser/orchestration behavior only, not native scientific qualification.

### 3a. Capture the actual installed native environment

Follow [the exact environment-capture commands](docs/research-math-20261006/NATIVE_RUNTIME_CAPTURE.md#ordered-local-commands). The committed `actionmesh/prepare_native_runtime.py` emits a CPU-only harness plan; its inner collector writes the actual interpreter, Conda/package inventory and hashed runtime JSON. Retain its attempt evidence and copy the completed `inputs/native-runtime/` directory back to this checkout without rewriting paths or replacing earlier captures.

This does not initialize CUDA or verify the physical device. Keep the current host GPU inventory separately; native protocol and official replay qualification remain pending. A capture alone never authorizes scoring.

### 4. Restore and verify one original development asset

Recover one complete original ActionBench development case and its official GT from the user's existing evidence store. This is controller file transfer, not a scientific workload: preserve the original files read-only, copy them into the clean delivered checkout, and then make every subsequent test/control/scoring workload use the harness.

Use an actual completed W0 unit under the verified protocol fingerprint `3c3fe5a7f9b0ec7b75ce32d2ab08a21bcd36cfc64ae7e9a39b88517d08072d0d`. Inspect that unit's `generation.receipt.json`; verify `status=complete`, the fingerprint, and its recorded SHA-256/byte count for the generation `report.json` and `sequence.npz`. Verify the same UID's `surfaces.npy` against `protocol.json` `generation.data_records` and ActionBench revision `2796071cbe6248422fcbeab3101fa9f9886cb7b9`.

Copy the verified bytes to these exact project-relative locations:

```text
inputs/original-case/report.json
inputs/original-case/sequence.npz
inputs/gt/<actual-uid>/surfaces.npy
inputs/original-provenance/generation.receipt.json
inputs/original-provenance/protocol.json
```

After transfer, recompute all hashes and compare them to the original receipt/protocol before using a builder. Record both source and destination paths/hashes in the return packet. The committed `scripts/research_evidence_20261006/export_feedback.py` is a full historical evidence exporter, not required merely to copy these three admitted bytes and not to be run raw as a substitute for this verification. If the original receipt-bound bytes are absent, return `blocked_missing_original_asset_bytes` with the inspected host paths; do not use the small checkpoint archive, a fixture, or regenerated data.

### 5. Build the strong simple controls

After the original case is staged and verified, create a plan with:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_mesh_controls.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --source-sequence "$project_dir/inputs/original-case/sequence.npz" \
  --run-id baseline-controls-001 \
  --plan-dir "$project_dir/plans/baseline-controls-001" \
  --sigma 1.0 \
  --wall-seconds 600
```

Source-inspect the generated commands and run the plan with `run_harness.py` plus its exact printed digest. Locate the completed attempt from the harness/native receipts rather than guessing its UUID. The attempt workspace's `actionmesh/control-output/` must contain three paired arms for the same sample UID and frame set:

| Arm | Definition | Purpose |
|---|---|---|
| `native` | original prediction, copied byte-for-byte | original baseline |
| `world_gaussian` | deterministic world-coordinate Gaussian perturbation | equally cheap non-body-aware perturbation |
| `body_gaussian` | deterministic canonical/body-coordinate Gaussian perturbation | cheap body-aware comparator |

Required frozen control parameters are inference seed inherited from the source report (`42` for the retained W0 case), scorer seed `44`, sigma `1.0`, all 16 frames, and identical topology/metadata. Validate hashes and arm manifests. Then use controller file transfer to copy that immutable completed `control-output/` into `$project_dir/inputs/three-arm-case/`, recheck every manifest hash, and record the source attempt path. Do not copy a partial or failed attempt.

### 6. Freeze native scoring inputs and prepare the request

First generate the request only; this does not run the scorer:

```bash
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.control_scoring request \
  --root "$project_dir" \
  --source-case "$project_dir/inputs/original-case" \
  --controls-dir "$project_dir/inputs/three-arm-case" \
  --ground-truth "$project_dir/inputs/gt/$actual_uid/surfaces.npy" \
  --repo-root "$project_dir/actionmesh/repo" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --output "$project_dir/plans/control-scoring-request-001.json"
```

The request builder verifies the source report, released population membership, exact native-byte arm, three arm reports/sequences, topology/timeline/anchor, GT UID, and current scorer/adapter hashes. Retain its printed request digest.

Next create the source-backed frozen qualification protocol described in
`BASELINE_SCORING.md`. The runtime JSON is already returned but must still match
the executing interpreter and current GPU. Freeze and hash at minimum:
ActionBench revision, sample UID/split, three arm manifests, evaluator
module/function and source closure, `n_pts_chamfer=100000`, `n_pts_icp=10000`,
rotation count `24`, ICP iterations `200`, scorer seed `44`, device policy,
framework/library versions, dependency locks, and the one physical GPU UUID.
Do not approve a request with unknown or mismatched fields. Qualification rules
must be prospective and source-backed; do not derive a permissive threshold from
this development asset.

### 6a. Establish independent official-versus-faithful parity

Before changing the native contract to `faithful_harness`, follow
[`ACTIONBENCH_SCORER_PARITY.md`](docs/research-math-20261006/ACTIONBENCH_SCORER_PARITY.md).
The first frozen protocol revision must use the exact official scorer descriptor.
Build the parity plan only after that full protocol passes the installed native
verifier, then execute the plan only through `run_harness.py`.

The parity unit invokes the official ActionBench dataset CLI and the existing
faithful wrapper separately for each of the same three prediction arms. It
retains GLBs, source copies, patch identity, raw official CSV/summary, raw
faithful JSON and logs. All three official metrics require zero absolute
tolerance. A passed sidecar permits a new frozen protocol revision using the
faithful descriptor plus `verification_ref`; it does not qualify any baseline,
control or candidate and does not replace trusted nonce-bound replay.

Create the admitted scientific plan with the actual resolved values:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_control_scoring.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --request "$project_dir/plans/control-scoring-request-001.json" \
  --protocol "$project_dir/plans/$protocol_file" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --run-id native-controls-development-001 \
  --plan-dir "$project_dir/plans/native-controls-development-001" \
  --group "$frozen_group" \
  --gpu-uuid "$gpu_uuid" \
  --wall-seconds "$reviewed_wall_seconds" \
  --ram-mib "$admitted_ram_mib" \
  --cpu-cores "$admitted_cpu_cores"
```

Review `native.json` and `harness.json`, capture the printed digest, and execute that exact harness plan. Plan generation or request generation is not a score.

### 7. Native replay and strict output binding

The admitted plan already uses `python -m research_math.control_scoring score` as its inner executor. That command starts a fresh `research_census_eval.py` process for each of two passes over each arm; the wrapper in turn calls the pinned official ActionBench source. Do not launch either command outside the admitted plan and do not replace it with another scorer.

If `prepare_control_scoring.py` rejects the protocol/runtime, return `blocked_missing_source_backed_native_protocol` or `blocked_native_runtime_mismatch` with the exact validation error. A completed developmental scoring plan still leaves `native_contract_qualified=false`; trusted official/harness parity and nonce-bound replay remain separate Local acceptance evidence, as specified in `BASELINE_SCORING.md`.

The admitted scoring protocol must reference the passed parity sidecar before
using the faithful harness. For every arm and pass, retain raw score output plus:

- sample UID and split;
- protocol and device manifest hashes;
- pass manifest;
- evaluator entrypoint, source closure, and hashes;
- mesh, GT, report, stdout, and stderr hashes;
- command, exit code, wall time, peak VRAM, and device samples.

Run the committed strict validator in `actionmesh/research_math/control_scoring.py`. A score is eligible only if all binding checks pass and official-source identity is verified. Any disagreement between frozen request, native output, and recomputed validator is a failure, not a value to average.

### 8. Qualification decision and natural-failure analysis

Apply the prospective rule in `BASELINE_SCORING.md`. Report paired arm values and deltas on the one development asset; do not generalize to a dataset-wide conclusion. Natural failure analysis must distinguish coordinate-frame effects from generic smoothing/perturbation and name the observed applicability boundary.

This round may establish baseline/native-scoring qualification. It may not claim a candidate success or increment the 0/15 candidate result count.

### 9. Monitor and collect within the hard window

For every executed harness plan, use its source-inspected status interface and retain the live record/status files. Stop launching scientific work when 1,800 seconds remain. Collection includes final status, attempts, logs, runtime/device telemetry, output hashes, protocol/runtime manifests, raw scores, integrity report, and `git rev-parse HEAD`/dirty patch.

Do not use a historical two-arm timing estimate. The current cached scoring plan explicitly marks its cost ineligible for the full queue. First measure a complete current multi-arm experimental unit including generation or source restoration, control/candidate preparation, scoring, integrity replay, and collection. Only then may a later round derive a queue that fits `28,800 - 1,800 = 27,000` seconds.

## Debug table

| Failure | Inspect first | Evidence to retain | Correct response |
|---|---|---|---|
| Environment/import error | attempt `stderr`, Conda package list, ActionBench revision, `actionmesh/requirements-inference.txt` | failing command, environment export, source hashes | repair the compatible native env; do not introduce Docker |
| OOM/device mismatch | `execution-context.json`, `device-samples.jsonl`, harness `record.json` | UUID, driver, total/free/peak VRAM, competing processes | reconcile the current device and rerun the same frozen plan; do not silently change science parameters |
| Shape/metadata failure | `check_arm_arrays`, `simple_mesh_controls.py`, arm manifests | frame/topology/metadata hashes and full traceback | correct the earliest producer/staging defect and regenerate all affected arms |
| Native score mismatch | frozen request, raw native output, evaluator source closure, strict integrity report | all six stdout/stderr streams and per-pass hashes | reject the affected score; resolve source/protocol/output identity before replay |
| Mixed or weak baseline effect | paired raw metrics and the development asset | arm-level values/deltas, visuals if available, selection provenance | report the natural failure boundary; do not tune on held-out data |
| SSH/session loss | harness state/lock/record and remote process table | last status, PID/PGID, attempt timestamps | recover through the controller; never start a duplicate attempt blindly |
| Window exhaustion | harness timing/status and collection ledger | elapsed and remaining budget | stop launching at the reserve boundary and collect evidence |
| Concurrent Git update | current `origin/main`, expected parent, local patch | old/new head and conflict paths | re-read current state, reconcile additively, and push without force |

## Required return packet

Create `rounds/20261006-baseline-qualification/windows/<actual-window-id>/REVIEW_PACKET.md` only after an actual run. It must link or enumerate:

1. delivered commit, execution commit, `git status`, and any patch;
2. resolved host/path/environment/device facts;
3. every plan, approved digest, attempt, command, exit code, stdout/stderr, and status record;
4. complete current three-arm unit wall time and peak VRAM;
5. dataset/sample/GT/evaluator/protocol/runtime identities and hashes;
6. raw native scores, strict integrity report, paired deltas, and the qualification decision;
7. failures, repairs, reruns, evidence limitations, and next executable step.

Push the source-safe review packet and receipts to `main` with expected-head protection, read back the commit, and return the exact GitHub path and commit. Do not commit credentials or large/private raw assets; preserve their immutable host paths and hashes instead.

## Candidate ledger (outside this round)

Conditional selection order is `c02, c01, c10, c13, c14, c04, c03, c20, c11, c12, c15, c05, c08, c06, c07`. C02 has a development operator; C14 has comparator/control code. Neither is a completed candidate validation. Every candidate still requires an implementation/design card, simple baseline, ablation, frozen native protocol, full-unit timing, reproducible result, and evidence-based conclusion after the baseline gate closes.
