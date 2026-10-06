# Local Agent Runbook — 4D baseline/native qualification

## Scope and current truth

This runbook is the Local Codex entry point for round `20261006-baseline-qualification`. It covers acceptance of the committed control/scoring software, recovery and verification of one original development asset, construction of the three-arm baseline unit, native-scoring replay, integrity validation, and evidence return.

Current state is `generated_unexecuted`. There are 20 mathematical constructions and 15 conditionally selected candidates, but **0/15 candidate implementations have a complete validation design and 0/15 have native results**. C02 has a corrected development operator and software tests only; it is not admitted here as a new-method arm.

Read these files at the delivered commit before acting:

- `AGENTS.md`
- `rounds/20261006-baseline-qualification/WEB_HANDOFF.md`
- `docs/research-math-20261006/CURRENT.json`
- `docs/research-math-20261006/STEPWISE_PROGRESS.md`
- `docs/research-math-20261006/BASELINE_CONTROLS.md`
- `docs/research-math-20261006/BASELINE_SCORING.md`
- `docs/research-math-20261006/SCORING_OUTPUT_INTEGRITY.md`
- `docs/research-math-20261006/revisions/20261006-mechanism-boundaries/selection.json`

## Roles and unresolved host facts

| Item | Authoritative status |
|---|---|
| Controller | Local Codex on the user's computer, with the authenticated Git checkout and configured SSH access |
| Compute | Separate Linux GPU host; ordinary Conda processes; no remote Codex/GPT required |
| Repository checkout | Resolve an absolute fresh path locally; do not reuse a dirty or running tree |
| GPU-host project path | Unknown. `/root/rivermind-data/actionmesh-repro` is historical evidence only and must be verified before use |
| SSH alias/endpoint | Unknown. Reuse the user's already configured authorized target; do not guess or extract credentials |
| Installed skill directory | Unknown. Locate the complete current installed `research-autopilot` package and record its absolute path |
| Conda env/interpreter | Unknown. Inspect and reuse a compatible existing environment; no Docker |
| GPU | User-authorized target is one RTX 2080 Ti; record current UUID, driver, total/free VRAM, and competing processes before launch |
| ActionBench revision | `2796071cbe6248422fcbeab3101fa9f9886cb7b9` |
| Original sample and GT paths | Pending Local restoration/verification from the original evidence store; do not invent |
| Time budget | 28,800 s hard window, 1,800 s collection reserve; current complete three-arm unit time is unmeasured |

These unknowns are explicit Local resolution steps. If the original development asset, GT, official evaluator source, or compatible environment cannot be restored, stop before scoring and return the exact missing item and inspected locations.

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

### 4. Restore and verify one original development asset

Recover one complete original ActionBench development case and its official GT from the user's existing evidence store. Verify the ActionBench source revision, sample UID, all 16 frame meshes, cameras, SMPL parameters, masks, GT file, and source hashes. Preserve the original outputs read-only.

The earlier exporter documentation contains historical paths and is not itself an admitted executable plan. Do not run it raw. If there is no committed harness plan that can stage the required asset without mutating the source, return `blocked_missing_harnessed_asset_stage` with the inspected paths. Otherwise, run the source-inspected staging plan through `run_harness.py` and retain its complete attempt evidence.

### 5. Build the strong simple controls

After the original case is staged and verified, create a plan with:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_mesh_controls.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --run-id baseline-controls-001 \
  --input-dir "$original_case_dir" \
  --plan-dir "$project_dir/plans/baseline-controls-001"
```

Source-inspect the generated commands and run the plan with `run_harness.py` plus its exact printed digest. The output must contain three paired arms for the same sample UID and frame set:

| Arm | Definition | Purpose |
|---|---|---|
| `native` | original prediction, copied byte-for-byte | original baseline |
| `world_gaussian` | deterministic world-coordinate Gaussian perturbation | equally cheap non-body-aware perturbation |
| `body_gaussian` | deterministic canonical/body-coordinate Gaussian perturbation | cheap body-aware comparator |

Required frozen control parameters are seed `44`, sigma `1.0`, all 16 frames, and identical topology/metadata. Validate hashes and arm manifests before scoring.

### 6. Freeze native scoring inputs and prepare the request

Create the scoring input/check plan:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_control_scoring.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --run-id baseline-native-score-001 \
  --case-dir "$three_arm_case_dir" \
  --gt-path "$gt_path" \
  --protocol-json "$protocol_json" \
  --runtime-json "$runtime_json" \
  --evaluator-source "$evaluator_source" \
  --plan-dir "$project_dir/plans/baseline-native-score-001"
```

The protocol/runtime files must be created from source inspection, not placeholders. Freeze and hash at minimum: ActionBench revision, sample UID/split, three arm manifests, evaluator module/function and source closure, `n_pts=100000`, ICP subsample `10000`, rotation count `24`, ICP iterations `200`, seed `44`, device policy, framework/library versions, and the one physical GPU identity. Do not approve a request with unknown or mismatched fields.

Run the generated plan through `run_harness.py`. It must emit a reviewer-visible dry-run report/request and integrity checks; it must not be counted as a score.

### 7. Native replay and strict output binding

Use the official/native ActionBench scoring entrypoint identified by source inspection. Execute all three arms under the same frozen protocol/device manifest. If a separate native runner is not yet represented by an admitted harness plan, stop and return `blocked_missing_harnessed_native_runner`; do not replace it with a surrogate scorer.

For every arm and pass, retain raw score output plus:

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

Do not use a historical two-arm timing estimate. First measure the complete current three-arm unit, including scoring and integrity collection. Only then may a later round derive a queue that fits `28,800 - 1,800 = 27,000` seconds.

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
