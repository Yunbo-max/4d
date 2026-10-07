# Local Agent Runbook — 4D baseline/native qualification

## Scope and current truth

This runbook is the Local Codex entry point for round `20261006-baseline-qualification`. It covers acceptance of the committed control/scoring software, recovery and verification of one original development asset, construction of the three-arm baseline unit, native-scoring replay, integrity validation, and evidence return.

Latest execution evidence is
`docs/research-math-20261006/longgoal-20261007/README.md`. It records 201/201
software checks at the returned revision, exact-zero official/faithful parity
for all nine one-UID/three-arm metric comparisons under the explicit GPU-forward
plus upstream CPU-backward compatibility policy, and successful CPU
finalization. This is engineering scorer-equivalence evidence only: no native
scientific contract was qualified and no candidate ran. This revision adds new
unit-manifest checks, so rerun current-revision software acceptance, but do not
repeat the already finalized parity unless one of its bound source/input hashes
has changed. Continue from snapshot/data staging after verifying the retained
evidence hashes.

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
- `docs/research-math-20261006/actionbench-qualification-source-audit.json`
- `docs/research-math-20261006/actionbench-full128-snapshot-contract.json`
- `docs/research-math-20261006/actionbench-full128-dataset-semantics-contract.json`
- `docs/research-math-20261006/actionbench-current-release-unit-contract.json`
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
The generated native plan must list `actionmesh/finalize_actionbench_parity.py`
in `code_refs`; the parity rejection tests import that controller finalizer. If
it is absent, reject the plan rather than treating an earlier 127-test run as
acceptance of the current revision.
The returned 201/201 suite predates the unit-manifest checks added here. Run the
current suite once; this does not invalidate or require repetition of the
separately finalized scorer parity because this change does not modify its bound
implementation or inputs.

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

### 6a. Establish independent official-versus-faithful parity

Before changing the native contract to `faithful_harness`, follow
[`ACTIONBENCH_SCORER_PARITY.md`](docs/research-math-20261006/ACTIONBENCH_SCORER_PARITY.md).
The official release has no per-asset numerical performance threshold, so do
not copy its 128-object aggregate means into a one-asset scientific protocol. A
`metric >= 0` domain-validity rule is also forbidden because the installed
evaluator would treat it as baseline/control qualification.

Parity itself is now separated as an engineering implementation-equivalence task:
the builder emits a hash-pinned one-UID sample manifest and a scorer-equivalence
contract with no scientific qualification fields, then emits a
`purpose=engineering` native plan with no `protocol_ref`. Source-inspect those
two generated inputs and both plans, then execute only through `run_harness.py`.
This may establish exact scorer implementation equivalence; it cannot qualify a
baseline, control or scientific score.

The parity unit invokes the official ActionBench dataset CLI and the existing
faithful wrapper separately for each of the same three prediction arms. It
retains and verifies GLBs, source copies, patch identity, exact subprocess
commands, input hashes, raw official CSV/summary, raw faithful JSON and logs.
All three official metrics require zero absolute tolerance. A passed sidecar is
limited to its exact sample manifest and permits a later frozen protocol using
the faithful descriptor plus `verification_ref` only when the scientific manifest
matches. That later protocol still requires prospective source-backed performance
qualification rules and installed-verifier acceptance. A different scientific
manifest requires matching parity evidence.
Parity does not qualify any baseline, control or candidate and does not replace
trusted nonce-bound replay.

Run this engineering parity step before attempting to author the scientific
qualification protocol. After a pass, promote and hash the **complete** output
bundle to the exact project-relative target specified by the attestation; copying
a summary alone is invalid because its record, raw-output and prediction refs must
remain resolvable. Then follow the CPU-only finalization plan in
`ACTIONBENCH_SCORER_PARITY.md`: use
`actionmesh/prepare_actionbench_parity_finalization.py`, inspect its pinned input
closure, and execute its exact digest through `run_harness.py`. Do not invoke the
finalizer directly. Promote the receipt-bound
`faithful-harness-verification.json` from that completed attempt into the already
promoted bundle and verify its hash. Only that post-promotion sidecar may be
referenced by a later scientific protocol. Finalization must fail if the promoted
attestation or any declared output is absent from the completed parity receipts.

### 6b. Freeze the scientific qualification protocol

Only after parity bundle promotion, create the source-backed frozen qualification
protocol described in `BASELINE_SCORING.md`. The runtime JSON is already returned
but must still match the executing interpreter and current GPU. Freeze and hash
at minimum: ActionBench revision, sample UID/split, three arm manifests,
evaluator module/function and source closure, `n_pts_chamfer=100000`,
`n_pts_icp=10000`, rotation count `24`, ICP iterations `200`, scorer seed `44`,
device policy, framework/library versions, dependency locks, and the one physical
GPU UUID. Do not approve a request with unknown or mismatched fields.
Qualification rules must be prospective and source-backed; do not derive a
permissive threshold from this development asset. If no such rule is available,
record `blocked_missing_source_backed_native_protocol` after completing parity;
do not treat that scientific blocker as a reason to skip engineering parity.

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

### 6b. Conditional full-population reproduction path

Read `docs/research-math-20261006/actionbench-full128-reproduction-contract.json`
and both its source review and
`docs/research-math-20261006/actionbench-full128-generation-source-audit.json`
before preparing any dataset-wide replacement for the
blocked one-UID scientific protocol. The contract now fixes the previously
missing prospective agreement rule: require all 128 released UIDs, 128 successes,
zero failures, and all three official aggregate means within the half-open
three-decimal intervals represented by the current official README row. This is
full finite-population reproduction, so it claims no sampling confidence interval.

Do not dispatch it yet. It intentionally has no run-plan builder because the
complete GT/prediction manifests, official ActionMesh generation source and
weights, current runtime/device qualification, full-unit timing, and trusted
official replay are not bound. In particular, seed `42` is the ActionMesh
generation seed; the official scorer sampling seed remains `44`. The paper-v2
values are a different versioned target and may not be silently substituted.
The nine new unit checks for the fail-closed contract helper are part of the
current whole-suite acceptance and are unexecuted until Local runs step 3.

The bounded generation audit resolves the **current public release candidates**:
ActionMesh Git `d5c01f5045df55819e337369c9617f603c667e00`, TripoSG submodule
`fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`, ActionBench dataset
`2796071cbe6248422fcbeab3101fa9f9886cb7b9`, and the four current Hub
revisions listed in the audit. Do not relabel those current revisions as the
unpublished leaderboard revisions. The official tree has no 128-UID generation
driver or prediction bundle, and `snapshot_download` is unpinned. The published
row supplies seed `42` and the non-fast variant but not code/weight revisions,
dtype, low-RAM choice, hardware, dependency lock, retry policy, or output
manifest. Therefore an admitted future attempt must be named a versioned
**current-public-release reproduction**, unless new primary evidence binds the
historical row.

Do not launch even one full-population generation unit until the four Hub
snapshots and dataset are materialized at immutable revisions with complete
file/hash manifests. Inspect the real GPU first. The official README states a
32 GB default and 12 GB `--low_ram` requirement and gives no RTX 2080 Ti result;
neither `--low_ram` nor `--dtype float16` is source-bound to the published row.
Record an OOM or unsupported dtype as the observed natural failure. Do not
silently change `--fast`, dtype, low-RAM mode, configuration, population, or
metrics to make it fit. After provenance staging, measure one complete non-fast,
seed-42, 16-frame generation/export/scoring unit and its peak VRAM. Only that
current complete-unit receipt can price a later 28,800-second plan with the
1,800-second collection reserve.

After confirming disk capacity and before inference, pin the current public Hub
snapshots rather than allowing the upstream entrypoint to resolve moving `main`:

```bash
hf download facebook/actionbench --type dataset \
  --revision 2796071cbe6248422fcbeab3101fa9f9886cb7b9 \
  --local-dir "$project_dir/inputs/actionbench-2796071c"
hf download facebook/ActionMesh \
  --revision fb69228ba8a4df684907b5d259cff3c22fb722f1 \
  --local-dir "$actionmesh_source/pretrained_weights/ActionMesh"
hf download VAST-AI/TripoSG \
  --revision 2c1c516d22d58db486a058d98d31bb6177344e06 \
  --local-dir "$actionmesh_source/pretrained_weights/TripoSG"
hf download facebook/dinov2-large \
  --revision 47b73eefe95e8d44ec3623f8890bd894b6ea2d6c \
  --local-dir "$actionmesh_source/pretrained_weights/dinov2"
hf download briaai/RMBG-1.4 \
  --revision 2ceba5a5efaec153162aedea169f76caf9b46cf8 \
  --local-dir "$actionmesh_source/pretrained_weights/RMBG"
```

Here `actionmesh_source` must be the detached official checkout at
`d5c01f5045df55819e337369c9617f603c667e00`, with the TripoSG submodule at
`fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`. Preserve each CLI receipt and a
recursive regular-file manifest with size and SHA-256. Reject an HTML response,
LFS pointer, empty/nonmatching directory, moving branch, or submodule mismatch.
These acquisition commands do not authorize inference; they only stage the
candidate current-release bytes for a later harness-owned plan.

After all five `hf download` commands finish, admit the exact local bytes with
the committed CPU-only plan. This is the only approved snapshot-manifest path;
do not invoke `research_math.snapshot_admission` directly:

```bash
snapshot_run_id=actionbench-full128-snapshot-admission-001
"$python_bin" "$project_dir/actionmesh/prepare_actionbench_snapshots.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --contract "$project_dir/docs/research-math-20261006/actionbench-full128-snapshot-contract.json" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --source-root "$actionmesh_source" \
  --dataset-root "$project_dir/inputs/actionbench-2796071c" \
  --actionmesh-root "$actionmesh_source/pretrained_weights/ActionMesh" \
  --triposg-root "$actionmesh_source/pretrained_weights/TripoSG" \
  --dinov2-root "$actionmesh_source/pretrained_weights/dinov2" \
  --rmbg-root "$actionmesh_source/pretrained_weights/RMBG" \
  --run-id "$snapshot_run_id" \
  --wall-seconds 1800 \
  --ram-mib 2048 \
  --cpu-cores 1 \
  --plan-dir "$project_dir/plans/$snapshot_run_id"

"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/$snapshot_run_id/harness.json" \
  --root "$project_dir" \
  --execute \
  --approved-plan-digest "$approved_plan_digest"
```

Before execution, inspect both emitted plans and confirm the command contains the
five resolved absolute roots above, `gpu_count: 0`, exactly one attempt, and no
network/download/inference/scorer command. The admission rejects a dirty or wrong
official Git tree/submodule; missing or wrong per-file Hugging Face revision
metadata; unresolved LFS pointers; HTML responses; links/special files; empty
snapshots; changed-while-hashing files; any UID difference; and any UID without
exactly `camera.json`, `surfaces.npy`, and frames `00.png` through `15.png`.

The successful attempt output is
`inputs/actionbench-full128-snapshots/admission.json` inside that attempt's
isolated workspace and is located by the native receipt's `output_refs`. Preserve
the receipt, logs, plan digests and output SHA-256. Promote it to the same
project-relative target only if that target is absent; if it already exists,
compare hashes and stop on any difference rather than overwriting it. Return the
actual file counts, byte totals and manifest digests for all five snapshots.
`admitted_engineering_snapshot` proves only local byte/revision/file closure. It
does not prove model loading, runtime compatibility, inference, scoring,
leaderboard identity, a complete prediction inventory, or dispatch readiness.

After promoting the successful snapshot admission, run the separate byte-bound
dataset-semantics pass. This pass must consume that exact admission; it is not a
replacement for snapshot admission and may not read a mutable or differently
downloaded dataset root:

```bash
semantics_run_id=actionbench-full128-dataset-semantics-001
"$python_bin" "$project_dir/actionmesh/prepare_actionbench_dataset_semantics.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --contract "$project_dir/docs/research-math-20261006/actionbench-full128-dataset-semantics-contract.json" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --snapshot-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-root "$project_dir/inputs/actionbench-2796071c" \
  --run-id "$semantics_run_id" \
  --wall-seconds 3600 \
  --ram-mib 2048 \
  --cpu-cores 1 \
  --plan-dir "$project_dir/plans/$semantics_run_id"

"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/$semantics_run_id/harness.json" \
  --root "$project_dir" \
  --execute \
  --approved-plan-digest "$approved_plan_digest"
```

Before execution, inspect both plans and confirm one CPU-only attempt, no network,
model, inference or scorer command, and an output path of
`inputs/actionbench-full128-snapshots/dataset-semantics.json`. The runner rechecks
the admission-bound size/SHA-256 of every consumed `camera.json`, `surfaces.npy`
and PNG, rejects parent links and mutation, then checks all 128 tensors for
shape `(16,100000,6)`, floating finite values and normalized position bounds. It
also checks the released four-key camera schema/proper rotation/positive focal
lengths and validates all 2,048 PNGs as square RGBA files with complete chunk
CRCs and compressed streams, requiring frame encoding consistency within each
sample while recording any cross-sample variants. Preserve the attempt and output hashes; promote the output
only into an absent target and stop on any existing-content difference.

`admitted_engineering_dataset_semantics` closes only released-input parsing and
structure. The public statement that point indices are tracked through time is
not independently proved by this structural pass, and no normal-length or alpha-
occupancy threshold was invented. Report its actual wall time, per-UID semantic
digest and any failing UID/path. It remains zero-GPU, zero-score and cannot
authorize generation, a scientific protocol or queue sizing.

After promoting both admissions, freeze the exact first calibration unit before
any current-release inference. This separate CPU pass prospectively selects the
first canonical released UID, revalidates its 18 admitted files, rechecks the
official clean Git revision/tree, hashes the required generation/scorer source
closure, and binds all four admitted model-manifest digests:

```bash
unit_manifest_run_id=actionbench-current-release-unit-manifest-001
"$python_bin" "$project_dir/actionmesh/prepare_actionbench_unit_manifest.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --contract "$project_dir/docs/research-math-20261006/actionbench-current-release-unit-contract.json" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --snapshot-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --source-root "$actionmesh_source" \
  --dataset-root "$project_dir/inputs/actionbench-2796071c" \
  --run-id "$unit_manifest_run_id" \
  --wall-seconds 900 \
  --ram-mib 1024 \
  --cpu-cores 1 \
  --plan-dir "$project_dir/plans/$unit_manifest_run_id"

"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/$unit_manifest_run_id/harness.json" \
  --root "$project_dir" \
  --execute \
  --approved-plan-digest "$approved_plan_digest"
```

Promote only the receipt-bound output to
`inputs/actionbench-full128-snapshots/unit-manifest.json`. Its UID must be
`000-000_03b69da8d2c94b5999bcf2605ee2ecd9`; generation remains seed 42 and
scoring remains seed 44. The manifest freezes a later boundary containing
non-fast/non-low-RAM bfloat16 generation, native/world-Gaussian/body-Gaussian
arms, all three official score rows, integrity collection, wall time and peak
VRAM. It emits no GPU plan. OOM or unsupported bfloat16 must be preserved; do
not silently enable low-RAM, float16 or fast mode.

After the three promoted admissions and a fresh runtime capture match the active
interpreter/GPU UUID, generate (but do not yet execute) the complete-unit plan:

```bash
complete_unit_run_id=actionbench-default-complete-unit-001
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.complete_unit_plan \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --contract "$project_dir/docs/research-math-20261006/actionbench-current-release-unit-contract.json" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --snapshot-contract "$project_dir/docs/research-math-20261006/actionbench-full128-snapshot-contract.json" \
  --snapshot-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --unit-manifest "$project_dir/inputs/actionbench-full128-snapshots/unit-manifest.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --source-root "$actionmesh_source" \
  --dataset-root "$project_dir/inputs/actionbench-2796071c" \
  --weights-root "$weights_root" \
  --run-id "$complete_unit_run_id" \
  --gpu-uuid "$gpu_uuid" \
  --wall-seconds 27000 \
  --plan-dir "$project_dir/plans/$complete_unit_run_id"

"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/$complete_unit_run_id/harness.json" \
  --root "$project_dir" \
  --execute \
  --approved-plan-digest "$approved_plan_digest"
```

Before approving the digest, inspect that the plan has one exclusive physical-
GPU task, a 27,000-second workload limit inside a 28,800-second window, no retry,
and the default non-fast/non-low-RAM BF16 argv. Its native receipt must enumerate
the full deterministic success closure: generator logs/arrays/16 GLBs and the
PyTorch3D `grid_normal.mp4` preview, all three
control sequences/reports, 48 official exported GLBs, three raw CSV/summary/
backend records, copied official scorer sources, integrity manifests and host/
device telemetry. A plan declaring only `result.json` is obsolete and must be
rejected. Preserve any partial failure rather than manufacturing the absent
success files.

The first frozen default attempt returned a retained natural OOM at Stage I with
an observed 20,673 MiB sampled peak. Its separately reviewed repair is
`complete-lowram-r7`: non-fast FP16 with the official low-RAM path, still seed 42,
all 16 frames and all three score arms. The terminal harness completed in
1,330.47 seconds with all three official rows, 117 receipt outputs, 121 runner
pre-result files and a sampled 10,255 MiB GPU peak. Do not start another GPU
attempt; proceed through the CPU-only admission below.

After—and only after—`complete-lowram-r7` returns `completed`, build the
CPU-only evidence-admission plan. The prospective contract is already frozen at
the running plan digest, so a different run, UID, profile or edited output cannot
be substituted:

```bash
source_unit_run_id=complete-lowram-r7
admission_run_id=admit-complete-lowram-r7
unit_plan_digest=a94aa69f8605266587f56f0977740001bbd21971463bb5b78d4e20eaf2bf574b

"$python_bin" "$project_dir/actionmesh/prepare_complete_unit_admission.py" \
  --root "$project_dir" \
  --contract "$project_dir/docs/research-math-20261006/actionbench-complete-unit-admission-contract.json" \
  --harness-plan "$project_dir/plans/$source_unit_run_id/harness.json" \
  --harness-report "$project_dir/runs/harness/$source_unit_run_id/report.json" \
  --native-plan "$project_dir/plans/$source_unit_run_id/native.json" \
  --native-receipt "$project_dir/runs/attempts/$source_unit_run_id/receipt.json" \
  --approved-unit-plan-digest "$unit_plan_digest" \
  --output "$project_dir/inputs/complete-unit-admissions/complete-lowram-r7.json" \
  --skill-dir "$skill_dir" \
  --plan-dir "$project_dir/plans/$admission_run_id" \
  --run-id "$admission_run_id" \
  --wall-seconds 900 \
  --ram-mib 4096 \
  --cpu-cores 2

"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/$admission_run_id/harness.json" \
  --root "$project_dir" \
  --execute \
  --approved-plan-digest "$admission_plan_digest"
```

Before executing, inspect that this second plan has `gpu_count: 0`, one attempt,
no retry and only one output sidecar. A pass must bind the canonical source
harness/native plan, state, report, task, receipt and attempt; exact 117 receipt
files; exact 121 nested pre-`result.json` files (116 declared files plus five
retained CPython 3.12 evaluator caches); four completed stages; three
official score arms; and error-free one-second host/device samples. Keep the raw
source unit immutable on the GPU host. The sidecar permits later queue pricing
for this exact FP16 low-RAM engineering unit only; it is not a baseline/native
scientific qualification and does not itself generate the queue.

The one-UID engineering parity/finalization chain is now complete and retained at
`00b30fd`; do not rerun it merely because the downstream manifest is new.
Inventory these full-population prerequisites without launching generation or scoring. Return
their exact paths/revisions/hashes, snapshot and semantic admissions, and measured
storage/semantic-pass requirements plus the unit manifest. The complete-unit
default runner has a retained OOM and the explicit FP16 low-RAM repair completed
generation before entering official scoring. Only a terminal successful receipt
followed by the CPU-only complete-unit admission can price the eight-hour queue.
One-UID parity timing or any CPU admission timing cannot.

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
