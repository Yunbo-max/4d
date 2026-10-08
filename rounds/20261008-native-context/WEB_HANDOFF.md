# Native context and continuous handoff source delivery

Status: **generated_unexecuted**. GPU STOP remains effective. No installation,
project test, native generation, scorer, GPU task or supervisor process was
started by this authoring round. Resolve the delivered main commit from the
GitHub delivery receipt; the authoring base is
`1da40c748a71f9687bd2c5958ee6705671632d10`. Integration preserves the concurrent
`1bc3c973e0c6f23c94a32be7d30a06de4a812cd8` capture-only delivery.

## Concrete source changes

| Component | Source | Acceptance still required |
| --- | --- | --- |
| Actual Stage-II integration | `actionmesh/research_math/pipeline_decoder_observer.py` | Original method called exactly once; original input/output objects preserved; cleanup on failure; real lazy-loaded model path |
| Complete raw context | `actionmesh/research_math/decoder_observer.py` | Full latents/time/query/normal tensors, dtype/device/autocast/inference/gradient modes, identity and hashes |
| Full-context replay and separate source-time query | `pipeline_decoder_observer.py::replay_window` | All original target outputs and bounded mesh coordinates agree before requesting the source time; no C01 correction or scientific qualification |
| Paired official generation entry | `actionmesh/research_math/native_context_runner.py` | Exact frozen assets/profile; complete unobserved/observed 16-frame export; explicit tolerance; raw replay, failure evidence and total resource envelope |
| Finite batch supervision | `scripts/research_supervisor.py` | Exact harness identity, failure isolation, finite repair dependencies, STOP, immutable deadline and same-task recovery |

Read the [supervisor interface and manifest](../../docs/research-math-20261006/longgoal-20261007/SUPERVISOR_HANDOFF.md)
and [context capture rationale](../../docs/research-math-20261006/longgoal-20261007/NATIVE_CONTEXT_CAPTURE_PLAN.md).
The supervisor's source does not configure an online repair agent or transfer
results by itself. Unknown faults remain explicit when that service is absent.

The existing `complete_unit_plan --capture-decoder` path, its
`PipelineDecoderCapture` archive format and its 119-output three-arm receipt
remain available. The new `native_context_runner` is a separate, unpriced paired
instrument: two fresh official generations followed by raw-context replay,
without scoring. Its `PipelineDecoderObserver` records exact window time mapping
in a distinct replay format. Do not interchange their capture directories or
attach both wrappers to one pipeline. Both use the shared decoder hooks.

## Implementation sequence and source review scope

- Preserve the existing 20-candidate pool and 15 selected identities.
- Author engineering tests for original argument/output identity, multiple
  windows, callback refusal on replay, failure cleanup, tamper, replay mismatch,
  and unchanged full context in the separate source-time query.
- Implement the real Stage-II boundary, then a paired official-entry runner.
- Add supervision around `run_harness.py`, preserving inner retries and IDs.
- Source-review and publish the changes together. Local execution remains a
  separate pending acceptance step, not a claimed red/green test result.

Candidate C01/C02 advancement remains dependent on the existing Parent,
Natural Gate 0, IPCG and native input obligations recorded in the
[current selection](../../docs/research-math-20261006/revisions/20261006-mechanism-boundaries/selection.json).
This instrument supplies a real route to obtaining missing inputs; it does not
replace the legal protection operator for C02, sparse mode/transport interfaces,
natural-gap evidence, strong/simple controls or full G01 design for 15 methods.

## Local CPU software acceptance

Read `AGENTS.md` and `LOCAL_AGENT_RUNBOOK.md` at the delivered commit. Restore
the known Local controller → SSH target → Conda interpreter/skill paths; no
remote LLM installation is needed. Reuse valid [pinned assets and verification
steps](../../LOCAL_AGENT_RUNBOOK.md#download-datasets-and-models).

After staging this exact clean revision, use the existing acceptance builder:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_control_scoring_checks.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --run-id native-context-supervisor-acceptance-001 \
  --plan-dir "$project_dir/plans/native-context-supervisor-acceptance-001"
```

Read the generated source closure, then run that exact printed digest through
the existing harness under Local authority. It requests zero GPUs. Keep all
failures and report the actual observed tests; do not copy a historical count.
The source closure includes the new modules, test files, supervisor script and
the concurrent capture-only generation entry. The frozen test command binds
`RESEARCH_AUTOPILOT_SKILL_DIR` to the builder's selected complete skill path.
Software fixtures are not native benchmark or method-effect evidence.

## Native execution remains stopped

Do not call the new runner directly or enqueue GPU work from this document.
Its CLI is an inner harness command. After explicit GPU resumption, Local still
must freeze the instrumentation plan with current runtime/device, exact assets,
profile, tolerances, code closure, complete output inventory and a finite budget
for two full generations plus replay. The old single three-arm calibration
price cannot price this different instrument. A GPU plan/approval is not created
by this source handoff. This instrument does not alter or reuse the base
complete-unit 117-output contract or the separate 119-output capture unit.

The instrument is restricted to the current forward, source-first,
unsubsampled, direct-prediction path. Unsupported windows, residual mode,
callback-dependent replay, missing identities and storage overflow reject
explicitly. Original native coordinate clamping is retained separately from
raw decoder outputs; no correction/clipping is hidden inside a candidate.
The default 64 MiB per-window bound includes both stored input copies, raw output
and full geometry payload; it excludes safetensors headers and JSON. Resource
sampling and the separately frozen attempt envelope still bound the full run.

## Failure and return

On a new acceptance failure, retain the harness plan, attempt, full stdout and
stderr and exact source hashes. Inspect the named implementation boundary;
do not increase tolerances or remove required inputs to make a result agree.
On native mismatch, preserve both complete sequences and all raw replay files.
On uncertain supervisor dispatch, retain its same campaign/plan identity and
use the existing harness reconciliation; never reset it or invent a new run ID.

Return a single source/version-bound Local packet with commands actually run,
logs, failures, resource observations and the distinct software/native statuses.
All 15 candidate native outcomes remain pending. The existing hourly authoring
automation is separate from a running Local supervisor and does not prove that
any native task is active.

## Paired instrument plan and raw collection continuation

Source base: `ece1ce883ea106e7d2d30ef23a3712a81cba9e49`. Still
**generated_unexecuted**, with GPU STOP effective and no additional budget.
`actionmesh/prepare_native_context.py` now supplies the missing plan-only entry.
It accepts only the existing calibration-unit manifest, never a Full128 UID or
historical pricing plan. The old 117-/119-output contracts are unchanged.

`research_math.native_context_delivery` gives the paired plan three direct
receipt outputs: `actionmesh/context-output/result.json`, `raw-manifest.json`
and `raw-evidence.tar`. On successful comparison the runner requires all native
raw files, packages every regular raw output (including optional renders),
rehashes archive members and detects source-file changes during collection.
The raw tree includes two complete 16-frame generations, exact captured inputs
and outputs, topology/time mapping, replay, requests/logs/reports, environment/
dependency bytes, initial/final input verification and resource samples.
The result remains scientifically unqualified even when every comparison agrees.

A failure or hard timeout may leave only a partial raw tree. Retain it under the
actual harness attempt workspace. Missing archive is a collection gap on a
failed attempt, never permission to recreate a success or rerun the same unit.
Source-time bytes are required only for a successful instrument that requested
them; a paired/replay disagreement remains `comparison_mismatch`.

### Local acceptance and plan-only commands

Use the existing CPU acceptance builder, with the new source-specific identity:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_control_scoring_checks.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --run-id native-context-delivery-acceptance-001 \
  --plan-dir "$project_dir/plans/native-context-delivery-acceptance-001"

"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/native-context-delivery-acceptance-001/harness.json" \
  --root "$project_dir" --execute --approved-plan-digest "$approved_plan_digest"
```

Set `approved_plan_digest` to the builder's actual printed digest after reviewing
its CPU-only source closure. Record observed outcomes; Web ran none of these
checks. New tests cover the actual installed `make_plan` and `_stage` path,
input/dependency relocation, output closure, corrupt/missing/link evidence,
conditional source-time output and collection footprint. Fixtures qualify
software behavior only.

After current software acceptance, Local may prepare the instrument **without
executing it**. Restore absolute `source_root`, `dataset_root`, `weights_root`,
`gpu_uuid`, `project_dir`, `skill_dir` and `python_bin` from the runbook/actual
host. The environment must be the current canonical
`inputs/native-runtime/environment.json`, with its matching dependency inventory.
Before emission, set `instrument_wall_seconds`, `instrument_cpu_cores` and
`instrument_ram_mib` from a reviewed allocation within the remaining existing
budget. No paired-instrument timing measurement exists; do not derive this
allocation from the old 1,664-second single-unit price. The hard maximum is
27,000 seconds, with 1,800 seconds reserved for controller collection/return.
The two generations, replay, verification, raw tar writing and hashes all count
inside the instrument budget. Raw collection duplicates disk payload; retain
space for both the tree and tar in addition to existing assets.

```bash
"$python_bin" "$project_dir/actionmesh/prepare_native_context.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --plan-dir "$project_dir/plans/paired-native-context-001" \
  --run-id paired-native-context-001 \
  --contract "$project_dir/docs/research-math-20261006/actionbench-fp16-lowram-unit-contract-v1.json" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --snapshot-contract "$project_dir/docs/research-math-20261006/actionbench-full128-snapshot-contract.json" \
  --snapshot-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --unit-manifest "$project_dir/inputs/actionbench-full128-snapshots/unit-manifest.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --source-root "$source_root" --dataset-root "$dataset_root" --weights-root "$weights_root" \
  --gpu-uuid "$gpu_uuid" --instrument-wall-seconds "$instrument_wall_seconds" \
  --cpu-cores "$instrument_cpu_cores" --ram-mib "$instrument_ram_mib" \
  --atol 0 --rtol 0 --max-capture-bytes 67108864 --source-time-query
```

The unit-manifest path above must contain the exact retained calibration manifest
for the explicit FP16 child profile (R7 manifest SHA-256
`b551cbcac49b9c71f7be5becfa331745c03bb96feae758847858a1b723047ae6`); if it is absent, restore its receipt-bound
bytes to that absent target or retain the missing-input failure. Never substitute
a Full128 manifest or silently change precision. Zero tolerances prospectively
ask whether replay is exact; a mismatch is retained and does not permit
post-result tolerance relaxation. Source review of this command is not evidence
that a plan has been emitted. Do not execute or enqueue this GPU plan while STOP
remains in force. No `--execute` command for it is issued by this handoff.

The outer harness's `total_wall_seconds` equals the instrument budget;
`window_seconds` is that budget plus 1,800 and is reporting cadence, not a second
timeout enforcement mechanism. Supervisor campaign limits must independently
preserve the actual overall deadline/reserve. Sampled host telemetry stops before
raw packaging so its archived bytes remain immutable. `result.json` separately
retains before/after collection disk/RSS observations, explicitly not exact peaks.

### Delivery inventory and next authoring work

The maintained [15-item inventory](../../docs/research-math-20261006/longgoal-20261007/CANDIDATE_INPUT_AUDIT.json)
now records each exact selected ID, math/review reference, real source entry,
legal input/model interface, required output and all named controls/ablations,
shared official scorer, absent candidate acceptance command, completion evidence
and separate implementation/scientific gaps. No complete candidate method is
claimed: C02 is a dense projection primitive; C14 is a simple baseline; C01 has
a shared input instrument; the other 12 lack their complete method entries.
No command is invented for an unimplemented method.

Next Web work continues legal-input adapters and complete method/design
obligations where their scientific premises exist. Sparse modes, legal motion
surrogates and native gap/importance/collision evidence remain genuine missing
prerequisites for affected candidates. Current-source Local CPU acceptance is
pending independently. Full128 indices 10–15 remain an explicit coverage gap;
the calibration instrument neither fills that gap nor retries r9.

## C13 quadratic strong-control continuation

On exact source base `c41d291b50aa3d84ad495417724403132041ea0b`, a
read-only candidate/runtime review confirmed that none of the 15 candidates has
all Natural Gate 0/IPCG/native prerequisites. No candidate was promoted. The
review also identified C13's required quadratic acceleration comparator as an
independent, legal pre-admission deliverable and found an API-shape inconsistency
in native replay (`query[:3]` retained features instead of selecting XYZ).

Added `research_math.quadratic_acceleration_control` and its CPU-only plan
builder. The identity-metric control uses the supplied sequence timestamp units,
records the weight's fourth-power unit dependence, directly solves the anchored
quadratic objective and exports one complete 16-frame mesh arm with
unchanged topology/identity metadata, records source/code/output hashes and
numerical diagnostics, and emits a case manifest accepted by the existing
generic official adapter. It requires float32 vertices before export. It deliberately excludes GT, scorer state and learned
parameters. It is not C13's group-l2 candidate and does not close the natural
Gaussian/quadratic gap. The replay call now passes `query[..., :3]`; this is an
API-shape normalization with no numerical effect in the only supported upstream
direct mode, where `apply_displacement` ignores `vertex`. Its fixture rejects a
non-XYZ vertex tensor; Local replay still must verify equivalence.

Acceptance source was authored before implementation. Web performed static
AST/JSON/diff/source review only and did not run the tests, builder, control,
model, scorer or GPU. All changed source is `generated_unexecuted`. Local first
runs the current CPU software plan. It then freezes an explicit positive control
weight and first verifies a real retained receipt-bound source sequence using
the controller procedure in the runbook. The builder then hash-binds only the
staged sequence/report pair; it does not replace that upstream receipt check.
Its plan is one CPU attempt, zero retries and no GPU. Official scoring remains
separately gated and stopped. Candidate source
complete remains 0/15; candidate native results remain 0/15.
