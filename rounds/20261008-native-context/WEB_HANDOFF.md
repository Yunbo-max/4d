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
