# Stepwise continuation — 2026-10-06

## Evidence checked

Starting repository revision: `28ff4bde041cefab55f058f3c921edb304dc4f5c`.
The newly returned C02 implementation and supervised execution records are
developmental. The native diagnostic used 24 queries and frames 1/8/15;
it is not a complete ActionBench comparison. Neither it nor CPU tests verifies
candidate quality, Natural Gate 0, IPCG, or a final scientific outcome.

## Completed engineering correction

`project_protected_step` explicitly solves
`min .5 (p-d)^T W (p-d)` subject to `C p=0` and zero pinned coordinates.
Eliminating pinned variables gives free unconstrained center
`d_f + solve(W_ff, W_fp d_p)`. The prior implementation omitted the second term.
This affects dense metrics coupling pinned and free coordinates; diagonal
metrics and calls without pins retain their previous mathematical behavior.
The existing pseudoinverse constraint projection is applied to this corrected
center. The full metric objective is now stated in the public function contract.

A regression test first failed on the old code: `d=(1,0)`,
`W=((2,1),(1,2))`, `p_0=0` returned `(0,0)` instead of `(0,.5)`.
A second analytic test combines cross terms with an action constraint.
All 47 tests in the repository's `research_math` suite passed locally with NumPy.
These tests are engineering/algebra checks, not a benchmark or native replay.
The new revision has not been executed on the GPU host.

## Next work in order

1. Qualify the full native baseline and simple controls using released
   ActionBench assets and the existing official scorer, preserving full
   16-frame sampling, timestamps, topology, anchor and native scoring budgets.
   Reuse historical caches only after their source/input/output bindings verify.
   Existing exposed assets are development assets, not fresh confirmation.
2. Implement and measure ordinary temporal smoothing and body-frame smoothing
   as baseline qualification controls. Freeze their parameters before inspecting
   their scores. Retain failures and complete raw outputs, not only summary flags.
   Compare C02 with ordinary geometry correction and an equal-norm scalar step;
   observed directional protection must be distinguished from step shrinkage.
3. Run one complete comparison unit as a timing/resource pilot on the actual
   single 2080 Ti. Keep 1,800 seconds of the 28,800-second window for collection.
   Admit subsequent complete units only using measured multi-arm costs.
   Historical two-arm timing and the 22-second decoder diagnostic cannot price
   the new full comparison. Do not promise all 15 methods within eight hours.
4. Establish current native Natural Gate 0 and concurrent IPCG before new
   scientific candidate implementation or Gate A freezing. Baseline reproduction,
   native evaluation qualification and natural failure census may precede them.
5. Finish the selected 15 implementation/protocol pairs against their mathematical
   assumptions and required simple comparators. Then run native comparisons,
   confirmation and full result verification. Record scoped success, failure or
   inconclusive outcomes per method; never infer effectiveness from unit tests.

Current status: 20 conditional mathematical constructions and 15 conditional
specifications exist; C02 has a developmental operator implementation. Fully
verified candidate designs, complete new native method comparisons, and formal
candidate success/failure outcomes remain zero. This checkpoint does not change
the mathematical batch's gate fields or claim experiment generation is finished.

## Continuation: ordinary mesh controls — 2026-10-06

Resumed unchanged remote main `87b6ddb760c882ce893627bcfbc0a64d18f91b03`
and CURRENT revision mechanism-boundaries-r2; no additional GPU feedback was
present. No existing dirty checkout was reset or overwritten.

Implemented ordinary world-coordinate Gaussian and classical body-frame
Procrustes Gaussian as simple comparator adapters. They preserve all 16 original
frames/timestamps, vertex identity, shared faces and the exact first-frame anchor;
they receive no GT/camera/scorer transform. The native arm is a byte copy.
Failed pose fits remain in the three-arm manifest and prevent completion.
The CPU-only plan builder pins source sequence/report and code, stages actual
files through run_harness, and exports complete meshes rather than sparse probes.

Independent review identified ambiguous reflection-corrected pose fits; a failing
regression established the issue before repair. Integer-index admission and
nonfinite world-output rejection were also repaired with failing regressions.
The final whole research_math suite passes **66 checks**, software evidence only.
A labelled engineering staging integration exported all nine expected artifacts
without running any native benchmark/scorer. Its scope and source identities
are retained in the engineering evidence; it provides no scientific qualification.

See [BASELINE_CONTROLS.md](BASELINE_CONTROLS.md) for algorithms, actual local
preparation/harness commands, native-score obligations and return contents.
No numerical candidate criteria, full science protocol, native score or measured
multi-arm GPU timing was certified. The mathematical batch and CURRENT pointers
remain unchanged. Verified candidate designs remain **0/15**; complete new-method
native comparisons and formal scoped successes/failures remain **0**.

Next executable work: prepare verified cached native sequences with these simple
adapters on the original host; bind the full scorer/input/GT/runtime qualification
protocol, measure one full multi-arm unit, and analyze all development failures
before candidate admission. Meanwhile continue remaining baseline/source/protocol
preparation within scope. The lack of a connected GPU does not turn draft method
specifications into completed experiment generation.

## Continuation: three-arm native scoring handoff — 2026-10-06

Resumed main `d1e1ab887d73d9e79d76aa31a7363eee003bcbbb`, CURRENT mechanism-boundaries-r2 and latest feedback. No new GPU results appeared. Preserved current mathematical pool/selection and all prior evidence. This round follows the owner's Web/Local split: new code is **generated_unexecuted**, with Local acceptance pending. The previous 66 engineering passes remain historical and do not certify this revision.

Generated three-arm scoring request/runner source retaining full native budgets, current source/report/code/GT bindings, metadata/topology/anchor and failed arms. Each official census scorer pass uses a fresh process. All six pass records initialize first; repeated-score disagreement is inconclusive. Pinned source evidence is copied before scoring to survive forced interruption; GT stays referenced. Physical-device memory samples are observed samples, not an exact peak. Cached scoring excludes generation/control preparation and cannot price the eight-hour full queue.

Generated a scientific baseline-only outer-harness builder requiring source-backed frozen native protocol, exact faithful scorer/metrics/sampling/budget/all-arm bindings, real Local environment/dependency lock and physical GPU UUID. Generated a CPU whole-suite acceptance plan and **33 new engineering checks, none executed this round**. Independent source review found protocol-binding, dependency-closure and failure-preservation omissions plus a missing import; these were repaired in source. Final narrow review found no remaining material source issue; Local acceptance is still pending.

Entry: [BASELINE_SCORING.md](BASELINE_SCORING.md); [checkpoint](workflow-checkpoint-baseline-scoring.json); [source review](baseline-scoring-source-review.json). Actual request/preparation/collection code and precise Local prerequisites are delivered. This is not a qualified frozen native protocol or a ready eight-hour candidate queue. No native threshold, qualification PASS, measurement or successful scorer replay was invented.

Candidate implementation admission still awaits current Natural Gate0/IPCG and source-specific prerequisites. Verified candidate designs **0/15**, complete new-candidate native comparisons **0**, formal scoped successes/failures **0**. Next: Local software acceptance, original-evidence recovery, source-backed native protocol/runtime qualification, then full three-arm scoring/trusted official replay. Further Web source/protocol preparation remains useful without a connected GPU.

## Continuation: native output binding repair — 2026-10-06

Resumed exact main `5d05e9c9cb652d7110fc1a909f6dc634350a3820`, CURRENT and retained W0 feedback. No newer execution evidence appeared. Preserved mathematical pool/selection, historical source reviews/receipts and the original baseline-scoring handoff.

Source inspection found that the scorer subprocess output was checked only for case ID/denominator and exit status before metric extraction; its UID, exact native protocol and returned input/source hashes were not bound. Generated a dedicated parser/admission check and connected it before successful metrics enter readouts. Error rows retain the declared asset and native error; mismatches retain the raw output/logs as error passes. Official metric computation and frozen budgets are unchanged.

Authored 18 additional parser-only engineering checks, taking this test module from 33 to 51 checks. **No checks were executed**; historical 66 passes do not accept this revision. Independent source review found no critical/important integration blocker and identified two minor issues: canonical symlink paths and missing parser boundary checks. Both were addressed in source; review and current code refs are recorded in the new checkpoint. See [SCORING_OUTPUT_INTEGRITY.md](SCORING_OUTPUT_INTEGRITY.md) for Local acceptance, request regeneration and return obligations.

Status remains generated_unexecuted. Verified candidate designs **0/15**, new-candidate native comparisons **0**, formal scientific verdicts **0**. This closes a source-level acceptance gap, not native qualification, Natural Gate0/IPCG, full multi-arm timing or the eight-hour queue. Next: Local whole-suite acceptance followed by the already-required source-backed native baseline qualification/replay; independent Web source preparation continues when useful.

## Local Codex execution handoff — 2026-10-06T20:49:19Z

Material delivery: added root `AGENTS.md`, root `LOCAL_AGENT_RUNBOOK.md`, and immutable `rounds/20261006-baseline-qualification/WEB_HANDOFF.md`. The handoff fixes the controller/compute split, exact read order, harness-only project execution, clean exact-commit checkout, three-arm baseline work units, frozen ActionBench revision and scorer parameters, strict per-pass provenance, 28,800-second hard window with 1,800-second collection reserve, causal debug table, and actual-run return packet.

Host-specific facts that the Web agent cannot verify are explicit Local resolution steps rather than invented values: SSH target, current GPU checkout, installed complete `research-autopilot` path, compatible Conda interpreter, original development asset/GT locations, device UUID/driver/VRAM, official evaluator source closure, and the native-runner harness plan. Missing harnessed asset staging or native running is now a named stop condition; a surrogate scorer is forbidden.

Local document validation checked presence, JSON validity, hashes, and required honest-state/budget/blocker tokens. No repository test, GPU workload, or official scorer was executed in this Web round. Scientific counters therefore remain unchanged: 20 constructions, 15 conditional selections, 0/15 complete candidate validation designs, 0/15 native candidate results, and baseline/native qualification pending.

Next executable step: Local Codex resolves the exact delivered commit and environment, source-inspects the plan and harness, then runs `actionmesh/prepare_control_scoring_checks.py` through the installed `run_harness.py` using the printed approved-plan digest. Its outcome is engineering acceptance only and must be returned with attempt logs and hashes before native replay.

## Local runbook source correction — 2026-10-06T21:57:21Z

Source inspection at commit `3cd9a2d3aac63032b9ce642f45da17766c2cfef0` found two execution-blocking documentation defects. The previous runbook invoked `prepare_mesh_controls.py` with nonexistent `--input-dir` and omitted required `--source-sequence`/`--sigma`; it also invoked `prepare_control_scoring.py` with nonexistent case/GT/protocol/runtime/evaluator flags instead of first creating a request and then passing `--request`, `--protocol`, `--environment`, group, GPU and resource fields. Both command sequences are now matched to the current CLIs.

The source review also established that a harnessed native runner is not missing: `prepare_control_scoring.py` emits the scientific harness plan, whose inner executor is `python -m research_math.control_scoring score`; that executor launches two fresh pinned census/official scorer processes for each of the three arms. The false runner blocker was removed. The actual pending prerequisites are receipt-bound original source/GT bytes, a source-backed frozen native qualification protocol, a matching runtime/device manifest, and trusted official/harness replay evidence.

The restoration step now uses the already-allowed controller file-transfer boundary: Local verifies the original W0 generation receipt and protocol, copies only its bound `report.json`, `sequence.npz`, and matching ActionBench `surfaces.npy` into the clean checkout, and rechecks hashes before any harnessed workload. Missing original bytes is recorded as `blocked_missing_original_asset_bytes`; fixtures, the small checkpoint archive and regenerated data are forbidden substitutes.

Corrected source/test/exporter paths and immutable blob identities are recorded in `local-runbook-source-review.json`. JSON/schema-token checks and command/source review were performed; Web did not execute repository tests, GPU work or native scoring. Scientific state remains unchanged: baseline qualification pending, candidate validation designs 0/15, native candidate results 0/15. Next executable step remains Local parser-only acceptance, followed by receipt-bound asset restoration and the corrected three-arm control plan.

## Installed native environment capture — 2026-10-06T22:53:19Z

Resumed main `e1f3f8f4fbe3a05c0b2b5b175c5d5258bc7c7dd0` and its current ledger/handoff. No newer execution return was present. Added `actionmesh/research_math/native_runtime.py` and CPU-only `actionmesh/prepare_native_runtime.py` to produce the exact installed interpreter, five dependency versions, sanitized Conda/package inventory and hashed runtime JSON required by the existing scientific plan. The builder emits a foreground admitted metadata task; no CUDA, model or native scorer is launched by preparation.

The output paths are stable relative to the isolated attempt root and final project checkout. The Local handoff gives exact generation/execution and receipt-bound copy steps. Existing captures are preserved; required dependency conflicts, non-Conda/malformed metadata, output escapes and invalid device labels fail rather than creating a ready runtime. This captures installed identities, not a solver lock, verified device identity or evaluator compatibility.

Eight engineering checks were authored and included in the whole-suite source closure. Static source syntax and command/consumer/staging review were performed; **no software check, generated collector, GPU work or native scorer was executed**. Source self-review is recorded in `native-runtime-source-review.json`; no independent execution acceptance is claimed.

Candidate complete designs and native results remain **0/15**. Next: Local whole-suite acceptance, actual environment capture, original receipt-bound source/GT restoration and source-backed native protocol/trusted replay. Full multi-arm timing and the eight-hour candidate queue remain pending.
