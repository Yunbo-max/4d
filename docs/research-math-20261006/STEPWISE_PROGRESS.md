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

## Returned setup accepted; independent scorer parity authored — 2026-10-07

Resumed exact main `c6d20a0acd2a39be03d9351a86381051d39c94d3` and read the
new Local packet at
`rounds/20261006-baseline-qualification/windows/20261006T2316Z-baseline-qualification/REVIEW_PACKET.md`.
Its source-safe archive was independently hash-checked as
`06ccb3de0223a4578d2da2c3fdf493cd73d9c55f66ca6b1141d9acdebeadc6b6`.
The accepted setup facts are 127/127 engineering checks, the captured native
runtime, one receipt-bound original development UID, all three completed simple
control arms, and a strictly validated scoring request. No official score, GPU
scorer replay, native decision or candidate result ran. Those completed setup
steps are no longer scheduled for blind repetition.

Authored an independent ActionBench official path instead of comparing two
invocations of the same wrapper. A format-only adapter exports all 16 float32 NPZ
frames to the official GLB layout, proves exact vertex/face round-trip, applies
only the already-documented CPU RNG device compatibility change in an isolated
copy of the pinned sources, and starts official `evaluate_dataset.py` in a fresh
process. It retains every GLB, original and patched source hashes, exact patch,
command, CSV, summary and logs. A separate process invokes the existing faithful
wrapper on the identical manifest, GT, device, seed and budgets.

The harnessed parity runner covers native, world-Gaussian and body-Gaussian arms
and requires exact equality for `cd_3d`, `cd_4d` and `cd_motion`; zero tolerance
is frozen before results. It emits a native-evaluator-compatible faithful-harness
verification sidecar only after all nine comparisons and device telemetry pass.
Twelve parser/identity/rejection checks were added, but **none were executed by
Web**. All new source remains `generated_unexecuted` until Local acceptance.

The plan builder intentionally requires a complete source-backed native protocol
that already passes the installed verifier with the exact official scorer
descriptor. This prevents the parity unit from bypassing the still-missing
released sample/split definition and prospective baseline/control qualification
rules. After parity, Local must freeze a new protocol revision with the faithful
descriptor plus verification ref, then perform nonce-bound official replay.

This delivery closes an executable-source gap, not native qualification. It does
not increment the candidate ledger: complete candidate validation designs remain
**0/15**, native candidate results **0/15**, and formal scoped outcomes **0**.
The parity unit's six scorer invocations are not a complete candidate experiment;
its timing cannot price the 27,000-second launch budget. Next work is authentic
protocol/threshold freezing, Local software acceptance, then this parity plan.

## Continuation: source-bound parity admission — 2026-10-07

Resumed exact remote main `741be89e28cebc52ea84ac1aa345cd0ae383e217`;
no newer Local/GPU feedback was present. Official ActionBench dataset/API, pinned
README/evaluator source and ActionMesh paper v2 were rechecked. The release binds
128 assets, 16 frames and 100,000 tracked surface points per frame. For the
returned development UID, the released `surfaces.npy` LFS SHA-256 exactly matches
the retained GT hash `25881f...823e`.

The source review found that ActionBench publishes only full-128 aggregate
leaderboard means, not a per-asset qualification threshold. Applying those means
to the one development UID would be a scope error. The prior parity builder also
required a fully verified native protocol before it could create the
official-versus-faithful sidecar, producing a circular dependency while the
faithful protocol itself needs that sidecar.

Generated a narrower parity admission. The builder now writes and pins a
single-UID sample manifest plus a source/scorer contract binding the released
population, exact GT, official sources, native budgets, three metric names and
both scorer descriptors. It explicitly records `qualification_rules=not_supplied`
and `native_contract_qualified=false`; changing these or any identity rejects the
run. Three new rejection checks were authored. **Web executed no tests, scorer or
GPU workload**; Local whole-suite acceptance and the actual parity run remain
pending.

This unblocks measurement of scorer equivalence without fabricating scientific
qualification. A passed zero-tolerance parity sidecar still does not authorize
baseline/control acceptance: prospective source-backed decision rules, a complete
installed-verifier protocol and nonce-bound trusted replay remain mandatory.
Candidate validation designs/results/verdicts remain **0/15**. Next executable
step: Local accepts the changed suite, rechecks device/disk, generates and reviews
the parity-only plan, then runs it through the installed harness and returns all
raw official/faithful outputs.

## Continuation: parity admission independent-review repair — 2026-10-07

Independent source review found the preceding narrower admission was still not
executable: `run_experiments.py` rejects every `purpose=scientific` plan lacking
`protocol_ref`. It also found that the parity runner could write a sidecar after
metric equality without validating the official adapter/source/input/output
provenance or the faithful wrapper's protocol/source/input bindings. Finally,
the builder pinned the source-evidence file but did not verify its contents
against the current request.

The builder now generates a complete installed-verifier protocol alongside the
admission, manifest and native definition. Its source-backed `cd_3d >= 0` rules
express only the published native-distance output domain; explicit guardrails
forbid effect/candidate claims and retain `native_contract_qualified=false`.
This satisfies the scientific execution framework without inventing a
one-asset performance threshold. Plan construction now passes that protocol to
`native.make_plan`, uses a declared native arm role, and cannot complete unless
the installed protocol verifier accepts all source, split, metric, scorer,
sampling, budget, role and qualification-reference bindings.

Before metric comparison, the runner now validates each process command/cwd/
exit/log binding. The official result must additionally bind the exact adapter,
seed/device, frozen denominator, original and patched evaluator hashes, reviewed
compatibility patch, UID/frame count, sequence/GT, every GLB, export manifest,
CSV/summary and inner official CLI execution. The faithful result is passed
through the existing strict native-output validator, which binds protocol,
device, evaluator and official-source hashes, sequence, GT, generation report
and pass manifest. A mismatch prevents the verification sidecar.

Nine parity checks are newly authored relative to remote main (21 checks in the
module total), but **none were executed by Web**. No scorer or GPU workload ran;
official scores, complete multi-arm timing, candidate designs/results/verdicts
remain 0. The next executable step remains Local whole-suite acceptance,
device/disk recheck, source inspection of all four generated parity contract
files, then exact-digest harness execution with complete raw return.

## Continuation: reject false native qualification — 2026-10-07

A second independent review invalidated the attempted generated scorer-validity
protocol. Besides an incompatible paired/project aggregation combination, the
substantive problem is that the installed native evaluator does not honor prose
guardrails: it would interpret `cd_3d >= 0` as ordinary baseline/control
qualification and could mark every finite nonnegative score qualified. That
would turn a domain-validity fact into a false scientific result.

The generated protocol/admission path was removed. Parity again requires a
separately authored `--protocol` that passes the installed verifier, uses the
exact official scorer and request sampling, cites the verified source evidence,
declares a `scorer-qualification` execution role, and contains genuine
source-backed performance rules. The builder and runner explicitly reject the
known domain-only `cd_3d >= 0` substitute. Because official sources reviewed so
far publish aggregate 128-object means but no one-asset threshold, parity plan
generation remains honestly blocked rather than relabeled as engineering.

The useful code advance is retained: source evidence is checked against the
request/GT/current official files, and sidecar emission now requires exact outer
command/log binding plus complete official adapter/source/patch/input/GLB/output
provenance and the faithful wrapper's strict protocol/source/input validation.
Two new checks cover execution-command and unbound-adapter rejection (14 checks
in the parity module total); **Web executed none**. Official scores, GPU work,
complete multi-arm timing, candidate designs/results/verdicts remain 0. The
next executable step is to establish a legitimate source-backed native protocol;
without it Local should return `blocked_missing_source_backed_native_protocol`.

Final static review also found and repaired four fail-closed details: the native
sample manifest is now a direct scientific-plan input; both scorers execute from
the exact descriptor cwd; any qualification rule citing the negative threshold
review is rejected, regardless of metric/operator spelling; and the full
sampling policy plus `predictions_per_sample=1` are bound in both builder and
runner. These remain authored, unexecuted safeguards.

## Continuation: separate scorer equivalence from scientific qualification — 2026-10-07

The installed research workflow distinguishes a protocol-free engineering run
from a scientific run that may advance evidence gates. Official-versus-faithful
scorer equality belongs to the former: it checks evaluator implementation
equivalence, while baseline/control efficacy still belongs to the latter. The
previous requirement for a performance-threshold protocol before parity was
therefore removed without weakening the scientific gate.

The new builder emits a `purpose=engineering` plan with no `protocol_ref`, plus
an immutable one-UID sample manifest and scorer-equivalence contract. Those
inputs bind the exact returned request and digest, released population and GT,
all three report/prediction artifacts, official source review, runtime, native
budgets, both scorer descriptors and zero tolerances. The descriptors use
project-relative arguments so contract identity survives isolated harness
staging. The runner rejects scientific qualification fields and verifies the
current interpreter, five dependency versions, physical GPU UUID and
`CUDA_VISIBLE_DEVICES` before any scorer process.

Independent review found and closed three additional evidence gaps: the contract
is now self-contained rather than relying only on plan-level inputs; a successful
run emits only a bundle attestation, then promotion of the complete bound output
tree and a separate hash-validating finalizer produce the scientific-consumer
sidecar; and the Local runbook executes engineering parity before trying to
author the presently unavailable performance-threshold protocol. The later
scientific plan still requires genuine prospective source-backed decision rules,
installed-verifier acceptance and nonce-bound trusted replay. A `cd_3d >= 0`
domain rule and 128-object aggregate means remain forbidden substitutes.

Seven parity/finalizer checks were authored, bringing that module to 21 checks. Web parsed
the changed Python sources and checked the patch statically, but executed no test,
scorer or GPU workload. No official score or scientific conclusion was produced;
candidate validation designs/results/verdicts remain **0/15**. The next Local
step is whole-suite software acceptance at the delivered revision, device/disk
recheck, generation and source inspection of the parity contract/plans, then the
exact-digest harness run and complete-bundle return. Scientific baseline scoring
remains blocked after parity until a genuine source-backed protocol exists.

A final compatibility review caught that the canonical relative descriptor used
inside the isolated parity workspace differs from the absolute descriptor that
the unchanged scientific scorer admission requires. The finalizer now converts
only that command representation using the existing request-pinned
`contract_scorer_command`; all source/code identities remain unchanged. It also
requires the exact stable bundle target and resolves the record, evidence, raw
official/faithful outputs and predictions before writing the single-use consumer
sidecar. Separately, the parity runner re-resolves the complete frozen contract
after all six subprocesses, so a scorer-side mutation prevents pass status and
attestation emission.

The promotion finalizer is also cryptographically tied back to execution: it
requires the exact approved outer-plan digest, completed harness report, native
plan and native receipt. It verifies the engineering/no-protocol identity, one
completed `scorer-parity` attempt, immutable input/code refs, the complete
declared-output inventory and the harness-recorded attestation, record, evidence
and raw-output hashes. Predictions must be receipt-bound attempt inputs. Thus an
internally consistent reconstructed output tree cannot mint a consumer sidecar.

## Continuation: repair current software-acceptance import closure — 2026-10-07

No new Local/GPU return followed `202cb6e`. Static inspection of the next required
whole-suite plan found a concrete execution blocker: the parity test module now
imports the root-level `finalize_actionbench_parity.py`, but
`prepare_control_scoring_checks.py` did not include that file in the native
plan's `code_refs`. The isolated harness workspace would therefore fail during
test import, before exercising the new parity/finalizer rejection checks.

The acceptance builder now owns a single explicit `acceptance_sources(root)`
inventory, includes the finalizer, and rejects a missing source before plan
emission. One regression check asserts that both the finalizer and parity test
module are staged. The Local handoff now requires inspecting this exact code ref
and explicitly says that the historical 127/127 result does not accept the
current revision.

Web parsed the two changed Python files and observed 149 authored `test_*`
methods statically; it did **not** execute the suite, scorer or GPU workload.
The observed Local count may differ through loader behavior and must be recorded
from the harness run. Scientific state is unchanged: parity is unexecuted,
official scores are zero, and candidate designs/results/verdicts remain **0/15**.
Next: Local regenerates the acceptance plan at the delivered revision, confirms
the finalizer ref, runs its exact approved digest, then proceeds to parity only
after that current-revision acceptance passes.

## Continuation: harness the post-parity finalizer — 2026-10-07

No new Local/GPU return followed `32bef65`. Source review found that the parity
procedure still instructed Local to invoke `finalize_actionbench_parity.py`
directly after bundle promotion. That is an executable evidence-collection step
outside the required single harness owner, even though the parity scorer itself
was already harnessed.

The new `prepare_actionbench_parity_finalization.py` generates a second,
CPU-only engineering plan. It recursively pins the promoted bundle, nested file
references, canonical outer/native plans and receipts, task/state/result records,
original attempt record, and every declared parity output. Its only declared
output is the single-use `faithful-harness-verification.json`; Local must promote
that receipt-bound sidecar back into the already complete bundle and verify its
hash. Direct finalizer execution is now forbidden in the protocol, runbook and
round handoff.

Three focused regression checks were authored for nested-reference closure,
canonical execution-origin paths, and the finalization source inventory. Web
parsed the four changed Python files and observed 152 authored `test_*` methods
statically; it did **not** execute them, the finalizer, either harness, a scorer,
or a GPU workload. Scientific state remains unchanged: parity is unexecuted,
official scores are zero, and candidate designs/results/verdicts remain **0/15**.
Next: Local runs current software acceptance, parity, complete bundle promotion,
then the new CPU-only finalization plan before any scientific protocol can cite
the sidecar.

## Continuation: exhaust the official one-asset qualification source path — 2026-10-07

Resumed exact remote main `9c1d7ef41927060472a4f161626798f13e0f330e`;
no newer Local/GPU packet was present. The current official ActionMesh main head,
its complete non-truncated repository tree, ActionBench README/evaluator files,
ActionBench-specific code searches and README history were inspected against the
retained source bytes. The retained seven ActionBench files exactly match the
current upstream Git blobs. The tree contains no predictions, per-sample score
table, CSV/JSON result artifact or threshold file. Searches for committed
ActionBench CSVs and thresholds returned zero results; the two `results` hits are
only the README and evaluator output writer.

This independently confirms the prior negative finding rather than weakening it:
the published leaderboard values are means over all 128 objects (ActionMesh seed
42), not one-object qualification thresholds. They remain forbidden for the
returned development UID, and metric nonnegativity remains forbidden as a false
scientific gate. The exact source identities, query URLs/counts, relevant official
history and adjudication are frozen in
`actionbench-qualification-source-audit.json`.

The audit identifies one legitimate but not-yet-ready source-matched alternative:
a new prospective full-128, seed-42 reproduction protocol using the official
scorer and published aggregate observations, with its own predeclared tolerance/
uncertainty rule and complete per-sample retention. It cannot reuse the existing
one-UID parity sidecar, one-UID timing or current single-asset inputs. No protocol,
test, scorer or GPU workload was executed; candidate designs/results/verdicts
remain **0/15**. The immediate Local order is unchanged: current software
acceptance, one-UID engineering parity, complete bundle promotion and CPU-only
finalization. Scientific scoring then remains blocked unless the full-population
protocol prerequisites are deliberately completed.

## Continuation: freeze the full-128 reproduction rule — 2026-10-07

No newer Local/GPU packet followed `af4d36b`. The current official README,
evaluator, retained 128-UID population and prior source audit were joined to close
one specific gap in the source-matched alternative: its prospective agreement
rule. The official source's seed 42 is the ActionMesh **generation** seed; the
pinned evaluator's sampling seed remains 44. The contract now keeps those
identities separate instead of using 42 for both.

The generated conditional contract requires the complete released population:
128 total, 128 successful, zero failed, 16-frame rows. All three official means
must fall inside the half-open three-decimal intervals represented by the current
README row: CD-3D `[0.0535,0.0545)`, CD-4D `[0.0845,0.0855)`, and CD-M
`[0.1525,0.1535)`. This is a full finite-population reproduction check, so it
claims no sampling confidence interval; the decision is the intersection of all
three metrics. The distinct paper-v2 values are retained as a non-target version,
not silently reconciled.

Added `actionbench_full_reproduction.py` plus nine authored rejection/decision
checks. The helper rejects incomplete/success-only denominators, one-UID reuse,
seed conflation, paper-target substitution, nonfinite/failed rows, CSV-summary
inconsistency and any missed metric interval. It never runs the scorer and always
returns `scientific_effect_qualification=false`.

This freezes a rule, not a runnable native protocol. Complete 128-object GT and
prediction manifests, ActionMesh source/weights/effective configuration, current
Conda/device qualification, installed-verifier protocol, full-unit timing/VRAM,
and nonce-bound official replay remain missing. Therefore no 28,800-second queue
was generated. Web performed only source/static review; it executed **0** tests,
scorer calls or GPU workloads. The current observed authored test-method count is
161, but only Local harness execution can accept it. Candidate designs, native
results and formal outcomes remain **0/15**, **0/15**, and **0**.

## Continuation: resolve the current public ActionMesh generation chain — 2026-10-07

No newer Local/GPU packet followed `29c8da1`. The full-128 alternative's next
source prerequisite was audited against the current official ActionMesh Git tree
and Hugging Face repositories. The current public chain is now explicit:
ActionMesh Git `d5c01f5045df55819e337369c9617f603c667e00`, TripoSG submodule
`fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`, ActionBench dataset
`2796071cbe6248422fcbeab3101fa9f9886cb7b9`, plus immutable current revisions
for `facebook/ActionMesh`, `VAST-AI/TripoSG`, `facebook/dinov2-large` and
`briaai/RMBG-1.4`. The Local runbook now gives exact revision-pinned `hf download`
commands and requires complete file/size/SHA-256 manifests before inference.

That resolution does **not** establish exact replay of the published leaderboard
generation. The complete official tree has no ActionBench batch-generation
driver or published prediction bundle. Its model downloader omits revisions and
trusts any nonempty local directory. The leaderboard supplies the non-fast
ActionMesh variant, all 128 objects and generation seed 42, while the public
entrypoint defaults to seed 44; it does not state the actual code/weight
revisions, dtype, low-RAM choice, hardware, dependency lock, retry policy or
prediction hashes. A future run using today's pinned releases must therefore be
reported as a versioned **current-public-release reproduction**, not the
unpublished historical run.

The hardware boundary is also now explicit. The official release documents
32 GB VRAM in default mode and 12 GB with `--low_ram`, but no RTX 2080 Ti result;
neither low-RAM nor float16 is source-bound to the published row. Local must
capture the actual device/runtime and preserve OOM or unsupported dtype as a
natural failure, not silently switch `--fast`, precision, configuration,
population or metric. Only after immutable staging may it measure one complete
non-fast, seed-42, 16-frame generation/export/scoring unit and peak VRAM. No
28,800-second queue was generated, because one-UID parity timing cannot price
this unit and the required 1,800-second collection reserve remains unbudgeted.

Web executed **0** tests, inference units, scorer calls and GPU workloads. The
existing 161 authored checks remain unaccepted until the Local harness run.
Candidate validation designs/results/formal outcomes remain **0/15**, **0/15**,
and **0**. The immediate Local order remains current whole-suite acceptance,
one-UID engineering parity, complete-bundle promotion and CPU-only finalization;
then return device/disk facts and pinned snapshot manifests before any full-128
timing plan is authored.
