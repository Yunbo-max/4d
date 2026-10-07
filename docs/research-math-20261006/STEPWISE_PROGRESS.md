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

## Continuation: make current-release snapshot staging fail closed — 2026-10-07

No newer Local/GPU packet followed `3fb78a5`. The resolved current-public-release
chain previously had commands to download immutable revisions, but no executable
admission step that could reject a partial cache, moving-revision residue or an
incorrect 128-object layout. Added a versioned snapshot contract, a CPU-only
harness plan builder and `research_math.snapshot_admission` to close that
engineering gap without launching inference or scoring.

The generated admission binds the official ActionMesh Git revision/tree and
TripoSG submodule, the ActionBench dataset revision and all four model revisions.
Every admitted Hugging Face content file must have matching local-dir revision
metadata and is recorded by relative path, size, SHA-256 and ETag. It rejects
missing/wrong metadata, metadata-parent or content symlinks, special files,
unresolved LFS pointers, downloaded HTML, empty snapshots and mutation during
hashing. For ActionBench it additionally requires the exact ordered 128-UID
digest and exactly `camera.json`, `surfaces.npy`, and `imgs/00.png` through
`15.png` for every released UID.

Static review caught and repaired two important first-draft defects: metadata
parent symlinks were not initially rejected, and the contract did not initially
recheck the canonical UID-list digest independently of the harness input ref.
The plan builder now also requires the contract's population path/SHA-256 to
equal the actual committed file. The only remaining source-level caveat is
explicit: the large external directories are read by their reviewed absolute
paths and bound by the resulting manifest rather than copied into the attempt
workspace, so this step is engineering staging and later scientific consumers
must revalidate the admitted bytes.

Eighteen rejection/closure and plan-admission checks were authored, raising the
static observed `research_math` test-method count from 161 to 179. Web performed
AST/JSON/hash and diff checks only; it executed **0** project tests, downloads, inference
units, scorer calls or GPU workloads. The plan remains `generated_unexecuted`.
Local must first run current whole-suite acceptance and the existing one-UID
parity/promotion/finalization chain, then download the five pinned snapshots,
inspect and execute the new CPU-only plan, and return its receipts plus five
file-count/byte-total/manifest-digest summaries.

Even a successful `admitted_engineering_snapshot` will not identify the
unpublished leaderboard environment, prove model/runtime compatibility, create
the missing 128-by-16 prediction inventory, or qualify an official result.
Therefore candidate validation designs/results/verdicts remain **0/15**,
**0/15**, and **0**; complete-unit timing/VRAM and the 28,800-second queue remain
blocked.

## Continuation: bind the full released dataset semantics — 2026-10-07

The immutable snapshot admission deliberately stopped at bytes, revisions and
file closure. The next source-backed prerequisite was therefore not model launch:
it was proving that the exact admitted ActionBench bytes have the tensor, camera
and frame semantics assumed by the official evaluator and generation input path.
The official README fixes 128 samples, 16 RGBA frames, a
`(16,100000,6)` position-plus-normal tensor and normalized position space. The
released `projection.py` fixes the four camera keys, their shapes and Blender
projection convention. A browsed released sample confirmed that schema but was
not treated as population-wide evidence.

Added `research_math.actionbench_dataset_semantics` and a CPU-only harness plan
builder. The pass refuses to operate without a successful exact-revision snapshot
admission. For every one of the 128 UIDs it rereads and rehashes the bound
`camera.json`, `surfaces.npy` and 16 PNGs, so post-admission mutation or a swapped
dataset fails before parsing. It rejects symlinked parents, wrong NumPy shape or
dtype, nonfinite values, positions outside the documented cube, malformed camera
keys/shapes/rotation/focal values, and non-square/non-RGBA/internally inconsistent/
truncated, undecodable or CRC-invalid PNGs. Cross-sample image encoding variants
are recorded rather than rejected. Its output retains per-UID input-manifest and semantic
digests plus numeric summaries, but sets scientific qualification and dispatch
readiness false.

The contract intentionally does not invent a unit-normal tolerance or alpha-mask
occupancy threshold because the public sources do not publish them. It also does
not claim to prove that point indices are truly semantically tracked through time;
that remains a documented dataset property outside what structural validation can
establish. Eighteen new fixture/rejection/plan checks were authored, raising the
static observed `research_math` test-method count from 179 to 197. Web performed
source inspection and static AST/JSON/hash review only: **0** project tests,
dataset downloads/loads, inference units, scorer calls or GPU workloads.

Local must still execute current whole-suite acceptance, then the earlier one-UID
parity/promotion/finalization chain. After the five pinned downloads and successful
snapshot admission, it may execute this single-attempt semantic plan and return
the exact receipt, output hash, failing UID/path if any, per-UID semantic digest
and measured wall time. A pass closes released-input structure only. It does not
authorize model loading or scoring, cannot price a complete multi-arm unit, and
does not change candidate designs/results/verdicts from **0/15**, **0/15**, and
**0**. The 28,800-second queue remains blocked pending a complete current
generation/export/scoring timing and peak-VRAM receipt.

## Continuation: freeze the first complete-unit identity — 2026-10-07

While this revision was being prepared, Local returned `00b30fd`: 201/201
software checks, all nine one-UID three-arm official/faithful parity comparisons
at exact zero difference under a versioned deterministic compatibility backend,
and completed CPU finalization. Those observations establish engineering scorer
equivalence only; they do not qualify the one-UID scores or any candidate. The
next unresolved binding before a current-public-release resource attempt was the exact unit identity: a later
runner must not choose a favorable object, reinterpret the official output
inventory, omit simple controls, or switch to a memory-saving variant after
seeing a failure. The official source closure confirms that the input is the 16
released RGBA frames and the required prediction output is `mesh_00.glb` through
`mesh_15.glb` plus the two deformation arrays.

Added a prospective current-release unit contract and CPU-only manifest pass.
The calibration object is fixed before outcomes as the first UID in the retained
canonical sorted population,
`000-000_03b69da8d2c94b5999bcf2605ee2ecd9`. After successful snapshot and
dataset-semantic admissions, the pass revalidates that UID's exact 18 admitted
files, the clean official Git revision/tree and 13 required source files, and
binds the complete-manifest digests for all four model snapshots. It refuses an
existing output instead of overwriting it.

The later timing boundary is now explicit: non-fast, non-low-RAM `bfloat16`
generation at seed 42; native, world-Gaussian and body-Gaussian arms; official
scoring of all three at seed 44; integrity collection; and complete wall/peak-
VRAM evidence. OOM, unsupported dtype, dependency/model, preprocessing, export,
control, scorer and timeout failures are retained. Automatic low-RAM, float16,
fast, UID, seed, revision, frame, point or arm changes are forbidden. Eleven
focused checks were authored, including path-escape rejection. After reconciling
the concurrent Local determinism update, the merged tree contains 212 statically
observed `research_math` test methods; none of the 11 new checks has run yet.

This closes only the manifest/specification layer. The current revision emits no
GPU plan and the complete-unit runner is still unimplemented. Web parsed the new
Python/JSON sources and ran `git diff --check`, but executed **0** project tests,
dataset/model loads, inference units, scorer calls or GPU workloads. Local must
run current software acceptance for the newly added checks, retain rather than
repeat the finalized parity, then run the snapshot, semantic and unit-manifest
admissions. A later reviewed runner must return the
actual full three-arm unit time and peak VRAM before any 28,800-second queue can
reserve 1,800 seconds for collection. Candidate designs/results/verdicts remain
**0/15**, **0/15**, and **0**.

## Continuation: bind the complete-unit raw evidence closure — 2026-10-07

Concurrent work after the unit-manifest commit implemented the default non-fast,
non-low-RAM BF16 generation/export/three-arm official-scoring runner, a one-GPU
harness plan, host/device telemetry and descendant cleanup on timeout. Local's
latest returned software acceptance at base revision `b3a4782` is 224/224. No
complete GPU unit has been dispatched: snapshot, dataset-semantic and unit-
manifest admissions remain prerequisites.

Review found one pre-dispatch evidence defect. The native plan declared only
`actionmesh/unit-output/result.json` as an output. The runner's JSON contained a
nested hash list, but that did not make the raw generation arrays/GLBs, three
control sequences, three official CSV/summary/backend bundles, copied evaluator
sources and telemetry direct members of the harness/native receipt. A successful
summary could therefore not by itself prove complete promotion of the raw unit.

The plan now derives an exhaustive 117-file deterministic successful-output
inventory from the prospectively frozen UID. It binds generator logs, arrays,
16 GLBs and the PyTorch3D `grid_normal.mp4` preview;
the three control arms; all 48 official-export GLBs; three raw official score
bundles; six copied evaluator files; integrity manifests; and host/device logs.
Two new rejection checks exercise exact set equality and prove the plan derives
the closure from the frozen manifest UID. They are authored but unexecuted by
Web, so current-revision Local acceptance is required. Natural failures remain
valid failed attempts with partial retained evidence; absent success files must
not be fabricated.

The Local runbook and handoff now contain the exact complete-unit plan/harness
commands. The first attempt remains the frozen default BF16 configuration. The
ledger's earlier low-RAM FP16 next-step wording was corrected: that variant is a
separately reviewed repair only after an observed default failure, never an
automatic fallback. This change executes **0** tests, inference, scorer or GPU
workloads and does not change candidate designs/results/outcomes from **0/15**,
**0/15**, and **0**. No queue is generated until the full receipt returns actual
wall time and observed peak VRAM.

## Continuation: freeze complete-unit admission before scoring returns — 2026-10-07

Local advanced the admitted current-release path substantially. The full 128
snapshot, dataset-semantics and one-UID unit-manifest passes are complete, and
the current software suite returned 228/228. The frozen default non-fast BF16
attempt retained a natural Stage-I attention OOM after 135.94 seconds with a
sampled 20,673 MiB peak; this is not converted into a score. The explicitly
versioned `fp16-lowram-v1` repair then completed all 30 temporal denoising steps,
decoded and rendered all 16 frames, and exited generation successfully after
792.54 seconds. At the latest committed observation, its three official score
rows were still pending at contract authorship. The subsequently returned
terminal evidence contains all three successful rows, 117 receipt-bound outputs
and a 121-file runner inventory: the 116 declared non-result files plus five
CPython 3.12 evaluator caches created by the official scorer. No second GPU
attempt was launched by Web.

The remaining post-run evidence boundary was not closed. Although the running
native plan declares 117 outputs, no independent pass yet requires the terminal
harness/native records, rehashes every declared raw file, compares the runner's
own pre-result inventory, and freezes the measured unit cost. Accepting
`result.json` or a completed receipt alone would allow a missing raw score bundle,
changed file or stale resource summary to become a queue-pricing input.

Added a prospective admission contract bound before the score outcome to
`complete-lowram-r7`, its approved plan digest, the frozen UID and
`fp16-lowram-v1`. Added a CPU-only finalizer and plan builder. A pass requires one
completed harness task/job/attempt, canonical plan/state/report/task/receipt/
attempt records, the exact 117 receipt files, the exact 121 files listed before
`result.json` was written (including the five exact evaluator caches), all four
completed stages, all three finite official
metric rows, and error-free one-second host/device sampling. It emits one
single-use measurement sidecar. It cannot qualify a scientific effect or native
benchmark, cannot increment a candidate, and does not generate a queue.

Twelve focused rejection/closure checks were authored first. Review then closed
three fail-open edges: the builder now recognizes only the official per-case
external ground-truth diagnostic reference while rejecting other escaping refs;
raw host/device JSONL is reparsed to reproduce identity, counts, peaks, cadence
and workload coverage; and the source execution context, remapped command/cwd,
process guard and generation/scoring stage records are independently bound.
Queue approval is explicitly false. The tree now has 245 statically observed
`test_*` methods: 228 are covered by the last returned suite; the five observer
checks and twelve new admission checks remain unexecuted. Web
performed AST/JSON/hash inspection only and ran **0** project tests, finalizers,
scorers or GPU work. Candidate native results and formal outcomes remain
**0/15** and **0**. Next: Local runs current whole-suite acceptance and the exact
CPU-only admission digest against the returned terminal source records.
Only its admitted measurement can feed a later 27,000-second workload queue with
the required 1,800-second collection reserve.

## Continuation: freeze Full128 queue pricing without dispatch — 2026-10-07

No newer Local result superseded the terminal `complete-lowram-r7` engineering
unit. Its elapsed time remains 1,330.465551 seconds and sampled peak remains
10,255 MiB, but the prospective CPU-only complete-unit admission is still
unexecuted. Therefore this continuation does not treat the unit as admitted and
does not generate an executable queue.

Added a second fail-closed boundary for the first scheduling decision after
admission. `research_math.actionbench_queue_pricing` accepts only the exact
`admitted_engineering_complete_unit` sidecar for `complete-lowram-r7`, rehashes
the admission contract plus every retained origin/output reference, revalidates
the frozen full-population reproduction contract and canonical 128-UID census,
and rejects scientific or queue claims. A CPU-only plan builder stages the full
transitive reference closure in the standard harness with one attempt, no retry
and no GPU allocation.

The pricing rule is prospective and deterministic. It applies 25% operational
headroom to the measured complete unit:

`ceil(1330.4655511886813 * 5 / 4) = 1664 seconds`.

The hard 28,800-second window retains 1,800 seconds for collection, leaving a
27,000-second workload budget. Thus a full window contains
`floor(27000 / 1664) = 16` complete units, plans 26,624 workload seconds and
leaves 376 seconds of additional workload slack. The exact 128-UID canonical
population partitions into eight consecutive 16-UID windows with no omissions,
reordering or success selection.

Six pricing rejection/partition checks and two plan-boundary checks were authored
first, raising the statically observed `research_math` `test_*` method count from
245 to 253. Web executed **0** project tests, finalizers, scorers or GPU work.
The conditional output deliberately says `queue_priced: true` only after Local
executes the admitted CPU pass, while retaining `queue_approved: false`,
`queue_generated: false`, `dispatch_ready: false`, and every scientific/candidate
flag false. At authoring time even `queue_priced` remains false because the source
admission is absent.

Next: Local runs all 253 current software checks, then the exact complete-unit
admission plan. Only a successful sidecar may feed the pricing plan. The returned
pricing manifest freezes capacity and UID windows but still cannot be dispatched;
a later revision must implement and admit the per-UID full reproduction runner,
qualify the official baseline/replay path, and preserve the full denominator.
Candidate native results and formal outcomes remain **0/15** and **0**.

## Continuation: bind each Full128 UID to admitted pricing — 2026-10-07

The conditional queue still has no admitted pricing receipt, executable window
plan or dispatch authorization. This continuation therefore does not create a
queue or run generation/scoring. It closes the next software boundary that was
previously missing: the calibration runner could only accept the prospectively
fixed first UID and could not safely represent the other 127 samples.

Added `research_math.actionbench_full128_unit`, which recomputes the prospective
pricing receipt from its three pinned sources and therefore rechecks the complete
unit's transitive origin/output evidence. A selected UID must occur in the exact
named ordered window. The freezer then verifies the admission-bound FP16 low-RAM
template, clean source revision/tree and required files, four model snapshot
identities, and the UID's exact 18 dataset files against both snapshot and semantic
admissions. Its single-use output retains every queue/science flag as false.

The existing complete-unit runner now has an explicit Full128 mode. The pricing
path must be canonical, `root`, pricing, UID and window ID must be supplied as one
bundle, and its wall limit must exactly equal the admitted per-unit price. The
same revalidation runs again after scoring for final integrity. Twelve focused
checks were authored first, bringing the statically observed `test_*` count from
253 to 265. Web executed **0** tests, finalizers, scorer calls or GPU workloads.

This is not yet an executable queue: no harness window compiler has been authored
or admitted, and baseline/replay qualification remains outstanding. Local's next
step remains the full 265-test software acceptance, complete-unit admission and
CPU-only pricing pass in that order. Candidate native results and formal outcomes
remain **0/15** and **0**.

## Continuation: compile canonical Full128 windows without duplicating active work — 2026-10-07

The newest Local return supersedes the earlier prospective admission/pricing
status. `admit-complete-r9b` admitted the historical three-arm engineering unit,
and `price-full128-r9` produced the canonical pricing receipt: 1,664 seconds per
unit, 16 units and 26,624 workload seconds per 28,800-second window, plus the
separate 1,800-second collection reserve. These remain engineering receipts, not
official full-population qualification or candidate evidence.

The same return reports `population-gpu-current-r9` already launched on the
single GPU for canonical indices 1 through 9 using the concurrent population
builder. Web last sees its state at 2026-10-07T15:39:08Z with one running and
eight pending. That timestamp is not current liveness evidence. No second attempt
or overlapping window is generated here; Local must reconnect to the same run ID
and digest, then collect, carry, fail, or accept each UID explicitly.

Added `prepare_actionbench_full128_window.py`, the missing canonical harness
compiler for later windows. It recomputes the pricing receipt and its transitive
evidence before plan creation, checks the exact current 8 × 16 partition, emits
one no-retry native plan per UID using the Full128 complete-unit runner mode, and
places all 16 tasks under one single-GPU outer harness. The frozen limits are
`max_parallel_tasks=1`, `max_tasks_per_gpu=1`,
`max_gpu_task_seconds=26624`, `total_wall_seconds=27000`, and
`window_seconds=28800`. The 27,000-second driver cutoff makes the final 1,800
seconds unavailable to new work and therefore preserves it for collection.
Every native timeout equals the admitted 1,664-second
price; no fast/dtype/low-RAM fallback or outcome-based UID omission is introduced.

Eight focused checks were authored first for exact UID order, false approval and
dispatch flags, operational budget/collection-reserve closure, attempt-relative
staged argv, exact current environment/dependency closure, and the complete
sixteen-plan no-retry set. They raise the statically inferred merged-tree method
count from 270 to 278. Web
executed **0** tests, plan builders, scorers or GPU workloads. A canonical window
plan therefore remains ungenerated and unapproved. Local's next legal step is to
reconcile the live r9 batch, run all 278 current software checks, and only then
emit/freeze a non-overlapping window plan for source review. Window 01 stays
blocked until indices 0 through 15 have explicit dispositions. Candidate native
results and formal outcomes remain **0/15** and **0**.

## Continuation: make active-batch overlap protection executable — 2026-10-07

The latest remote `main` observed for this continuation is `b047114`. Its r9
engineering ledger reports five completed units, one running unit and three
pending units over canonical indices 1 through 9. These are retained engineering
baseline attempts, not candidate results. Their changing status does not permit
another plan to duplicate, retry or outcome-select any of those UIDs.

Source review found one fail-open boundary in the newly authored canonical
window compiler: the runbook required overlap reconciliation, but the compiler
did not consume a reconciliation artifact. A caller could therefore emit window
01 even though it intersected r9. This continuation adds a mandatory canonical
sidecar, exact schema and SHA-256 closure. The sidecar must include
`population-gpu-current-r9`, a 64-hex plan digest, the complete ordered range
`[1,10)`, one canonical disposition per UID, and immutable source references.
Only `completed`, `running`, `pending` and `failed` are accepted. Every one of
those states reserves the UID: a target-window intersection now fails before the
plan directory is created.

Seven focused checks were authored before the implementation for nonoverlap,
missing/incomplete/duplicate retained ranges, identity/claim drift, canonical
campaign/status binding, campaign/native-plan digest closure, and rejection of
additional unreviewed source runs, and archive-root-safe closure. The expected merged-tree total becomes 285
methods after Local merges this revision. The sole r9 campaign digest is
independently recomputed; all nine native plans require exact SHA-256 references
and independently recomputed plan digests, and are retained in every new unit's
input closure. Validated historical JSON is staged as opaque byte-pinned evidence
so its original run-relative inner paths cannot be rebound to current-root files.
This compiler revision intentionally accepts exactly one retained
r9 handoff; supporting any later retained run requires reviewed code.
Under the Web/Local role split, Web executed **0** project tests, plan builders,
scorers or GPU workloads; the code and checks remain `generated_unexecuted`.
Static AST, JSON, hash-reference and diff validation are the only Web checks.

The sidecar permits plan generation only. It does not approve a queue or prove
current liveness, so Local must re-reconcile the host immediately before any
later execution decision. Window 01 remains blocked by construction; remaining
indices in 0 through 15 require a no-omission continuation rather than a
success-selected replacement. Candidate native results and formal outcomes stay
**0/15** and **0**.
