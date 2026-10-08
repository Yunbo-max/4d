# Stepwise continuation — 2026-10-06

Latest source handoff: [Native decoder capture](NATIVE_DECODER_CAPTURE.md).
Earlier entries below are dated history; the final entry supersedes their current-status wording.

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

## Continuation: derive retained task dispositions from harness state — 2026-10-07

The newest verified remote checkpoint reports six completed r9 engineering
units, one running unit and two pending units. Population-006 has a successful
117-file engineering receipt, while population-007 is the active generation
process. These remain baseline engineering attempts; they neither qualify the
official full-population benchmark nor test a candidate method.

Review of the overlap guard found one remaining evidence gap. The canonical
`STATUS.json` carries aggregate counts, but the retained repository archive does
not yet contain `runs/harness/population-gpu-current-r9/state.json`. Aggregate
counts cannot prove which exact task owns which status: swapping a completed and
running task preserves all four totals. Therefore a manually written
`dispositions` list is not sufficient evidence for plan generation.

Added `prepare_actionbench_active_batch_reconciliation.py`. It has no status-list
input. It reads the exact canonical pricing, r9 campaign, aggregate status and
research-harness state snapshot; derives all nine ordered dispositions from the
state task map; hashes every source; then invokes the window compiler's complete
reconciliation validator before writing the sidecar. The compiler now requires
that same state reference, exact batch ID and plan digest, the complete task-key
set `population-001` through `population-009`, and equality between every
disposition and its corresponding task status. A per-task state swap is rejected
even when aggregate counts remain unchanged.

Three focused checks cover complete state admission, state-derived sidecar
generation and same-count task swaps, raising the Full128 window test module to
18 methods and the expected merged `research_math` total from 285 to 288. The
author-side isolated checks were observed green in the transient partial checkout
using import-only stubs for source modules absent from that checkout; this is not
the required Local whole-suite software acceptance. Python compilation and JSON
parsing passed. No plan was generated, no GPU or official scorer was invoked, and
queue/science flags remain false.

The new guard intentionally blocks today: the exact r9 state snapshot is not in
the retained archive. At the next safe r9 boundary, Local must copy that file
without modification to its canonical archive path, run current whole-suite
acceptance, then invoke the sole builder command recorded in the reconciliation
contract. Only the resulting immutable sidecar may feed a nonoverlapping Full128
window. Candidate native results and formal outcomes remain **0/15** and **0**.

## Continuation: freeze state and observer status as one immutable pair — 2026-10-07

The latest observed r9 engineering ledger has advanced to seven completed units,
one running unit and one pending unit at 2026-10-07T17:59:21.929Z. Population-007
completed generation and official scoring with exit code zero; its receipt and
all 117 declared output hashes were independently verified. Population-008 then
reached Stage I 27/30. These are still current-release engineering baseline attempts, not
candidate-method evidence or full-population qualification.

Source review found a second concurrency boundary in the new reconciliation
chain. The sidecar required a hash-bound harness `state.json`, but its independent
aggregate reference still pointed to the global `STATUS.json`. That file is
intentionally rewritten by the watcher after every observed transition. A valid
sidecar would therefore become unreproducible from the next checkout even though
its state evidence had not changed.

Added `prepare_actionbench_active_batch_snapshot.py` and changed the compiler
profile to an archived `status-snapshot.json`. The Local-only builder double-reads
both live files, rejects either file moving during capture, validates the r9 run
ID/digest/task set, derives completed/running/pending/failed counts from the task
map, requires exact equality to the observer snapshot and writes the original
bytes exactly once to two immutable archive paths. Ordinary publication failure
removes a partial pair; existing outputs are never overwritten. Only after this
pair exists may the separate reconciliation builder derive the nine ordered UID
dispositions.

Two focused checks were authored first for stable-byte preservation and
count-incoherent rejection without partial output, raising the expected merged
software suite from 288 to 290 methods. Under the Web/Local role split these new
checks remain `generated_unexecuted`; Web ran no project tests, snapshot builder,
GPU work or scorer. Static source/JSON validation and GitHub byte readback are the
only author-side acceptance intended for this round.

The current Local software receipt still covers only 258 pre-a408422 checks, so
it cannot qualify this code. At the next safe r9 boundary Local must refresh the
observer ledger, run the then-current suite (subsequently superseded by 291 checks
below), execute the immutable snapshot
builder and then the reconciliation builder. Window generation remains blocked
until Web reads back that exact pair and sidecar. Candidate native results and
formal outcomes remain **0/15** and **0**.

## Continuation: close the Full128 software-acceptance import set — 2026-10-07

The newest retained observer record reports eight completed r9 engineering units,
one running unit and no pending or failed units at 2026-10-07T18:15:41.051Z.
Population-008 completed generation and scoring with exit code zero, and all 117
receipt-declared outputs were independently rehashed. Population-009 was at Stage
I 19/30. These remain current-release engineering baseline attempts rather than
candidate-method or native scientific evidence.

New Local feedback also preserved a direct macOS/Python 3.9 discovery attempt:
259 checks ran with 9 failures and 13 errors. The attempt used neither the Linux
harness acceptance path nor a complete compatible checkout, so it is retained as
diagnostic evidence only. Two following commits corrected the one genuinely stale
Full128 rejection assertion; the remaining output includes platform, dependency,
missing-source and fixture mismatches and cannot be converted into acceptance.

Source inspection then found a deterministic failure in the *formal* isolated
acceptance plan: its explicit root-source inventory did not include
`prepare_actionbench_full128_window.py`,
`prepare_actionbench_active_batch_snapshot.py`, or
`prepare_actionbench_active_batch_reconciliation.py`, although the current window
test module imports all three. Added a focused closure check first, then added
exactly those modules to `ROOT_SOURCES`. The plan remains CPU-only, single-attempt,
and fail-closed when any source is missing.

The new check raises the expected current suite from 290 to 291 methods. Web
performed source/AST/JSON/hash review only and did not execute the check or any
project suite, model, scorer or GPU work. Local must regenerate the software plan
at the delivered revision, inspect all three new `code_refs`, and obtain a clean
291-method Linux harness receipt before the immutable r9 state/status capture and
reconciliation sidecar can be accepted. Candidate native results and formal
outcomes remain **0/15** and **0**.

## Continuation: verify terminal r9/r10 return and isolate window-preparation blocker — 2026-10-07

The returned archive at commit `04ea448` was downloaded and its SHA-256
independently recomputed as
`34967a5fd1d259be0398aebd538e8f9347223ffd9128ee7ba63e84dda0f4a126`.
All 1,617 manifest payloads were checked for declared size and SHA-256:
5,448,428 bytes total, zero missing or mismatched files. The archive excludes
images and model weights by design; their admitted host manifests remain
separate evidence.

The Linux harness receipt records 294 software checks, zero failures/errors and
exit code zero. Its 72 `code_refs` were compared one by one with the current
code tree: zero mismatches. The later `712fb29` head adds only a monitor note,
so this software acceptance still covers every executable ref. It closes
engineering acceptance only, not scorer, benchmark or candidate qualification.

The immutable r9 pair is terminal at
`2026-10-07T19:12:37.209259Z`: 9 completed, 0 running, 0 pending, 0 failed.
State, observer snapshot and reconciliation SHA-256 values are respectively
`c6615919...b862`, `744c4b46...75b` and `4ad874da...97a`. Eight CPU-only
r10 manifest batches cover 118 unique indices 10-127, with 236 log refs checked,
no nonempty stderr, no inference and no GPU workload. These are engineering
baseline/input facts; candidate results remain 0/15.

The first plan-only window-02 attempt exposed the next real blocker before any
GPU launch. The retained r7 admission recursively names five historical source
files whose hashes differ from the current checkout, while the retained
environment binds the older physical GPU. Source tracing shows that
`build_input_ref_closure` currently sends both historical pricing/admission
evidence and current executable inputs through the same live-root recursive
closure. Rewriting the old hashes would invalidate the timing evidence; accepting
the old GPU UUID would invalidate current-device admission. The generic closure
also rejects historical absolute refs outside the current checkout.

No production repair is committed in this checkpoint because the exact plan-only
failure packet and a Local failing regression are not yet returned. The legal
next step is to finish that read-only inspection, preserve the original bytes,
then test-first separate validated opaque historical pricing evidence from the
current live execution closure. A fresh GPU-4910 environment capture remains
mandatory. Queue generation, approval and dispatch stay false; scientific
qualification and all 15 candidate experiments remain open.

## Continuation: audit repaired archive closure through the real unit runtime — 2026-10-07

Local returned a substantive repair at `240bd0d`: the historical R7 archive
closure now distinguishes exact old evidence from the current checkout, and the
return reports a TDD sequence of 302/302 green, two expected new red cases, then
304/304 green under the installed CPU-only harness. It also reports an officially
validated, unexecuted Window-02 plan for canonical indices 16-31: 16 serial
single-GPU units, 26,624 priced workload seconds inside the 28,800-second outer
window, plan digest
`d7e5558a08aafcc1c996ebae558aa74b77eda42c9179ff1bb01291feccab6ab9`.
The plan keeps approval, dispatch and science flags false. The r9 terminal 9/9
ledger and 118 CPU-only r10 manifests are unchanged, and indices 10-15 remain an
explicit no-omission coverage gap.

The new static review does not yet accept Window-02 for dispatch. The compiler
validates the pricing receipt against the exact historical checkout and uses the
archive manifest while collecting staged input refs. However, the emitted native
command carries only `--root ..` and the current staged pricing path. At actual
unit startup, `complete_unit_runner.verify_prerequisites` calls the real
`freeze_unit` with that current attempt root, and `freeze_unit` immediately calls
`verify_pricing_receipt(root, pricing)`. No historical root, archive manifest or
equivalent resolver crosses that runtime boundary. Consequently, successful
plan construction does not prove that a real unit can complete its pre-inference
freeze. This is an inferred fail-closed prerequisite risk, not a claimed
executed failure.

The current tests leave the same boundary open: runner coverage mocks
`freeze_unit`, while native-command coverage requires staged current paths but
does not require runtime historical-evidence plumbing. The next legal step is a
non-mocked CPU red test over the real runner-to-freezer path with distinct
current and historical roots. The repair must revalidate the pinned historical
revision, clean tracked state, archive-manifest hash and every consumed old byte
inside each unit, while keeping current code, GPU UUID, environment, models,
dataset and scorer bound to the current attempt. Only after the complete CPU
suite returns raw red/green receipts and an exact regenerated plan may Web review
dispatch again.

The committed `LOCAL_EXECUTION_RECEIPT_R2.md` is a useful source-safe summary,
but the three newly reported raw receipt JSON files/logs and exact historical
archive manifest are not on `main`; their digests occur only in that summary.
Independent receipt replay therefore remains pending. Web performed static
source, JSON and SHA-256 review only: zero project tests, builders, scorers,
models or GPU workloads. Native scientific qualification, trusted replay,
candidate native results and formal outcomes remain **0/15** and **0**.
## Continuation: accept UID008 parity return without upgrading native qualification — 2026-10-07

The new compact UID008 return was read from exact main commit
`6d630ef5fe0d01b364eaa73e2f375e2c63393331`; its repository bytes were
rehash-checked. It reports one current-device parity attempt and a CPU-only
finalization attempt with exit code zero. For native, world Gaussian and body
Gaussian, all three reported official metrics are numerically identical to the
corresponding faithful metrics and all nine reported absolute differences are
zero. This is useful scorer-equivalence evidence for the single frozen UID and
the exact reported source/runtime closure.

The receipt also explicitly says that its 127-file promoted bundle was not
transferred. Therefore Web could not independently rehash its 112 parity output
references, 127 promoted files or nested sidecar references, nor replay either
scorer. No nonce-bound trusted replay or same-scorer repeat was returned. The
result is accepted only as a compact engineering summary; it does not qualify
the native benchmark contract, any baseline/control effect, Full128 reproduction
or a candidate method. Candidate native results and formal outcomes remain
**0/15** and **0**.

The separate Full128 dispatch blocker is unchanged: the compiler verifies the
historical pricing closure while building the plan, but its emitted unit command
does not pass the historical evidence root or manifest into the runner that calls
`freeze_unit`. A focused RED-test patch now makes both arguments mandatory at the
command and runner boundaries. The patch was dry-run against the exact reviewed
source and applies cleanly, but Web did not execute project tests. Under the
test-driven boundary, no production repair was authored before an actual RED
receipt exists.

Next executable work is two-track and ordered. Local must transfer the complete
immutable UID008 bundle for independent rehash, then return fresh nonce-bound and
same-scorer-repeat records before native qualification is considered. Separately,
Local must apply the RED patch in an isolated exact-commit checkout, preserve the
focused failures, add a non-mocked runner-to-freeze closure test, and only then
implement the minimal double-root propagation fix. Focused GREEN, current
whole-suite acceptance and a regenerated complete-unit validation are mandatory
before any GPU dispatch.

## Continuation: connect complete decoder capture to the official generation path

Resumed exact main `1da40c748a71f9687bd2c5958ee6705671632d10` in a fresh
checkout, preserving the dirty older workspace. The user's feedback requires
actual model interfaces and complete output paths instead of treating algebra
functions or document/test counts as delivered methods. The selected pool and
unclosed native qualification/Natural Gate 0/IPCG are retained; this work is
baseline interface engineering, not candidate implementation or a gate bypass.

The existing observer only accepted an already loaded decoder. Added an
instance-local pipeline adapter that attaches at `_decode_displacement`, after
official low-RAM lazy loading. It retains full latent/timestep/source/target/query
tensors including original vertex normals, raw decoder output, anchor topology
and every returned target mesh. It preserves original call/output objects,
validates full coverage and output-to-mesh semantics, restores hooks/methods on
failure, and prevents a caught failed window from becoming a complete session.

The complete-unit plan and runner now have an opt-in `--capture-decoder` path.
A separate child entry calls the actual upstream `run_actionmesh` using the
unchanged profile and generation arguments; it archives all captured windows.
The existing three-arm control and official-scoring stages still execute in a
later admitted run. A distinct engineering task role and two additional declared
outputs separate this 119-output unit from the old 117-output timing admission.
Full128 plus capture is rejected; no old per-unit cost is applied. Observer
plans enforce the 27,000-second driver cutoff and 1,800-second collection reserve.

Also reconciled the stale runtime-root blocker with R3's raw archive. Its
green-r2 archive SHA-256 is
`2d7b0290711ed37044f57083633eb9347f15be2669eeab022f532738977f9a68`;
all 21 payloads in its manifest match size/SHA-256, the receipt matches
`61e0a4113e9c1e16b6581f784c94823931b46b5d29982dd6139e9ac9c0a88b4d`,
and the retained stderr says 311 checks, OK. At the resumed commit one of the
72 pinned sources already differed (the subsequent macOS path normalization).
This delivery changes additional sources. Thus the repair exists and its prior
test evidence is real, while current-tree software acceptance remains pending.
No historical records or hashes were rewritten.

New capture tests and source are generated, unexecuted. Web performed source,
AST/JSON, retained-evidence hashing and diff checks only; no project test, plan
builder, model, scorer or GPU workload ran. Full native capture, observer-free
equivalence and full-context replay still require actual Local execution. GPU
stop remains in force. Next executable step: the documented CPU harness
acceptance; then eligible plan-only preparation and evidence restoration.
Candidate complete designs/results remain 0/15 and formal outcomes remain 0.

## Continuation: native replay and finite supervisor source — 2026-10-07T23:24:50Z

Preserved the concurrent `1bc3c97` capture-only unit and added a separate paired official-generation instrument, complete original decoder-context replay and guarded source-time query. The new observer retains exact native frame mapping, gradient/autocast modes, original argument/output identity and full geometry. Persisted payload accounting includes both decoder input copies; partial hook registration rolls back. Both pipeline wrappers reject simultaneous attachment.

Added a finite campaign controller around the installed `run_harness.py`: pinned dependencies, verified terminal receipts, bounded preapproved repair children, same-task recovery, an unchanged overall deadline, driver-lock waiting and STOP handoff. It is source only, not an installed or running supervisor. CPU acceptance stages both capture entries, new replay modules, supervisor and test sources with the selected installed skill path.

Source review, AST/JSON parsing and whitespace checks are distinct from runtime acceptance. No project test, builder, model, scorer, supervisor campaign or GPU task was executed by this authoring round. All new source remains `generated_unexecuted`. The paired instrument needs its own budget and output closure; old calibration pricing is not reused. The 15 candidate implementations/full designs are not complete, and no scientific gate or native result advances. See `rounds/20261008-native-context/WEB_HANDOFF.md` for the concrete next Local acceptance step.

## Continuation: make paired native context deliverable through the harness — 2026-10-08

Resumed exact main `ece1ce883ea106e7d2d30ef23a3712a81cba9e49` in a new
clean detached worktree; preserved prior dirty workspaces. The historical
pricing-root fix is present and was not reimplemented. GPU STOP remains in force.

Added the missing plan-only `prepare_native_context.py` for the existing
calibration UID. It pins current environment/dependency records, all actual
runtime source imports and six native prerequisites; emits one engineering
attempt with no retry; preserves an explicit separate two-generation/replay
budget and collection reserve; and never consumes the Full128 queue price.
Runner runtime metadata is checked before loading and again at final verification.

Closed raw-output delivery: the runner now validates a complete successful raw
inventory and binds every emitted regular file into a tar plus manifest, with
archive-member rehash and change detection. Result/manifest/archive are direct
native receipt outputs. Failed or forcibly interrupted work retains its partial
attempt tree. Independent review found collection disk/RSS was outside the old
telemetry interval; added separate before/after collection observations and
explicitly scoped sampled telemetry, without claiming exact peaks.

Authored Local tests using the actual installed plan/staging code (no mock at
that boundary), raw archive round-trip, required/missing/link outputs, source-time
conditions, dependency tamper, budget admission and archive disk accounting.
Static parsing and whitespace checks only were performed; no project test,
plan builder, model, scorer, supervisor or GPU task was executed by Web.
Independent final source review found no remaining critical issue; Local
acceptance is still pending.

Expanded the existing CANDIDATE_INPUT_AUDIT.json into the maintained 15-item
source/design delivery inventory after a separate read-only audit. It preserves
all selected IDs and actual math references, names actual entries and missing
adapters, required outputs and control arms, scorer linkage, absent candidate
acceptance commands and separate implementation/scientific obligations. C02 is
still a dense primitive, C14 a baseline, C01 has only shared capture/replay, and
no candidate is marked complete. Native candidate results remain 0/15.

Current host remains the existing hourly authoring task
`6ac527ad50cc8191a4964d35ae0b826e`, not an installed continuous Local supervisor.
This is a partial code delivery toward the unchanged full-source stage goal;
the task must remain enabled. Next: Local CPU acceptance at the delivered
revision; Web continues legal input adapters and method/design work without
inventing natural-gap/importance/collision evidence. Full128 10–15 coverage,
UID008 raw/repeat/trusted replay and all scientific gates remain explicit gaps.

## Continuation: C13 quadratic strong control — 2026-10-08

Resumed exact main `c41d291b50aa3d84ad495417724403132041ea0b` in a clean
worktree and retained the existing 15-candidate selection. Independent source
review confirmed that every candidate still lacks scientific admission; source
complete candidates therefore remain 0/15.

Implemented C13's named quadratic-acceleration strong comparator as a separate
CPU-only path, without relabelling it as the group-trend candidate. It consumes
a controller-verified real full `sequence.npz`, then hash-binds the staged
sequence/report pair, constructs second differences in the supplied timestamp
units, uses an identity observation metric and a direct frame-zero-anchored
quadratic solve, preserves all
16 frames/topology/identity metadata, records numerical and provenance details,
and emits a single case manifest for the existing official adapter. Float32 is
required before export; reports record the timestamp unit convention and the
weight's fourth-power unit dependence. Added a
single-use zero-GPU harness plan with one attempt and zero retry plus Local
contracts for affine fixed points, spike damping, nonuniform time, invalid
inputs, identity preservation and fail-closed hashes. Also corrected native
replay's vertex slice from the batch axis to XYZ.

Web did not execute project tests, plan builders, control computation, model,
scorer or GPU. Static source/AST/JSON/diff checks only; the delivery is
`generated_unexecuted`. Next executable action is the current CPU acceptance
plan, followed by the new CPU plan over the exact retained real sequence with an
explicitly frozen positive weight. Official scoring and all GPU work remain
stopped. C13 still lacks its group-l2 solver, primal/dual certificate,
multi-arm native collection and natural event-preservation evidence. The replay
XYZ change is an API-shape normalization only: direct mode ignores the vertex
argument, so its numerical equivalence remains a Local replay obligation.

## Continuation: paired-context consumer — 2026-10-08

Resumed exact remote main `b4923b4d3a1efd3936982ded85f8be2a1bfa05af`.
Natural Gate 0/IPCG remain open for all selected candidates, so no candidate was
promoted or implemented out of order. Continued the independent P0 interface
work needed by C01 and later methods: downstream tasks can now consume a terminal
paired-context bundle without access to the producer's live workspace.

The new consumer pins and rehashes `result.json`, `raw-manifest.json` and
`raw-evidence.tar` as three explicit harness-staged paths, snapshots those staged
bytes, and binds the expected producer UID/GPU/generation identity/source-time
mode. It verifies the exact unqualified result/replay schemas and matching
comparisons, checks every required regular tar member and its size/hash, applies
frozen archive/metadata/file/per-member/total expansion ceilings plus pre-copy free-space
preflight, rejects path escape/link/sparse/special/duplicate/order drift, and extracts with controlled exclusive writes
into a new root. A CPU-only one-attempt/zero-retry harness plan declares all
inputs and extracted outputs. Replay stage status now preserves a comparison
mismatch instead of unconditionally reporting completed.

Two independent static reviews caught and drove closure of the initial live
parent-directory staging bug, scope overclaim, reopen race and missing archive/
expansion limits. The resulting tests exercise actual installed staging and controller-copy
mutation but remain authored/unexecuted. Web performed only static source,
AST/JSON/hash/diff review; no plan builder, project test, archive consumption,
model, scorer or GPU was run. Local next runs common CPU acceptance, then the
new consumption plan against an actual receipt-bound three-file bundle. This
does not change 0/15 candidate source completion or 0/15 native results.

## Continuation: C13 group-acceleration candidate core — 2026-10-08

Resumed literal remote main `30ceaa06d0426aeda7ce6de591182c666e7af689`
and preserved the revised 15-candidate identities/order. C13 was selected for
the next legal source increment because its complete native mesh/timestamp
boundary already exists and no sparse mode, Markov kernel, GT/event label or
hidden decoder input must be invented.

Authored the reviewed group-l2 acceleration solver with an explicit shared
temporal SPD metric, retained off-diagonal anchor/free terms, exact whole-frame
anchor elimination, true 3D group proximal/dual balls, scaled-ADMM residuals and
a dual-feasible Fenchel lower bound/gap. Added a fail-closed full-sequence and
certificate exporter plus a one-attempt/zero-retry/zero-GPU plan builder using
the existing harness. The source preserves all 16 frames, topology, vertex
mapping, non-vertex arrays and float32 official-export dtype. Identity metric is
the explicit native specialization; a reviewed external metric is a separately
hash-pinned NPZ input. Supplied `0..15` native loader time is not described as
physical video wall-clock time.

Authored acceptance coverage before implementation for affine fixed points,
anchor exactness, 3D rotation equivariance, non-diagonal SPD influence, dual
gap/feasibility, malformed inputs, stale hashes, single-use output and plan
closure. Web ran only AST/diff/static checks, not project tests, builders, solver,
model, scorer or GPU. The implementation and tests are
`generated_unexecuted`; Local acceptance is pending.

Updated the maintained 15-item table and Local handoff. C13 no longer lacks its
core solver, primal/dual diagnostics or one-arm complete-sequence export, but it
still lacks a candidate-specific hash-bound B0/B*/Gaussian/quadratic/group
native scoring/collection plan, observed Local execution, Natural Gate 0/IPCG,
natural event evidence, frozen parameters/statistics and official results.
Accordingly `full_method_source_complete=false`, complete candidates remain
0/15 and native candidate results remain 0/15. GPU STOP remains effective.

## Continuation: C13 five-role comparison request — 2026-10-08

Resumed literal remote main `7266cd5441266dab03a5e728cc3ef0d9a429a306`.
Two independent read-only reviews confirmed that current evidence still lacks
Natural Gate 0/IPCG, a prospectively frozen strongest B*, actual family split,
numeric effect/noninferiority criteria and a design-verified candidate protocol.
Those absences block scientific dispatch, but not the engineering request layer.

Added `research_math.c13_native_comparison`. Its request operation consumes an
externally frozen ordered B0/B*/Gaussian/quadratic/group inventory, pins and
rehashes every retained report/sequence plus the group certificate, and checks
the shared UID, inference seed, original source bytes, 16 float32 frames,
topology, timestamps, vertex identity, extra metadata and exact frame-zero
anchor. B0 must be the exact source sequence. B* must have a separate
prospective decision with no C13 score-derived selection; an explicit alias to
an already declared physical arm shares one case instead of creating a false
independent observation. A preparation error retains its logical role and
report while contributing no physical scoring case. The request keeps a fixed
five-role logical denominator and explicitly forbids zero imputation; result
validation, contrasts, confidence intervals and verdicts are not implemented in
this engineering-only layer.

Acceptance tests were authored first for the freeze, alias, source/native
identity, physical duplicate, topology/timeline and failure-denominator contracts. In accordance
with the Web execution boundary they were not run; Web performed only AST,
JSON, diff and source inspection. Source remains `generated_unexecuted` and the
request is hard-coded `dispatch_ready=false`. No scientific scorer plan, project
test, mesh generation, metric, model, supervisor or GPU task ran. The next
implementation gap is the admitted official runner plus receipt-bound raw
collection, but its scientific plan must continue to fail closed until the
named gate/design/protocol artifacts exist. Candidate completeness and native
results remain 0/15; GPU STOP remains effective.

## Continuation: C13 official scoring/collection chain — 2026-10-08

Resumed literal remote main `b90a853a03abf4d569ba7773c489e712ea147b95` and
continued the concrete C13 gap rather than repeating historical pricing/root
repairs. Added `research_math.c13_native_scoring`, its source-level behavior
tests, and `prepare_c13_native_scoring.py` with plan-boundary tests.

The runner creates a second hash-bound request for the released population, GT,
current adapter, deterministic CPU-kNN entry and six official source files. At
execution it stages each unique physical case once, invokes the existing
official adapter once, validates source/compatibility/seed/device/16-frame/input/
output identities, preserves all failed cases, expands B* aliases without a
second measurement, and computes only available descriptive group-minus-control
deltas. It never imputes a failed metric, computes a confidence interval, or
issues a scientific verdict.

Collection now has a deterministic regular-file tar, sorted SHA-256 inventory,
fixed file/member/expanded/archive/metadata limits, exact request/comparison/GT
cross-links and a standalone delivery validator. The released GT is referenced
but not redistributed. A scorer error or interruption leaves its actual partial
raw tree for collection; the same experiment is not silently retried.

The plan builder uses the installed method verifier at C13 `before dispatch`,
the installed native protocol verifier, the exact five-role contract, base plus
selected one-UID sample/GT/scorer closure, strict native dependency lock and
actual physical GPU UUID. Independent plan review found that the first draft
used a non-contract arm role, downgraded confirmation evidence, omitted the base
sample when a selected sample existed, and described admission without enforcing
it. The repaired boundary now binds every arm revision/implementation to the
frozen comparison, preserves the protocol's confirmation mode, emits one
`treatment` confirmation attempt with zero retry, and requires one hash-bound
admission containing Gate 0 PASS, compatible IPCG, a nonoverlapping independent-
family split, positive/numeric outcome criteria and explicit exact-UID/GPU
single-attempt resume authorization. With GPU STOP active that authorization is
absent, so no runnable digest can be emitted. Only `result.json`,
`raw-manifest.json` and `raw-evidence.tar` form the returned interface.

Tests were authored before production source but could not be run under the Web
execution boundary; therefore there is no RED/GREEN evidence and no test-pass
claim. Static AST/JSON/diff inspection only; no project code, model, scorer or
GPU ran. C13 is now source-chain complete but `generated_unexecuted` and Local
unverified. Source-chain complete candidates are 1/15; Local-verified candidates
and native candidate results remain 0/15. GPU STOP remains effective.

## Continuation: C14 corotational residual candidate core — 2026-10-08

Resumed literal remote main `cc84100d01cdbab1fa0744d2cd87654354604621`
without repeating the completed C13 scoring work. The maintained selection ranks
C14 fifth. Its math card explicitly requires frozen pose factoring plus a
declared zero-preserving residual repair; the pre-existing `smooth_body` Gaussian
was only the named strong simple control and therefore did not count as C14.

Authored `research_math.corotational_residual_candidate`. It estimates proper
uniform-vertex Kabsch factors from the predicted complete mesh relative to
centered frame zero, rejects rank/reflection ambiguity, freezes those factors,
and solves an explicit anchored nonuniform-time XYZ-group-TV problem for the
body residual. Reconstruction uses the unchanged frozen poses and preserves
frame zero, all 16 frames, topology, vertex identity, non-vertex arrays and the
float32 official-export dtype. Inputs exclude GT, cameras, labels, evaluator ICP,
scorer state and learned weights. World/body Gaussian controls remain separate.

The exporter retains pose singular values/RMS, observed and repaired residuals,
ADMM objective/residual/tolerance/conditioning/termination evidence, source/code
hashes and a one-case candidate manifest. A nonconverged solve is retained as an
incomplete arm without substituting the Gaussian control. Added
`prepare_corotational_residual_candidate.py`, a single-use existing-harness CPU
plan with explicit parameters, one attempt, zero retry and zero GPUs. The common
acceptance closure now includes the builder.

Tests were authored before production source for rigid fixed points, group-TV
method identity, SO(3)/rank refusal, complete identity/hash export, failed
nonconvergence and plan resource/retry bounds. Web performed static AST/JSON/
diff inspection only; it did not run project tests, the builder, solver, model,
scorer or GPU. All source is `generated_unexecuted`/`authored_not_run`.

C14 still lacks its prospective B0/B*/world-Gaussian/body-Gaussian/candidate
comparison freeze and candidate-specific official-scoring/raw-collection plan,
as well as Local acceptance, Natural Gate 0/IPCG, family split, numeric criteria
and native results. It therefore remains source-incomplete. Source-chain complete
candidates stay 1/15 (C13 only), Local-verified and native results stay 0/15,
and GPU STOP remains effective.

## Continuation: C14 official scoring/collection chain — 2026-10-08

Resumed literal remote main `6a1149f5f1853d07404ab4089445c0143dfacecf`
and closed the concrete source gap named above. Added the C14-specific prospective
five-role comparison, official ActionBench runner/raw collector, fail-closed
common-harness plan builder and source-level acceptance tests. The five roles are
B0, B*, world Gaussian, body Gaussian and the actual corotational residual
candidate; body Gaussian is never treated as the candidate.

The comparison pins complete native identity, current role implementations,
body-pose evidence and a recomputed candidate pose/solver certificate. The
scorer de-duplicates every byte-identical physical sequence, preserves all failed
roles in the denominator, binds the admitted ActionBench snapshot/semantics/GT,
validates all 16 exported GLBs and returns a
bounded tamper-evident three-file bundle without a confidence interval or verdict.
The plan builder requires current design verification, installed-schema-verified
Gate 0/IPCG, a source-derived independent-family split, protocol-bound prospective
numeric effect/noninferiority criteria, exact arm implementations, strict runtime
closure and an expiring exact-request/run/environment/GPU authorization that is
reserved once, recoverable only for identical partial plan state, validated
against the installed native/harness schemas and finalized against both plan
digests. A separate immutable launch ticket is a declared staged input, avoiding
self-reference while binding the request/run/GPU/expiry; the scorer gates all
output/GPU activity on that staged ticket and its authorization closure. The
controller may launch only through `launch_c14_native_scoring.py`, which validates
the final consumption and installed plan pair, holds one owner lock and records
an exact recoverable claim, injects its hash/path for scorer verification, then
delegates to the existing harness. The
protocol still permits one confirmation attempt and zero retry.

Tests were authored before production source but Web did not run them, so no
RED/GREEN or pass claim exists. Static AST/JSON/hash/source review only; no project
test, builder, scorer, model or GPU ran. C14 is source-chain complete but
`generated_unexecuted`; Local acceptance and all scientific prerequisites/results
remain open. Source-complete candidates are now 2/15 (C13 and C14), Local-verified
and native results remain 0/15, and GPU STOP remains effective.

## Continuation: C02 full source chain — 2026-10-08

Resumed exact remote main `4aaaf0bcb65f45be03d17e509d8c2d5003e2e387`.
Implemented the missing C02 construction rather than wrapping the old dense
`project_protected_step(d,C,W)` primitive. The new candidate derives its shared
matrix-free ARAP repair, fixed topology-geodesic patches, barycentric-area W and
patch-centroid-velocity protection from a receipt-bound predicted native mesh,
then exports geometry-only, equal-W-norm scalar and protected complete sequences
plus a recomputable NPZ of areas/patches/three steps; rank/nullity,
orthogonality/rho/local gain and enforced float32 diagnostics live in each
role report and are independently recomputed at freeze time.

Added its CPU artifact builder, prospective five-role comparison, official
ActionBench request/scorer/raw-archive profile, authorization-gated plan builder
and stable single-owner launcher. Parameterized only C14's scorer transport,
archive and recovery substrate; C02 method construction/certification and roles
remain candidate-specific. Added source-level Local acceptance for projection,
patching, native identity and a small real candidate-to-freeze path. Updated the
common CPU acceptance closure, README/AGENTS/runbook/current handoff and 15-item
inventory.

Web performed AST/JSON/diff/source review only and did not import/run the project,
tests, builder, ARAP solver, model, scorer or GPU. The source remains
`generated_unexecuted`. No scientific plan was emitted because C02 still lacks
Local acceptance, design verification, Gate 0/IPCG, prospective B*, actual family
records, numeric G01 criteria, strict live runtime admission and exact-attempt GPU
resume authorization. Source-complete candidates are 3/15 (C02, C13, C14);
Local-verified and native results remain 0/15. GPU STOP remains effective and no
supervisor process was installed or started.

## Continuation: C01 same-context source chain — 2026-10-08

Resumed literal main `7e03902dc592e88545a182dfd37e3c5b3ef33dca` in an isolated
source worktree. The installed read-only method evidence validator reported the
unchanged 20-card pool/15 selected and the C01 code-generation boundary ready;
it granted no scientific authorization. Re-read the actual ActionMesh direct
clamp/query semantics, native capture/replay, official motion-Chamfer source,
ActionMesh Section 3.3 and the retained SFG collision notes. Original C01 math
and ranking were preserved. C01 remains a diagnostic/baseline construction,
with no novelty or native performance claim.

Added the receipt-bound complete native-context consumer/corrector, raw and
mean-bias controls, exact source anchor and 16-frame export, explicit no-clipping
bounds policy, certificate reconstruction and CPU artifact plan. Native query
casting is part of the implemented decoder definition and is retained explicitly.
The fifth raw-uncorrected role isolates the native clamp from bias subtraction;
this extends the four-role draft without removing a comparison. Added prospective
freeze assembly, official scoring/raw collection, full archived native context
and failed reports, authorization-gated plan/launcher and a dedicated retained-
native CPU acceptance builder. Every actual input file is explicitly staged.

Independent static review identified and repaired inherited shared C01/C02/C14
profile-import omissions, missing deterministic-kNN staging, and a circular
protocol/analysis hash binding. The analysis now binds a protocol-core digest;
the final protocol pins the analysis and downstream criteria/authorization pin
the full protocol. No existing run or frozen protocol was rewritten. Added
isolated import and real admission-record construction test source, plus actual
capture/replay/consume/_stage/candidate software integration and retained-native
comparison acceptance source. These tests were authored, not run.

Historical R3's 311/311 receipt and R9's nine completed baseline units were
restored as historical scoped evidence. They do not cover this source. The
historical pricing path fix is already present; indices10–15 remain a separate
no-omission obligation and UID008 still lacks its returned raw package/trusted
replay. No experiment was repeated or launched.

Current source-chain count is 4/15 (C01,C02,C13,C14), Local-verified candidates
0/15 and native candidate results 0/15. C01's current producer scope is seed42,
one complete 16-frame context; original proposed seeds314/2718 and complete G01
remain additional source/design qualification obligations. Eleven candidate
source chains and stage-wide G01/supervisor acceptance obligations remain.
GPU STOP is unchanged; no supervisor installed/started. Existing host is hourly
automation `6ac527ad50cc8191a4964d35ae0b826e`, not a verified continuously running
process. Continue the retained selection at C10's legal differential target/common
lift/pinned solver or another independently ready missing component; do not
restart ranking or count Local-unexecuted source as method verification.

## Continuation: C10 integrable-gradient full source chain — 2026-10-08

Resumed literal main `4d5269a3e8815089d4d6a60769a8d762374ae4e8` and preserved
the existing 20-card pool, 15 selected identities and priority order. The
installed read-only method validator reported math20/20, selected15 and the C10
code boundary ready; code/design/results authorization remained absent. Primary
source review confirmed that gradient-domain/Poisson editing and ARAP are
classical solver classes, so this delivery makes no novelty claim.

Implemented a geometry/topology-only common differential target from one exact
receipt-bound native prediction. Edge orientation, anchor-derived positive
weights and per-component lowest-ID pins are deterministic; `h[0]=0` and the
same W is held across all 16 frames. Three distinct physical constructions now
exist: spanning-forest direct lift, qualified per-face proper-rotation ARAP with
independent local reconstruction, and the actual C10 matrix-free pinned weighted
Poisson solve. All complete arms preserve every non-vertex source array, exact
frame zero and pins, and recompute projection certificates from the final
float32 output. No clipping or hidden fallback converts a method failure into a
result.

Added the zero-GPU, zero-retry artifact plan with a fixed bounded deterministic
archive so terminal failed arms can be collected successfully while retaining
candidate `status=incomplete`. Validation rejects source symlinks, requires the
sibling completed report, binds the installed implementation and exactly matches
embedded arms to physical reports. Added the prospective five-role comparison,
qualified-ARAP method identity, physical content deduplication, failed-role
denominator, official ActionBench scoring/raw closure, full frozen-input
snapshot, admission-gated plan and stable single-owner launcher.

Independent static review closed earlier frame-zero, fixed-W, final-float32,
failed-artifact settlement, exact-report, source-path and qualified-ARAP issues.
Only AST/JSON/diff/source inspection was performed. No project test, builder,
solver, model, scorer, download, inference or GPU task ran. C10 is therefore
source-chain complete but `generated_unexecuted`; Local acceptance and native
result remain absent.

The maintained count is now 5/15 source-complete (C01,C02,C10,C13,C14), 0/15
Local-verified and 0/15 native results. Ten method source chains, complete G01
and the persistent supervisor continuation layer remain. The current supervisor
source is only a finite pre-frozen campaign controller: per-item missing-input
isolation, periodic owner/status heartbeat, GPU-idle anomaly observation,
lost-ack restart reconciliation, safe stop/resume/status commands, continued
waiting for newly ready reviewed items and non-mocked recovery coverage remain
open. It is not installed or running. GPU STOP and the frozen budget/protocol
remain unchanged; continue the next independently ready method (C04).

## Continuation: C04 robust motion source chain and bounded supervisor recovery — 2026-10-08

Resumed exact main `fd4c69055fbd827a91c3f21181b480188bf947ff`; every reused local
source blob matched its remote tree. Restored current AGENTS/runbook/handoff,
R3's historical 311/311 software receipt and R9's terminal nine baseline units.
The historical pricing-root implementation is present; no duplicate repair or
baseline/scorer run was launched. Indices10–15 and UID008 raw/trusted replay
remain their existing obligations. Read-only current method-evidence validation
confirmed the unchanged math20/20 pool, selected15 and C04 selection boundary;
it granted no scientific admission. Primary robust-optimization text, ActionMesh
Section3.3 and the actual native benchmark/chamfer sources were jointly read.

C04 now constructs shared ARAP d from the real native mesh, an exact kinetic
surrogate gradient/Hessian, a declared strictly positive diagonal sensitivity
ellipsoid and its anchored Euclidean conic projection. Projection gap and
stationarity precede separately certified float32 trust/backtracking. The
three complete physical arms preserve topology/time/vertex identity and exact
frame0; scalar strength agreement is explicitly tolerance-qualified. Structured
rejection history and construction/export failures remain in the terminal
archive and five-role denominator. Prospective B*, official scorer/raw closure,
exact-attempt admission/launcher and real-input CPU acceptance are connected.
Semantic review is bound to prospective method parameters and actual held-out
development receipts, not future confirmation outputs; probability/native
safety claims remain unavailable. C04 is only this positive-diagonal
specialization, not a generic PSD optimizer or qualified scientific result.

Also repaired inherited source integration defects: C10 now stages its terminal
archive pair; shared B0 contract validation binds the actual explicit generator
source and available producer/command evidence instead of demanding a candidate-
only report field absent from native reports. Non-B0 report hashes remain strict.
Corrected the C10 scorer-request CLI card to its real --comparison/data/source
arguments. These repairs have only source review, not new execution evidence.

Independent static reviews repaired lost rejection/export evidence, omitted
ARAP import and archive staging closure, semantic receipt/UID/parameter/time
bindings, and STOP/resume/readiness races. The supervisor now waits within a
fixed immutable campaign for exact known inputs, isolates blocked dependencies,
retains heartbeat/status and process identity, and reconciles lost acknowledgement
without reissuing native trials. It remains finite: reviewed append-only task
extension and configured actual runtime/online repair-agent integration are
unfinished. No supervisor was installed or started. Authored recovery tests use
the actual harness for two batches, failure continuation, input arrival,
interruption and lost acknowledgement; none were executed.

Source-chain delivery is 6/15 (C01,C02,C04,C10,C13,C14); Local acceptance and
native candidate results remain 0/15. Nine source chains, full G01 numeric/split/
qualification closure, broader producer seeds and supervisor integration remain.
Only source/AST/JSON/diff/hash checks were performed. GPU STOP, cumulative budget
and one-attempt/zero-retry scientific protocols are unchanged. The actual host
is existing hourly task `6ac527ad50cc8191a4964d35ae0b826e`, conversation
`6ac0b6a1-5e28-83e9-98d6-30f1536deef1`; a continuous hosted process is not verified.
Next source work follows retained C03, then independent ready selected methods.
Local starts at the C04 runbook and supervisor handoff at the eventual exact
read-back commit, preserving all failed/live evidence and returning actual logs.
