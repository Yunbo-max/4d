# Local Agent Runbook — 4D baseline/native qualification

## Scope and current truth

### Current source delivery — paired replay and supervisor

### C03 development calibration core — partial source, not a native chain

Current source is `generated_unexecuted`. Entry:
`actionmesh/prepare_c03_calibration.py`; numerical core:
`research_math/correlated_calibration.py`; artifact boundary:
`research_math/c03_calibration_artifacts.py`. This is a real weighted affine
solver, not the historical `correlated.py` synthetic demonstration. Do not run
that demonstration as C03 acceptance.

The core implements the revised `20261006-native-loss/c03-math.json` s8–s12:
full, diagonal and intercept-only fits, each with squared or smoothed vector-norm
loss; unpenalized intercept; positive slope ridge; fixed positive tau; explicit
weights; exact free-gradient convergence; actual and quadratic-surrogate descent.
It retains rejected/nonconverged frame fits and never substitutes another arm.
The squared convention is one-half weighted squared error plus one-half ridge;
the smoothed convention is the weighted smoothed norm plus one-half ridge.
Weights are not renormalized. Label fitting is data calibration, not zero-shot.

`fit_bundle` produces six roles × fifteen nonanchor frame fits with byte-bound
policy/data/correspondence evidence and disjoint UID/family partitions.
`apply_bundle_arrays` takes only raw 16-frame coordinates, the observable
same-context residual, original anchor and frozen bundle. It has no GT input or
refitting route. It preserves the anchor exactly in float64 and retains each
application failure independently. This array API is not yet an admitted
C01-context-to-native-export adapter.

**Remaining C03 source gaps:** a scientifically qualified development
correspondence/label-bank producer from actual ActionMesh queries and tracked GT;
receipt-bound C01 context-to-C03 complete mesh export with float32 and bounds
checks; nine-role B0/B*/six fits/unit-C01 prospective freeze, native scorer and
raw collection; retained-native acceptance. Do not fill these gaps with arbitrary
nearest neighbours or claim official ICP supplies material correspondence. This
partial delivery does not increase the full-source-chain count (still 6/15).
Native Gate 0/IPCG, numeric criteria, family audit and G01 remain pending.

First run the existing CPU software-acceptance builder at this exact delivered
revision, using a new unique run ID and the emitted approved-plan digest. Its
source closure now includes the C03 builder and both new test modules. New tests
cover independent normal-equation expectations, restricted vector IRLS, failure
and serialization behavior, declared split rejection and the actual `_stage`
operation followed by a fresh staged subprocess after removal of live fixture
inputs. These are software fixtures only, not native data or native evidence.
None have been executed by Web.

For later real development-bank fitting, the input format is deliberately
explicit. NPZ keys are `reference_residual[N,3]`, `labeled_error[15,N,3]`,
`weights[N]` and Unicode `uids[N]`. Each declared development UID must have
positive weight; rows must contain exactly the prospective development partition.
The policy keys and validation live in `validate_policy`, including seed42,
frames0..15, `plain_pointwise_no_asset_specific_icp`, disjoint `uid_to_family`,
all five estimator parameters, bank SHA-256 and named correspondence-evidence
SHA-256 values. Every evidence file must be an explicit input argument. There is
no generated native bank or approved policy in this delivery; retain that blocker
rather than manufacture one. The source checks metadata and hashes, not the
scientific truth of a correspondence review.

Once the real bank and its reviewed policy exist, on the existing GPU host with
native Conda, resolve `c03_data`, `c03_policy`, `c03_evidence_file` to those exact
retained files under `project_dir`, and `c03_evidence_name` to its exact policy
key. Resolve `c03_wall_seconds`/`c03_ram_mib` within the existing remaining budget.
For multiple evidence files repeat the real `--evidence-file NAME PATH` arguments.
The following emits a one-attempt, zero-retry, zero-GPU plan; it does not execute:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_c03_calibration.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --data "$c03_data" --policy "$c03_policy" \
  --evidence-file "$c03_evidence_name" "$c03_evidence_file" \
  --run-id "$c03_run_id" --plan-dir "$project_dir/plans/$c03_run_id" \
  --wall-seconds "$c03_wall_seconds" --ram-mib "$c03_ram_mib"
```

After source review, use only the existing `run_harness.py` with this plan and
its actual emitted digest, under the existing Local authority and budget. Do not
call the inner fitter directly. The declared receipt output is
`actionmesh/c03-calibration-output/fit-bundle.json` inside the actual attempt
workspace; derive its absolute path from the harness receipt, not a guessed UUID.
On failure preserve this JSON, stderr/stdout and exact input/code refs. Parameter
or label changes require a new version; do not retry a frozen scientific attempt.
Return those raw records plus execution SHA to the current round packet. Inspect
`role_results[role].failures` and `diagnostics.history` for nonconvergence; never
raise tolerances or switch loss silently. A bank/hash/split rejection requires
restoring the exact intended bytes or scientific review of a versioned change.
No model download, scorer, supervisor installation or GPU resume is authorized
by this source handoff. Existing asset acquisition cards below remain unchanged.

### C01 same-context correction and retained-native acceptance

The C01 source chain uses the frozen native 16-frame, seed-42 context. It is
**generated_unexecuted**. It adds no GPU authorization, learned weights or data
source. Obtain its inputs through the existing pinned ActionMesh/ActionBench
[asset acquisition](#download-datasets-and-models), paired context producer with
source-time query enabled, and receipt-bound bundle consumer. Keep GPU STOP:
if that real producer bundle does not yet exist, this input-dependent acceptance
waits; source/software work for independent candidates continues.

The original native float32 anchor is X. The recorded query may cast X to the
model dtype; define F_t(X) as that exact composite native query. C01 outputs
X+F_t(X)-F_a(X), keeping frame zero exactly X and all original faces, vertex IDs
and frame IDs. The mean control subtracts mean_v(F_a-X) from target coordinates.
A raw-uncorrected arm is necessary because original B0 clamps coordinates but
neither the correction nor its controls silently clip. The frozen five roles are
`b0,b_star,raw_uncorrected,mean_bias,self_map_subtraction`. B* is selected without
C01 outcomes; aliases may only reference simple arms. CD-M is the official
first-frame-correspondence coordinate metric, not a velocity loss. Constant
per-vertex bias subtraction preserves target-target vertex differences; it does
not mathematically guarantee any Chamfer improvement or establish novelty.

First run the common current-source CPU software acceptance. Its engineering
capture fixture exercises the actual observer/replay/transport and installed
cross-root staging boundary, deletes live input files and checks that correction
uses only staged bytes. This is software evidence, not a native experiment.
The optional retained-native test is skipped by the common suite until its
explicit real input is supplied; a skip never counts as native acceptance.

After a completed producer receipt and successful CPU consumption exist, retain
the whole `context-consumed` directory returned by that attempt under the project
root. Set `c01_context_root` to its actual copied location; bind the recorded
result/manifest/archive hashes to the producer receipt using the existing bundle
consumption procedure. Use the original frozen bounds `-1 1` and prospectively
choose `preserve_and_report` or `reject`; neither clips. Set CPU time/RAM from the
remaining authorized budget and actual host, not from a guessed GPU profile.
Emit the one-attempt, zero-retry CPU artifact plan:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_self_map_candidate.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --context-root "$c01_context_root" \
  --plan-dir "$project_dir/plans/c01-artifacts-001" --run-id c01-artifacts-001 \
  --coordinate-bounds -1 1 --bounds-policy "$c01_frozen_bounds_policy" \
  --wall-seconds "$c01_cpu_seconds" --ram-mib "$c01_cpu_ram_mib"
```

Inspect its printed digest and run only through the installed `run_harness.py`.
Each raw file is its own declared, rewritten input argument. Return the complete
`actionmesh/c01-self-map-output` from the actual attempt: candidate/manifest,
certificate, retained context, and every role's report/sequence. Rejected roles
retain an error report and no scoreable sequence; the failed attempt remains
failed, with no fallback or automatic retry. Copy its actual files back before
using them for comparison; never fabricate a successful receipt for a partial
attempt. Set `c01_candidate_json` to the returned `candidate.json` under the
project root. Run the dedicated real-input CPU acceptance plan:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_c01_native_acceptance.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --artifact-candidate "$c01_candidate_json" \
  --plan-dir "$project_dir/plans/c01-native-acceptance-001" \
  --run-id c01-native-acceptance-001 \
  --wall-seconds "$c01_acceptance_seconds" --ram-mib "$c01_cpu_ram_mib"
```

Dispatch only its printed CPU harness digest. This plan stages the retained real
artifact/context bytes and explicitly selects the real-input test; it must not
skip. It verifies correction/certificate reconstruction and full comparison
reassembly across a different root using the installed staging implementation.
Its engineering B* choice is only a test fixture; it cannot supply a scientific
B* decision. Preserve the exact tested commit, plan, attempt and stdout/stderr.

For the actual prospective comparison create the five-role
`c01-native-comparison-freeze` described by `c01_native_comparison.py`. All three
operation arms share one certificate. B0 uses the context's exact
`raw/observed/sequence.npz`, report and adjacent generation identity; its source
implementation ref is `native_context_runner.py`. A failure has a null sequence
ref plus its actual bounded error report. Then assemble the official request:

```bash
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c01_native_comparison request \
  --root "$project_dir" --freeze "$project_dir/inputs/c01/comparison-freeze.json" \
  --output "$project_dir/inputs/c01/comparison-request.json"
"$python_bin" -m research_math.c01_native_scoring request \
  --root "$project_dir" --comparison "$project_dir/inputs/c01/comparison-request.json" \
  --ground-truth "$dataset_root/data/$c01_uid/surfaces.npy" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --dataset-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --repo-root "$actionmesh_source" --timeout-seconds "$frozen_per_case_timeout" \
  --output "$project_dir/inputs/c01/scoring-request.json"
```

The comparison assembler recomputes the method from real retained tensors. To
run the actual prospective freeze through the harness, repeat the dedicated
acceptance builder with a fresh stable run ID and `--freeze
"$project_dir/inputs/c01/comparison-freeze.json"`. It pins that freeze, decision
and every referenced input, runs the real-input acceptance, and then writes
`actionmesh/c01-comparison-request.json` as a declared output. Copy the exact
receipt-bound request into `inputs/c01/comparison-request.json` before assembling
the scorer request. The comparison CLI above is the inner operation, not a
request for Local to design a new job or run a method outside the harness.
Official scoring and its plan/launcher are **blocked by GPU STOP**. Future
admission requires current design verification, Natural Gate 0/IPCG, independent
source-derived families, prospective B*, exact protocol-bound numeric cd_motion
minimum effect and cd_3d/cd_4d NI margins, strict environment closure, and an
expiring `single_c01_scoring_attempt` authorization for the exact GPU/run/request.
Only after those exist, use `prepare_c01_native_scoring.py` with the same
`--root --request --protocol --method-batch --environment --admission --skill-dir
--plan-dir --run-id --group --gpu-uuid --wall-seconds --ram-mib --cpu-cores` flags as
the C02 command below, then `launch_c01_native_scoring.py --root ... --consumption
... --skill-dir ... --approved-plan-digest ...`. Never call `score` directly or
bypass the controller via the generic harness entry.

The receipt-bound scoring outputs are `result.json`, `raw-manifest.json` and
`raw-evidence.tar`. The C01 archive includes every frozen context input and failed
role report under `frozen-inputs/` in addition to the complete official exports.
Validate them with `python -m research_math.c01_native_scoring validate-delivery`
using the exact `--root --request --gpu-uuid --result --manifest --archive
--expected-request-digest --expected-result-sha256 --expected-manifest-sha256
--expected-archive-sha256` arguments documented for C02 below. No individual-unit
readout is a confidence interval or scientific verdict.

The current conditional G01 draft is in the C01 specification's
`g01_authoring_draft`. Family memberships, numeric task effects/NI, measured
complete-unit cost and scientific gates remain unresolved. The original wider
seed policy `[42,314,2718]` remains a design obligation: this frozen producer only
implements seed 42, so 314/2718 require a separately reviewed producer extension,
not a silent seed substitution. One 16-frame window is the exact native scope;
changed/multiple windows are rejected rather than incorrectly stitched.

Debug from `candidate.json`, all role reports and the retained context. Hash or
replay mismatches require restoring the exact producer bytes, never relaxing the
tolerance. Bound rejection is a recorded preparation failure, not a reason to
clip. Wrong frame/window/query mapping requires repairing the producer adapter
and requalifying changed source. Keep old failed attempts and any live roots.

### Shared C01/C02/C14 protocol-analysis binding

New source uses an acyclic sequence: first author the protocol core, compute
`analysis_protocol_core_digest(protocol)` from the shared plan module, and put
that value in analysis.`protocol_core_digest`. It excludes only
`analysis_plan_ref`, `protocol_digest` and `frozen_at`. Then hash the final analysis
file into protocol.`analysis_plan_ref` and freeze the full protocol with the
installed canonical `protocol_hash`. Outcome criteria and GPU authorization
continue to bind that **final** protocol digest. Never insert the final protocol
digest into the analysis it hashes; that creates a cycle. Every scientific
criterion/split/identity remains checked. No already-run protocol was changed.
Shared profile loading now avoids importing another candidate before applying
its own profile, and the deterministic kNN implementation is explicitly staged.
All these repairs require new Local acceptance; historical passes do not cover them.

### C02 complete protected-geometry source chain

C02 now has a distinct end-to-end source path rather than only the dense
`protected_projection.py` reference operator. The method consumes a completed,
receipt-bound native `sequence.npz`/`report.json` pair. It uses no GT, scorer,
camera, video label or hidden model state. Frame-zero topology determines
deterministic geodesic patches and barycentric vertex areas; the shared
matrix-free ARAP repair is computed once, then the same desired step is exported
as geometry-only, equal-W-norm scalar and patch-centroid-velocity protected arms.
`certificate.npz` stores patches, areas and all three step arrays. Each role's
`report.json` stores rank/nullity, weighted orthogonality, rho, local-gain and
pre/post-float32 diagnostics, including the enforced native-coordinate tolerance.

All source and tests are **generated_unexecuted**. First run the common CPU
software acceptance at this exact revision and retain its observed results. Then
freeze every numerical setting on development inputs before emitting the CPU
artifact plan; there are no silent defaults:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_protected_geometry_candidate.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --source-sequence "$project_dir/inputs/original-case/sequence.npz" \
  --run-id c02-protected-artifacts-001 \
  --plan-dir "$project_dir/plans/c02-protected-artifacts-001" \
  --patch-count "$frozen_patch_count" \
  --arap-weight "$frozen_arap_weight" \
  --temporal-weight "$frozen_correction_velocity_weight" \
  --iterations "$frozen_arap_iterations" \
  --cg-tolerance "$frozen_cg_tolerance" \
  --cg-max-iterations "$frozen_cg_max_iterations" \
  --wall-seconds "$c02_cpu_wall_seconds"
```

Inspect the printed plan: exact sequence/report refs; candidate, ARAP and package
source refs; nine declared outputs; one CPU attempt; zero retries; zero GPUs.
Execute only its printed digest via installed `run_harness.py`. Preserve a common
repair failure as an incomplete attempt. Do not replace it with geometry-only,
the scalar control or `protected_projection.py`. On success retain
`candidate.json`, `manifest.json`, `certificate.npz` and all three role directories.

Before any scorer plan, create a prospective `c02-native-comparison-freeze`
with exact ordered roles `b0,b_star,geometry_only,strength_matched_blend,
protected_step`. Freeze B* without C02 outcomes; it may alias only B0,
geometry-only or strength-matched when method identity and bytes match. Each
physical role pins report, sequence and implementation; B0 additionally pins
the adjacent native producer `command.json`, whose UID/seed/script hash are
validated; all three C02-produced roles pin the same certificate. Then assemble
the CPU comparison request:

```bash
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c02_native_comparison request \
  --root "$project_dir" \
  --freeze "$project_dir/inputs/c02/c02-native-comparison-freeze.json" \
  --output "$project_dir/inputs/c02/c02-native-comparison-request.json"
```

This recomputes the common ARAP repair, patch construction and projection and
retains a five-role logical denominator while de-duplicating byte-identical
physical sequences. It does not score or admit the candidate. Next bind the
admitted ActionBench snapshot, semantics, population, exact UID GT and official
source closure, still without scoring:

```bash
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c02_native_scoring request \
  --root "$project_dir" \
  --comparison "$project_dir/inputs/c02/c02-native-comparison-request.json" \
  --ground-truth "$dataset_root/data/$c02_uid/surfaces.npy" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --dataset-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --repo-root "$actionmesh_source" \
  --timeout-seconds "$frozen_per_case_timeout" \
  --output "$project_dir/inputs/c02/c02-native-scoring-request.json"
```

GPU STOP forbids the following scientific builder/launcher today. After an
explicit future exact-attempt resume, Local must first supply current C02
`design_verified`, the frozen five-role native contract, Natural Gate 0 PASS,
compatible IPCG/importance decision, source-derived independent-family records,
protocol-bound positive absolute cd_3d effect, numeric cd_4d/cd_motion
noninferiority margins, strict environment/dependency closure, and an expiring
single-use authorization bound to the exact request/run/group/environment/GPU.
Only then use the following exact builder/controller flow; never invoke
`research_math.c02_native_scoring score` or the generic harness CLI directly:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_c02_native_scoring.py" \
  --root "$project_dir" \
  --request "$project_dir/inputs/c02/c02-native-scoring-request.json" \
  --protocol "$project_dir/inputs/c02/native-protocol.json" \
  --method-batch "$project_dir/inputs/c02/method-batch.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --admission "$project_dir/inputs/c02/scientific-dispatch-admission.json" \
  --skill-dir "$skill_dir" --plan-dir "$project_dir/plans/c02-native-001" \
  --run-id c02-native-001 --group "$c02_group" --gpu-uuid "$gpu_uuid" \
  --wall-seconds "$c02_wall_seconds" --ram-mib "$c02_ram_mib" \
  --cpu-cores "$c02_cpu_cores"

"$python_bin" "$project_dir/actionmesh/launch_c02_native_scoring.py" \
  --root "$project_dir" --consumption "$c02_final_consumption" \
  --skill-dir "$skill_dir" --approved-plan-digest "$c02_plan_digest"
```

After the three returned bytes are copied back, validate the exact delivery:

```bash
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c02_native_scoring validate-delivery \
  --root "$project_dir" \
  --request "$project_dir/inputs/c02/c02-native-scoring-request.json" \
  --gpu-uuid "$gpu_uuid" \
  --result "$c02_return/result.json" \
  --manifest "$c02_return/raw-manifest.json" \
  --archive "$c02_return/raw-evidence.tar" \
  --expected-request-digest "$c02_request_digest" \
  --expected-result-sha256 "$c02_result_sha256" \
  --expected-manifest-sha256 "$c02_manifest_sha256" \
  --expected-archive-sha256 "$c02_archive_sha256"
```

The plan has one confirmation attempt, zero retry,
and exactly `result.json`, `raw-manifest.json`, `raw-evidence.tar` as receipt
outputs. All 16 GLBs and archived reports/sequences/certificates must validate.
This is still not a confidence interval, scientific verdict or native result.

### Paired-context bundle consumption

The returned paired-context archive now has a fail-closed CPU consumer. Use it
only after the producing harness attempt is terminal and the three receipt output
hashes have been copied from that attempt. The builder itself revalidates the
complete tar/member inventory before it emits a plan; the inner task repeats the
same validation and writes a new single-use extraction. It never loads a model,
scores a mesh or changes qualification state.

```bash
"$python_bin" "$project_dir/actionmesh/prepare_native_context_consumption.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --bundle-root "$project_dir/inputs/native-context-bundle" \
  --run-id native-context-consume-001 \
  --plan-dir "$project_dir/plans/native-context-consume-001" \
  --expected-result-sha256 "$result_sha256" \
  --expected-manifest-sha256 "$manifest_sha256" \
  --expected-archive-sha256 "$archive_sha256" \
  --expected-uid "$producer_uid" \
  --expected-gpu-uuid "$producer_gpu_uuid" \
  --expected-generation-identity-sha256 "$generation_identity_sha256" \
  --expected-source-time-query true \
  --max-files "$frozen_max_files" \
  --max-unpacked-bytes "$frozen_max_unpacked_bytes" \
  --max-member-bytes "$frozen_max_member_bytes" \
  --max-archive-bytes "$frozen_max_archive_bytes" \
  --max-metadata-bytes "$frozen_max_metadata_bytes" \
  --wall-seconds 900
```

Take `producer_uid`, GPU UUID and the generation-identity hash from the retained
producer receipt/raw manifest review, not from a new run. Set source-time to the
exact producer setting. Freeze the five positive archive/metadata/expansion ceilings above the
observed signed inventory (without reducing or regenerating it), and record the
values in the Local receipt. The archive ceiling is checked before its private
snapshot is copied, and free space is preflighted for the bounded copy plus
extraction. The emitted command passes the three files
separately so the native staging layer rewrites every one to its immutable
attempt copy; a parent-directory argument is intentionally forbidden.

Inspect the emitted input/code refs, zero-GPU resources, one attempt, zero retry
and every declared `actionmesh/context-consumed/raw/...` output. Execute only the
printed digest through the installed `run_harness.py`. Promote the completed
attempt's `actionmesh/context-consumed/` directory by controller file transfer,
rehash it and record its receipt. `bundle-consumption.json` must retain
`native_context_qualified=false`, `scientific_effect_qualification=false`,
`replay_qualified=false`, `candidate_methods_tested=false` and
`dispatch_ready=false`. Any mismatch blocks this consumer only; keep
the original producer attempt immutable and do not regenerate it.

### Latest method-adjacent source — C13 quadratic strong control

The C13 specification's required quadratic-acceleration comparator now has an
independent CPU-only implementation and harness plan. It directly solves the
identity-metric control
`0.5||Y-Yhat||_F^2 + 0.5*weight||D2_timestamp Y||_F^2` with exact frame-zero
anchoring, uses strictly increasing timestamps in the supplied sequence units,
records that the weight scales in those units to the fourth power, and preserves the full 16-frame
topology/identity arrays and emits one manifest case consumable by the existing
official ActionBench adapter. This is a comparator only, not C13's group-l2
trend method and not scientific admission. Source and tests are
`generated_unexecuted`; Web did not run them.
The adapter rejects before output when the float64 solve exceeds its recorded
relative backward-error threshold `100*T*eps` or condition-number ceiling
`1/sqrt(eps)`; the observed values and limits remain in the harness failure log.

Because current sources changed, first run the CPU software acceptance in step
3 at the delivered revision and retain its actual result. Then use the controller
checks in step 4 to restore and independently verify the receipt-bound real
`sequence.npz` before emitting this plan. The builder itself pins only that
staged `sequence.npz` and its adjacent generator `report.json`; their agreement
does not independently prove the upstream receipt/protocol:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_quadratic_acceleration_control.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --source-sequence "$project_dir/inputs/original-case/sequence.npz" \
  --run-id c13-quadratic-control-001 \
  --plan-dir "$project_dir/plans/c13-quadratic-control-001" \
  --weight "$frozen_quadratic_weight" --wall-seconds 600
```

`frozen_quadratic_weight` has no silent default and depends on the fourth power
of the supplied timestamp unit: record both value and unit in the development
protocol before plan emission. Inspect the two pinned source refs,
two input refs, four declared outputs, zero-GPU resources, one attempt and zero
retries. Execute only the printed digest through `run_harness.py`. The output
path inside the completed attempt is
`actionmesh/quadratic-control-output/`; retain `controls.json`, `manifest.json`
and the arm report/sequence. Do not score or tune it while GPU STOP is in force.
After later native authorization and scorer qualification, the manifest is an
input to `official_actionbench_adapter.py`; official scoring is not included in
this CPU plan.

### C13 candidate source — anchored group acceleration

The distinct C13 candidate source is
`actionmesh/research_math/group_acceleration_candidate.py`; its plan-only entry
is `actionmesh/prepare_group_acceleration_candidate.py`. It implements the
reviewed convex group-l2 acceleration objective rather than reusing the
quadratic control. The scalable observation metric is an explicit temporal SPD
matrix shared over vertex/XYZ columns, so non-diagonal temporal and anchor/free
cross terms are retained without allocating a dense `3TV x 3TV` matrix. Whole
pinned frames are eliminated exactly. The solver exports a dual-feasible 3D
group-ball certificate, Fenchel lower bound/gap, ADMM residuals and termination
reason. The current native specialization deliberately freezes `M=I` and frame
zero; a separate metric NPZ is also supported and hash-pinned when a reviewed
metric exists.

This source is **generated_unexecuted**. First run the current common CPU
software-acceptance plan and retain its actual result. Then restore and verify a
receipt-bound real `sequence.npz`/`report.json` pair. Before plan emission,
freeze every positive parameter and the supplied timestamp-unit interpretation
in the development protocol; there are no silent solver defaults:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_group_acceleration_candidate.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --source-sequence "$project_dir/inputs/original-case/sequence.npz" \
  --identity-observation-metric \
  --run-id c13-group-candidate-001 \
  --plan-dir "$project_dir/plans/c13-group-candidate-001" \
  --group-weight "$frozen_group_weight" --rho "$frozen_admm_rho" \
  --absolute-tolerance "$frozen_abs_tolerance" \
  --relative-tolerance "$frozen_rel_tolerance" \
  --gap-tolerance "$frozen_gap_tolerance" \
  --max-iterations "$frozen_max_iterations" \
  --wall-seconds "$c13_cpu_wall_seconds"
```

For a reviewed external metric, replace `--identity-observation-metric` with
`--observation-metric "$project_dir/inputs/c13/metric.npz"`; that NPZ must
contain only `observation_metric[T,T]` and becomes a third input ref. Inspect
the printed plan for the exact sequence/report (and optional metric) refs,
three code refs, five declared outputs, zero GPUs, one attempt and zero retry.
Execute only its reviewed digest through `run_harness.py`. Retain
`candidate.json`, `manifest.json`, the complete group sequence, report and
`certificate.npz`. A nonconverged solver is an incomplete retained attempt; do
not substitute the quadratic/Gaussian arm or retry it as the same frozen trial.
The builder caps `c13_cpu_wall_seconds` at 26,940 so its 60-second outer harness
allowance keeps the complete CPU artifact plan within 27,000 seconds.

The emitted manifest contains only `group_acceleration`; existing native,
Gaussian, quadratic and prospectively frozen B* artifacts remain separate and
must first be combined by the candidate-specific engineering request assembler
below. The scientific scoring plan is not yet admissible. Do not call the official GPU
adapter while STOP is effective. CPU artifact completion would still not prove
Natural Gate 0, IPCG, event preservation, Local method verification or a native
effect.

### C13 five-role comparison request — engineering only

After the real B0/Gaussian/quadratic/group artifacts exist, create a prospective
`c13-native-comparison-freeze` JSON. It must have `version=1`, candidate ID
`4d-math-20261006-c13`, `frozen_at`, the exact UID/inference seed, scoring seed
44, primary `cd_motion`, guardrails `[cd_3d,cd_4d]`, exact source sequence/report
refs, and the ordered roles `b0,b_star,gaussian,quadratic_acceleration,
group_acceleration`. Each physical role supplies nonempty `method_id` plus exact
`report_ref` and `sequence_ref`; an observed preparation failure supplies its
report ref and `sequence_ref:null`. B* may instead carry only `alias_of` to a
named simple-control role (B0, Gaussian or quadratic). The separate
`b_star_decision_ref` uses exact kind/version/candidate/UID/inference-seed and
timezone-aware `decided_at`, names the selected role and method, records
`selected_without_c13_native_outcomes=true`, and includes nonempty pinned
`selection_basis_refs`; it carries its own canonical `decision_digest`. Compute
both digests over each object before adding its digest field. A physical B* must
have a matching report `method_id`, and byte-identical physical B* output must
instead use `alias_of`. Do not derive B* from official scores.

Then, inside the compatible Local environment and still without GPU execution:

```bash
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c13_native_comparison request \
  --root "$project_dir" \
  --freeze "$project_dir/inputs/c13/c13-native-comparison-freeze.json" \
  --output "$project_dir/inputs/c13/c13-native-comparison-request.json"
```

The command rehashes the complete closure, requires exact B0 source bytes,
checks each completed arm against the same 16-frame float32 topology, timeline,
vertex identity and frame-zero anchor, and requires the group certificate.
Aliased B* maps to the same physical case and does not create a second
measurement. A failed physical preparation remains a logical role with no
scoring case. The output must retain `generated_unexecuted=true`,
`native_qualified=false`, `scientific_verdict=not_computed` and
`dispatch_ready=false`.

This is not a harness execution plan and it does not call ActionBench. The later
scientific builder must additionally receive real Natural Gate 0/IPCG evidence,
a frozen Gate-A/G01 candidate protocol, verified method design, actual family
split/IDs, non-null effect and noninferiority criteria, fixed runtime policy and
Local GPU identity. Those artifacts do not currently exist; GPU STOP remains
effective.

### C13 official scoring and raw collection — source complete, not admitted

The candidate-specific scorer/collector is now
`actionmesh/research_math/c13_native_scoring.py`; its only plan entry is
`actionmesh/prepare_c13_native_scoring.py`. First create the scorer closure in
the compatible Local environment without executing the GPU scorer:

```bash
dataset_root="$project_dir/inputs/actionbench-2796071c"
source_root="$actionmesh_source"
population="$project_dir/actionmesh/research_overnight/assets/actionbench_population.json"
test -d "$dataset_root/data"
test -d "$source_root"
test -f "$population"

cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c13_native_scoring request \
  --root "$project_dir" \
  --comparison "$project_dir/inputs/c13/c13-native-comparison-request.json" \
  --ground-truth "$dataset_root/data/$c13_uid/surfaces.npy" \
  --population "$population" \
  --repo-root "$source_root" \
  --timeout-seconds "$frozen_per_case_timeout" \
  --output "$project_dir/inputs/c13/c13-native-scoring-request.json"
```

This revalidates the complete comparison request and pins the released
population/revision, exact UID GT, existing adapter, deterministic CPU-kNN
entry and all six official ActionBench source files. The per-case timeout times
the number of unique physical cases must be at most 27,000 seconds. The output
still has `generated_unexecuted=true`, `native_qualified=false`,
`scientific_verdict=not_computed` and `dispatch_ready=false`.

Do not build or run the scientific plan while GPU STOP is active. After an
explicit later resume, the actual C13 Natural Gate 0/IPCG, prospective B*,
family split/IDs, numeric effect/noninferiority criteria, method implementation
and design reviews must be present. Freeze them in one
`c13-scientific-dispatch-admission` record whose path/SHA references bind the
Natural Gate 0 record, importance decision, nonoverlapping family split,
positive primary effect plus both numeric guardrail margins, and an explicit
`single_c13_scoring_attempt` GPU-resume authorization. That authorization must
name the same candidate, UID, comparison, protocol digest, physical GPU UUID
and at least the requested wall budget. It is not permission to resume any
other task. Then use the exact native runtime receipt and physical GPU UUID:

The custom records use canonical SHA-256 of sorted compact JSON with their own
digest field omitted. `c13-family-split` version 1 carries `candidate_id`,
`benchmark_revision`, nonempty disjoint `development_ids` and
`confirmation_ids`, a complete `family_by_uid`, `frozen_at`, and
`split_digest`; no family may cross the two sets and the scoring UID is in the
confirmation set. `c13-outcome-criteria` version 1 carries `candidate_id`, the
comparison-file SHA as `comparison_digest`, `primary_metric=cd_motion`, ordered
`guardrail_metrics=[cd_3d,cd_4d]`, positive finite `min_effect`, finite
nonnegative margins for both guardrails, `independent_unit=asset_family`,
`frozen_at`, and `criteria_digest`. `c13-gpu-resume-authorization` version 1
carries the same candidate/UID/comparison, exact `protocol_digest`, GPU UUID,
`scope=single_c13_scoring_attempt`, integer `max_gpu_task_seconds`,
`status=authorized`, timezone-aware `authorized_at`, and
`authorization_digest`. The outer admission has an exact schema: kind/version,
candidate/UID/benchmark/comparison, exact protocol and method-batch refs, the
five evidence refs, `status=admitted`, timezone-aware `admitted_at`, and
`admission_digest`. Do not fabricate any of these records to satisfy the parser.

```bash
"$python_bin" "$project_dir/actionmesh/prepare_c13_native_scoring.py" \
  --root "$project_dir" \
  --request "$project_dir/inputs/c13/c13-native-scoring-request.json" \
  --protocol "$project_dir/inputs/c13/c13-native-protocol.json" \
  --method-batch "$project_dir/inputs/c13/c13-method-verification-batch.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --admission "$project_dir/inputs/c13/c13-scientific-dispatch-admission.json" \
  --skill-dir "$skill_dir" \
  --plan-dir "$project_dir/plans/c13-native-scoring-001" \
  --run-id c13-native-scoring-001 --group "$frozen_c13_group" \
  --gpu-uuid "$actual_gpu_uuid" --wall-seconds "$c13_scoring_wall_seconds" \
  --ram-mib "$confirmed_ram_mib" --cpu-cores "$confirmed_cpu_cores"
```

The builder calls the installed method boundary at `--before dispatch`, requires
the protocol's exact `group_acceleration` treatment, prospective `b_star`
baseline, and `b0`/Gaussian/quadratic controls, verifies the official scorer
source closure, binds each contract arm's revision/implementation to the frozen
comparison, validates the strict native dependency inventory, and reserves 300
seconds for validation/collection. It preserves the protocol's confirmation
mode and emits one treatment task, one confirmation attempt and zero retries.
The base sample manifest and any selected manifest are both retained. While
GPU STOP is active the required authorization is absent, so no runnable digest
can be emitted. Inspect both plan files and execute only
the printed digest through the installed `run_harness.py`; never invoke
`research_math.c13_native_scoring score` directly.

The retained attempt output is exactly:

- `actionmesh/c13-scoring-output/result.json`
- `actionmesh/c13-scoring-output/raw-manifest.json`
- `actionmesh/c13-scoring-output/raw-evidence.tar`

After the harness copies those three files out, take their exact SHA-256 values
from the retained attempt/output refs and validate the transported bytes before
opening or reporting them:

```bash
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c13_native_scoring validate-delivery \
  --result "$returned_c13_dir/result.json" \
  --manifest "$returned_c13_dir/raw-manifest.json" \
  --archive "$returned_c13_dir/raw-evidence.tar" \
  --expected-request-digest "$frozen_c13_request_digest" \
  --expected-result-sha256 "$receipt_result_sha256" \
  --expected-manifest-sha256 "$receipt_manifest_sha256" \
  --expected-archive-sha256 "$receipt_archive_sha256"
```

On failure, preserve the attempt directory. Inspect harness `state.json`,
`events.jsonl`, task stdout/stderr and the archived `adapter-execution.json`,
`validation.exception.log` and per-case official execution files. Do not rerun
the same frozen scientific attempt; a code repair requires a new source version,
new reviewed plan and separate authorization while retaining the failed bytes.

The archive contains the staged unique physical cases, scorer request,
comparison request, official report/work tree, GLBs, CSV/summary, execution and
stdout/stderr evidence. Released GT is excluded but hash-bound. The external
manifest fixes member count, per-member, total-expanded, archive and metadata
limits; `validate-delivery` rehashes every regular tar member and all cross-links.
Aliased B* shares the physical measurement but remains a fifth logical role.
Preparation/scoring errors remain failures in the denominator and are never
imputed as zero. Even a validated bundle remains unqualified and carries no
confidence interval or scientific verdict until E04/live native replay.

### C14 candidate core — frozen corotational group-TV residual

C14 now has a candidate implementation distinct from the existing body Gaussian
control. `research_math.corotational_residual_candidate` uses only the retained
complete predicted sequence. It estimates one proper uniform-vertex Kabsch pose
relative to centered frame zero, rejects rank/reflection ambiguity, freezes all
pose factors, solves an anchored nonuniform-time XYZ-group-TV problem for the
body residual, and reconstructs the complete original sequence. It consumes no
GT, camera, label, evaluator ICP, scorer state or learned weights. The Gaussian
world/body functions in `simple_mesh_controls.py` remain separate controls.

This source is **generated_unexecuted**. Run current common CPU software
acceptance first. Then independently verify the receipt-bound real
`sequence.npz`/`report.json`, freeze every positive solver parameter and the
timestamp-unit interpretation in the development protocol, and emit only this
zero-GPU artifact plan:

The frozen objective uses the discrete sum of `||delta_u/delta_t||` groups with
no quadrature `delta_t` factor. Treat it as a cadence-sensitive loader-clock
derivative penalty, not a continuous-time TV integral or a cross-cadence claim.

```bash
"$python_bin" "$project_dir/actionmesh/prepare_corotational_residual_candidate.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --source-sequence "$project_dir/inputs/original-case/sequence.npz" \
  --run-id c14-corotational-candidate-001 \
  --plan-dir "$project_dir/plans/c14-corotational-candidate-001" \
  --weight "$frozen_group_tv_weight" --rho "$frozen_admm_rho" \
  --absolute-tolerance "$frozen_abs_tolerance" \
  --relative-tolerance "$frozen_rel_tolerance" \
  --max-iterations "$frozen_max_iterations" \
  --wall-seconds "$c14_cpu_wall_seconds"
```

Inspect the exact sequence/report refs, two code refs, five declared outputs,
one attempt, zero retries and zero GPUs, then execute only the printed digest
through `run_harness.py`. Retain `candidate.json`, `manifest.json`, the complete
sequence, pose/solver `certificate.npz` and arm report. Nonconvergence is a
failed retained attempt; do not substitute the Gaussian control or rerun the
same frozen trial. The builder caps its inner job at 26,940 seconds so the outer
60-second allowance remains within 27,000 seconds.

The C14 source chain now continues through an explicit prospective comparison,
official scorer/raw collector and admitted plan builder. First create the exact
`c14-native-comparison-freeze` and prospective `c14-b-star-decision` records.
The five ordered logical roles are `b0`, `b_star`, `world_gaussian`,
`body_gaussian`, `corotational_residual`. B0 must be the exact source pair; B*
must be selected before C14 outcomes and may alias only an already-declared
simple control; the candidate row must pin its report, sequence and
`certificate.npz`. Then emit the comparison request without scoring:

```bash
dataset_root="$project_dir/inputs/actionbench-2796071c"
source_root="$actionmesh_source"
population="$project_dir/actionmesh/research_overnight/assets/actionbench_population.json"
test -d "$dataset_root/data"
test -d "$source_root"
test -f "$population"

cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c14_native_comparison request \
  --root "$project_dir" \
  --freeze "$project_dir/inputs/c14/c14-native-comparison-freeze.json" \
  --output "$project_dir/inputs/c14/c14-native-comparison-request.json"

"$python_bin" -m research_math.c14_native_scoring request \
  --root "$project_dir" \
  --comparison "$project_dir/inputs/c14/c14-native-comparison-request.json" \
  --ground-truth "$dataset_root/data/$c14_uid/surfaces.npy" \
  --population "$population" \
  --dataset-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --repo-root "$source_root" --timeout-seconds "$frozen_per_case_timeout" \
  --output "$project_dir/inputs/c14/c14-native-scoring-request.json"
```

Both requests remain `generated_unexecuted`, `native_qualified=false` and
`dispatch_ready=false`. The comparison validates same UID/seed/source bytes,
all 16 float32 frames, topology, timeline, vertex identity, exact frame-zero
anchor, current method implementations, body-Gaussian `poses.npz`, and the
candidate certificate by recomputing its pose factors and reconstruction. The
scoring request binds the admitted released snapshot, dataset-semantics receipt,
population/GT, current adapter, deterministic CPU-kNN entry and all six official
ActionBench sources. Preparation failures remain in the five-role denominator;
every byte-identical completed role shares one physical score and failures are
never zero-filled.

GPU STOP is active, so do not build or execute the scientific plan now. After a
separate explicit resume, supply a design-verified C14 method batch/protocol and
a hash-bound `c14-scientific-dispatch-admission`. Its referenced records are
`c14-family-split`, `c14-outcome-criteria` and
`c14-gpu-resume-authorization`; they have the same fail-closed structural rules
as the C13 records above but must name C14, the exact C14 comparison/protocol,
source-derived independent asset families, positive finite `cd_motion` effect, finite
nonnegative `cd_3d`/`cd_4d` margins and scope
`single_c14_scoring_attempt`. Gate 0 PASS and compatible importance/IPCG evidence
must pass the installed schema validators. The resume authorization additionally
binds the exact request, method batch, environment, run/task identity, expiry,
STOP acknowledgement and nonce. The builder creates one canonical recoverable
reservation, writes or exactly resumes the same dirty/native/harness plan state,
issues a non-circular immutable launch ticket that is an actual staged input,
then validates both plans with the installed runtime and finalizes one controller
receipt bound to their paths and digests. The `score` entrypoint rejects all
output/GPU activity unless the staged ticket, staged request, reservation and
unexpired original authorization cross-validate and the controller injects the
hash of its canonical final-consumption launch claim. Do not fabricate these records
merely to satisfy the parser.

```bash
"$python_bin" "$project_dir/actionmesh/prepare_c14_native_scoring.py" \
  --root "$project_dir" \
  --request "$project_dir/inputs/c14/c14-native-scoring-request.json" \
  --protocol "$project_dir/inputs/c14/c14-native-protocol.json" \
  --method-batch "$project_dir/inputs/c14/c14-method-verification-batch.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --admission "$project_dir/inputs/c14/c14-scientific-dispatch-admission.json" \
  --skill-dir "$skill_dir" \
  --plan-dir "$project_dir/plans/c14-native-scoring-001" \
  --run-id c14-native-scoring-001 --group "$frozen_c14_group" \
  --gpu-uuid "$actual_gpu_uuid" --wall-seconds "$c14_scoring_wall_seconds" \
  --ram-mib "$confirmed_ram_mib" --cpu-cores "$confirmed_cpu_cores"
```

Plan construction is not launch authority. After inspecting the emitted native
and harness plans, confirm the canonical consumption receipt exists and use only
the controller gate below. It validates the final receipt/plan pair, takes a
nonblocking single-owner lock, creates or resumes the exact launch claim, and
then delegates to the installed harness. Do not invoke the generic
`run_harness.py --execute` entry for C14.

```bash
"$python_bin" "$project_dir/actionmesh/launch_c14_native_scoring.py" \
  --root "$project_dir" \
  --consumption "$project_dir/inputs/c14/authorization-consumption/$c14_authorization_sha.json" \
  --skill-dir "$skill_dir" \
  --approved-plan-digest "$approved_c14_harness_plan_digest"
```

The builder requires `corotational_residual` as treatment, prospective B* as
baseline and B0/world Gaussian/body Gaussian as controls, binds every arm's
revision/implementation, verifies the strict Local runtime/scorer closure,
reserves 300 seconds for validation/collection and emits one confirmation
attempt with zero retry. An interrupted build may resume only when the retained
dirty patch and any existing native/harness plan exactly equal the reconstructed
state; a different state fails closed. Never call the `score` operation directly,
and never bypass `launch_c14_native_scoring.py`. The only
returned files are `actionmesh/c14-scoring-output/result.json`,
`raw-manifest.json` and `raw-evidence.tar`. Validate transported receipt hashes:

```bash
"$python_bin" -m research_math.c14_native_scoring validate-delivery \
  --root "$project_dir" \
  --request "$project_dir/inputs/c14/c14-native-scoring-request.json" \
  --gpu-uuid "$actual_gpu_uuid" \
  --result "$returned_c14_dir/result.json" \
  --manifest "$returned_c14_dir/raw-manifest.json" \
  --archive "$returned_c14_dir/raw-evidence.tar" \
  --expected-request-digest "$frozen_c14_request_digest" \
  --expected-result-sha256 "$receipt_result_sha256" \
  --expected-manifest-sha256 "$receipt_manifest_sha256" \
  --expected-archive-sha256 "$receipt_archive_sha256"
```

C14 is therefore source-chain complete but still `generated_unexecuted` and
Local-unverified. A validated raw bundle remains unqualified and contains no
confidence interval, gate decision or scientific verdict until official replay,
E04 and the frozen paired family/multiplicity analysis complete. GPU STOP remains
effective.

### C15 protected low-rank chain — source complete, not executed

C15 consumes one receipt-bound complete native prediction and a prospective
orthonormal temporal basis. The default concrete Q construction is exact `e0`
plus fixed DCT-II modes on frames 1..15; the frozen rank must come from allowed
development evidence and never from confirmation outcomes. A custom
development/input-only basis is accepted only with the same hash-bound evidence
contract. The candidate preserves frame zero exactly and preserves all Q
coefficients within the frozen float32 relative tolerance, then soft-thresholds
only the complementary residual. The two operation controls retain only the
common frame-zero constraint: same-lambda SVT and a truncated SVD whose exported
numeric rank matches the candidate under the frozen relative rank tolerance.

First author `inputs/c15/basis.npy` and
`inputs/c15/basis-evidence.json` prospectively. The evidence must have kind
`c15-protection-basis-evidence`, version `1.0.0`, a timezone-aware `frozen_at`,
the exact basis SHA-256, policy/rank/construction rule, `prospective_freeze=true`,
`confirmation_outcomes_used=false`, and current project-relative path/SHA-256
`source_refs`. For `analytic_anchored_dct`, the basis must byte-decode to the
exact output of `build_anchored_dct_basis(16, K)` and `K<16`; do not tune K or
lambda on confirmation scores.

After current-source CPU acceptance, emit the zero-GPU artifact plan:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_protected_lowrank_candidate.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --source-sequence "$project_dir/inputs/original-case/sequence.npz" \
  --basis "$project_dir/inputs/c15/basis.npy" \
  --basis-evidence "$project_dir/inputs/c15/basis-evidence.json" \
  --basis-source-policy analytic_anchored_dct \
  --lambda-value "$frozen_c15_lambda" \
  --residual-rank-policy candidate_export_numeric_rank \
  --basis-orthogonality-tolerance "$frozen_basis_orthogonality_tolerance" \
  --protected-coefficient-tolerance "$frozen_float32_protection_tolerance" \
  --numeric-rank-tolerance "$frozen_relative_rank_tolerance" \
  --max-artifact-bytes "$frozen_c15_artifact_limit" \
  --run-id c15-protected-lowrank-001 \
  --plan-dir "$project_dir/plans/c15-protected-lowrank-001" \
  --wall-seconds "$c15_cpu_wall_seconds"
```

Inspect every input/code ref, the three terminal outputs, one attempt, zero
retry and zero GPUs. Execute only the printed plan/digest through the installed
harness. Retain `candidate.json`, `artifact-archive.json` and `artifact.tar`.
The archive contains all three reports and only the sequence/certificate pairs
that actually completed; an individual failure stays a failed logical role.
Successful terminal artifact collection is not method success.

Generate the Gaussian arm with the existing `prepare_mesh_controls.py` command
in section 5, using the same source sequence and a prospectively frozen sigma.
Then create the exact `c15-native-comparison-freeze` and
`c15-b-star-decision`. The ordered denominator is `b0`, `b_star`, `gaussian`,
`unprotected_svt`, `rank_matched_tsvd`, `protected_residual_svt`. Freeze the
basis/evidence, lambda, both numerical tolerances, archive closure, all role
reports/outputs/implementations, and choose B* before any C15 native outcome.
Emit requests only:

```bash
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c15_native_comparison request \
  --root "$project_dir" \
  --freeze "$project_dir/inputs/c15/c15-native-comparison-freeze.json" \
  --output "$project_dir/inputs/c15/c15-native-comparison-request.json"

"$python_bin" -m research_math.c15_native_scoring request \
  --root "$project_dir" \
  --comparison "$project_dir/inputs/c15/c15-native-comparison-request.json" \
  --ground-truth "$dataset_root/data/$c15_uid/surfaces.npy" \
  --population "$population" \
  --dataset-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --repo-root "$actionmesh_source" \
  --timeout-seconds "$frozen_per_case_timeout" \
  --output "$project_dir/inputs/c15/c15-native-scoring-request.json"
```

The official primary metric is `cd_motion`; `cd_3d` and `cd_4d` are guardrails.
Byte-identical completed arms are scored once, but all six logical roles remain
in the denominator and errors are never zero-imputed. Requests remain
`generated_unexecuted`, `dispatch_ready=false` and `native_qualified=false`.

GPU STOP remains active. Only after a separate explicit resume and exact C15
design verification, Natural Gate 0/IPCG, source-derived family split, numeric
effect/noninferiority criteria, complete G01 protocol, strict runtime closure
and single-use GPU authorization may Local build and launch scoring:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_c15_native_scoring.py" \
  --root "$project_dir" \
  --request "$project_dir/inputs/c15/c15-native-scoring-request.json" \
  --protocol "$project_dir/inputs/c15/c15-native-protocol.json" \
  --method-batch "$project_dir/inputs/c15/c15-method-verification-batch.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --admission "$project_dir/inputs/c15/c15-scientific-dispatch-admission.json" \
  --skill-dir "$skill_dir" \
  --plan-dir "$project_dir/plans/c15-native-scoring-001" \
  --run-id c15-native-scoring-001 --group "$frozen_c15_group" \
  --gpu-uuid "$actual_gpu_uuid" --wall-seconds "$c15_scoring_wall_seconds" \
  --ram-mib "$confirmed_ram_mib" --cpu-cores "$confirmed_cpu_cores"

"$python_bin" "$project_dir/actionmesh/launch_c15_native_scoring.py" \
  --root "$project_dir" \
  --consumption "$project_dir/inputs/c15/authorization-consumption/$c15_authorization_sha.json" \
  --skill-dir "$skill_dir" \
  --approved-plan-digest "$approved_c15_harness_plan_digest"
```

Never call `score` or generic harness execution directly. Preserve the stable
single-owner launch claim and failed attempt. Returned evidence is the bounded
`result.json`, `raw-manifest.json` and `raw-evidence.tar`; validate it with
`python -m research_math.c15_native_scoring validate-delivery` using the same
arguments as C14, substituting the exact C15 request, digest, GPU and receipt
hashes. No C15 project test, artifact plan, scorer, model or GPU task was run by
Web authoring; Local acceptance and native results remain pending.

The same handoff now includes `actionmesh/prepare_native_context.py`, its exact
plan-only command, raw archive/manifest collection and current CPU acceptance
commands. Read its **Paired instrument plan and raw collection continuation**
section. The new runner requires the canonical environment and pinned dependency
inventory before model loading and at final verification. Plan emission remains
unapproved; GPU STOP and the existing budget remain unchanged. The maintained
[15-item delivery inventory](docs/research-math-20261006/longgoal-20261007/CANDIDATE_INPUT_AUDIT.json)
separates concrete implementation gaps from scientific prerequisites and has no
invented candidate acceptance commands.

Read [the native-context/supervisor handoff](rounds/20261008-native-context/WEB_HANDOFF.md)
first for this revision. GPU STOP is still effective. New source is
`generated_unexecuted`; Web ran no project tests, generation, scoring or GPU work.
This adds an independent paired official-generation instrument, complete decoder
replay and finite campaign supervision. The capture-only unit below is retained
with its own archive and receipt format. The supervisor delegates to the existing
harness and has not been installed or started. No candidate is promoted.

### Latest source handoff — decoder capture

### Fixed-inventory supervisor recovery

Use [SUPERVISOR_HANDOFF.md](docs/research-math-20261006/longgoal-20261007/SUPERVISOR_HANDOFF.md)
for the exact v1/v2 manifest, status/heartbeat, bounded `--watch-ready`, `--stop`
and guarded `--resume` interface. All commands retain one campaign identity and
original deadline/collection reserve; a lost acknowledgement is reconciled
against the exact harness/native receipts and PID/start/boot identity. Unknown
launch state is blocked, never retried. The current shared CPU acceptance stages
the supervisor and its actual-harness recovery tests. No test or deployment is
claimed here. Reviewed append-only task admission and actual online repair-agent
connection are still missing; GPU STOP remains in force.

### C04 robust motion protection and retained-native acceptance

Status: **generated_unexecuted**. Use the same pinned ActionMesh weights,
ActionBench snapshot, native loader and official scorer in [asset acquisition](#download-datasets-and-models).
No new model, dataset, GPU permission or budget is introduced. C04 reads a
receipt-bound completed `sequence.npz` and its sibling `report.json`; all 16
native frames, timestamps, faces and vertex identities must be present.

The source specialization uses the existing common ARAP repair `d` and the
inference-only surrogate `f(X)=sum ||X[t+1]-X[t]||^2/(2 V duration dt[t])`.
Its exact temporal Hessian gives a finite-step bound. A strictly positive,
diagonal, acceleration-weighted sensitivity shape defines a deterministic
ellipsoid: it is neither an arbitrary-PSD solver nor a calibrated probability
region. The actual conic projection minimizes `.5||delta-d||^2` with frame-zero
pins and `|g0.delta|+r||sqrt(S)delta||<=epsilon`. Projection gap/stationarity are
reported before trust/backtracking. Final float32 exports receive separate
surrogate/finite-budget checks. The scalar control matches the final robust
L2 update norm within the recorded 0.5% quantization tolerance; this is not
literal bit-exact norm equality. No native motion guarantee follows from it.

Run the current shared zero-GPU software acceptance first. Then set
`source_sequence` to the actual retained complete native sequence under
`project_dir`. The following variables come from the prospective C04 parameter
record and measured Local resource envelope, never a confirmation-score search.
The existing design permits at most two tuned numerical parameters with three
values each; for this specialization they are radius and epsilon. Freeze ARAP,
shape-floor/gain, solver and finite-step settings before that comparison; apply
matched tuning/cost to the simple controls. Exact values, source-derived family
split and numeric native effect/noninferiority margins still require their
recorded evidence before scientific dispatch.

```bash
"$python_bin" "$project_dir/actionmesh/prepare_robust_motion_candidate.py" \
  --root "$project_dir" --skill-dir "$skill_dir" --source-sequence "$source_sequence" \
  --run-id c04-artifacts-001 --plan-dir "$project_dir/plans/c04-artifacts-001" \
  --arap-weight "$c04_arap_weight" --temporal-weight "$c04_temporal_weight" \
  --iterations "$c04_arap_iterations" --cg-tolerance "$c04_cg_tolerance" \
  --cg-max-iterations "$c04_cg_max_iterations" \
  --shape-floor "$c04_shape_floor" --shape-gain "$c04_shape_gain" \
  --radius "$c04_radius" --epsilon "$c04_epsilon" \
  --trust-radius "$c04_trust_radius" --finite-budget "$c04_finite_budget" \
  --absolute-tolerance "$c04_absolute_tolerance" --relative-tolerance "$c04_relative_tolerance" \
  --max-iterations "$c04_conic_max_iterations" --max-backtracks "$c04_max_backtracks" \
  --coordinate-bounds "$c04_lower" "$c04_upper" --bounds-policy "$c04_bounds_policy" \
  --max-artifact-bytes "$c04_archive_bytes" --wall-seconds "$c04_cpu_seconds"
"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/c04-artifacts-001/harness.json" --root "$project_dir" \
  --execute --approved-plan-digest "$approved_plan_digest"
```

Use the digest actually emitted by the builder after reviewing zero GPUs,
8192 MiB CPU RAM, one attempt/zero retries and remaining campaign allowance.
Do not execute the candidate module directly. The three receipt outputs are
`actionmesh/c04-robust-output/{candidate.json,artifact-archive.json,artifact.tar}`.
Return that attempt's complete `c04-robust-output` directory without changes,
including `manifest.json`, `common-target.npz` and all three role directories.
The archive binds every materialized member and terminal failed reports. Also
retain the exact original sequence/report at their original project-relative
paths; do not rewrite hashes or source refs when copying between host roots.
Set `c04_candidate_json` to the returned `candidate.json`. Then emit the real
artifact acceptance plan, which reconstructs the method from these native inputs:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_c04_native_acceptance.py" \
  --root "$project_dir" --skill-dir "$skill_dir" --artifact-candidate "$c04_candidate_json" \
  --run-id c04-native-acceptance-001 --plan-dir "$project_dir/plans/c04-native-acceptance-001" \
  --wall-seconds "$c04_acceptance_seconds" --ram-mib "$c04_acceptance_ram_mib"
"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/c04-native-acceptance-001/harness.json" --root "$project_dir" \
  --execute --approved-plan-digest "$approved_plan_digest"
```

Use this second builder's digest. It rejects a skipped/empty acceptance suite;
common engineering fixtures alone cannot close this real-input acceptance.
Incomplete candidates retain failed logical roles; a missing archive after a
hard interruption remains a failed attempt and collection gap, not permission
to rerun or manufacture a terminal artifact. Logs/receipts resolve from the
actual harness state and attempt workspace, never an invented attempt UUID.

After prospective B* selection, construct a `c04-native-comparison-freeze`
using `c04_native_comparison.py`'s exact fields and all five roles:
`b0,b_star,deterministic_protection,strength_matched_repair,robust_conic_protection`.
Use the same metadata-only acceptance builder in explicit request-preparation
mode; it validates terminal artifact integrity, retains failures, and assembles
both requests inside one zero-GPU harness job. This mode does not declare all
methods accepted. Preserve the prospective freeze and its complete decision/
semantic evidence under the project root; the builder hashes their closure.

```bash
"$python_bin" "$project_dir/actionmesh/prepare_c04_native_acceptance.py" \
  --root "$project_dir" --skill-dir "$skill_dir" --artifact-candidate "$c04_candidate_json" \
  --freeze "$c04_freeze" --ground-truth "$dataset_root/data/$c04_uid/surfaces.npy" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --dataset-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --repo-root "$actionmesh_source" --timeout-seconds "$frozen_per_case_timeout" \
  --run-id c04-requests-001 --plan-dir "$project_dir/plans/c04-requests-001" \
  --wall-seconds "$c04_request_seconds" --ram-mib "$c04_acceptance_ram_mib"
"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/c04-requests-001/harness.json" --root "$project_dir" \
  --execute --approved-plan-digest "$approved_plan_digest"
```

Use that builder's actual digest. The declared outputs are
`actionmesh/c04-comparison-request.json` and `actionmesh/c04-scoring-request.json`.
Copy them back preserving their project-relative paths and hashes; every input
closure file retains its path too. Providing only `--freeze` creates just the
comparison request. Supplying scoring arguments requires their complete set.
The official source root is reconstructed from the explicitly staged source
file; the job never falls back to a live source checkout. These CPU validators
perform actual method reconstruction, which is why their execution belongs to
this harness job. They never call the official scorer or grant GPU admission.

Official execution remains stopped. A future plan requires current C04
`design_verified`, Natural Gate 0/IPCG, independent-family split, protocol-equal
numeric CD-3D effect and CD-4D/CD-M noninferiority criteria, the frozen native
contract, exact runtime/source closure and an expiring exact-run/GPU
`single_c04_scoring_attempt` resume. It additionally requires actual held-out
development evidence for the surrogate/set semantics, bound to a prospective
method specification and implementation. That review cannot depend on future
confirmation output or assert probability coverage. No such evidence is
created by this source delivery.

Only then use `prepare_c04_native_scoring.py` with its explicit
`--root --request --protocol --method-batch --environment --admission --skill-dir
--plan-dir --run-id --group --gpu-uuid --wall-seconds --ram-mib --cpu-cores` fields,
then `launch_c04_native_scoring.py --root ... --consumption ... --skill-dir ...
--approved-plan-digest ...`. The launcher preserves its exact stable claim,
checks final consumption, and delegates to the existing harness. Never use the
generic harness CLI or `score` directly for this authorized scientific plan.
Return the receipt-bound `result.json`, `raw-manifest.json`, `raw-evidence.tar`,
actual full logs, complete five-role denominator, all source/data/runtime hashes
and validation result. Deduplicated physical output does not remove logical roles.

Debug projection failures from `projection` gap/stationarity, fixed tolerances
and common repair; inspect `finite_step` and every `rejected_trials` entry for
backtracking failures. Preserve the entire failed attempt and frozen params.
Do not fix failure by dropping a role, changing dtype/frames or relaxing a budget.
Missing semantic/family/criterion evidence blocks scientific scoring while
independent source work continues. All Local tests/native comparisons remain
pending until their actual exact-version receipts are returned.

### C10 integrable-gradient chain — source complete, not executed

C10 now has a distinct method-to-official-score source chain. It does not reuse
the older external `d/C/W` projection as the candidate. The CPU artifact builder
consumes one completed receipt-bound native `sequence.npz` and its sibling
`report.json`, constructs a single geometry-only differential target, and holds
one anchor-derived positive edge-weight vector fixed across all 16 frames. It
exports deterministic direct lift, qualified face-local ARAP and the pinned
matrix-free integrable solve. All use the same `h` and are evaluated under the
same `W`; direct lift does not use `W` in its construction. Per-component
lowest-ID pins, frame zero, topology, timing arrays and vertex identities are
preserved exactly. A method failure remains a failed logical role but is still
collectable through the bounded terminal archive.

First run the common current-source CPU acceptance plan described above. Then,
with an actual retained native source and a new single-use directory, prepare
the candidate artifact plan only; the example numbers are design inputs that
must be reviewed and frozen, not defaults to copy silently:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_integrable_gradient_candidate.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --source-sequence "$source_sequence" \
  --run-id c10-artifact-001 --plan-dir "$project_dir/plans/c10-artifact-001" \
  --target-strength 0.25 --max-relative-change 0.10 \
  --absolute-tolerance 1e-10 --relative-tolerance 1e-8 \
  --max-iterations 2000 --coordinate-bounds -2 2 \
  --bounds-policy preserve_and_report --max-artifact-bytes 1073741824 \
  --wall-seconds 7200
```

Inspect the emitted `harness.json`, its exact digest, zero-GPU resources and
three stable receipt outputs before Local decides whether to execute it through
the installed harness. Never call `research_math.integrable_gradient_candidate`
directly. On return, retain `candidate.json`, `artifact-archive.json` and
`artifact.tar`; materialize the archive only with
`materialize_candidate_archive`, using explicit archive/member byte ceilings,
then run the validator against the materialized `candidate.json`. `incomplete`
means one or more physical methods failed and must stay in the five-role
denominator; it is not a successful candidate outcome.

After prospective B* selection and an exact C10 freeze exist, build the logical
comparison request without scoring:

```bash
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.c10_native_comparison request \
  --root "$project_dir" --freeze "$c10_freeze" --output "$c10_request"
"$python_bin" -m research_math.c10_native_scoring request \
  --root "$project_dir" --comparison "$c10_request" \
  --ground-truth "$dataset_root/data/$c10_uid/surfaces.npy" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --dataset-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --repo-root "$actionmesh_source" --timeout-seconds "$frozen_per_case_timeout" \
  --output "$c10_scoring_request"
```

The request fixes B0, prospective B*, direct lift, qualified local ARAP and the
pinned candidate; content-identical physical arrays are scored once, while all
five logical roles and terminal preparation failures remain in the analysis
denominator. Official scoring still requires C10 `design_verified`, installed
validator evidence for Gate 0/IPCG, source-bound independent family split,
protocol-equal G01 numeric criteria, strict runtime closure, exact ActionBench
snapshot/GT/scorer identities and an explicit expiring
`single_c10_scoring_attempt` authorization for the named physical GPU. Only
then may Local invoke `actionmesh/prepare_c10_native_scoring.py` using its
`--root/--request/--protocol/--method-batch/--environment/--admission/--skill-dir`
and plan/resource arguments, inspect the emitted digest, and launch through
`actionmesh/launch_c10_native_scoring.py`. Direct scorer or generic harness CLI
execution is forbidden. Current status is `generated_unexecuted`: no Local
test, candidate computation, scorer or GPU execution has occurred.

The latest implementation entry is
[NATIVE_DECODER_CAPTURE.md](docs/research-math-20261006/NATIVE_DECODER_CAPTURE.md).
It adds the opt-in `complete_unit_plan --capture-decoder` path around the actual
official pipeline, with a complete capture archive plus the existing three-arm
generation/scoring outputs. It is `generated_unexecuted`; follow that entry's
CPU harness acceptance commands first. GPU execution remains stopped.

The historical R3 green-r2 archive now contains raw plans, receipt and logs for
311/311 checks. Web rehashed all 21 manifest payloads and the receipt; the real
runtime historical-root fix is present. At base `1da40c7`, however, one of its
72 code refs already differs (`actionbench_full128_unit.py`, later macOS parent
normalization); this delivery changes further code. Do not label the current
tree accepted from that historical receipt. The old historical-root defect is
not the current source blocker. Canonical input staging, current-source
acceptance, same-device raw scorer evidence and trusted replay remain pending.

The paragraphs and command cards below retain earlier round history. Their old
294-test and pre-repair status descriptions do not supersede this section,
CURRENT's latest handoff or the raw R3 return. Do not rewrite historical hashes
or blindly repeat completed unchanged work. The observer has a different task
role and 119-output inventory, so it cannot reuse Full128's old price/admission.

This runbook is the Local Codex entry point for round `20261006-baseline-qualification`. It covers acceptance of the committed control/scoring software, recovery and verification of one original development asset, construction of the three-arm baseline unit, native-scoring replay, integrity validation, and evidence return.

Latest execution evidence is summarized in
`docs/research-math-20261006/actionbench-full128-r9-r10-return-review.json`.
The returned archive passed a complete 1,617-file size/SHA-256 audit. The Linux
harness ran 294/294 software checks successfully, and all 72 receipt-pinned
`code_refs` match the current code tree. The immutable r9 pair is terminal at
9/9 completed; its reconciliation and eight CPU-only r10 manifest batches
covering indices 10-127 are returned. These are engineering/software/input facts
only: no current-device native scientific contract was qualified and no
candidate ran. Do not repeat the accepted suite or finalized parity unless a
bound source/input hash changes. Window-plan preparation is currently blocked by
historical-r7 reference closure and a stale GPU environment identity; do not
launch GPU work while resolving it.

There are 20 mathematical constructions and 15 conditionally selected candidates.
C01, C02, C10, C13 and C14 now have **source-complete, generated_unexecuted** method-to-
official-score chains, so source completion is **5/15**; **0/15 are Local-verified
and 0/15 have native results**. C02's new complete chain remains unadmitted and
unexecuted; the earlier dense operator alone is still only a reference primitive.

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
| GPU-host project path | Historical r9 tree: `/root/actionmesh-research-staging/4d-longgoal-r9`; R3 retains its terminal snapshot (9 completed, none running/pending/failed, observed 2026-10-07T19:12:37Z). Preserve it and use an isolated checkout; no current liveness is inferred |
| SSH alias/endpoint | Unknown. Reuse the user's already configured authorized target; do not guess or extract credentials |
| Installed skill directory | Unknown. Locate the complete current installed `research-autopilot` package and record its absolute path |
| Conda env/interpreter | Last returned as `/root/rivermind-data/actionmesh-repro/inference-env/bin/python`, a Conda-backed venv; revalidate captured package/dependency hashes; no Docker |
| GPU | Current host is RTX 2080 Ti, UUID `GPU-4910bb04-2d00-ca81-f64a-2031568758ad`, 22,528 MiB total; the older calibration UUID is a different physical card. Recheck identity/free VRAM/processes before launch |
| ActionBench revision | `2796071cbe6248422fcbeab3101fa9f9886cb7b9` |
| Original sample and GT paths | Returned UID `000-048_45e57349f062416aaf11f2c31587da16`; GT remains host-only at SHA-256 `25881f0d7a9f41578f77ba6be70f810ddcbcf4236a6834713ea197ea2916823e` |
| Time budget | 28,800 s hard window, 1,800 s collection reserve; admitted complete-unit measurement 1,330.465551 s, frozen per-unit timeout 1,664 s, 16 units / 26,624 workload seconds |

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

### 3. Reuse current parser-only acceptance; rerun only after source changes

Historical `software-status-observation-r1` records 294/294 checks; later R3
records 311/311 on its explicitly pinned old closure. Neither accepts this changed
source tree. Preserve both receipts and run current software acceptance once from
a clean delivered checkout under the existing Local harness authority:

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
It must also list all three root modules imported by the current Full128 window
checks: `prepare_actionbench_full128_window.py`,
`prepare_actionbench_active_batch_snapshot.py`, and
`prepare_actionbench_active_batch_reconciliation.py`. Missing any one would make
the isolated harness workspace incomplete even if the controller checkout can
import it.
The returned 294/294 suite includes the unit-manifest, Full128 snapshot,
reconciliation and import-closure checks. It remains engineering evidence only
and does not require repetition of the finalized scorer parity. A later source
change requires a new receipt whose complete `code_refs` match that revision.

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
The nine unit checks for the fail-closed contract helper are included in the
returned 294/294 whole-suite acceptance.

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

<a id="download-datasets-and-models"></a>
### Download datasets and models

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

After—and only after—the sidecar above is promoted at its canonical path, build
the CPU-only pricing plan. This pass rehashes the admission's entire transitive
evidence closure again and partitions the immutable 128-UID population; it does
not launch generation, scoring or any GPU work:

```bash
pricing_run_id=price-full128-from-complete-lowram-r7

"$python_bin" "$project_dir/actionmesh/prepare_actionbench_queue_pricing.py" \
  --root "$project_dir" \
  --contract "$project_dir/docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json" \
  --admission "$project_dir/inputs/complete-unit-admissions/complete-lowram-r7.json" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --output "$project_dir/inputs/actionbench-full128-queue/pricing.json" \
  --skill-dir "$skill_dir" \
  --plan-dir "$project_dir/plans/$pricing_run_id" \
  --run-id "$pricing_run_id" \
  --wall-seconds 900 \
  --ram-mib 2048 \
  --cpu-cores 1

"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/$pricing_run_id/harness.json" \
  --root "$project_dir" \
  --execute \
  --approved-plan-digest "$pricing_plan_digest"
```

Before execution, inspect one CPU-only task, one attempt, no retry,
`max_gpu_task_seconds: 0`, and the single canonical output. With the returned
1,330.465551-second complete-unit measurement, the frozen 5/4 operational
headroom gives `ceil(1330.465551 * 5 / 4) = 1664` seconds per unit. The
27,000-second workload budget therefore admits 16 complete units (26,624
seconds), leaving 376 seconds of workload slack in addition to the separate
1,800-second collection reserve. The complete population must appear exactly
once in canonical order across eight 16-UID windows.

The returned pricing output still says `queue_approved: false`,
`queue_generated: false`, `dispatch_ready: false`, and all scientific/candidate
qualification flags false. It is a capacity and partition freeze, not an
executable queue.

The per-UID freezer, Full128 runner mode, and a fail-closed harness window
compiler are now authored. They are **unexecuted by Web**. The concurrently
returned Local ledger also records the already-running engineering batch
`population-gpu-current-r9` over population indices 1 through 9. Do not create a
second attempt for any of those UIDs and do not modify its frozen digest. First
reconnect to the same host, inspect that exact run ID/digest and collect or carry
its real state. A stale status file or SSH shell is not liveness evidence.

The returned reconciliation is immutable and hash-verified, and the current
whole software suite passed 294/294 through the Linux harness. Do not repeat
either unchanged. A plan-only window-02 attempt is now blocked because the
historical r7 admission is recursively treated as current live input and because
the environment binds the older GPU UUID. Preserve every historical hash, return
the exact offending-ref list, and capture a fresh current-device environment.
Only after a test-first closure repair is accepted may the builder emit window
02; it performs no workload:

Before invoking the compiler, freeze the live harness state and its matching
aggregate status as one coherent immutable evidence pair. Do not copy only one
file and do not point the sidecar at the mutable global `STATUS.json`. The
snapshot builder double-reads both inputs, checks exact run/digest/task identity
and count agreement, and writes each original byte sequence exactly once:

```bash
live_r9_root=/root/actionmesh-research-staging/4d-longgoal-r9

"$python_bin" "$project_dir/actionmesh/prepare_actionbench_active_batch_snapshot.py" \
  --root "$project_dir" \
  --live-state "$live_r9_root/runs/harness/population-gpu-current-r9/state.json" \
  --live-status "$project_dir/docs/research-math-20261006/longgoal-20261007/STATUS.json"
```

This creates immutable `state.json` and `status-snapshot.json` beneath
`docs/research-math-20261006/longgoal-20261007/resumed-evidence-r9/4d-longgoal-r9/runs/harness/population-gpu-current-r9/`.
If the watcher advances either input or the counts differ, the command fails
without an admissible pair; refresh/reconcile the source files rather than editing
the archive. Then generate the immutable reconciliation at
`inputs/actionbench-full128-queue/active-batch-reconciliation.json` with the
sole canonical builder:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_actionbench_active_batch_reconciliation.py" \
  --root "$project_dir"
```

Follow
`docs/research-math-20261006/actionbench-full128-active-batch-reconciliation-contract.json`:
bind the exact pricing receipt, include `population-gpu-current-r9`, its frozen
plan digest `6bfec342ca9b2eb5b3f8174e2cf9be597d4b540da6615f62cd13bb48a0011ac1`,
the fixed range `[1,10)`, all nine ordered canonical UID dispositions, and SHA-256
references to the retained canonical campaign plan, immutable
`status-snapshot.json` and
research-harness `state.json` snapshot.
This compiler revision accepts exactly that one r9 source run; do not append a
later retained run without a reviewed compiler revision. The compiler parses
all three sources, recomputes the campaign digest, verifies every task's native-plan
SHA-256 and native plan digest, and matches the run, task IDs, exact indices,
observation time, disposition counts and every per-task status. A swap between
two task states is rejected even when aggregate counts are unchanged. All nine native plans are retained in
each newly generated unit's transitive input closure.
These retained JSON files are staged as already-validated opaque evidence: do
not re-resolve their original run-relative inner paths against the current
project root.
Use only `completed`, `running`, `pending`, or `failed`; do not omit a failure or
completed UID to make a target window available. The compiler rehashes these
sources and rejects every target window that intersects any retained UID. This
sidecar permits plan generation only; refresh host state again before any later
dispatch decision.

```bash
set -eu
window_id=full128-window-02
window_run_id=full128-window-02-r10
historical_archive=/root/actionmesh-research-staging/4d-historical-closure-r2
historical_root=/root/actionmesh-research-staging/4d-r7-pricing-84a94
historical_manifest="$project_dir/evidence-preparation/closure-r7-original-MANIFEST-2000.json"
source_manifest="$historical_archive/evidence-preparation/closure-r7-original-MANIFEST-2000.json"

# This prepared project root already contains the immutable staged manifest
# and 170 unique archive blobs at their original archive_path values. Check
# the original source and staged copy before planning; the builder then
# rehashes every archived byte. If either source or staged evidence is absent,
# stop and repair the staging through the reviewed evidence-transfer process.
test "$(sha256sum "$source_manifest" | cut -d ' ' -f 1)" = \
  3e65c9347aab4b67329df72f6e14800a510b6c79d02ecabf337f0d0d105a1eb0
test "$(sha256sum "$historical_manifest" | cut -d ' ' -f 1)" = \
  3e65c9347aab4b67329df72f6e14800a510b6c79d02ecabf337f0d0d105a1eb0
cmp "$source_manifest" "$historical_manifest"
test "$(git -C "$historical_root" rev-parse HEAD)" = \
  84a94a60779b86e477c3488929097b76fdcebfec
test -z "$(git -C "$historical_root" status --porcelain --untracked-files=no)"

"$python_bin" "$project_dir/actionmesh/prepare_actionbench_full128_window.py" \
  --root "$project_dir" \
  --plan-dir "$project_dir/plans/$window_run_id" \
  --skill-dir "$skill_dir" \
  --pricing "$project_dir/inputs/actionbench-full128-queue/pricing.json" \
  --contract "$project_dir/docs/research-math-20261006/actionbench-current-release-unit-contract.json" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --snapshot-contract "$project_dir/docs/research-math-20261006/actionbench-full128-snapshot-contract.json" \
  --snapshot-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --unit-manifest "$project_dir/inputs/actionbench-full128-snapshots/unit-manifest.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --active-batch-reconciliation "$project_dir/inputs/actionbench-full128-queue/active-batch-reconciliation.json" \
  --source-root "$source_root" \
  --dataset-root "$dataset_root" \
  --weights-root "$weights_root" \
  --historical-root "$historical_root" \
  --historical-manifest "$historical_manifest" \
  --run-id "$window_run_id" \
  --window-id "$window_id" \
  --gpu-uuid "$gpu_uuid"

"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/$window_run_id/harness.json" \
  --root "$project_dir" \
  --freeze > "$project_dir/plans/$window_run_id/reviewed-harness.json"
```

Inspect exact equality of the reviewed plan digest, 16 canonical UIDs, sixteen
1664-second no-retry native plans, `max_gpu_task_seconds: 26624`, one physical
GPU, `max_parallel_tasks: 1`, `total_wall_seconds: 27000`, and
`window_seconds: 28800`. The compiler reports `queue_generated: true` only when
Local actually emits such a plan; that does not set `queue_approved` or
`dispatch_ready`. The 27,000-second driver cutoff is the operational workload
boundary; the remaining 1,800 seconds of the 28,800-second reporting window are
reserved for collection and cannot be consumed by task-start overhead. Do not dispatch it until the historical-evidence/live-input closure repair,
fresh current-device environment, native/scoring qualification, trusted replay,
and final device/disk admission are recorded. Software acceptance and active-batch
reconciliation are already satisfied and should not be repeated unchanged. Window 01 intersects the
retained r9 indices 1 through 9 and is rejected by code, including after those
units complete or fail; never duplicate them. Any remaining indices in 0 through
15 require a separate no-omission continuation rather than an outcome-selected
replacement window.

For this repaired plan, also verify that every generated native task stages the
hash-pinned R7 manifest and all 170 unique `archive_path` blobs into its attempt
input closure. The runtime freezer rechecks every matching manifest row inside
the attempt root; a plan that omits any blob is not acceptable even if the outer
plan validates. Keep indices 10 through 15 explicitly covered by a separate
no-omission continuation; do not treat window 02 alone as full-128 coverage.

The one-UID engineering parity/finalization chain is now complete and retained at
`00b30fd`; do not rerun it merely because the downstream manifest is new.
Inventory these full-population prerequisites without launching generation or scoring. Return
their exact paths/revisions/hashes, snapshot and semantic admissions, and measured
storage/semantic-pass requirements plus the unit manifest. The complete-unit
default runner has a retained OOM and the explicit FP16 low-RAM repair completed
generation before entering official scoring. Only a terminal successful receipt
followed by the CPU-only complete-unit admission can feed the separate pricing
pass above. One-UID parity timing or any CPU admission timing cannot. Pricing
still does not create an executable queue.

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

Conditional selection order is `c02, c01, c10, c13, c14, c04, c03, c20, c11, c12, c15, c05, c08, c06, c07`. C01, C02, C10, C13 and C14 have source-complete but generated-unexecuted candidate/comparison/scoring chains; none has Local acceptance, scientific admission or a native result. Every candidate still requires its applicable implementation/design evidence, simple baseline, ablation, frozen native protocol, full-unit timing, reproducible result, and evidence-based conclusion after the baseline gate closes.
