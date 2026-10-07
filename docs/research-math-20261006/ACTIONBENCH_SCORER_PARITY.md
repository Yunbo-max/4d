# ActionBench official-versus-faithful scorer parity

Status: `generated_unexecuted`
Scientific effect status: no scores, baseline decision, candidate result, or full-unit timing accepted

## Purpose and boundary

This handoff adds an independent native scorer path before the existing
`research_census_eval.py` faithful harness can be admitted. The official path:

1. reads the same frozen one-arm manifest and NPZ sequence;
2. exports all 16 float32 frames to the GLB layout required by the pinned
   ActionBench `evaluate_dataset.py`;
3. rejects any vertex or face change on GLB round-trip;
4. starts the pinned official dataset CLI in a fresh process at the unchanged
   `100000` Chamfer / `10000` ICP budgets and seed `44`;
5. retains official CSV, summary, stdout, stderr, command, copied evaluator
   source and all GLBs; and
6. normalizes the official CSV to JSON without implementing a metric.

The official source samples predicted surfaces on CPU before moving points to
CUDA but passes that CPU device to the CUDA RNG-state API. Both paths therefore
apply the same reviewed compatibility-only change in
`sample_mesh.get_baryc_sampling_mesh`: the RNG device list is empty for CPU and
unchanged for CUDA. The copied official source, original hashes, patched hashes
and exact replacement are retained. Any other upstream source shape fails.

The parity task runs official CLI and faithful harness separately for each of
`native`, `world_gaussian` and `body_gaussian`. `cd_3d`, `cd_4d` and `cd_motion`
must match with absolute tolerance zero. A positive run writes
`parity-bundle-attestation.json`; a failed or partial run writes no attestation.
The scientific-consumer `faithful-harness-verification.json` is created only
after the complete bundle is promoted and revalidated at its stable target.

Parity is not baseline qualification. It neither supplies source-backed
baseline/control decision rules nor replaces the nonce-bound trusted replay.
The retained metric values are infrastructure evidence only at this stage.

## Engineering admission before execution

The official dataset and paper publish the released population and metric
definition, but no per-asset numerical pass threshold. Their leaderboard values
are means over all 128 objects and must not be reused as a threshold for this one
development UID. The exact captured sources and this negative threshold finding
are frozen in `actionbench-official-source-evidence.json`.

Scorer equivalence is implementation qualification, not a baseline or method
performance experiment. It therefore runs as a protocol-free `purpose=engineering`
harness task. The builder emits two immutable ordinary inputs:

- a one-UID parity sample manifest binding the exact request/digest, released
  population and UID, GT, all three report/prediction refs, complete 16-frame
  sampling and native budgets; and
- a scorer-equivalence contract binding the source review, exact official and
  faithful descriptors, environment, three frozen prediction artifacts and zero
  tolerances.

That contract is fail-closed against `baseline_qualification`,
`control_qualifications`, criteria, contrasts or any effect threshold. It carries
`scientific_effect_qualification=false` and `native_contract_qualified=false`.
The runner recomputes the entire contract from the request and current files; a
changed sample, source, scorer, runtime or budget rejects before scoring.

This separation does not weaken the scientific boundary. A nonnegativity rule
such as `cd_3d >= 0` remains forbidden as baseline/control qualification, and the
128-object aggregate mean remains invalid as a one-asset threshold. A later
scientific scoring plan still requires its own genuine source-backed protocol,
installed-verifier acceptance and nonce-bound replay. The current environment and
GPU UUID must match before this engineering parity task can run.

## Harness-only Local commands

Use a clean checkout at the delivered commit and restore the source-safe
evidence archive plus the separately retained GT. Replace only values named in
angle brackets with reviewed facts:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_actionbench_parity.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --request "$project_dir/plans/control-scoring-request-000-048.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --run-id actionbench-official-faithful-parity-001 \
  --plan-dir "$project_dir/plans/actionbench-official-faithful-parity-001" \
  --gpu-uuid <current-gpu-uuid> \
  --wall-seconds <reviewed-parity-limit> \
  --ram-mib <reviewed-host-ram> \
  --cpu-cores <reviewed-cpu-count>

"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/actionbench-official-faithful-parity-001/harness.json" \
  --root "$project_dir" \
  --execute \
  --approved-plan-digest <printed-plan-digest>
```

Plan generation starts no scorer. It writes
`actionbench-parity-sample-manifest.json` and
`actionbench-scorer-equivalence-contract.json` inside the plan directory; inspect
those files, both plans and every command before approving the digest. Recheck
GPU UUID/processes and free disk because
the last return reported only about 2.4 GB free. This task retains 48 GLBs plus
six score passes, so measure its actual output size as well as wall time and
observed VRAM.

## Acceptance and return

Accept scorer parity only when:

- the outer and native harness receipts are complete and bind the approved plan;
- each wrapper process record exactly matches the reviewed command, working
  directory, exit code and retained stdout/stderr;
- each official invocation executed the retained patched copy of the pinned
  official CLI and binds its source hashes, compatibility patch, exact command,
  UID, sequence, GT, CSV, summary, export manifest and logs;
- each faithful output binds its request parameters, device, evaluator/source hashes,
  sequence, GT, generation report and pass manifest;
- every exported GLB round-trips with exact vertices and faces;
- both scorers succeeded on all three arms;
- all nine metric comparisons have absolute difference `0.0`;
- device samples bind the current physical GPU UUID with no telemetry error; and
- the post-promotion `faithful-harness-verification.json` matches the generated
  parity-only sample manifest and both exact scorer descriptors.

The sidecar additionally hash-binds the final record/evidence, request,
population, GT, runtime and each arm's prediction plus official/faithful raw
output. Its scope is exactly the one-UID manifest. It is not global ActionBench
parity and must not be reused by a scientific protocol with a different sample
manifest; rerun matching parity for that manifest instead.

Do not relax the zero tolerance after seeing a mismatch. Inspect, in order, the
GLB export manifest, executed official source hashes and patch record, runtime
packages, raw official CSV, raw faithful JSON, commands, and logs. Preserve a
failure as parity failure; do not average the two paths or select the preferred
value.

After a pass, do **not** copy only the sidecar. Copy the entire successful
`<attempt>/workspace/actionmesh/actionbench-parity-output/` directory, byte for
byte, to the previously absent
`$project_dir/actionmesh/actionbench-parity-output/` target recorded in the
bundle attestation. Recompute every `record_ref`, `parity_evidence_ref`, prediction and raw
official/faithful output hash against the promoted tree; retain the original
harness attempt and promotion receipt. A partial promotion leaves the sidecar
inadmissible.

Do not run the finalizer directly. After that complete copy, generate a second,
CPU-only engineering harness plan that pins the promoted bundle and the complete
canonical parity execution record:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_actionbench_parity_finalization.py" \
  --root "$project_dir" \
  --request "$project_dir/plans/control-scoring-request-000-048.json" \
  --contract "$project_dir/plans/actionbench-official-faithful-parity-001/actionbench-scorer-equivalence-contract.json" \
  --output "$project_dir/actionmesh/actionbench-parity-output" \
  --parity-harness-plan "$project_dir/plans/actionbench-official-faithful-parity-001/harness.json" \
  --parity-harness-report "$project_dir/runs/harness/actionbench-official-faithful-parity-001/report.json" \
  --parity-native-plan "$project_dir/plans/actionbench-official-faithful-parity-001/native.json" \
  --parity-native-receipt "$project_dir/runs/attempts/actionbench-official-faithful-parity-001/receipt.json" \
  --approved-parity-plan-digest <the-exact-digest-approved-for-the-parity-run> \
  --skill-dir "$skill_dir" \
  --run-id actionbench-parity-finalization-001 \
  --plan-dir "$project_dir/plans/actionbench-parity-finalization-001" \
  --wall-seconds 300 --ram-mib 2048 --cpu-cores 1
```

Capture the newly printed finalization digest, inspect the generated plan, and
execute that exact digest through `run_harness.py`. From the completed native
receipt, locate the one declared output
`faithful-harness-verification.json`; copy that receipt-bound file into the
already promoted bundle target and verify the source/destination hashes match.
Do not guess an attempt ID or rerun the finalizer directly.

```bash
"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/actionbench-parity-finalization-001/harness.json" \
  --root "$project_dir" --execute \
  --approved-plan-digest <the-exact-finalization-plan-digest>
```

The harnessed finalizer re-resolves every bound hash, requires the exact promoted target,
and verifies the approved outer-plan digest, completed harness report, native
plan/receipt, one engineering scorer-parity attempt, complete declared-output
inventory and the harness-recorded attestation hash. It converts only the
path-canonical parity descriptor into the unchanged
absolute faithful descriptor required by `control_scoring.py`; its generated
`faithful-harness-verification.json` is the only sidecar eligible for later
scientific protocol admission. This finalization task requests zero GPUs and
does not qualify a scientific score.

Then freeze a complete scientific protocol whose scorer is the
faithful descriptor plus that `verification_ref`, whose sample manifest exactly
matches the parity manifest, and whose prospective qualification rules are
independently source-backed. If the scientific manifest differs, run a matching
parity unit first. Run native protocol verification. Only then may the existing
three-arm scientific scoring plan be prepared.
Trusted live replay remains mandatory and accepted values come from the official
path. This parity unit is not a complete candidate multi-arm experiment and its
timing cannot price the later eight-hour queue.

## Qualification-source audit — 2026-10-07

The current official repository head and complete ActionBench source tree were
re-inspected after the parity/finalization handoff. The seven-file ActionBench
tree matches the retained evaluator bytes exactly and contains no committed
prediction bundle, per-sample score table, CSV/JSON result artifact, or
qualification-threshold file. Official code searches for ActionBench CSV files
and thresholds returned no result; the only `results` hits are the README and
the evaluator that writes a caller's output. The release history exposes the
leaderboard edits, but the published values remain means over all 128 objects.
See `actionbench-qualification-source-audit.json` for the exact head/tree/blob
identities, queries, returned paths, and scope adjudication.

This closes the bounded source search without changing the gate: the returned
one-UID development asset still has no primary-source absolute qualification
rule. The source-matched alternative now has a conditional prospective rule in
`actionbench-full128-reproduction-contract.json`: all 128 released objects must
succeed, and all three current-README means must lie in their published
three-decimal intervals. ActionMesh generation seed 42 and official evaluator
sampling seed 44 are separate identities. This finite-population reproduction
rule has no sampling confidence interval and is not an efficacy threshold.

The path remains non-dispatchable until it binds complete per-sample GT and
prediction manifests, exact generation source/weights/configuration, current
runtime and full-unit resource measurements, an installed-verifier native
protocol, and fresh trusted official replay. It cannot reuse this one-UID parity
sidecar or timing, or apply either aggregate table as a per-asset threshold.
