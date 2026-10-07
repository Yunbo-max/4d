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
`faithful-harness-verification.json` in the exact identity/tolerance form needed
by the installed native evaluator. A failed or partial run writes no sidecar.

Parity is not baseline qualification. It neither supplies source-backed
baseline/control decision rules nor replaces the nonce-bound trusted replay.
The retained metric values are infrastructure evidence only at this stage.

## Admission before execution

Do not execute from the current setup evidence alone. First freeze a complete
source-backed native protocol that passes the installed
`scripts/_native_eval.py:verify_protocol` with the scorer set to the exact
official descriptor returned by
`research_math.actionbench_parity.scorer_descriptors`. This deliberately blocks
the task while authentic sample/split definitions and prospective
baseline/control thresholds are missing. Do not invent a one-asset threshold.

The protocol must retain the released ActionBench revision
`2796071cbe6248422fcbeab3101fa9f9886cb7b9`, its released sample manifest and GT
inventory, the three official metric paths, the current request's sampling
parameters and budgets, and source-backed qualification rules. The selected
group must expose this exact official scorer descriptor. The environment and
GPU UUID must match the returned native runtime and current physical device.

## Harness-only Local commands

Use a clean checkout at the delivered commit and restore the source-safe
evidence archive plus the separately retained GT. Replace only values named in
angle brackets with reviewed facts:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_actionbench_parity.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --request "$project_dir/plans/control-scoring-request-000-048.json" \
  --protocol "$project_dir/plans/<frozen-official-scorer-protocol>.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --run-id actionbench-official-faithful-parity-001 \
  --plan-dir "$project_dir/plans/actionbench-official-faithful-parity-001" \
  --group <frozen-group> \
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

Plan generation starts no scorer. Source-inspect both plans and every command
before approving the digest. Recheck GPU UUID/processes and free disk because
the last return reported only about 2.4 GB free. This task retains 48 GLBs plus
six score passes, so measure its actual output size as well as wall time and
observed VRAM.

## Acceptance and return

Accept scorer parity only when:

- the outer and native harness receipts are complete and bind the approved plan;
- each official invocation executed the retained patched copy of the pinned
  official CLI and returned exactly one requested UID;
- every exported GLB round-trips with exact vertices and faces;
- both scorers succeeded on all three arms;
- all nine metric comparisons have absolute difference `0.0`;
- device samples bind the current physical GPU UUID with no telemetry error; and
- `faithful-harness-verification.json` matches the frozen official contract's
  sample manifest and both exact scorer descriptors.

Do not relax the zero tolerance after seeing a mismatch. Inspect, in order, the
GLB export manifest, executed official source hashes and patch record, runtime
packages, raw official CSV, raw faithful JSON, commands, and logs. Preserve a
failure as parity failure; do not average the two paths or select the preferred
value.

After a pass, copy the verification sidecar into a stable project-relative
location, hash it, and freeze a second protocol revision whose scorer is the
faithful descriptor plus that `verification_ref`. Re-run native protocol
verification. Only then may the existing three-arm scoring plan be prepared.
Trusted live replay remains mandatory and accepted values come from the official
path. This parity unit is not a complete candidate multi-arm experiment and its
timing cannot price the later eight-hour queue.
