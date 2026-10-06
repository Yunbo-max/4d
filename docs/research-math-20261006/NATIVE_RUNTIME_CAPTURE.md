# Installed native environment capture

Status: **generated_unexecuted**. This supplements the existing baseline qualification handoff; it supplies neither a qualified evaluator nor candidate results.

The scoring plan validates five installed dependency versions against a hashed runtime file. Local can now capture those versions and the installed Conda/package inventory through a bounded CPU-only harness task rather than hand-writing runtime JSON. Use the already-resolved remote Conda interpreter and controller-observed physical GPU UUID.

## Ordered Local commands

First run the whole parser/software acceptance suite from `LOCAL_AGENT_RUNBOOK.md` at the delivered commit. The suite includes eight new runtime inventory checks; their fixture metadata are engineering inputs only.

On the GPU host:

```bash
"$python_bin" "$project_dir/actionmesh/prepare_native_runtime.py" \
  --root "$project_dir" \
  --skill-dir "$skill_dir" \
  --run-id baseline-native-runtime-001 \
  --plan-dir "$project_dir/plans/baseline-native-runtime-001" \
  --gpu-uuid "$gpu_uuid"
```

Inspect the generated plan and use its printed digest:

```bash
"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/baseline-native-runtime-001/harness.json" \
  --root "$project_dir" --execute \
  --approved-plan-digest "$runtime_plan_digest"
```

The outer plan reserves one CPU, 1,024 MiB RAM, zero GPUs, and at most 180 seconds. The inner task has a 120-second limit, one attempt, no retry. Its foreground command is `python -m research_math.native_runtime` in the staged attempt's `actionmesh` directory. It initializes no CUDA device and imports no native scorer or torch module.

Locate the completed attempt via the actual harness/native receipts. Verify its declared output hashes. Copy the attempt workspace's `inputs/native-runtime/` directory, preserving relative paths, into the clean project checkout using controller file transfer. Refuse to overwrite an existing different capture; record both source and destination paths and hashes. The dependency reference already uses the final project-relative path, so do not rewrite the JSON after transfer.

Outputs:

- `inputs/native-runtime/dependencies.json`: installed Python distribution names/versions and Conda name/version/build/subdir records, with no channel or download URL export.
- `inputs/native-runtime/environment.json`: actual interpreter/prefix, five required package versions, supplied GPU UUID, UTC capture time, and the SHA-256 of the dependency inventory.

Pass `--environment "$project_dir/inputs/native-runtime/environment.json"` to the existing scoring-plan builder. That builder rechecks current interpreter, package versions, supplied UUID and every dependency reference before admission.

## What the capture establishes

It records installed metadata and its byte identity. It does not certify native library importability, CUDA compatibility, physical GPU identity, native source authenticity, official/harness parity or scientific qualification. Both `gpu_identity_verified` and `native_contract_qualified` remain false. Keep current GPU/driver/VRAM/process inventory and the official scorer acceptance evidence separately.

This dependency inventory is not a solver lock or an exact reinstall recipe: upstream package artifacts and hashes remain distinct evidence. A changed environment requires a new capture and affected Local acceptance; do not reuse a stale file by editing versions to match.

## Failure and return

Missing required packages, conflicting installed versions, absent/corrupt/incomplete Conda metadata, malformed UUIDs and escaping output paths stop the capture. Existing output directories are preserved. Read the attempt stderr and the resolved interpreter before repairing the actual environment; do not install arbitrary latest versions or a container runtime.

Return the two files, the exact source/plan/harness digests, all attempt logs and the separate current host inventory in the existing window review packet. Interrupted or failed attempts remain retained. Native protocol/source qualification, original source/GT restoration and trusted native replay remain pending; candidate designs and native results stay **0/15**.
