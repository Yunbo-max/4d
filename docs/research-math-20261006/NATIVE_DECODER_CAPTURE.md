# Native pipeline decoder capture — source delivery, not execution

Status: **generated_unexecuted**. GPU work remains stopped. No candidate,
native-context qualification, scorer qualification or replay is accepted here.
This is a complete opt-in capture path for the existing engineering unit, not
an implementation of C01/C02 or any of the 15 selected methods.

## What this closes

`decoder_observer.py` could capture an already loaded decoder, but no real
generation entry attached it. The official low-RAM pipeline loads the temporal
decoder only after Stage I. Attaching to its initial `None` value cannot work.
The new `PipelineDecoderCapture` wraps that pipeline instance's
`_decode_displacement` boundary and resolves the loaded decoder at each call.
It calls the original method with the original arguments and returns the same
mesh list; neither queries nor predictions are substituted.

The inspected native sources are the pinned public release at
`d5c01f5045df55819e337369c9617f603c667e00`, retained under `actionmesh/repo/`:

- `inference/video_to_animated_mesh.py`: official configuration, generation and export.
- `actionmesh/pipeline.py`: lazy loading, complete window context, output ordering.
- `actionmesh/model/temporal_autoencoder.py`: five-input forward signature,
  direct/residual interpretation and clamping.
- `actionmesh/preprocessing/mesh_processor.py`: original XYZ and vertex normals.
- `actionmesh/io/mesh_io.py`: final official mesh and deformation export.

The prior CANDIDATE_INPUT_AUDIT and mathematical selection are reused. No new
method, ranking, benchmark, threshold or Gate 0/IPCG status is introduced.

## Source-to-output path

| Source | Responsibility |
|---|---|
| `actionmesh/research_math/complete_unit_plan.py --capture-decoder` | Generates the distinct `observer-complete-<profile>-three-arm-unit`, engineering role `native-context-capture`; never executes it |
| `actionmesh/research_math/complete_unit_runner.py` | Existing admission, device/resource monitoring, complete native generation, three controls, official scoring and final integrity; Full128 plus capture is rejected |
| `actionmesh/observe_actionmesh_generation.py` | Calls the actual upstream `run_actionmesh` using the same fixed non-fast configuration and seed; wraps the live pipeline, then archives the complete capture |
| `actionmesh/research_math/pipeline_decoder_capture.py` | Records every original decoder window, full query normals, raw output, anchor topology and returned window geometry |
| `actionmesh/research_math/tests/test_pipeline_decoder_capture.py` | CPU fixtures for lazy attachment, full capture, failures, restoration, bounds, topology, command preservation, output inventory and pricing separation; unexecuted |

Each window retains `anchor.npz`, `geometry.npz`, `window.json`, and the existing
observer's `identity.json`, `inputs.safetensors`, `tensors.safetensors` and
`record.json`. It checks that the five captured inputs are the actual pipeline
inputs, that query XYZ covers every anchor vertex in order, and that native
direct/residual conversion reproduces every returned vertex without rounding
away differences. All returned faces must match the original topology.

The unit receipt declares the previous 117 outputs **plus**
`actionmesh/unit-output/decoder-capture.tar.gz` and
`actionmesh/unit-output/decoder-capture/manifest.json`. The archive retains the
entire capture directory; the manifest lists every payload's path, size and
SHA-256. Partial captures, original decoder exceptions and logs remain on failed
attempts. An incomplete call, swallowed window failure, zero observed calls,
changed topology or overwritten output cannot become a completed capture.

For the frozen 16-frame pipeline, the native window normally has 16 latent
times and 15 decoded target times: `drop_first=True` leaves the original anchor
in the mesh bank. The observer retains that anchor separately. It does not
insert an unobserved source-time decode or call a 15-target output a 16-target
decode. Final generation/export and all three score arms still contain 16 frames.

Bounds: two decoder windows maximum; 64 MiB tensor payload and 64 MiB geometry
budget per window. Existing observer storage contains an input copy and a combined
input/output copy, and the archive adds another retained copy. These are payload
bounds, not measured peak RAM/disk/VRAM. A bound failure is retained; no sampling,
subsetting, clipping change or automatic retry is permitted.

## Local acceptance and plan generation

Read AGENTS, LOCAL_AGENT_RUNBOOK and the baseline round handoff at the delivered
commit. Preserve old attempts and use a fresh exact-commit checkout. Restore
`project_dir`, `skill_dir`, `python_bin`, source/dataset/weight paths and actual
GPU UUID from the existing authorized host. Downloads/admission remain exactly
the existing runbook paths and immutable versions; this feature needs no new
model, benchmark, package installation or source modification.

First perform CPU software acceptance through the installed harness:

```bash
software_run_id=decoder-capture-software-r1
"$python_bin" "$project_dir/actionmesh/prepare_control_scoring_checks.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --run-id "$software_run_id" --plan-dir "$project_dir/plans/$software_run_id"
```

Inspect the emitted source inventory, including the new generation entry and
both capture modules. Use the builder's actual printed digest:

```bash
"$python_bin" "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/$software_run_id/harness.json" --root "$project_dir" \
  --execute --approved-plan-digest "$approved_plan_digest"
```

Record the actual result/count, failures and logs. No test or red/green outcome
is claimed by this Web delivery. The fixtures exercise the capture integration;
they do not instantiate ActionMesh or prove real model equivalence.

After current software acceptance and canonical input restoration, the following
command **only prepares** the opt-in unit; GPU execution remains stopped:

```bash
capture_run_id=native-decoder-capture-r1
cd "$project_dir/actionmesh"
"$python_bin" -m research_math.complete_unit_plan \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --contract "$project_dir/docs/research-math-20261006/actionbench-current-release-unit-contract.json" \
  --population "$project_dir/actionmesh/research_overnight/assets/actionbench_population.json" \
  --snapshot-contract "$project_dir/docs/research-math-20261006/actionbench-full128-snapshot-contract.json" \
  --snapshot-admission "$project_dir/inputs/actionbench-full128-snapshots/admission.json" \
  --dataset-semantics "$project_dir/inputs/actionbench-full128-snapshots/dataset-semantics.json" \
  --unit-manifest "$project_dir/inputs/actionbench-full128-snapshots/unit-manifest.json" \
  --environment "$project_dir/inputs/native-runtime/environment.json" \
  --source-root "$source_root" --dataset-root "$dataset_root" \
  --weights-root "$weights_root" --gpu-uuid "$gpu_uuid" \
  --run-id "$capture_run_id" --plan-dir "$project_dir/plans/$capture_run_id" \
  --wall-seconds 27000 --capture-decoder
```

The generation profile comes from the admitted unit manifest. Do not silently
replace its BF16/default or explicitly reviewed FP16/low-RAM profile. Generation
seed remains 42, official sampling seed 44, controls sigma 1.0, and scorer budgets
are unchanged. Inspect one no-retry GPU task, 119 declared outputs, the staged
observer entry, exact physical UUID, driver hard cutoff of 27,000 seconds and
28,800-second reporting window with 1,800 seconds reserved for collection.
Only an eligible later Local run may execute its exact reviewed harness digest.

This is a newly instrumented unit. It cannot be inserted into Full128 plans,
accepted by the old 117-output admission, or priced using the old 1,664-second
unit allowance. Retain its whole-unit time, stage times, observed memory and disk
if it is eventually run; capture overhead is not assumed to be zero.

## Remaining scientific boundary and return

Same-object output identity and tensor capture are not observer-free equivalence.
Local must separately compare a matched uninstrumented native execution under
the same code, input, weights, precision and device, and replay the full saved
decoder context. This delivery deliberately records `native_context_qualified`,
`replay_qualified` and candidate/scientific qualification as false. It does not
manufacture latent correspondences, a source-time response, attention modes,
legal motion targets or benchmark efficacy from captured tensors.

Return the exact execution commit/patch, harness/native plans and receipts,
observed attempt paths, capture archive and manifest, all existing three-arm
raw scores and resource logs. Rehash archive members against the manifest after
transfer and retain failed attempts. Do not guess a future attempt directory.
Current UID008 raw parity/repeat/nonce replay, natural failure evidence and
candidate Gate 0/IPCG remain independent pending work; GPU stop remains in force.
