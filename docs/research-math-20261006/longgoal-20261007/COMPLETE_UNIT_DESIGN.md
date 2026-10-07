# Current-public-release complete unit

Status: design; not executed or scientifically qualified.

The existing user authorization is to run and debug the full research workflow on
one RTX 2080 Ti, using existing code wherever possible. This bounded execution
adapter will invoke official current-release inference, export all 16 meshes,
then run the already verified scorer on native, world-Gaussian and body-Gaussian
arms. It does not introduce a new research method or reduce the population.

## Admission and identity

Require the committed full128 snapshot and dataset-semantics admissions before
launch, and revalidate referenced bytes. Official source must be clean at
`d5c01f5045df55819e337369c9617f603c667e00`, including TripoSG submodule
`fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`. Existing extracted source lacks Git
metadata, so stage a separate genuine Git checkout; retain the old source.
Require all four pinned model caches and the full 128-UID input manifest.

Choose the lexicographically first released UID for prospective timing, without
looking at its score. One sample is a runtime measurement only; it cannot satisfy
the all-128 benchmark agreement rule or the 15 candidate outcomes.

## Generation

Use official `inference/video_to_animated_mesh.py` with seed 42, non-fast mode,
16 original RGBA frames, stage 0=100, stage 1=30, faces=40000,
floaters threshold=.02, guidance=7.5, anchor=0. Set `--low_ram --dtype float16`
explicitly as the versioned RTX-2080-Ti current-release configuration. This is not
an assertion about the unpublished leaderboard dtype or low-RAM settings.
Use offline verified model directories and preserve the resolved configuration,
source/runtime hashes, exact argv/cwd/environment, GPU UUID and device samples.
Do not replace the official pipeline with the older census helper: that helper
skips background-model loading for valid alpha and executes stages separately.

## Export, arms and score

Official mesh_00.glb through mesh_15.glb and deformation arrays must exist and
be finite with consistent topology before controls can be prepared. Retain hashes
and a lossless sequence representation for the existing mesh-control module.
Reuse the source-bound world/body controls and their exact parameters. Run all
three scores via the explicit strict CUDA-forward/upstream CPU-KNN-backward
backend that passed 9/9 parity comparisons. Preserve its numerical-runtime caveat.
Never silently restore the old atomic CUDA backward or lower metric budgets.

## Execution and evidence

Add a committed plan builder that only validates/adapts inputs and emits the
research-autopilot harness plan. Run the workload only through that harness with
its inspected digest. One GPU task, no GPU sharing; bounded attempt, immutable
output directory; failures/OOMs retained. Collect total wall time for generation,
export, controls, all three scores and integrity checking, plus observed peak
VRAM. Derive later queue size only from this complete measured unit; keep a
1,800-second collection reserve in each 28,800-second window. Continuation of the
overall user goal is not stopped by the window boundary.

Implementation checks must reject missing/stale admissions, wrong UID/frame set,
wrong source/config, missing/invalid export, scorer mismatch and partial score
success. The real complete GPU unit remains the decisive integration check.
