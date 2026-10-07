# Native decoder context needed by the fifteen-method program

Status: engineering capture core and five CPU hook tests authored; unexecuted.
The native observer/replay admission has not run.
This does not change the frozen default-release calibration unit.

At ActionMesh `d5c01f5045df55819e337369c9617f603c667e00`,
`ActionMeshPipeline.generate_mesh_animation` fetches the whole Stage-I latent
window and invokes `_decode_displacement` with the actual window time IDs,
normalized source and target times, and anchor mesh. `_decode_displacement`
constructs query vertices plus normals and invokes `temporal_3D_vae` with:
`latent`, `framestep`, `source_alpha`, `target_alphas`, and `query`.
These are the sufficient *observed arguments* for replay of that decoder call;
they are not evidence that an arbitrary reduced or reordered context is equivalent.

The normal 16-frame decode drops the anchor from target output times. The anchor
GLB is the original anchor mesh, not a saved self-query `F_a(X)`. Consequently
C01/C03 cannot obtain their same-context self residual from exported GLBs alone.
They require a separate source-time decoder query under the captured complete
window and identical query normals. Preserve this distinction in the method API.

Proposed engineering observer: record the actual decoder forward keyword tensors
with a PyTorch forward-pre-hook after the official pipeline loads the temporal
VAE, plus source/model/config identity, prediction mode, anchor topology and
output frame mapping. Copy detached tensors to CPU without changing arguments,
RNG state, precision, schedule, or model parameters. Store non-pickle tensors
and an explicit dtype/shape/hash manifest. Do not capture full attention matrices.
The official direct prediction mode and its subsequent coordinate bounds must
both be recorded; corrected coordinates must not be silently clipped.

Admission experiment after the baseline runtime is qualified:

1. Bound observer overhead and compare observed native outputs with the matched
   unobserved path under the exact same release/config/seed. Retain disagreement.
2. Replay all recorded non-anchor target queries and compare decoder output with
   the retained native sequence in the same coordinate system, separating official
   float32 GLB quantization from model dtype. Freeze the equality/tolerance policy
   before seeing the comparison; do not choose it to hide a discrepancy.
3. Query the source time using the identical full latent/window/query context,
   recording raw field output and any official bound operation separately.
4. Only after these engineering checks create a `NativeContext` consumed by
   C01/C03 and other qualified field-based methods. No inference labels, GT,
   scoring ICP transforms or score outputs enter this artifact.

This observer alone does not provide `SparseModes` or a `LegalMotionSurrogate`.
Those interfaces still need independently justified model outputs or input-video
measurements and frozen eligibility rules. A saved tensor is not their admission.

## Implementation checkpoint, 2026-10-07

`actionmesh/research_math/decoder_observer.py` captures the five actual named
forward tensor arguments and raw output with PyTorch pre/post hooks. Hooks
return no replacements. CPU snapshots preserve dtype (including BF16) and have
safetensors hashes, shape/dtype metadata, original device and autocast/inference
state. Originals are never cast or edited. Capture storage is bounded to 64 MiB
of tensor payload per call. The bound excludes serialization overhead and the
retained separate input copy, so disk can approach twice that bound. Successful
captures remain `captured_unqualified`. Failures preserve captured inputs.

Five engineering checks cover nonmutation/RNG/output preservation, byte bounds,
failed forward cleanup, positional arguments/BF16, and tamper rejection. The
first test-plan launch was rejected with `HARNESS_HOST_DRIVER_ALREADY_ACTIVE`
while complete-lowram-r7 was executing. This is not a test failure or pass.
No attempt was made to bypass the host supervisor lock or stop the GPU run.
Run the original red plan after the current GPU task terminates, then transfer
the authored core, generate a new green plan, and run the complete suite.
The remote red plan pins the test with no implementation in its code closure.

Outstanding integration remains explicit: attach the observer to the loaded
Stage-II module just around `_decode_displacement`, bind anchor faces and the
complete generation manifest, source/config/weights and frame mapping; compare
observed and unobserved native outputs; replay all original queries; only then
issue the same-source-time query. No candidate can consume the core's outputs
until those checks establish the `NativeContext` interface. No new GPU observer
run has been launched and no context from the live baseline is retroactively
claimed captured.
