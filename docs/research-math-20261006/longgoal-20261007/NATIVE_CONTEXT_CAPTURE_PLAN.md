# Native decoder context needed by the fifteen-method program

Status: source-inspected engineering design; not implemented or executed.
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
