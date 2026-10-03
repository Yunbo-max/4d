# Native interfaces for hypotheses 2, 9 and 10

These are executable inference mechanisms and constructed behavioral tests.
They do not establish better natural-data quality, a full benchmark result, a
cross-model result, or a measured <=22 GB peak. No adapter loads weights, reads
GT, trains a model, or downloads data. The caller owns frozen model loading,
hash-qualified input provenance, resource caps and evaluation.

## 2: composed decoder queries and anchored graph consensus

```python
from research_ten.actionmesh_adapter import ActionMeshDecoderAdapter
from research_ten.m02_cycles import probe_cycles

adapter = ActionMeshDecoderAdapter(
    decoder, latents, context_times, device="cuda:0", autocast=True)
result = probe_cycles(
    adapter, anchor_vertices, faces, probe_times, query_ids,
    source_normals=prepared_anchor_normals,
    normal_fn=official_mesh_normal_function,
    max_query_points=32768)
```

`decoder` must already be in eval mode with every parameter frozen.
The adapter also accepts an optional `step_callback(step, total)` forwarded to
the native decoder for progress and budget accounting. Native callbacks occur
before each target; a callback alone does not prove CUDA completion.

`latents`
has shape `[T,N,D]` or `[1,T,N,D]`; `context_times[T]` is the full ordered latent
clock. The adapter signature is
`adapter(source_time, target_times, positions[Q,3], normals[Q,3], vertex_ids[Q])`
and returns absolute positions `[K,Q,3]`. Source and target alpha are normalized
against the **full context clock**, matching official `get_scaling` and
`apply_scaling`. The native decoder's `apply_displacement` determines whether
its predicted output is direct or residual. No extra spatial alignment occurs.

`probe_cycles` uses one immutable latent context and real supplied times.
It queries A to all other selected times, including the vertices needed for
the selected vertices' complete face one-ring. At each B it recomputes mesh
normals, then queries B to every other selected time on the same material IDs.
This includes A->B->A and A->B->C against A->C. The default normal function is
area weighted; official ActionMesh uses trimesh's mesh normals. Native parity
therefore requires prepared source normals and the official normal callback,
as in `native_smoke.py`. Invalid selected intermediate normals fail explicitly.

The result contains `direct`, `composed`, `consensus`,
`multi_reference_average`, `smoothing`, residuals, times, query IDs, support IDs
and per-call costs. For T selected times, Q selected vertices and S support
vertices, decoder work is S(T-1)+Q(T-1)^2 point-target pairs. The budget is
checked before any query. Consensus solves data fidelity plus displacement
edge residuals with the original anchor eliminated from the linear solve.
The simple smoothing control uses squared adjacent velocity with 1/dt² weights;
the averaging control reuses exactly the composed queries. Cycle agreement
alone cannot establish accuracy or correct motion.

## 9: fresh three-branch guidance at each native sampler state

```python
from actionmesh.scheduler.guidance import ClassifierFreeGuidance
from research_ten.m09_guidance import make_torch_guidance

base = ClassifierFreeGuidance(
    guidance_at_inference=[[0, 0], [0, 1], [1, 1]],
    guidance_scales=[1.0, 7.5])
hook = make_torch_guidance(base, mode="projection", epsilon=1e-12, seed=42)
old = pipeline.cf_guidance
try:
    pipeline.cf_guidance = hook
    # Run the normal live pipeline._denoise_latents(...) or scheduler.denoise(...).
finally:
    pipeline.cf_guidance = old
print(hook.records)
```

Modes are `projection`, `scalar`, `norm_matched`, `random`. The wrapper requires
the exact enabled branch order unconditional/mesh/mesh+video and two scales;
it rejects a two-branch configuration instead of silently changing the native
baseline. It delegates official condition expansion and masks, then hooks
`aggregate_cfg` on every newly evaluated sampler state. It never reads cached
velocities. For each unknown frame, flatten all tokens/channels, define
u=vm-v0 and w=vmv-vm, and replace w with
w-min(<u,w>,0)u/(||u||²+epsilon) only when ||u||²>epsilon. Known frames,
near-zero u and nonconflicts retain the scalar formula. The native sampler
continues to lock known latent values; changing a known frame's unused velocity
is not used as a substitute for testing that invariant.

`norm_matched` preserves w's direction with projected-w norm. `random` uses
a fresh seeded random direction with the projection correction's norm.
All modes evaluate all three live branches at every step. FP32 subtraction,
dot products and accumulation avoid intermediate FP16 overflow; the final
velocity deliberately returns to the input dtype and rejects overflow.
Thus low-precision rounding need not match an older all-half scalar path
bitwise. The wrapper does not mutate input branches. Upstream three-branch
CFG has the same scalar mathematical formula; its in-place v0 side effect
is not claimed to be a mathematical bug.

`run_live_flow` is a NumPy additive-flow reference with a strictly decreasing
clock and explicitly supplied nonnegative step distances. It is not the native
solver. Native two-step tests establish wiring and fresh calls, not official
30-step mesh quality or memory feasibility for the full protocol.

## 10: backward information through shared latent windows

```python
from research_ten.actionmesh_adapter import ActionMeshWindowAdapter
from research_ten.m10_windows import rollout_windows

callback = ActionMeshWindowAdapter(
    pipeline, full_input, full_context, representation_id,
    autocast=True, step_callback=step_callback)
result = rollout_windows(
    callback, full_times, window_size=16, overlap=4,
    anchors={0: original_anchor_latent},
    representation_id=representation_id,
    future_reliable=reliability_mask,
    backward_updates=4, max_rollouts=8, seed=42)
```

`ActionMeshWindowAdapter(pipeline, input_data, context, representation_id, *,
autocast=True, step_callback=None)` requires a loaded eval/frozen Stage-I
denoiser. `context[T,S,D]` and the full `ActionMeshInput` must share the clock.
The callback receives `(indices, known, direction, seed, representation_id)`,
where `indices` are global frame IDs and `known` maps global IDs to hard latent
values. It returns `{"latents": [window_length,N,D], "representation_id": ...}`.
It invokes the real `_denoise_latents` with a fresh native `LatentBank` and the
window's actual input/context. At least 16 frames per native window are required
by `ActionMeshInput`. The rollout driver shifts the final window backward to
keep its full requested length without repeating or dropping real frames;
the final overlap can exceed the requested minimum. It records all window IDs
and actual overlap sizes. W=16/O=4 gives four windows for 52 frames and six
windows for 70 frames; set max_rollouts to cover forward and backward calls.
Times must stay distinct under native float32 and LatentBank's 1e-5 matching.

The driver makes at least four forward window calls. Only caller-provided
original anchors are immutable. It first reruns the final window against its
caller-marked-reliable terminal tail, freeing its generated source overlap.
Then each earlier window receives the newly updated overlap from the revised
later window. Without a reliable boundary, or without a revised later window,
reverse propagation abstains. The default four backward updates cover four
windows, for eight total calls; both budgets are explicit. Callback outputs
must preserve all supplied conditions bitwise. No latent averaging occurs.

The terminal boundary is a **forward prediction used as a pseudo-observation**,
not GT and not automatically trustworthy. Reliability is supplied externally,
not inferred or calibrated here. The returned `backward_boundaries` records
this provenance and every source of revised overlap. The constructed test
changes evidence only in the final window and verifies that earlier overlaps
change only after backward propagation, while original input anchors stay fixed.

Representation identity is a caller assertion to be justified with actual
common token/anchor/coordinate provenance. Equality of a string does not prove
that independently regenerated latents are aligned. Native long-data testing
still needs >=4 full windows, qualified reliable future constraints, fixed
forward-only and equal-compute rerun controls, Stage-II re-decoding after latent
updates, official evaluation, and measured resource use. The current cached
16-frame census does not meet that long-sequence requirement.

## Verification status

`PYTHONPATH=actionmesh python3 -m unittest research_ten.tests.test_model_methods -v`
has 17 passing NumPy tests locally and two skipped Torch tests in this revision. The latter test
full-clock decoder semantics, NumPy/Torch guidance parity, caller-input purity
and finite FP16 extremes when Torch is available. Constructed demos explicitly
set natural-data validation claims false. Live native GPU smoke results are
reported separately by the root runner, not assumed from these local tests.
