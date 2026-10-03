# ActionMesh time reversal: bounded source audit, 2026-10-02

Status: **feasible with a small explicit adapter; unsafe with only `anchor_idx=15`**. No model edits, SSH, GPU, or model experiment performed. This is an API/mechanism audit, not evidence that reversal improves reconstruction or a new method claim. Scope is exactly 16 frames and subsampling level 1.

## Confirmed indexing bug

Paths below are relative to `/Users/yunbo/Documents/Codex/4d/actionmesh/repo/actionmesh/`.

- `model/utils/timesteps.py:95–97`: for total=size=16, `chunk_from(15,16,16,...)` produces `[15,0,1,...,14]`. It includes every time once; Stage I will generate all 16 entries (`pipeline.py:470–506`). This is anchor-first order, not descending time order.
- `pipeline.py:547–557`: Stage II selects sorted bank times using the same window and correctly gets source mesh at time 15.
- `pipeline.py:560–565` calls `interpolate_timesteps(...,drop_first=True)`.
- `model/utils/embeddings.py:234–242`: that helper builds ascending min-to-max times then removes numeric time 0, irrespective of which time is the source. Targets become `[1,...,15]`.
- `model/utils/storage.py:194–224`: MeshBank update defaults to `replace=False`. Existing source 15 remains; 1 through 14 are added; time 0 is absent. Only **15 meshes** result, missing reversed time 0, i.e. original physical frame 15.
- Correct 16-frame reverse targets are `[0,...,14]`, excluding actual source 15. Source alpha is 1, target alphas are those times divided by 15. Do not silently append a copy of the missing frame.

Pure CPU reproduction is saved in `census-time-reversal-index-repro-20261002.json`. It executes the actual function ASTs with a minimal NumPy torch-API shim and records source SHA256. Forward source 0 yields all 16 mesh times; reverse source 15 yields only 15 with missing time 0; explicit source exclusion yields all 16. This verifies the indexing logic, not PyTorch/model execution.

## Reuse and exact source identity

Reuse the existing `stage0.npz` and `prepared.npz` (`research_census_case.py:260–279`): anchor latent, anchor vertices/faces, saved query features, query vertex IDs, exact preprocessed frames/crop metadata, and per-image encoded contexts. Do not rerun image-to-3D, remesh, crop each reversal anew, or independently regenerate normals without checking them against saved features. Image encoder contexts can be permuted because they were computed per image; their attached physical frames must remain explicit.

For literal reversed input, attach the cached physical-frame-0 anchor latent/mesh to reversed time 15, reverse contexts and preprocessed frames, and use the Stage I window `[15,0,...,14]`. After denoising, use `_decode_displacement` (`pipeline.py:316`) with explicit target times excluding the true source instead of the buggy `generate_mesh_animation` output-time construction. Ordered reverse-clock results need flipping back to physical chronology. Check all 16 unique times, exact anchor bytes, unchanged faces, and query IDs.

The native helper recomputes query normals through `get_mesh_features` at `pipeline.py:354–356`. To guarantee feature identity, either require its reconstructed features to be bitwise equal to `anchor_query_features`, or call the same temporal VAE directly with the saved query tensor. The latter exact signature is `temporal_3D_vae(latent=window_latents, framestep=window_times, source_alpha=source_alpha, target_alphas=target_alphas, query=cached_query[None])`, followed by the existing `apply_displacement` under the same native direct-output config and FP16 autocast. No latent scaling changes. Preserve anchor output directly rather than decoding source-to-source.

## Permutation is not physical time reversal

Purely permuting **all** of latent rows, contexts, masks/noises and their unchanged timestamp labels should leave this attention computation equivariant up to numerical error. Inflated attention is noncausal (`model/utils/attention_processor.py:138`), and position is supplied through actual timestamps. Such a permutation-only canary is an adapter sanity check, not a research consistency result.

Actual reversal attaches timestamp `15-t` to each physical frame previously labelled `t`. That changes temporal RoPE relative signs (`model/temporal_denoiser.py:134–145`), and Stage II source/target alpha from `(0,t/15)` to `(1,1-t/15)` (`model/temporal_autoencoder.py:201–236`). Generic RoPE scores `q_i^T R(t_j-t_i) k_j` are not invariant to negating the relative time. A two-dimensional CPU counterexample is included in the reproduction JSON. Source/target alpha embeddings also have no explicit reversal symmetry. Thus the model is not architecturally guaranteed to be time-reversal equivariant, although learned near-equivariance remains possible and untested. Direction dependence by itself is not a reconstruction error: motions and priors can be asymmetric.

A cleaner adapter can keep physical storage order `[0,...,15]`, use descending timestamps `[15,...,0]`, and retain `anchor_idx=0` **as an array index**. This changes temporal embeddings while preserving per-physical-frame row/noise pairing automatically in Stage I. Stage II must still be explicit: bank ordering is sorted by time (`storage.py:78`), so retrieve latents in descending timestamp order, source alpha 1, targets `[14,...,0]/15`. Prepend the fixed physical-frame-0 mesh and outputs are already in physical chronology. Do not call native Stage II with `anchor_idx=0` here: its sorted bank indexing would seek the wrong source. Equivalence to literal reverse + anchor-last requires a permutation canary, not assumption.

## Randomness and controls

`pipeline.py:280–303` creates a seeded generator and calls `scheduler.get_noise`; `scheduler/scheduler.py:123–136` draws shared noise **even when corr_noise=0**, then draws independent frame noise. Bare `torch.randn([1,16,...])` with the same seed does not reproduce the native initial tensor. Use the native noise helper or cache its output.

Forward internal physical order is `[0,1,...,15]`. Literal reverse plus anchor-last internal physical order is `[0,15,14,...,1]`. Therefore the same numeric seed without noise-row remapping is **not** matched physical noise. Couple reverse noise using forward rows `[0,15,14,...,1]`, with mask and contexts mapped consistently. Same-physical-order/descending-timestamps avoids this extra row mapping but still records tensor hashes and exact initial latent checks. Native Stage I needs an explicit initial-noise adapter for literal reversed arrays; avoid hidden denoiser monkeypatches.

Minimal future controls: (1) forward native reproduction; (2) permutation-only with identical physical timestamps and matched noise, expected near-equality; (3) true time reversal with matched physical noise; (4) forward independent-noise control to quantify ordinary stochastic variation. Keep the same cached source, 30 steps, CFG 7.5, crop, model weights and all 16 frames. Independent-seed variation cannot replace the paired-noise reversal contrast. No GT selects seeds, directions, frames, or interpolation weights.

## Cost and bounded next decision

The cheapest first test is **Stage-II-only timestamp reversal** with every forward Stage-I latent frozen and reordered consistently: it isolates decoder direction sensitivity while preserving all shape support. It is not full-pipeline reversal. Existing first-case Stage II cost was ~48 seconds for 15 nonanchor targets. A complete reversed Stage I + II adds roughly the previously measured ~528 + 48 seconds, excluding reused Stage 0/encoder. These are estimates from a prior case, not measured adapter timings.

The tensor shapes and sequential low-memory model loading remain unchanged; the existing native peak was ~10,255 MiB. Staying under 22 GiB is plausible, not verified for the new adapter. Keep one GPU job at a time until its own measured profile exists; no backward pass is required. Start with the inexpensive permutation sanity check and decoder-only contrast before spending another denoising run. If reversal discrepancies are near numerical/permutation noise, stop this mechanism. If discrepancies exist but match independent-seed variation or worsen external geometry/motion metrics, do not promote them as useful uncertainty or steering.
