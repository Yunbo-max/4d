# Native GPU resume and dependency repair — 2026-10-09

The user explicitly resumed GPU work with “可以开始正式实验了”. This permits the engineering prerequisites for the requested formal experiment sequence; it does not establish scientific admission for any candidate.

## First actual attempt

- Run: `paired-native-context-seed42-20261009-001`.
- Source: `462b6f5e5d23a98d9e71e66e5bb6025d6f42c577`; accepted CPU suite852/846passed/6skipped.
- Harness digest: `ff290ea97c4ed6db71d7fa852132dfe4290e9ced3677219abbfa6ca30a06da03`.
- Device: Tesla T4, `GPU-305023ac-8457-0ac4-0432-4ff19b30de46`. Historical R7 manifest describes its original2080Ti; no hardware equivalence claimed.
- Frozen FP16 low-RAM R7 manifest restored byte-for-byte (`b551cbcac49b9c71f7be5becfa331745c03bb96feae758847858a1b723047ae6`). Its exact model file inventory is an additional hard-linked view of the verified complete snapshots; full downloaded assets are preserved. Original source revision and tracked clean status verified.
- Seed42;16frames;two generations then raw replay;atol=rtol=0;64MiB capture cap;1attempt/0retry;8CPU,32768MiB,1exclusiveGPU;12600second explicit allocation,1800collection reserve. No historical timing reused.
- Actual outcome: failed while importing Diffusers0.39.0 before the first model generation. Final prerequisite integrity recheck matched; runner elapsed100.6943887937814seconds. All actual partial files retained, with no fabricated success bundle.
- Raw archive: `/root/rivermind-data/4d-native-compatibility-20261009/paired-native-context-seed42-20261009-001-failed-raw.tar.gz`, SHA-256 `dc08e73cb9bb98775e81acf12b350a3bfa025bad4d8e55e6eecc1a1117fc44d3`.

## Root cause and prospective repair

Diffusers0.39 attention_dispatch imports postponed string annotations into Torch2.4 custom-op schema inference, which rejects the string `torch.Tensor`. CPU mathematical fixtures and package metadata capture did not exercise this native import chain.

An independent source review identifies0.36.0 as the next compatibility candidate: its dispatcher uses concrete annotations;0.37.0 adds postponed annotations on this path. [0.36.0 source](https://raw.githubusercontent.com/huggingface/diffusers/v0.36.0/src/diffusers/models/attention_dispatch.py), [0.37.0 source](https://raw.githubusercontent.com/huggingface/diffusers/v0.37.0/src/diffusers/models/attention_dispatch.py). This is a source-based hypothesis until actual verification. The old0.39 environment is preserved;0.36.0 is installed without dependency replacement in a separate venv using the Tsinghua mirror.

The committed CPU diagnostic stages the inspected official Python closure and checks imports plus the actual None/layer_norm attention constructor/forward variants with tiny CPU fixtures. It loads no checkpoint and initializes no CUDA context. Red/green results and refreshed runtime metadata are required before separately reviewing a new GPU plan. A proposal to use0.32.2 was superseded before installation; an initial reviewer concern about an unsupported default norm was retracted after tracing actual caller overrides.


## Observed repair validation

The actual old-environment CPU diagnostic reproduced the same infer_schema failure in4.306749767623842seconds. The0.36.0 diagnostic completed all7 imports and both actual attention variants in5.805101944133639seconds; CUDA remained uninitialized. Runtime metadata capture completed in0.3193948296830058seconds. Exact plans,receipts and outputs: [repair-evidence/RESULT.json](repair-evidence/RESULT.json).

The separately generated repair GPU plan is `paired-native-context-seed42-20261009-r2`, digest `e209c7377f8b32a5f3280a20fc80ce56e2ae38762598dac8fb6837599573b6ab`. It retains the original generation/profile,zero comparison tolerances,full16frames,three receipt outputs,1attempt/0retry and12600second bound. The candidate scientific state remains unadmitted. Its execution state will be recorded separately; the source builder always emits `dispatch_ready=false` and cannot itself authorize execution.


## Repair attempt completed

The independently reviewed r2 attempt completed under the single-attempt harness. The exact launch authority and device inventory remain in [controller launch review](repair-launch/controller-launch-review.json). The attempt used the repaired Diffusers0.36.0 environment on Tesla T4 UUID `GPU-305023ac-8457-0ac4-0432-4ff19b30de46`.

- Run: `paired-native-context-seed42-20261009-r2`; harness digest `e209c7377f8b32a5f3280a20fc80ce56e2ae38762598dac8fb6837599573b6ab`; native plan digest `07365dc75ab33b7048de2ea2ba32ed1d47767ead9bb62d4822bc7f3d1b3aa041`.
- Unobserved generation: completed, 1595.3498215284199 seconds. Observed generation: completed, 1599.4099076380953 seconds. Replay: completed, 139.4388423934579 seconds. Total instrument elapsed: 3430.5481796972454 seconds.
- Stage counts: 30/30 Stage I frames, 15/15 Stage II frames, and 1/1 source-time query. Both 16-frame GLB sequences and both grid videos were emitted.
- Paired comparison: exact topology, identity and coordinates match at `atol=rtol=0`, maximum absolute error `0.0`; both sequence hashes are `1b199107ec0e09b36231bd8d5322b11ea09d99d8b72208f1b93f516c4ba07894`.
- Replay raw and mesh comparisons match with maximum absolute error `0.0`. The replay report records `dtype_matches=false` for the mesh comparison, so the run remains `replayed_unqualified`.
- Final status: `completed_unqualified`, `final_integrity=matched`, `all_comparisons_match=true`, `replay_qualified=false`, `native_context_qualified=false`, `scientific_effect_qualification=false`, and `dispatch_ready=false`. This is engineering transport/replay evidence, not candidate admission or an official score.
- Receipt hashes: `result.json` `58145ee7792b28c31bcf45a18d9614bea2046978ee08211110428240c78a825d`; `raw-manifest.json` `8bdce2261f2a64498c2da02653d2200e4730e632fca6b827285fbde5996f16b3`; `raw-evidence.tar` `fcceacd171edbd20a9431b3bdccf84b7e5687414b8e1306f7ca86af793894953` (79,452,160 bytes). The complete retained attempt archive is `paired-native-context-seed42-20261009-r2-completion.tar.gz`, SHA-256 `c967f9eedcb318b94337b315a270f03a39216fcaa5c65a2cfa1ea52495184759`.

The GPU is now idle. The next step is CPU-only receipt-bound consumption and baseline/scorer parity; no candidate scoring or 15-idea scientific result is authorized by this engineering completion.
