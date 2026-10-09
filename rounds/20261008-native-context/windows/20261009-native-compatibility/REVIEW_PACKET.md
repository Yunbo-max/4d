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


## Repair attempt launched

The independently reviewed r2 attempt is now running. The exact read-only harness observation is [repair-launch/status-snapshot.json](repair-launch/status-snapshot.json); [controller launch review](repair-launch/controller-launch-review.json) retains the user instruction, exact GPU/current inventory, repair parent and parent hard deadline. Subsequent live logs show16 input frames loaded, background remover and TripoSG loaded, and the100-step generator advancing. This is live progress, not a completed result.
