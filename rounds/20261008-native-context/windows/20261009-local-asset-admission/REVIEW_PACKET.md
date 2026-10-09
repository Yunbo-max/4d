# Local asset acquisition and CPU admission — 20261009-local-asset-admission

## Observed result

At source commit `875d8e106a26be17bf80711b305d2b072523fcd8`, all five pinned
snapshots were downloaded and admitted on the SSH host. **2,356 files,
23,131,060,721 bytes** were hashed. All 128 ActionBench samples and their 2,048
PNG frames passed the byte-bound tensor/camera/image structural admission.
The prospective first-UID engineering manifest was frozen successfully.

| Snapshot | Files | Bytes | Revision |
| --- | ---: | ---: | --- |
| dataset | 2309 | 5,292,032,295 | `2796071cbe6248422fcbeab3101fa9f9886cb7b9` |
| actionmesh | 10 | 6,615,176,099 | `fb69228ba8a4df684907b5d259cff3c22fb722f1` |
| triposg | 11 | 7,946,494,238 | `2c1c516d22d58db486a058d98d31bb6177344e06` |
| dinov2 | 6 | 2,435,142,994 | `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c` |
| rmbg | 20 | 842,215,095 | `2ceba5a5efaec153162aedea169f76caf9b46cf8` |

| Actual run | Status | Native receipt seconds |
| --- | --- | ---: |
| `actionbench-snapshots-20261009-001` | completed | 86.413 |
| `actionbench-semantics-20261009-001` | completed | 46.601 |
| `actionbench-unit-manifest-20261009-001` | completed | 0.379 |

Every run used the installed Research_Autopilot runtime at
`bd368630ef623efa16c6c28fab894d692f82ee4d`, a committed plan builder, its reviewed
printed harness digest, one CPU, zero GPUs, one attempt and zero retries.
Snapshot/semantics RAM reservations were 2,048 MiB; unit-manifest RAM was 1,024
MiB. Inner ceilings were 1,800 / 3,600 / 900 seconds respectively, with 60 seconds
of outer allowance. No project workload was invoked outside the harness.

## Evidence and promotion

[RESULT.json](RESULT.json) contains exact plan/receipt/output references and
hashes. [outputs/](outputs/) retains the complete snapshot, dataset semantics and
unit manifests. [evidence/](evidence/) retains the three plans, receipts,
harness reports, actual attempts, process guards and stdout/stderr.

Remote working root: `/root/rivermind-data/4d-native-preparation`.
Each successful output was rehashed against its actual receipt and copied only
into an absent target under `inputs/actionbench-full128-snapshots/`.
No output JSON was rewritten after promotion.

Raw archive: `/root/rivermind-data/actionbench-assets-20261009-raw-evidence.tar.gz`

SHA-256: `b18a1f9abb944be142167c0544b96846a2693297862f992541aa7a7235fdc78a`

A matching local copy is under
`/Users/yunbo/Documents/4d/results/local-asset-admission-20261009/`.
The raw archive includes the complete actual plan, harness and attempt trees for
all three CPU jobs; large model/dataset assets remain on the SSH host.

Official ActionMesh revision/tree and the TripoSG submodule were checked:
`d5c01f5045df55819e337369c9617f603c667e00` /
`aae2874f931d21e038966ecfde312638fff2091a`, with submodule
`fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`. Tracked source was clean.
The source bundle's remote SHA-256 matched the local original before use.

## Acquisition repairs

Package tools were installed in a separate environment using the Tsinghua PyPI
mirror. Hugging Face downloads used `hf-mirror.com` with immutable revisions.

- Dataset pagination incorrectly linked back to the unreachable primary host.
  A controller-only CLI request hook routed those pagination requests through
  the configured mirror.
- Slow HTTP weight downloads were switched to the installed `hf_xet` transport;
  completed files, stopped-attempt logs and partial bytes were retained.
- DINOv2 and TripoSG Xet requests initially failed with HTTP 401. A metadata
  probe found a cached token expiration of `1791509056` at observed time
  `1791546974`; a cache-busting refresh returned expiration `1791547874`.
  The transport wrapper refreshes public download tokens without changing asset
  revisions. Both subsequent download commands completed successfully.
- GitHub submodule acquisition and large SSH uploads stalled. The exact local
  Git bundle was transferred as a verified prefix and missing tail, then its
  complete SHA-256 was checked before Git checkout. Failed partial transfers
  remain retained; no scientific attempt was retried or repaired after the fact.

[acquisition/](acquisition/) includes controller requests, scripts, exit codes,
source identity, progress observations and URL-redacted logs. Original download
logs and partial assets remain under
`/root/rivermind-data/4d-asset-acquisition-20261009/`.
The controller helpers only acquire or collect files; they are not scientific
execution or admission entrypoints.

## Scope and remaining boundary

**GPU STOP remains active. Native candidate results and Local method
verification remain 0/15.** The prior common software acceptance remains
843 passed / 6 retained-native checks skipped, delivered separately.

This delivery establishes engineering input byte/revision/file closure and
released-data structure. It does not establish tracked correspondence,
model loading, runtime/CUDA compatibility, official scorer qualification,
scientific admission, candidate effects or dispatch readiness.

The current physical device is Tesla T4 UUID
`GPU-305023ac-8457-0ac4-0432-4ff19b30de46`, not the historical RTX 2080 Ti.
The pre-admission observation showed zero used VRAM and zero GPU utilization.
No inference/scoring job, GPU campaign or running project supervisor was started.

Remaining requirements include source-derived independently reviewed G01 family
evidence, qualified native runtime and baseline/scorer evidence, candidate
Gate 0/IPCG/design admission, and an expiring exact-attempt GPU-resume
authorization for the observed device. The frozen UID
`000-000_03b69da8d2c94b5999bcf2605ee2ecd9` is an engineering calibration input;
no calibration GPU attempt or GPU plan was emitted by this delivery.
