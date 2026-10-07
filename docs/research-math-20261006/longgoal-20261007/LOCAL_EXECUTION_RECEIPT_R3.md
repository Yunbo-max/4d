# Local execution receipt R3 — runtime closure and R9 evidence

Recorded: 2026-10-07 UTC. Base checkout: `6d630ef5fe0d01b364eaa73e2f375e2c63393331`.

## Full128 runtime-pricing closure

The isolated branch adds a runtime double-root check for Full128 units. The
per-UID freezer now receives the exact historical R7 root and an attempt-staged,
hash-pinned archive manifest. Before inference it checks the historical Git
revision and tracked cleanliness, the manifest digest, every matching archived
blob, and the five canonical pricing files against both the historical checkout
and archive evidence. The native plan builder includes the two required runtime
arguments. The exact red/green harness archives are retained in
`runtime-pricing-closure-evidence/`.

The red run exposed the missing runtime resolver (310/311 passed). Green-r1 is
retained as a failed attempt because of a test-fixture placement error. Green-r2
passed **311/311**, with no failures, errors or skips; native and harness exit
codes were 0, GPU usage was 0%, and all 72 current source refs match the frozen
plan. Green-r2 harness digest:
`c33de8dedf07543a4f4017111879fea79010d83e9bd879947e84c6aab54b6ab0`.
Receipt SHA-256: `61e0a4113e9c1e16b6581f784c94823931b46b5d29982dd6139e9ac9c0a88b4d`.

An independent source review found no critical issue in the runtime fix. The
runbook now uses the verified historical root and manifest paths, fails fast on
preflight errors, and requires each native attempt input closure to include the
manifest plus all 170 unique archive blobs. Its command block passes `bash -n`.
The reviewer also noted a minor hardening gap: a symlink in a parent component
of the historical-root path is resolved and accepted. A CPU-harness TDD plan to
test this case was rejected by automatic review before the test or plan was
written remotely, so no unverified code change was retained; this remains a
follow-up.
The plan remains ungenerated and undispatched pending live canonical inputs and
scientific gates. Indices 10–15 still require a separate no-omission continuation.

## R9 terminal-state evidence

The isolated green-r2 root froze the retained `population-gpu-current-r9` state
and official status into an immutable pair. Snapshot execution receipt SHA-256:
`408ae70b7bd221bda2cbefe048d26309f8f1b1b3673a254e8dd33496097e92c4`. The
snapshot records 9 completed, 0 running, 0 pending and 0 failed. The exact
original observation time is `2026-10-07T19:12:37.209259Z`; it is not a new
status observation. State SHA-256:
`c6615919ff39b3c0445ccb0358f7c23af7026f4365b657c8e511ab27e564b862`; status
SHA-256:
`744c4b4630113719a75194caf53a6f40d072eeb45c84c24ad766dc33080ab75b`. All five
exported payloads passed local size/SHA verification; the export receipt is
retained at `r9-snapshot-only-r1/EXPORT_RECEIPT.json`.

At `2026-10-07T22:06:05Z`, the GPU was idle (0%, 0/22,528 MiB, 35°C) with no
compute, harness, native-runner or scorer process. Staging disk had 11.07 GB
free; the data volume had 1.32 GB free. No Full128 GPU work was started.

## Remaining gates

The installed checker lacks the raw UID008 request, sample manifest, parity
sidecar, original device samples and an independently verifiable GPU UUID, so an
exact same-configuration scorer repeat cannot yet be planned. The 8 missing
Full128 canonical inputs were found in the historical staging root, but the
attempt to stage them into the isolated green-r2 project root was rejected by
automatic review before writing. The user has been asked whether to authorize
that exact hash-checked, no-overwrite transfer. No reconciliation or Full128 plan
was generated. Baseline/native qualification and trusted replay remain pending;
the 15 candidate methods still have 0 native results.
