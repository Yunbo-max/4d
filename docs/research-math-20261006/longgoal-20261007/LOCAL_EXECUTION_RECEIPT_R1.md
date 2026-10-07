# Local execution receipt R1 — 2026-10-07

This receipt records independently checked engineering evidence. It does not claim native scientific qualification or candidate-method results.

## Current remote state (SSH target :30246)

At 2026-10-07 19:40:33 UTC, the r9 campaign `population-gpu-current-r9` was terminal at 9/9 completed. The driver and all nine worker PIDs were absent from `/proc`; the original SSH shells still existed but were not experiment processes. RTX 2080 Ti UUID prefix `GPU-4910…` was idle at 0% utilization and 0/22,528 MiB, with no compute process. Free disk was 16,728,072,192 bytes. The latest population-009 official log ended with `Success 1 / Failed 0`; the live check found no new error lines. No run was launched by this audit.

The retained canonical r9 state hash is `c6615919ff39b3c0445ccb0358f7c23af7026f4365b657c8e511ab27e564b862`. A read-only harness status observation captured at 19:12:37 UTC has SHA-256 `744c4b4630113719a75194caf53a6f40d072eeb45c84c24ad766dc33080ab75b`; the reconciliation sidecar is `4ad874da468837a18c02b73d84322c6a9937255dda76d9175445decea877897a`. These records describe an engineering baseline campaign, not a qualified scientific result.

All four model groups were rehashed on the host and matched their admitted manifests: ActionMesh 10 files / 6,615,176,099 bytes; TripoSG 10 / 7,946,492,674; DINOv2 4 / 1,217,526,906; RMBG 9 / 176,407,656. There were zero missing or mismatched files.

## CPU-only r10 input freeze

The installed harness completed eight CPU-only manifest batches: 118/118 unique ActionBench assets, indices 10–127, all exit code 0. This did not run inference or use the GPU. The final summary is at the staging path `/root/actionmesh-research-staging/4d-status-observation-r1/cpu-manifest-execution/final-summary.json`, SHA-256 `8f77eccbc29fa1e00a1835264877db623b0e711845b5d89e2570629d9938875b`. It records 236 checked stdout/stderr references and no nonempty stderr.

## Independent archive and qualification limits

The returned text-evidence bundle was independently checked against its embedded and external manifests: 1,617 payload files, 5,448,428 bytes, no images or model weights, all per-file sizes and SHA-256 values matched. Archive SHA-256: `34967a5fd1d259be0398aebd538e8f9347223ffd9128ee7ba63e84dda0f4a126`. External manifest SHA-256: `94142bfa23af6fa0bdeb7bc22c3043ccdc8a431432a3a77fe934fe61a046074a`. The bundle remains in the staging area; this GitHub commit contains this compact audit receipt, not the raw bundle.

The older complete-lowram-r7 three-arm baseline receipt (`ec72d1aa2301d73cbc62eb61bda4e588ebc7ebbb949c17f542b7dc3e21f43909`) has 139 bound references verified, but it used GPU UUID `GPU-b544…`, which differs from the currently inspected `GPU-4910…`. It is not a current-device qualification. Formal baseline/native qualification is still open, and candidate-method results remain **0/15**.

The current software/harness acceptance at source commit `b9ce349cede47587494190a821a66e3ab9678a2b` passed 294/294 checks. The source change is on `main` at commit `4bbc33852e63e458d8f909770d167f51c345e756`. Software acceptance is engineering evidence only.

## Next step

Audit exact UID/window coverage using the retained r9 terminal receipts and r10 manifests. Preserve already completed UIDs, form only nonoverlapping eligible windows, and keep all work CPU-only until the current-device native/scoring gates and the harness window compiler authorize execution. Do not interpret input manifests as predictions or method tests.
