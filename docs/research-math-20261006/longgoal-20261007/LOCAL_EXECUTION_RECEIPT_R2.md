# Local execution receipt R2 — 2026-10-07

This receipt records verified software and plan-preparation progress. It does
not claim baseline qualification, candidate results, or GPU execution.

## Supervision and current campaign state

The independent `gpu_watch_r9` agent performed a fresh read-only host check at
20:39 UTC. The retained `population-gpu-current-r9` harness state is terminal
at 9/9 completed, with no running, failed, or pending tasks. The single GPU was
idle, with no compute process. The official population-009 scorer log ended
successfully and had no newer error lines. No workload was started during this
receipt. The canonical state hash remains
`c6615919ff39b3c0445ccb0358f7c23af7026f4365b657c8e511ab27e564b862`.

The earlier CPU-only r10 manifest batches are also complete: 118/118 unique
assets at population indices 10–127, across eight completed batches. Their
retained summary hash is
`8f77eccbc29fa1e00a1835264877db623b0e711845b5d89e2570629d9938875b`.
All 118 outputs and receipts were rehashed successfully. These manifests are
input evidence, not inference results.

## Historical evidence closure repair

The historical R7 pricing verifier expected the executed harness plan at
`runs/harness/complete-lowram-r7/plan.json`; it incorrectly looked for the
unarchived builder input `plans/complete-lowram-r7/harness.json`. A regression
test now binds the retained plan path to the admission's exact hash. Two more
tests cover historical archives that contain newer, explicitly unmatched R9
runtime rows: those rows may remain in the pinned manifest, but cannot supply
historical bytes for any reference.

The manifest also omitted five scorer bytecode-cache files because its nested
inventory uses `path/bytes/sha256` rows instead of the normal `path/sha256`
reference shape. Each original cache was checked against the already pinned
runner `result.json` inventory and copied byte-for-byte from the retained R7
worktree into the separate, exact-commit historical verification clone. No
cache was regenerated and no tracked historical source was changed.

The remote installed harness results are:

| Run | Result | Outer plan digest | Receipt SHA-256 |
|---|---:|---|---|
| `closure-archive-acceptance-r4-20261007` | 302/302 passed | `de93cd00e128ef020978b6d858ffd414efbe11697c3031d060032e64687e9859` | `55dec8c677bef768a4ad4d11ae51920df35e9b9ff746240b2c25b6e22848653a` |
| `closure-archive-acceptance-r5-red-20261007` | 302 passed; 1 expected failure and 1 expected error in the two new tests | `d1570a4823a8663608dd28a328a93095d768604fd6bef2506c08d0eb78d492f7` | `dd378fb5de5b28d3360d304b6fa59882fd0c52a7d7f3e22f3f43222c7272144f` |
| `closure-archive-acceptance-r6-green-20261007` | 304/304 passed | `298a5ce7215d4570a80580b9a08132c3b2fa4723c40ec147a58da32659f2a90c` | `673c32ee701b84a57da85bc51c4bdb94e65366c890f51503c4d30a54499384c5` |

All three runs used the installed remote harness and zero GPU resources. R5's
one failure and one error were the expected TDD red checks for the two new
behaviors; R6 passed both. The R6 native attempt exited zero and its stderr was
empty.

## Full128 window plan

After the evidence-closure changes, the window-02 builder successfully created
and the official harness validated a plan for indices 16–31. It contains 16
serial single-GPU jobs, each with 232 input refs and 60 code refs; the priced
workload is 26,624 seconds within a 28,800-second outer window. Its outer plan
digest is `d7e5558a08aafcc1c996ebae558aa74b77eda42c9179ff1bb01291feccab6ab9`.
The plan is non-overlapping with the retained R7 index 0 and R9 indices 1–9.
Official validation returned `validated`; execution was not requested.

The plan itself reports `execution_started=false`,
`dispatch_ready=false`, `queue_approved=false`,
`native_scientific_qualification=false`, and
`scientific_effect_qualification=false`. Indices 10–15 remain an explicit
coverage gap for later planning; input manifests alone do not close it.

## Remaining qualification

The scorer parity artifacts in R8 are engineering evidence from an earlier GPU
allocation. Current-device native qualification, trusted replay, the full
finite-population baseline, and all 15 method comparisons remain incomplete.
Verified native candidate results remain **0/15**. No GPU workload or candidate
method was started in this continuation.
