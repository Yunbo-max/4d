# Web handoff — baseline/native qualification

Round: `20261006-baseline-qualification`  
Status: `generated_unexecuted`  
Parent main at authoring start: `766c955eae07f6e328033b196cb3b7dea88b096a`  
Delivered commit: resolve from the delivery receipt and verify by readback before execution.

This file is an immutable round handoff. Actual-run evidence belongs under `rounds/20261006-baseline-qualification/windows/<actual-window-id>/`.

## Current source supplement: full decoder capture

The original round history below is preserved. For the latest unexecuted source
delivery, read [NATIVE_DECODER_CAPTURE.md](../../docs/research-math-20261006/NATIVE_DECODER_CAPTURE.md)
and the runbook's latest-source section. The optional `--capture-decoder` plan
uses the real upstream generation entry, captures complete native windows, and
retains the three simple arms and official scoring. Its new archive/manifest
are direct receipt outputs. It is not a candidate, native-context qualification
or an approved GPU plan. GPU execution remains stopped.

R3's historical raw 311-test receipt is now available and its payloads rehashed;
the historical-root runtime fix exists. That receipt does not certify later
source changes. Local's next action is current CPU harness acceptance; new GPU
work, raw scorer replay, Gate 0/IPCG and all 15 candidate results remain pending.

## Scientific objective

Qualify the strong simple controls and official/native ActionBench scoring chain on one complete original development asset. Produce a source-bound, protocol-bound, per-pass replay that supports a narrow baseline qualification and natural-failure analysis. Do not execute or claim any new candidate arm in this round.

## Starting evidence

- Current batch: `4d-math-20261006-mechanism-boundaries-r2`
- Mathematical constructions: 20
- Conditional selection: 15
- Complete candidate validation designs: 0/15
- Native candidate results: 0/15
- Returned setup: 127/127 software checks, native runtime, original asset,
  three control artifacts and scoring request accepted from the latest review packet
- Scorer parity/finalization: returned at `00b30fd`; nine three-arm metric comparisons had exact zero difference under the explicit deterministic compatibility backend; engineering evidence only
- C02 corrected development operator: software-tested historically, not natively qualified

The historical software-test receipts are engineering evidence only. They do not close native qualification.

## Read order

1. `AGENTS.md`
2. `LOCAL_AGENT_RUNBOOK.md`
3. `docs/research-math-20261006/CURRENT.json`
4. `docs/research-math-20261006/STEPWISE_PROGRESS.md`
5. `docs/research-math-20261006/BASELINE_CONTROLS.md`
6. `docs/research-math-20261006/BASELINE_SCORING.md`
7. `docs/research-math-20261006/SCORING_OUTPUT_INTEGRITY.md`
8. `docs/research-math-20261006/ACTIONBENCH_SCORER_PARITY.md`
9. `docs/research-math-20261006/actionbench-qualification-source-audit.json`
10. `docs/research-math-20261006/actionbench-full128-reproduction-contract.json`
11. `docs/research-math-20261006/actionbench-full128-reproduction-source-review.json`
12. `docs/research-math-20261006/actionbench-full128-snapshot-contract.json`
13. `docs/research-math-20261006/actionbench-full128-dataset-semantics-contract.json`
14. `docs/research-math-20261006/actionbench-current-release-unit-contract.json`
15. `docs/research-math-20261006/actionbench-complete-unit-admission-contract.json`
16. `docs/research-math-20261006/actionbench-full128-queue-pricing-contract.json`
17. `docs/research-math-20261006/revisions/20261006-mechanism-boundaries/selection.json`

Also read [NATIVE_RUNTIME_CAPTURE.md](../../docs/research-math-20261006/NATIVE_RUNTIME_CAPTURE.md) before resolving the runtime JSON. Its new CPU-only capture is pending Local acceptance.

## Code and builders in scope

| Purpose | Path | Current role |
|---|---|---|
| Three-arm construction | `actionmesh/research_math/simple_mesh_controls.py` | deterministic `native`, `world_gaussian`, `body_gaussian` arms |
| Control plan builder | `actionmesh/prepare_mesh_controls.py` | emits harness plan for three-arm construction |
| Scoring/integrity core | `actionmesh/research_math/control_scoring.py` | request validation, arm/score checks, per-pass output binding |
| Runtime capture | `actionmesh/prepare_native_runtime.py`, `actionmesh/research_math/native_runtime.py` | CPU-only installed metadata collection; GPU identity and scorer qualification remain pending |
| Acceptance plan builder | `actionmesh/prepare_control_scoring_checks.py` | emits parser-only unit-test plan |
| Scoring plan builder | `actionmesh/prepare_control_scoring.py` | emits native-scoring request/integrity plan |
| Scorer parity | `actionmesh/prepare_actionbench_parity.py`, `actionmesh/research_math/actionbench_parity.py` | emits/runs protocol-free engineering equivalence with exact source/input/output binding |
| Full-128 reproduction guard | `actionmesh/research_math/actionbench_full_reproduction.py` | checks the frozen complete-population/current-README rule; never runs the scorer or authorizes dispatch |
| Full-128 snapshot admission | `actionmesh/prepare_actionbench_snapshots.py`, `actionmesh/research_math/snapshot_admission.py` | CPU-only immutable revision/byte/file closure; generated, unexecuted |
| Full-128 dataset semantics | `actionmesh/prepare_actionbench_dataset_semantics.py`, `actionmesh/research_math/actionbench_dataset_semantics.py` | revalidates admitted bytes and all 128 tensor/camera/RGBA structures; generated, unexecuted, no scientific qualification |
| Current-release unit manifest | `actionmesh/prepare_actionbench_unit_manifest.py`, `actionmesh/research_math/actionbench_unit_manifest.py` | freezes the first canonical UID, exact input/source/model identities and later three-arm output/timing boundary; CPU-only, generated, unexecuted |
| Complete-unit admission | `actionmesh/prepare_complete_unit_admission.py`, `actionmesh/research_math/complete_unit_admission.py` | rehashes returned terminal unit records, outputs and telemetry; CPU-only, generated, unexecuted |
| Full-128 queue pricing | `actionmesh/prepare_actionbench_queue_pricing.py`, `actionmesh/research_math/actionbench_queue_pricing.py` | consumes only an admitted complete unit and freezes eight ordered capacity windows; CPU-only, generated, unexecuted, never dispatches |
| Full-128 per-UID freeze | `actionmesh/research_math/actionbench_full128_unit.py`, `actionmesh/research_math/complete_unit_runner.py` | revalidates a priced UID/window and exact inputs; authored, unexecuted, no harness window compiler or dispatch |
| Unit tests | `actionmesh/research_math/tests/` | software acceptance only |
| Evidence exporter | `scripts/research_evidence_20261006/export_feedback.py` | full historical closure exporter; not required for the three-file controller transfer |

All test, control-construction, native-evaluation, and integrity workloads must go through the installed skill's `scripts/run_harness.py`. A committed plan builder may be invoked directly only to emit/check its plan. Controller-only Git, SSH, and file transfer remain outside the project harness.

## Work units and gates

| Unit | Required input | Completion evidence | Gate |
|---|---|---|---|
| Software acceptance | exact delivered checkout, installed skill | harness attempt with observed test count, exit 0, logs and hashes | engineering only |
| Original asset restoration | one complete dev sample, GT, ActionBench source | UID/split, 16 frames, manifests and hashes, read-only source preservation | required before controls |
| Three-arm controls | verified original case | arm manifests/hashes for identical UID/frames/topology; seed 44, sigma 1.0 | required before scoring |
| Frozen scoring request | verified arms, GT, evaluator, protocol/runtime manifests | dry-run/report, source closure and hashes, strict request validation | required before native replay |
| Official/faithful parity | returned request/runtime/GT and exact delivered source | three-arm zero-tolerance comparison plus manifest-scoped sidecar | engineering implementation qualification only |
| Native replay | official source and admitted harness plan | raw outputs for every arm/pass, logs, metrics, wall time and VRAM | scientific evidence |
| Integrity replay | all native outputs and frozen inputs | strict per-pass binding passes without mismatch | required for qualification |
| Qualification review | valid paired metrics | narrow decision plus natural-failure/applicability analysis | closes this round only |

Controller file transfer may copy receipt-bound original bytes into the clean checkout; it is not a scientific workload. If those bytes cannot be restored and verified, return `blocked_missing_original_asset_bytes`. The scorer-equivalence plan may run after current software acceptance without a scientific protocol, but it cannot establish an accepted benchmark score. Scientific scoring still requires a source-backed frozen native protocol, matching runtime/device evidence and trusted official/harness replay. Do not improvise a surrogate.

## Frozen constraints

- one physical RTX 2080 Ti; native/Conda; no Docker;
- ActionBench revision `2796071cbe6248422fcbeab3101fa9f9886cb7b9`;
- same development UID, split, frames, topology, cameras, SMPL parameters, masks, and GT for all arms;
- all 16 frames; seed `44`; Gaussian sigma `1.0`;
- native scorer parameters `n_pts=100000`, ICP subsample `10000`, rotations `24`, ICP iterations `200`;
- 28,800-second hard window with 1,800 seconds reserved for collection;
- complete current three-arm unit timing before any queue construction;
- no candidate dispatch and no increase to the 0/15 result count.

The device UUID, driver, framework versions, sample UID, split, asset/GT paths, official evaluator entrypoint, and source closure are intentionally unresolved until Local source inspection. They must be frozen and hashed before replay.

## Prospective decision boundary

Apply the exact success/failure/insufficient-evidence rules in `BASELINE_SCORING.md`. Compare paired values for the same asset and report all three arms. A one-asset result supports only qualification of the baseline/scoring chain and an asset-specific natural-failure observation. It does not establish dataset-wide efficacy and does not validate C02 or any other selected candidate.

## First executable step

After resolving the exact delivered commit, host, interpreter and installed skill
directory, execute the whole software acceptance plan at this changed revision;
the returned 201/201 run predates the new unit-manifest checks. Verify the
retained parity/finalization hashes from `longgoal-20261007` and do not rerun that
expensive one-UID scorer chain unless its bound source/input closure changed.
Continue with immutable full-128 staging, snapshot admission, dataset semantics
and the deterministic unit manifest. Scientific baseline scoring remains blocked
until a genuine source-backed protocol exists.

The bounded primary-source audit at
`docs/research-math-20261006/actionbench-qualification-source-audit.json`
confirms that the current official repository tree and history publish no
per-sample results or one-asset threshold. Do not repeat that search or invent a
rule. The source-matched full-128 alternative now has a prospective reproduction
rule: all 128 objects must succeed and all three official means must fall within
the three-decimal intervals represented by the current README row. This closes
the rule-definition gap only. It is not part of the one-UID parity unit and
remains non-dispatchable until complete GT/prediction manifests, exact ActionMesh
seed-42 generation source/weights/config, current resource measurement,
installed-verifier protocol acceptance and fresh replay are bound. The official
scorer sampling seed remains 44; do not replace it with the generation seed.

The follow-up primary-source audit at
`docs/research-math-20261006/actionbench-full128-generation-source-audit.json`
now resolves the current public Git, submodule, dataset and four model-repository
revisions, but also proves the remaining provenance boundary. The public tree has
no ActionBench batch-generation driver or published prediction bundle; model
downloads are unpinned; and the leaderboard does not state its exact code/weight
revisions, dtype, low-RAM choice, hardware, dependency lock or retry policy. Use
the resolved revisions only as candidates for a newly versioned
`current-public-release reproduction`, not an exact replay of the unpublished
leaderboard run. The entrypoint defaults to seed 44, so the target row requires
an explicit generation `--seed 42`; the scorer still uses sampling seed 44.

Before any dataset-wide launch, Local must return verified immutable dataset and
model snapshot manifests, the byte-bound full-128 dataset-semantics admission,
the receipt-bound deterministic unit manifest,
actual RTX 2080 Ti memory/runtime facts, and a measured
complete non-fast seed-42 16-frame generation/export/scoring unit. The official
release documents 32 GB default and 12 GB low-RAM requirements but no 2080 Ti
result, and does not bind low-RAM or float16 to the published row. Preserve OOM
or dtype incompatibility as a natural failure; do not silently switch variant or
precision. No eight-hour queue may be generated from one-UID parity timing.
The unit contract fixes UID
`000-000_03b69da8d2c94b5999bcf2605ee2ecd9` before outcomes and defines a later
complete boundary with native, world-Gaussian and body-Gaussian arms plus all
three official score rows. The GPU complete-unit runner and single-GPU plan are
authored. The default BF16 attempt is a retained Stage-I OOM; the separately
versioned FP16 low-RAM repair completed its 16-frame generation after 792.54
seconds and subsequently completed all three official rows in 1,330.47 seconds
with a sampled 10,255 MiB peak. The delivered plan
binds the entire successful output closure in its native receipt, not only
`result.json`. Snapshot, dataset-semantic and unit-manifest admission have now
returned successfully. Its timing/VRAM cannot
price a queue until the CPU-only `prepare_complete_unit_admission.py` path
verifies the prospective contract, canonical run records, exact 117 receipt
outputs, exact 121 nested outputs (116 declared plus five retained evaluator
caches), three score rows and resource telemetry. That
admitted measurement—not code, live progress or old parity timing—is the first
evidence eligible to price a later queue.

The next conditional CPU-only pass is now authored. After the complete-unit
admission succeeds, run `prepare_actionbench_queue_pricing.py` exactly as shown
in `LOCAL_AGENT_RUNBOOK.md`. It applies 25% operational headroom to the measured
1,330.465551-second unit, yielding a 1,664-second unit timeout, 16 complete units
per 27,000-second workload budget, 26,624 planned seconds, 376 seconds of extra
workload slack, and eight ordered windows for the 128 UIDs. The generated
pricing manifest explicitly keeps queue approval, queue generation, dispatch,
native scientific qualification and candidate testing false. An admitted
harness window compiler and official baseline/replay qualification are still
required before any window is executable.

The per-UID input freezer and an explicit Full128 mode in the complete-unit
runner are now authored. They recompute the pricing receipt, bind a UID to its
named window, rehash its exact 18 dataset files and require the admitted FP16
low-RAM profile plus the exact priced unit timeout. Run all 265 software checks
at the delivered revision. Do not call the module or runner directly: no
harness-owned window compiler exists yet, so the queue remains ungenerated and
non-dispatchable. Return the complete-unit admission and pricing receipts first.

## Runtime supplement — 2026-10-06

After software acceptance, run the exact CPU-only environment capture in `NATIVE_RUNTIME_CAPTURE.md`. Retain the actual attempt and transfer both output files to their same project-relative `inputs/native-runtime/` paths. The scientific scoring plan reads that captured runtime and checks its current package/interpreter bindings. Eight new engineering checks are authored, unexecuted; mathematical/native counters remain unchanged.

## Return contract

After an actual run, create and push:

`rounds/20261006-baseline-qualification/windows/<actual-window-id>/REVIEW_PACKET.md`

The packet must contain the exact result locator, execution SHA/dirty patch, resolved host facts, all plans/digests/attempts/logs, current three-arm wall time and peak VRAM, source/data/protocol hashes, raw native scores, integrity report, paired decision, failures/repairs, evidence limits, and next executable step. Read back the pushed commit before reporting it.
