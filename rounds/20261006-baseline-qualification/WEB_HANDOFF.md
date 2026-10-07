# Web handoff — baseline/native qualification

Round: `20261006-baseline-qualification`  
Status: `generated_unexecuted`  
Parent main at authoring start: `766c955eae07f6e328033b196cb3b7dea88b096a`  
Delivered commit: resolve from the delivery receipt and verify by readback before execution.

This file is an immutable round handoff. Actual-run evidence belongs under `rounds/20261006-baseline-qualification/windows/<actual-window-id>/`.

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
- Scorer-parity separation: generated, not run at the current revision
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
12. `docs/research-math-20261006/revisions/20261006-mechanism-boundaries/selection.json`

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
directory, execute the whole software acceptance plan at this changed revision.
Confirm its `code_refs` includes the post-promotion parity finalizer; historical
127/127 evidence predates that source and is not current acceptance.
Then source-inspect and run the protocol-free engineering parity plan from
`ACTIONBENCH_SCORER_PARITY.md`, promote its complete passed output bundle and run
the committed CPU-only post-promotion finalization harness; direct finalizer
execution is forbidden. Preserve its exact one-UID scope. Scientific
baseline scoring remains blocked until a genuine source-backed protocol exists.

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

## Runtime supplement — 2026-10-06

After software acceptance, run the exact CPU-only environment capture in `NATIVE_RUNTIME_CAPTURE.md`. Retain the actual attempt and transfer both output files to their same project-relative `inputs/native-runtime/` paths. The scientific scoring plan reads that captured runtime and checks its current package/interpreter bindings. Eight new engineering checks are authored, unexecuted; mathematical/native counters remain unchanged.

## Return contract

After an actual run, create and push:

`rounds/20261006-baseline-qualification/windows/<actual-window-id>/REVIEW_PACKET.md`

The packet must contain the exact result locator, execution SHA/dirty patch, resolved host facts, all plans/digests/attempts/logs, current three-arm wall time and peak VRAM, source/data/protocol hashes, raw native scores, integrity report, paired decision, failures/repairs, evidence limits, and next executable step. Read back the pushed commit before reporting it.
