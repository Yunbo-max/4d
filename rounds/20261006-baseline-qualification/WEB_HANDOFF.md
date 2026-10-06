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
- Baseline control software: generated, not run in the current state
- Native scoring/integrity software: generated, not run in the current state
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
8. `docs/research-math-20261006/revisions/20261006-mechanism-boundaries/selection.json`

## Code and builders in scope

| Purpose | Path | Current role |
|---|---|---|
| Three-arm construction | `actionmesh/research_math/simple_mesh_controls.py` | deterministic `native`, `world_gaussian`, `body_gaussian` arms |
| Control plan builder | `actionmesh/prepare_mesh_controls.py` | emits harness plan for three-arm construction |
| Scoring/integrity core | `actionmesh/research_math/control_scoring.py` | request validation, arm/score checks, per-pass output binding |
| Acceptance plan builder | `actionmesh/prepare_control_scoring_checks.py` | emits parser-only unit-test plan |
| Scoring plan builder | `actionmesh/prepare_control_scoring.py` | emits native-scoring request/integrity plan |
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
| Native replay | official source and admitted harness plan | raw outputs for every arm/pass, logs, metrics, wall time and VRAM | scientific evidence |
| Integrity replay | all native outputs and frozen inputs | strict per-pass binding passes without mismatch | required for qualification |
| Qualification review | valid paired metrics | narrow decision plus natural-failure/applicability analysis | closes this round only |

Controller file transfer may copy receipt-bound original bytes into the clean checkout; it is not a scientific workload. If those bytes cannot be restored and verified, return `blocked_missing_original_asset_bytes`. The admitted scoring plan already wraps `research_math.control_scoring score` and the pinned `research_census_eval.py`; remaining blockers are a source-backed frozen native protocol, matching native runtime/device evidence, and trusted official/harness replay—not a missing runner. Do not improvise a surrogate.

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

After resolving the exact delivered commit, host, interpreter, and installed skill directory, execute the parser-only acceptance plan exactly as specified in `LOCAL_AGENT_RUNBOOK.md`. Source-inspect the generated plan and harness before approving its digest.

## Return contract

After an actual run, create and push:

`rounds/20261006-baseline-qualification/windows/<actual-window-id>/REVIEW_PACKET.md`

The packet must contain the exact result locator, execution SHA/dirty patch, resolved host facts, all plans/digests/attempts/logs, current three-arm wall time and peak VRAM, source/data/protocol hashes, raw native scores, integrity report, paired decision, failures/repairs, evidence limits, and next executable step. Read back the pushed commit before reporting it.
