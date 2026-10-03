# Ten training-free 4D candidates: implementation plan

> Execution: parallel subagents implement disjoint modules; root integrates and reviews. The user supplied the full written method specification and explicitly requested all ten implementations.

**Goal:** Deliver executable numerical implementations of all ten hypotheses, their simplest controls, model adapters where needed, and honest validation receipts.

**Architecture:** A new `actionmesh/research_ten` package isolates interventions from the official backbone. NumPy implementations accept explicit observations/candidates; Torch adapters load lazily. Each method provides `demo() -> dict` returning JSON-safe constructed-control evidence; these are software checks, never natural-benchmark validation. Tests use unittest and independently check invariants. No test GT enters an inference API.

**Tech stack:** Python 3.10+, NumPy, optional existing PyTorch/ActionMesh for native adapters. No new global dependency installation.

**Spec:** [User's complete plan](user-ten-hypotheses-20261002.md).

## Global constraints

- Freeze model weights. Preserve source data, old results and negative evidence.
- Intervene one method at a time; account for every decoder query and rollout.
- Missing real observations, correspondences or natural long videos are explicit missing prerequisites, not fabricated data or scientific failure.
- Float64 numerical reference where practical; finite values, dimensions and masks validated.
- Original anchor/visible constraints are exact where specified; no arbitrary normalization removing true motion.
- Full contact detection/continuous collision guarantees and trained image scoring models are not implied by a numerical repair/selection API; document actual supplied inputs.

## Review focus

1. Rigid transforms, anchors and visible observations must remain unchanged where specified.
2. Zero confidence, all-invalid candidates, degenerate geometry and zero guidance must have explicit behavior.
3. Query/rollout budgets must count probes and updated sampling paths; no cached-velocity replay.
4. GT-only evaluation, candidate inputs and synthetic control truth must remain distinct.
5. Local numerical tests must not be reported as full natural-data GPU validation.

## Tasks

- [x] Geometry worker: `m01_elasticity.py`, `m05_scale.py`, `m08_contact.py` and owned tests. Implement observed elastic weights + local/global ARAP solve, robust stable-region similarity decomposition, local contact-normal repair with declared constraints. Provide uniform/motion/shuffled, bbox/global similarity, and plain feasible repair comparisons where applicable. Test rigid preservation, true translation, targeted strain and signed contact repair with bounded changes.
- [x] Tracking worker: `m03_correspondence.py`, `m04_occlusion.py`, `m07_sampling.py` and owned tests. Implement temporal dynamic programming over explicit candidates, two-sided visible-constrained trajectory recovery, and budget-accounted adaptive query allocation. Test identity continuity, no-endpoint abstention, exact visible constraints and no hidden extra queries.
- [x] Model worker: `m02_cycles.py`, `m09_guidance.py`, `m10_windows.py`, `actionmesh_adapter.py` if useful, and owned tests. Implement path-composition/cycle probes and anchored consensus, grouped cross-condition projection and controls, actual callback-based forward/backward rollout. Inspect official source before adapter implementation. Test analytic flows, anchor/time/normal correctness, unchanged non-conflict guidance and changed-state rollout. Run native checks only after root stages code remotely.
- [x] Root: `m06_selection.py`, registry, CLI, README and integration tests. Implement action-gated downstream-feasibility video selection on explicit measurable candidate features, reference selectors, and separate ranking diagnostic. Validate all-ineligible abstention, identity/action tradeoffs and no GT selector input. Every candidate has executable `demo` and a documented real input contract.
- [x] Integration: run full new package tests locally and on existing remote environment, execute all ten software demos, one actual native adapter check if cached inputs permit, independent cross-module review, fix defects, record exact validation scope and synchronize requested local repository.

For each implementation: write meaningful failing tests, run them, implement, re-run; log results. Shared package initialization, CLI and README are root-owned. Workers do not edit official code or commit/push. Root preserves user artifacts and stages only new package/docs/results.

## Execution rulings

- User's written plan plus “write code for all of these” supplies design and execution authorization. No repeated approval step.
- Independent modules can be implemented concurrently; each has a separate file and test ownership.
- No monolithic natural-data experiment or model-weight training is started merely to show that modules import.
