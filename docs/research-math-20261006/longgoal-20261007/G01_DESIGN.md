# Stage-wide G01 design for the selected 15

Status: **complete source design, `generated_unexecuted`; dynamic scientific
admission and all Local/native evidence remain pending**.  The canonical machine
record is [G01_DESIGN.json](G01_DESIGN.json).  It does not resume GPU authority,
freeze a candidate protocol by itself, or turn any of the 15 candidates into a
verified method.

## Fixed native task

- Benchmark: released `facebook/actionbench` revision
  `2796071cbe6248422fcbeab3101fa9f9886cb7b9`, all 128 UIDs and all 16 frames.
- Producer coverage: ActionMesh generation seeds `42`, `314`, `2718`; scorer
  sampling seed `44`.  Current retained source supports seed 42 only, so the
  other two seeds remain an explicit source/interface admission dependency.
- Official endpoint implementation:
  `actionmesh/repo/actionbench/evaluate_dataset.py` calling
  `benchmark.py:compute_chamfer_3d_4d`; 100,000 surface samples, 10,000 ICP
  samples, 24 initial rotations and 200 ICP iterations.
- Metrics: CD-3D, CD-4D and CD-M (`cd_motion`), all minimized.  Native output,
  sampling, alignment and denominators may not be replaced by a surrogate.
- Environment: the existing native Conda path, one user-stated RTX 2080 Ti and
  no Docker.  Local must observe the actual UUID, VRAM, CPU/RAM/disk and cost.

The released README/source/paper distinguish production-ready ActionMesh and
ActionBench's paired 16-frame tracked point clouds; the local source closure pins
the exact evaluator bytes.  The design does not claim that the current public
weights reproduce an unpublished paper run.

## Numeric decision rules

For lower-is-better paired differences `d = treatment - control`, the fixed
minimum effects are `0.002` for CD-3D-primary candidates and `0.003` for
CD-M-primary candidates.  Guardrail noninferiority margins are `0.002` CD-3D,
`0.004` CD-4D and `0.003` CD-M.

These are absolute native units fixed before candidate outcomes.  They use the
official leaderboard's task-scale separations: ActionMesh CD-3D `0.054` versus
TripoSG `0.056`, and full versus fast ActionMesh `0.085` versus `0.089` CD-4D
and `0.153` versus `0.156` CD-M.  They are not the earlier unsourced “5%/2%”
development lines.

There are 98 predeclared confirmatory tests:

- each candidate versus every declared non-treatment role on its primary metric;
- each candidate versus frozen B* on both guardrails.

Use one familywise alpha `0.05`, Bonferroni over all 98 tests, and a trusted live
project callback for the paired family-cluster bootstrap: 200,000 basic one-sided
resamples, RNG seed `20261006`.  The independent unit is a reviewed asset family;
frames, scorer samples, hyperparameters and repeated seeds are nested observations.
PASS requires the adjusted upper bound below `-Delta` for every indispensable
primary contrast and at or below the relevant guardrail margin.  Incomplete or
imprecise evidence is INCONCLUSIVE, not a favorable zero or a method KILL.

## Prospective family split

Local must create a complete `family_by_uid` mapping for all 128 UIDs from actual
upstream metadata and input/geometry identity evidence. Each unit contains an
exact `g01-upstream-family-metadata-receipt`: its JSON pointers are evaluated
against pinned native metadata, so a self-asserted unrelated label is rejected.
A separate exact `g01-geometry-identity-receipt` pins geometry files and extractor
source; metadata, extractor and geometry categories may not overlap. The validator
reconstructs the canonical basis and `family_id = SHA256(canonical family basis)`.
UID/prefix/hash/base64-derived keys, category swaps and scorer/candidate outcomes
are forbidden. Receipt/derivation authors cannot perform the independent review.

The validator then performs the only allowed split:

1. Every historically exposed UID and its whole reviewed family is development.
2. Sort all other family IDs by SHA-256 of canonical
   `["4d-math-restart-20261006-v1", family_id]`.
3. First 12 families are D1; next 12 are D2; all remaining families are locked
   confirmation. At least 30 independent confirmation families are required;
   fewer makes the design INCONCLUSIVE rather than permitting a degenerate
   one-cluster bootstrap. This is only a floor; the separate precision gate may
   still block PASS.
4. D1 alone selects method parameters and B*.  D2 checks transfer/assumptions.
   Confirmation is never opened for tuning.

`research_math.g01_design` rejects incomplete mappings, unreviewed derivations,
changed source bytes, split leakage and insufficient remaining confirmation
families.  `prepare_g01_acceptance.py` stages that check as a single-attempt,
zero-GPU task in the existing harness and refuses to overwrite an old split.

## Fifteen-candidate coverage

Every row also includes B0 and a prospectively frozen B*.  The listed controls
are indispensable operation-isolating comparisons, not optional smoke arms.

| ID | Treatment | Other required controls | Primary | Guardrails | Source state |
|---|---|---|---|---|---|
| C02 | protected step | geometry-only, strength-matched blend | CD-3D | CD-4D, CD-M | complete, unexecuted |
| C01 | self-map subtraction | raw uncorrected, mean bias | CD-M | CD-3D, CD-4D | complete, unexecuted |
| C10 | pinned integrable solve | direct lift, qualified local repair | CD-3D | CD-4D, CD-M | complete, unexecuted |
| C13 | group acceleration | Gaussian, quadratic acceleration | CD-M | CD-3D, CD-4D | complete, unexecuted |
| C14 | corotational residual | world/body Gaussian | CD-M | CD-3D, CD-4D | complete, unexecuted |
| C04 | robust conic protection | deterministic, strength-matched repair | CD-3D | CD-4D, CD-M | complete, unexecuted |
| C03 | full smoothed unsquared estimator | intercept/unit/diagonal/full estimators | CD-M | CD-3D, CD-4D | partial; legal labels/export absent |
| C20 | joint monotone phase/amplitude | phase, amplitude, simple lag | CD-M | CD-3D, CD-4D | blocked on legal motion target |
| C11 | rotation-preserving stretch projection | ARAP, elastic repair | CD-3D | CD-4D, CD-M | complete, unexecuted |
| C12 | exact quadratic admission | fixed damping, backtracking | CD-3D | CD-4D, CD-M | complete, unexecuted |
| C15 | protected residual SVT | Gaussian, unprotected SVT, rank-matched TSVD | CD-M | CD-3D, CD-4D | complete, unexecuted |
| C05 | joint spatial labels | localized/temperature means, independent top-1 | CD-3D | CD-4D, CD-M | blocked on legal modes/scores |
| C08 | endpoint bridge | local tracker, smoother, decoder cycle | CD-M | CD-3D, CD-4D | blocked on legal chain/endpoints |
| C06 | area-marginal transport | row softmax, vertex-density transport | CD-3D | CD-4D, CD-M | blocked on feature/cost/support producer |
| C07 | partial mass with native fallback | full transport, confidence fallback | CD-M | CD-3D, CD-4D | blocked on C06 producer/fallback mapping |

The complete design deliberately retains the six blocked methods.  Their missing
scientific inputs prevent their implementation/admission; they do not justify
removing a comparison or fabricating arrays. The exact candidate/control
construction inventories, including their local transitive implementation
imports and the canonical native support closure, for the nine delivered chains
are frozen in the canonical design. The native job `code_refs` must equal the
canonical union of that frozen closure and the official scorer refs: omissions
and extra executable source are both rejected. The other six carry a
null approved closure and therefore cannot create a B* admission until a reviewed
child design freezes their real implementation bytes.

## B* and fairness

Every declared selectable control other than the B* placeholder must first pass
its own implementation/native admission.  On D1, choose the lowest primary
family mean across those controls, with a lexical tie break; B* aliases that
exact role and is frozen before D2.  An unavailable strong control blocks
weakening B*. Each `g01-b-star-d1-control-result` must enumerate the complete
D1 UID × three-seed denominator. Every unit is a canonical
`g01-b-star-d1-native-receipt` binding the scored sequence, completed generation
receipt, exact candidate/control native admission, comparison/scoring requests,
official result, raw manifest and raw archive. The admission binds candidate
spec, role, UID, seed, sequence, implementation closure, one-attempt/zero-retry
harness receipt plus its canonical executed harness plan/report and native
plan/receipt, producer/generation identity, exact official scorer closure and
a distinct-author review. Admission review must precede the official score; score,
unit receipt, aggregate result and freeze timestamps must remain prospective in
that order. The validator rejects even a coherently rehashed source substitution,
checks every raw archive member, then recomputes seed →
UID → family means, requires exactly one unit per UID × seed for every candidate
× selectable-control pair, then recomputes all 15 choices. The freeze binds the
admitted split, all receipts, the selections and an independent review.
Confirmation never selects B*.

Each candidate's arms share source video, anchor/context, generation seed and all
legal upstream information.  At most two numeric method parameters with at most
three prespecified development values each are allowed; strong controls receive
matched information, tuning opportunity and time.  Any changed search rule is a
new child design.

## Failures, cost and continuation

The denominator is candidate × UID × generation seed × logical role.  Preserve
preparation, bounds, no-solution, timeout, export and official-score failures;
never retry the same scientific attempt, zero-impute or replace a failed role.
PASS requires valid native evidence for every indispensable role over the full
frozen confirmation inventory.  Byte-identical roles may share one physical
score only with a frozen digest alias; they remain separate logical records.

The existing 28,800-second window and 1,800-second collection reserve remain.
Use one attempt/zero retries.  Measure the first complete multi-arm D1 unit in
each cost class and admit later work using 1.35× the observed maximum complete-
unit time.  Never reduce arms, seeds, dtype, scorer settings or comparisons to
fit a window.  A reporting boundary does not reset cumulative time or budget.

## Local acceptance order

At the exact delivered revision:

1. Run the common zero-GPU software acceptance builder from
   `LOCAL_AGENT_RUNBOOK.md`; it stages this design, all 15 bound specs, official
   source closure, validator, planner and tests.
2. Produce all 128 exact metadata/geometry receipts and their nested pinned
   metadata, geometry and extractor refs, then `inputs/g01/family-derivation.json`, an
   independent `g01-family-derivation-review`, and
   `inputs/g01/family-evidence.json`.  Do not use candidate/scorer outcomes.
3. Emit, inspect and execute the CPU-only plan:

   ```bash
   "$python_bin" "$project_dir/actionmesh/prepare_g01_acceptance.py" \
     --root "$project_dir" --skill-dir "$skill_dir" \
     --design "$project_dir/docs/research-math-20261006/longgoal-20261007/G01_DESIGN.json" \
     --family-evidence "$project_dir/inputs/g01/family-evidence.json" \
     --output-split "$project_dir/inputs/g01/family-split.json" \
     --run-id g01-family-acceptance-001 \
     --plan-dir "$project_dir/plans/g01-family-acceptance-001"
   ```

   Execute only the printed exact digest through the installed harness.  It asks
   for zero GPUs and does not clear GPU STOP.
4. After qualified D1 only, create the reviewed `g01-b-star-freeze`; rerun the
   same acceptance path with `--b-star-selections` and a new output/plan identity.
5. Candidate protocol admission remains per-candidate and requires current Local
   software/native acceptance, source-complete method, Natural Gate 0/IPCG,
   exact arm implementations, the shared split/criteria, trusted live scorer and
   analysis callbacks, measured resources and a separate exact-attempt GPU resume.

Return the plan, attempt tree, stdout/stderr, exact split/freeze bytes, hashes and
all failures.  A successful CPU design check establishes only source/design
consistency, not method correctness, benchmark qualification or a scientific
result.
