# Stepwise continuation — 2026-10-06

## Evidence checked

Starting repository revision: `28ff4bde041cefab55f058f3c921edb304dc4f5c`.
The newly returned C02 implementation and supervised execution records are
developmental. The native diagnostic used 24 queries and frames 1/8/15;
it is not a complete ActionBench comparison. Neither it nor CPU tests verifies
candidate quality, Natural Gate 0, IPCG, or a final scientific outcome.

## Completed engineering correction

`project_protected_step` explicitly solves
`min .5 (p-d)^T W (p-d)` subject to `C p=0` and zero pinned coordinates.
Eliminating pinned variables gives free unconstrained center
`d_f + solve(W_ff, W_fp d_p)`. The prior implementation omitted the second term.
This affects dense metrics coupling pinned and free coordinates; diagonal
metrics and calls without pins retain their previous mathematical behavior.
The existing pseudoinverse constraint projection is applied to this corrected
center. The full metric objective is now stated in the public function contract.

A regression test first failed on the old code: `d=(1,0)`,
`W=((2,1),(1,2))`, `p_0=0` returned `(0,0)` instead of `(0,.5)`.
A second analytic test combines cross terms with an action constraint.
All 47 tests in the repository's `research_math` suite passed locally with NumPy.
These tests are engineering/algebra checks, not a benchmark or native replay.
The new revision has not been executed on the GPU host.

## Next work in order

1. Qualify the full native baseline and simple controls using released
   ActionBench assets and the existing official scorer, preserving full
   16-frame sampling, timestamps, topology, anchor and native scoring budgets.
   Reuse historical caches only after their source/input/output bindings verify.
   Existing exposed assets are development assets, not fresh confirmation.
2. Implement and measure ordinary temporal smoothing and body-frame smoothing
   as baseline qualification controls. Freeze their parameters before inspecting
   their scores. Retain failures and complete raw outputs, not only summary flags.
   Compare C02 with ordinary geometry correction and an equal-norm scalar step;
   observed directional protection must be distinguished from step shrinkage.
3. Run one complete comparison unit as a timing/resource pilot on the actual
   single 2080 Ti. Keep 1,800 seconds of the 28,800-second window for collection.
   Admit subsequent complete units only using measured multi-arm costs.
   Historical two-arm timing and the 22-second decoder diagnostic cannot price
   the new full comparison. Do not promise all 15 methods within eight hours.
4. Establish current native Natural Gate 0 and concurrent IPCG before new
   scientific candidate implementation or Gate A freezing. Baseline reproduction,
   native evaluation qualification and natural failure census may precede them.
5. Finish the selected 15 implementation/protocol pairs against their mathematical
   assumptions and required simple comparators. Then run native comparisons,
   confirmation and full result verification. Record scoped success, failure or
   inconclusive outcomes per method; never infer effectiveness from unit tests.

Current status: 20 conditional mathematical constructions and 15 conditional
specifications exist; C02 has a developmental operator implementation. Fully
verified candidate designs, complete new native method comparisons, and formal
candidate success/failure outcomes remain zero. This checkpoint does not change
the mathematical batch's gate fields or claim experiment generation is finished.
