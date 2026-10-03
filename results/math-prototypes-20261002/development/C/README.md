# C: contact-event mathematical prototype

Scope: constructed, one-endpoint NumPy controls. No natural interaction dataset, model inference, SSH, GPU, network, or installation.

Files:
- research_math/contact_events.py
- research_math/tests/test_contact_events.py

Run from the actionmesh directory:
- python -m unittest research_math.tests.test_contact_events -v
- python -m research_math.contact_events --output <new-file.json>

Verification:
- red.log: tests written first; explicit missing-module assertion failed, behavioral tests were skipped until the implementation existed.
- green-first.log: 12 tests passed (11 behavioral controls plus availability assertion).
- package-tests.log: all 40 research_math package tests passed at this checkpoint.
- controls-v1.json: six deterministic constructed controls.
- manifest.json: source, tests, output and log hashes.

Observed controls:
- Missing event, wrong partner, and reversed order: endpoint repair succeeds; the finite-grid monotone retiming comparator does not satisfy the constraints.
- Timing-only: both succeed.
- Insufficient displacement budget: both fail; direct repair reports a narrow analytic infeasibility certificate.
- Unknown observation: no negative-contact constraint is added; unchanged native output remains feasible.
- Same-information direct continuous solver equals the wrapper on all six controls. There is no evidence of a discrete-layer advantage in this exact-evidence setting.

Technical boundary:
K-best dynamic programming optimizes additive typed event edit costs only. It cannot claim a joint geometric optimum. Geometric feasibility is audited on the shared endpoint trajectory. Unknown/unobserved events use null operations; explicit noncontact evidence differs from missing annotations. Positive target constraints yield a convex quadratic endpoint problem under per-frame L2 displacement balls. Explicit noncontact projections are a heuristic and return unresolved unless a supported infeasibility certificate applies. Contact modes are event labels, not a physics model. Retiming results use a finite time grid and failure is not labeled a proof against arbitrary continuous retiming. No 4D/cross-model efficacy or novelty conclusion is supported.

