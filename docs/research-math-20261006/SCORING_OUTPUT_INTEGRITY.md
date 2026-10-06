# Native output binding repair — 2026-10-06

Status: **generated_unexecuted**. This baseline/evaluator repair resumes `main`
at `5d05e9c9cb652d7110fc1a909f6dc634350a3820`; no candidate is admitted.
The current mathematical revision remains `mechanism-boundaries-r2`.

## Source finding and generated repair

The three-arm runner previously checked a score file's `case_id`, declared
denominator and exit status, then copied its metrics. It did not bind the file's
UID, full sampling protocol, evaluator/source provenance or prediction/GT/report
hashes before producing its repeated-score and paired readouts. The request's
before/after hash checks do not by themselves bind those fields in the returned
score file. This finding is from source inspection, not an observed GPU failure.

`research_math/control_scoring.py:validate_native_output` now requires:

- Exactly one frozen asset with the current per-pass manifest path and hash.
- Exact UID, case/arm path, full native protocol including seed, and admitted device.
- The current census evaluator hash and, for successful rows, all six pinned
  official source hashes at the expected source directory.
- The same completed arm's mesh, original GT and generation-report paths/hashes.
- Zero scorer exit and three finite, nonnegative native metrics for success.

A native error row retains the asset and its error even if backend loading or
prediction loading prevented complete input/source provenance. It cannot enter
a complete pair. An inconsistent score file becomes an error pass, while the
original score JSON and process logs remain in the attempt for diagnosis.
The repair changes acceptance, not the official computation, input population,
sampling counts, scoring seed, model parameters or Gaussian sigma.

Added **18 parser-only engineering checks** to the existing 33 in
`test_control_scoring.py`. They cover native JSON identity/protocol/input/source
mismatches, failed preparation, nonzero success exits and preserved backend
errors, structured/path mismatches and canonical nonescaping symlink references.
Literal score fields and dummy source bytes are only parser inputs;
they are never sent to a benchmark or official metric function. The checks were
authored and have **not been executed** on Web. The initial 16 checks preceded
the main repair; two checks followed independent source review. Red/green and
whole-suite acceptance remain Local obligations; no passed-test count is implied.

## Local acceptance and continuation

Local fetches the exact delivered commit into an isolated checkout, preserving
running and dirty worktrees. Use the existing native Conda environment on the
GPU host under Local's SSH controller. No new environment or container is needed
solely for this source repair. The whole-suite builder already stages these files.
These are planned commands, not an execution report:

```bash
project_dir=/absolute/path/to/isolated-4d-checkout
skill_dir=/absolute/path/to/current-installed/research-autopilot
python "$project_dir/actionmesh/prepare_control_scoring_checks.py" \
  --root "$project_dir" --skill-dir "$skill_dir" \
  --run-id control-scoring-integrity-software-001 \
  --plan-dir "$project_dir/plans/control-scoring-integrity-software-001"
python "$skill_dir/scripts/run_harness.py" \
  "$project_dir/plans/control-scoring-integrity-software-001/harness.json" \
  --root "$project_dir" --execute --approved-plan-digest PRINTED_DIGEST
```

Resolve these host paths using the retained Local SSH profile; Web has not
observed the remote paths or interpreter. Retain the exact tested revision,
test count/result, full attempt logs and code hashes. A failed check leaves
acceptance pending; preserve its evidence and repair within scope.

Regenerate scoring requests/plans after Local software acceptance because
`control_scoring.py`'s bytes have changed. Do not rewrite old request hashes or
mutate an active attempt. Cached sequences can carry forward under their original
verified input/report bindings. Continue the source-backed frozen native
qualification and official/harness replay requirements in
[BASELINE_SCORING.md](BASELINE_SCORING.md); this parser repair does not satisfy them.

Return all six pass records, their raw native JSON/logs, source/input bundle,
request/plan hashes, actual Conda/dependency and GPU records, and the Local
software acceptance packet. A matching static file or repeated wrapper call
still does not certify official/harness parity or nonce-bound trusted replay.

## Scientific status

No new software checks, scorer processes, GPU jobs, replay or timing measurements
were run in this Web round. Verified candidate designs, complete new-candidate
native comparisons and formal candidate success/failure conclusions remain
**0/15**. Current Natural Gate 0 and IPCG remain pending. Source hashes establish
identity, not scorer authenticity or method efficacy.

Next: Local whole-suite acceptance, then existing baseline-only native
qualification with retained original development artifacts. Full-unit inference,
control preparation, scoring/replay and collection timing on the actual card
still precedes admission of an eight-hour candidate queue. Preserve the 1,800-second
collection reserve; neither cached scoring nor historical two-arm costs price it.
