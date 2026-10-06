# ActionMesh overnight census: 16/16 complete

2026-10-06. This is the completed result of the frozen 16-asset ActionBench
development census on one RTX 2080 Ti. Every asset has native ActionMesh
generation, a same-anchor stationary control, and both official scoring arms.

The published queue, summary and completion supervisor report all 16 pairs
complete. The completion collector validates receipts on the GPU workspace;
the raw artifacts and a final 16-pair verification packet have not been published
here. See [the returned-feedback review](../../docs/research-overnight/FEEDBACK-20261006.md).
No candidate method, training, or perception/QA suite was included in this
protocol. The original eight-hour window is retained as historical provenance;
the user-authorized continuation record allowed the finite inventory to finish
if needed without changing inputs, weights, seeds, inference settings, or metrics.
The actual recorded run finished at 02:02 UTC (03:02 Europe/London), about four
hours two minutes after launch and within the original eight-hour deadline.

| Paired contrast (native - stationary) | Mean | Bootstrap 95% interval |
|---|---:|---:|
| CD-3D | -0.024519 | [-0.035694, -0.015058] |
| CD-4D | -0.063427 | [-0.102656, -0.030184] |
| CD-motion | -0.141513 | [-0.207971, -0.081337] |

Negative values favor the native dynamic result over the stationary control for
these metrics. These are conditional development-census summaries, not a
formal method-efficacy or Gate A claim.

Key artifacts:

- `summary.json`: all 16 paired metrics and the frozen denominator.
- `REPORT.md`: human-readable report.
- `queue.json`: attempts and completion states; all 16 are complete.
- `protocol.json`, `window.json`: frozen parameters, inputs, source and weight hashes.
- `completion-authorization.json`, `completion-status.json`: continuation provenance.
- `completion-verification.json`: early progress/source verification at 22:22 UTC,
  with one verified pair; identical to `progress-20261005T222233Z.json`, not final verification.
- `FEEDBACK-REVIEW.json`: source, denominator, timing and arithmetic checks on
  the returned metadata; raw receipt replay remains pending.

Large model weights, caches, raw ActionBench downloads, and per-frame generated
outputs remain on the authorized GPU workspace and are not committed to Git.
