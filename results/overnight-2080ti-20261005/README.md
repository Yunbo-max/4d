# ActionMesh overnight census: 16/16 complete

2026-10-06. This is the completed result of the frozen 16-asset ActionBench
development census on one RTX 2080 Ti. Every asset has native ActionMesh
generation, a same-anchor stationary control, and both official scoring arms.

All 16 complete receipts were verified against the frozen protocol fingerprint.
No candidate method, training, or perception/QA suite was included in this
protocol. The original eight-hour window is retained as historical provenance;
the user-authorized continuation record allowed the finite inventory to finish
after that window without changing inputs, weights, seeds, inference settings,
or metrics.

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
- `completion-verification.json`: final receipt and source-hash verification.

Large model weights, caches, raw ActionBench downloads, and per-frame generated
outputs remain on the authorized GPU workspace and are not committed to Git.
