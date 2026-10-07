# Full128 runtime pricing closure evidence

These archives retain the exact harness plans, attempts, receipts, logs,
builder outputs, validation output and GPU observations for the CPU-only
runtime-closure regression campaign. Each archive includes an evidence manifest;
all manifest members were hash-checked before export.

| Run | Outcome | Archive SHA-256 |
|---|---|---|
| `runtime-pricing-closure-red-r1-20261007` | Expected red: 310/311 tests passed; the real double-root integration test exposed the missing runtime historical-root parameter. | `0fabb62443432188cb3bfe84a711afc18d25f2de37a62116d83837c2f8a55ede` |
| `runtime-pricing-closure-green-r1-20261007` | Failed: the new integration test passed, but one pre-existing test method had been placed on the wrong fixture class. Preserved as a failed attempt. | `fcc5de773c374f3148bf05a431ea7918ff59d7688acfa5ff4b5a2392bead06ea` |
| `runtime-pricing-closure-green-r2-20261007` | Passed: 311/311 tests, 0 failures/errors/skips; native and harness exit 0; 72 code refs bound; GPU stayed at 0%. | `2d7b0290711ed37044f57083633eb9347f15be2669eeab022f532738977f9a68` |

Green-r2 harness plan digest:
`c33de8dedf07543a4f4017111879fea79010d83e9bd879947e84c6aab54b6ab0`.
Native plan digest:
`3db2ae28c578fe772981700955ea0379abaf5e0e0ca3c9ecdcdd04a1cd676239`.
Green-r2 receipt SHA-256:
`61e0a4113e9c1e16b6581f784c94823931b46b5d29982dd6139e9ac9c0a88b4d`.
The current source tree independently matches all 72 green-r2 `code_refs`.

This is software acceptance only. It did not generate or score an ActionBench
asset, qualify the official scorer, complete trusted replay, authorize a GPU
queue, or test a candidate method. The regenerated Full128 plan still requires
all qualification gates and a separate no-omission continuation for indices
10–15.
