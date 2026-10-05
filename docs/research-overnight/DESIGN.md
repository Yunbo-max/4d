# 2080 Ti eight-hour native-benchmark investigation

User scope: one RTX 2080 Ti; eight hours; the user intends to start at
2026-10-05 23:00 Europe/London, with a reporting boundary at 2026-10-06 07:00.
The actual launch defines the hard eight-hour execution deadline. Downloads and
environment preparation use the separate `prepare` command before launch.
The old 22 GiB record is historical; the runtime must observe the actual card.

## Scientific scope and readiness

This is a bounded baseline/evaluator qualification and natural-case census.
No candidate has Natural Gate 0 PASS, IPCG CONCURRENT or frozen Gate A. The queue
must never invent these approvals or automatically launch a new method.

Generation: new ActionBench assets, excluding the eight inspected October 2
assets, selected by salted SHA256 order before new outputs. Compare unmodified
ActionMesh with a stationary first-frame control. Both use the same generated
anchor, 16 native frames and the official CD-3D/CD-4D/CD-M scorer. The control
tests whether predicted dynamics add task value; it is not the strongest
video-constrained deformation competitor and cannot close the tangent hypothesis.
Do not rerun the previously refuted unconditional surface correction or reversal.

Perception: optional only with supplied local 4D-Bench QA/video data and a local
Qwen2-VL-7B-Instruct checkpoint. Compare the official 18-frame concatenated
video input against three separately identified six-frame videos. Preserve
views [1,8,16], native integer-interval sampling, choices, max_new_tokens=128,
native first-option extraction and category aggregation. Use FP16/SDPA on the
2080 Ti, explicitly recording this common runtime configuration. No quantization,
frame reduction, novel interaction decoding or new QA labels are enabled.

## Capacity and complete work units

Single selected GPU, one GPU worker at a time. Generation requires >=11000 MiB
total and >=10769 MiB free, allowing 512 MiB above the historical 10257 MiB
generation peak. A stock 11 GiB card can qualify when sufficiently free. This is
one historical case, not a fit guarantee for new cases. Optional perception requires >=18000 MiB free and still needs
a real native-case calibration; these thresholds do not guarantee a fit.

Finite generation backlog: 16 new assets; initial complete-pair estimate 2400s.
Inference was historically ~650s; scoring/overhead of this new queue is unmeasured.
Estimate subsequent units conservatively from the maximum observed complete-unit
time, with a 1.35 multiplier. No promise that all 16 finish. Reserve 120s for saving.
Each generation unit includes generation, stationary control, both native scores,
artifact verification and a receipt. Each optional perception unit includes both
input arms, raw answers, native scoring, cost records and a receipt. Partial units
are pending comparisons, not reported gains. No sweep, training, paid API or rental.

## Evidence and decisions

Record protocol/source/data hashes, commands, actual GPU/versions, outputs,
failed attempts, latency/memory, all frozen IDs and scored/pending denominators.
Each completed unit gets scoped post-run verification. QA unit is the object;
generation unit is the asset. Report paired descriptive differences and uncertainty
only for complete pairs, and preserve missing units. A small or interrupted census
cannot establish population prevalence, substantive novelty or method efficacy.

STOP resource failures/OOM or unavailable native scorer; continue unrelated data
failures within the frozen inventory. Never silently replace failed IDs. Resume
the same output directory without resetting the deadline or discarding attempts.
Code/protocol/input changes reject resume. A new eight-hour window needs an
explicit new output path; it is not fresh scientific confirmation by itself.

Next scientific review: inspect actual failures and successes, qualify ordinary
video-constrained deformation and same-budget perception alternatives, audit
functional collisions, complete IPCG, then prospectively freeze Gate A.
