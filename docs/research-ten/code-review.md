# Implementation review receipt

Independent cross-reviews were performed on disjoint ownership groups; these are code reviews, not scientific novelty or Gate-A approval.

| Finding | Resolution and evidence |
|---|---|
| H6 huge finite weight sums overflowed and selected the wrong video | Regression failed before fix; normalize by maximum before sum; local and remote full suites pass. |
| H8 integer multiplication could wrap and trigger enormous hash-cell enumeration | Python integer product plus pre-cast range guard; explicit regressions cover dangerous finite geometry and unrepresentable cells. |
| H4 long gaps created unbounded dense least-squares systems | `max_gap_frames=256`; explicit abstention preserves samples beyond the reference solver's budget. |
| H7 allocated full point×control×3 distance arrays | Chunked distance work; tests preserve allocation/output across chunk partitions. |
| H10 frozen generated window starts blocked actual reverse information flow | Only original input anchors remain immutable; terminal pseudo-boundary starts a reverse pass whose updated overlaps condition preceding windows. Regression changes future context and observes earlier overlap change. Independent multianchor/budget check passed. |
| H9 half-precision subtraction could overflow before promotion | Promote branches before subtraction/accumulation; extreme finite FP16 and NumPy-parity checks pass in local and remote Torch. |
| H8 disconnected ARAP fixture was a weak comparator | Explicitly labeled `uniform_arap_disconnected_weak_control`; standard contact baseline tie retained. No stronger comparative claim. |

Native runner independently checked for full-context time scaling, material IDs, normal convention, live three-branch calls, equal initialization, hard anchor preservation, source mutation and GT leakage; no blocking issue found. External720-second wall cap used in addition to per-stage checks.

Final tests:71/71 on local installed Torch and71/71 on existing remote environment, no skips. H1 shuffled control nonconvergence remains visible and is not represented as solver success. Natural benchmark and strong external method qualification remain separate work.
