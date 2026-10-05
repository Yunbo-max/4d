# Candidate backlog and admission status

All four are hypotheses, not validated methods. Training-free execution and one
2080 Ti are fixed. None has Natural Gate 0 PASS, IPCG CONCURRENT or frozen Gate A.
The overnight baseline queue cannot enable them automatically.

| Priority | Idea | Mathematical mechanism | Evidence needed before implementation | Strong/simple comparisons |
|---|---|---|---|---|
| 1 | Separate normal and tangential motion correction | First-order surface-distance residual responds to `nᵀu`; a tangential component needs input-video correspondence rather than nearest-surface attraction. | Natural cases where material motion, rather than geometry/registration, is wrong; reliable input-only camera registration and correspondence confidence. | Unmodified ActionMesh, stationary anchor, ordinary video-constrained deformation, confidence masks. Previously negative unconditional surface/normal attraction does not satisfy this requirement. |
| 2 | View × time interaction | `I = s11 − s10 − s01 + s00` removes separable view/time terms only under the stated local decomposition; it can also amplify noise. For K=3, the proposed contrasts require `2K+2=8` visual forward passes. | Native hallucination errors, usable visual logit access, same-budget alternatives, functional novelty relative to temporal/spatial contrastive inference such as SEASON. | Native concatenated input, separate view input, temporal-only and view-only contrasts, same visual/token budget. |
| Reserve | Probabilistic reverse attention chain | For forward transition `C`, Bayes reverse is `R[j,u]=π[u]C[u,j]/Σvπ[v]C[v,j]`; a transpose alone need not be a normalized reverse conditional. | Real Fast4D attention extraction and native correspondence failures; code-level functional collision and IPCG. Fast4D code path is not yet qualified. | Transpose, dual softmax, Sinkhorn, matched compute. |
| Reserve | Correlation-aware view evidence | Under equal variance and pairwise correlation ρ, effective sample count is `m_eff=m/(1+(m−1)ρ)`. A similarity heuristic is not automatically an estimate of error correlation. | Native multi-view errors and a defensible input-only proxy for correlated error. | Averaging, duplicate removal, confidence weighting, matched image/token budget. |

Tonight produces evidence for the first two priorities using existing published
tasks and simple baselines. It does not test their proposed mechanisms. GT is
reserved for native evaluation; no GT alignment or material correspondence may
guide the predictor. A gain on this development cohort cannot establish novelty
or provide an independent confirmation set.

Next action after complete native receipts: inspect paired successes/failures,
qualify the stronger ordinary alternatives, freeze a six-field Parent Problem,
run Natural Gate 0 and functional collision review, complete IPCG, then freeze
Gate A with a full native small-N calibration and resource ledger.
