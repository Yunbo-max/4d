# Source-anchor transport cocycle: bounded mechanism scout

Date / literature cutoff: 2026-10-02. Agent: time_reversal_collision.
Scope: NEW candidate screening only; no implementation, SSH, GPU, or experiment launch. Four search queries used; three primary full texts scientifically read, two further primary papers attempted but inaccessible. Research-autopilot literature-evidence and collision-audit rules applied.

Project context supplied by root: frozen ActionMesh; real census currently eight assets and two seeds; direction diagnostic pending; earlier observations suggest Stage-I shape errors dominate. These project facts were not reverified by this literature-only scout. Candidate novelty and usefulness are **not established**.

## Decision and claim boundary

**Do not promote a new cocycle mechanism.** Composed 3D deformation cycles, cycle-based source selection, and reconstruction residual alternatives have direct precedents. The potentially useful empirical question is narrower: does this particular frozen decoder's source-change inconsistency predict natural material-correspondence errors beyond surface mismatch and seed spread?

A diagnostic-only study remains reasonable if it fits the parent problem; it is not permission to implement or launch. Formal collision clearance remains **INCONCLUSIVE_EXPAND_SEARCH**, because recent tracking/reconstruction coverage and two primary full texts remain incomplete. This is a retrieval/prosecution packet, not independent adjudication or an irreversible kill decision.

## Candidate computation, frozen for comparison

For a fixed latent sequence Z and source/target decoder call, write:

- F_ab(x,n;Z) = x + D(Z,a,b,x,n), with normal input and coordinate conventions made explicit.
- Baseline X_k^v = F_0k(X_0^v,n_0^v;Z).
- Composed prediction Y_kt^v = F_kt(X_k^v,n_k^v;Z).
- Direct-versus-composed residual r_kt^v = ||Y_kt^v - X_t^v|| / L.
- Return residual c_k^v = ||F_k0(X_k^v,n_k^v;Z) - X_0^v|| / L.

L should be a fixed non-GT scale, such as the original anchor bounding-box diagonal. Do not normalize by the predicted displacement itself, which can make small-motion cases unstable or reward suppressed motion. An aggregation across k, t, or vertices is a separate design choice and must be fixed before looking at evaluation errors.

These are **decoder consistency residuals**. Calling them calibrated uncertainty, posterior variance, or a physical-correspondence certificate needs additional evidence. Fixed topology preserves the identity of a predicted vertex track; it does not establish that the track follows the correct physical material point.

## Closest verified evidence

| Primary source | Version / read depth / exact locator | Mechanism comparison |
|---|---|---|
| [Unsupervised cycle-consistent deformation for shape matching](https://vovakim.com/papers_small/19_SGP_DeepDeformCycles.pdf), Groueix et al., SGP 2019 | arXiv:1907.03165v1, 2019-07-06; D2. §3 Eq. 1; §4.2.2 Eqs. 3–6, PDF p. 4; §4.3, p. 5; §5.2.3, p. 8. | Direct collision with generic composed-deformation cycles: two- and three-shape loops regularize learned maps. It also selects source shapes at test time using two-cycle loss, so using cycle error as a quality criterion is not new. The paper projects intermediate outputs onto the next shape because raw predicted points may be off-surface. Its broader selection comparison reports deformation distance outperforming the other criteria. Different setting: inter-object shape matching with a trained pairwise network, rather than frozen temporal decoding of one video. |
| [ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion](https://arxiv.org/html/2601.16148v1), Sabathier et al., 2026 | arXiv:2601.16148v1, 2026-01-22; D2. §3.3 “Formulation,” particularly source/target time tokens and source queries; §3.4 “Animation extrapolation.” | The decoder already accepts arbitrary source/target framesteps and source mesh queries with normals. Training queries lie on source surfaces. It also reuses the prior chunk's last output as a reference for long sequences. This supports the feasibility of source changes as a probe; it supplies no verified cocycle-calibration result. A source change alone is therefore a native decoder capability, not a novel transport mechanism. |
| [Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](https://arxiv.org/html/2407.01521v1), Zhang et al., 2024 | arXiv:2407.01521v1, 2024-07-01; D2. §2.2 Eq. 3; §3.1 Eqs. 5–9 and Algorithm 1; Appendix A Eqs. 13–15 and Algorithm 2. | DAPS alternates a diffusion-derived clean estimate, approximate conditional sampling using the observation likelihood, and re-noising. It supports latent-space updates through a decoder. A cycle score alone is not its measurement model. Adopting a similar inference loop with a cycle energy would be a transferred sampler plus a new empirical energy choice, not a new posterior-sampling principle. Posterior guarantees cannot be inherited simply by naming self-consistency a likelihood. |

Each source summary is below 200 words; no extended quotation is used.

## Tracking and recent correspondence leads not fully verified

- **Track, Check, Repeat: An EM Approach to Unsupervised Tracking**, Harley et al., CVPR 2021. [Official PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Harley_Track_Check_Repeat_An_EM_Approach_to_Unsupervised_Tracking_CVPR_2021_paper.pdf). D1 discovery only: a primary PDF search excerpt mentions cycle-consistency for unsupervised tracking. HTML/PDF direct opens failed. It is a required follow-up for the tracking-quality/pseudo-label filtering comparison; this report makes no detailed algorithmic or test-time-refinement claim about it.
- **EchoMatch: Partial-to-Partial Shape Matching via Correspondence Reflection**, Xie et al., CVPR 2025. [Attempted official record](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_EchoMatch_Partial-to-Partial_Shape_Matching_via_Correspondence_Reflection_CVPR_2025_paper.html). D0/D1 only: title/authors/venue discovered through a survey bibliography; primary HTML/PDF access failed. No novelty judgment relies on the survey's mechanism description.

For discovery only, the [2026 nonrigid-shape correspondence survey](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70397) was opened to locate EchoMatch's primary citation. It is not scientific evidence for any candidate claim. Broader unsupervised 3D point tracking, reversible dynamic-mesh representations, and 2025–2026 4D correspondence systems were not adequately covered under the bound. No “no collision found” conclusion is warranted.

## Strongest counterarguments: mathematical analysis, not measured outcomes

**Self-consistent wrong material motion.** Let h_t be any invertible parameterization from a common domain to each predicted surface, and define F_ab = h_b composed with inverse(h_a). Then every return cycle and every direct-versus-composed residual is exactly zero. Yet h_t may contain arbitrary time-varying surface reparameterizations that disagree with actual material trajectories. In a symmetric shape, geometry can be correct while material rotation is wrong. An identity family and a globally factorized rigid family are simpler zero-residual examples. Therefore no universal positive error bound or calibrated uncertainty follows from small cycle residual alone.

**Common error is invisible.** All paths use the same Z and the same decoder; they are not independent observations. A wrong Stage-I shape or shared semantic swap can propagate coherently through every call. This is especially relevant if the parent problem remains dominated by Stage-I geometry.

**Query-distribution shift can dominate the signal.** The composed query X_k can deviate from the surface represented by Z_k. Recomputed normals can also disagree with that latent surface, especially around contact, foldovers, collapsed triangles or occluded parts. A high residual may reflect source-query invalidity rather than a bad direct material track. Source point error can be amplified by local sensitivity of F_kt. The converse also matters: a large residual can arise even when direct F_0t is accurate.

**Projection is a control, not a free cure.** Querying at the closest point on the intermediate Stage-I surface tests the off-surface explanation, but can jump across touching limbs or change the physical point being tracked. Keep the projection displacement and normal change visible. Do not silently redefine the candidate around projected queries after inspecting results.

**Optimization can game the score.** Shrinking motion, reverting to rigid motion, or enforcing maps through a common canonical representation can improve consistency without improving physical correspondence. Refinement and uncertainty must therefore be assessed separately.

## Smallest discriminating evidence packet (not a launch plan)

1. **Validate the queried state.** Freeze Z, X_0, faces, source point IDs, and normals policy. Record source/target index conventions. Measure F_00(X_0)-X_0 and F_kk(X_k)-X_k alongside nontrivial cycles, so general identity-query defects are not misread as path-specific failures.
2. **Measure the confound directly.** For every composed source query, retain distance to the decoded Stage-I surface, normal mismatch where meaningful, motion amplitude, temporal lag and local deformation/triangle quality. Compare the cycle signal with these inexpensive predictors.
3. **Check real incremental prediction.** On natural clips, evaluate whether the residual predicts correspondence/trajectory error beyond those predictors and paired-seed spread. Use asset-level paired summaries and report every asset; dense vertices are not independent replicates. Eight assets and two seeds support a pilot, not broad calibration claims.
4. **Separate ranking from probabilities.** Start with error ranking/risk-versus-coverage and a fixed bad-track definition. If a probability of failure is claimed, reserve held-out assets for calibration and coverage. Already inspected census cases remain developmental.
5. **Avoid selecting the easiest claim after results.** Preselect source k and target t coverage, aggregation and the task endpoint. No post-hoc anchor or asset exclusion to rescue the signal.
6. **If refinement is later considered, demand external improvement.** Keep true trajectory quality, geometric fit, motion amplitude and topology/triangle quality visible. Lower cycle energy alone is not success.

## Equal-budget alternatives and falsifiers

| Alternative | Fair comparison / decisive implication |
|---|---|
| Unmodified ActionMesh / identity correction | Establish the correction baseline. For a cycle-only score, explicitly demonstrate that an identity map can have zero residual while failing moving clips. This is a negative control, not a competitive motion solution. |
| Global rigid fit | Fit without GT using the same predicted surfaces or observations. Factorized rigid maps satisfy cycles by construction; compare actual material trajectory and geometry error. Gains explained by suppressing nonrigid motion are a no-go for the proposed explanation. |
| Existing simple surface/ARAP correction | Use the already authorized baseline settings and actual measured runtime; do not increase its degrees of freedom selectively. If geometric correction removes the residual and the predictive gain, the score may only expose surface mismatch. |
| Two-seed spread / same-budget additional seed | Use the same physical anchor and comparable output queries when comparing trajectories. With only two seeds, characterize disagreement rather than claiming reliable variance. Compare marginal value per added decoder/denoiser call and wall time. |
| Direct output versus fixed reanchor / uniform composition average | Generate the same candidate paths for all selection rules. Compare cycle-based selection with a fixed anchor rule, uniform averaging, and selection by intermediate/final surface distance. This separates useful scoring from additional predictions. |
| Cheap quality scores | Surface distance, displacement magnitude, temporal lag and local mesh quality must be competitive uncertainty baselines. If residual ranking adds no information beyond them, the proposed uncertainty mechanism lacks necessity. |

GPU decoder calls and CPU ARAP steps are not interchangeable budgets; report both wall time and calls, including normal construction and correspondence queries. No 22 GiB feasibility measurement was made.

Prospective no-go: the score is mainly explained by source-surface mismatch; low-residual high-error natural cases dominate; ranking adds no held-out information over cheap/seed scores; or refinement improves internal cycles while leaving material accuracy unchanged or harming motion. These are falsifiers, not observed findings.

## Exact search and capture protocol

Provider: web.run search index; version not exposed. Search date/cutoff 2026-10-02. No domain/date filters; default relevance order; response_length=long; English; no pagination. Exactly four queries:

1. `3D shape correspondence deformation cycle consistency unsupervised source target composed deformation`
2. `4D point tracking reconstruction cycle consistency uncertainty test time refinement DAPS`
3. `DAPS cycle consistency test time refinement 4D tracking diffusion posterior sampling`
4. `point tracking cycle consistency uncertainty test time EchoMatch 3D`

The queries were issued in two paired batches. Follow-ups used direct primary opens, find operations, and one survey bibliography lookup. No further search query was issued. At most five primary-paper families were pursued: Groueix, ActionMesh, DAPS, Harley, EchoMatch. Full scientific reads succeeded for the first three. Additional direct urllib attempts for the two unavailable CVF PDFs failed with DNS errors; no paper bytes were downloaded.

Raw provider responses for both searches and all substantive access batches are stored in the companion `source-anchor-cocycle-20261002-captures.json`. This is a bounded search snapshot, not a fully normalized literature database or saturation certificate. No independent blinded adjudication, no implementation, and no experiment launch occurred.

## Allowed claims

Allowed: source-change cycle residual is an untested ActionMesh consistency diagnostic with close generic predecessors and a specific query-validity confound; an asset-level test of incremental predictive value could decide its usefulness.

Not allowed: first cocycle correspondence method; new cycle-based confidence principle; guaranteed material correctness; calibrated uncertainty; Bayesian posterior from self-consistency alone; or demonstrated improvement.

