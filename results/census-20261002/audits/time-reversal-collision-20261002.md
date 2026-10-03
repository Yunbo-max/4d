# Bounded collision audit: time reversal in frozen ActionMesh

Audit date/cutoff: 2026-10-02. Agent: time_reversal_collision. Scope: literature and mechanism screening only; no SSH, GPU runs, or implementation. Used the research-autopilot literature-evidence and collision-audit references. This is a retriever/prosecutor evidence packet, not an independent blinded adjudication.

## Decision

**Treat reversal as a diagnostic and a strong baseline; do not currently present forward/reverse consensus as a new method.** There are direct mechanism precedents for inference-time temporal fusion, a denoiser reversal-disagreement objective, and group averaging. A meaningful ActionMesh-specific result would need to establish a naturally prevalent, consequential error signal beyond seed variance, then beat matched-cost seed ensembling and fixed reversal TTA. Applying known symmetrization to mesh outputs alone is a transfer/replication result.

Formal status: **INCONCLUSIVE_EXPAND_SEARCH** for novelty clearance. Four queries and at most six primary-source families were the hard budget, so coverage is deliberately incomplete. There is enough verified evidence to challenge broad originality claims; there is not enough to claim a complete collision adjudication or an irreversible KILL. No claim that this exact ActionMesh configuration has already been published.

## Frozen candidate elements

- C1: With the same physical anchor mesh, evaluate the same video in forward and reversed temporal order; move the anchor index from 0 to 15 in a 16-frame sequence; map outputs back to common chronological order.
- C2: Use the discrepancy to diagnose temporal-order bias, rather than ordinary stochastic variation.
- C3: Improve outputs through inference-time symmetry/consensus guidance or averaging deformation predictions at shared source queries.
- C4: Remain within a frozen ActionMesh workflow and 22 GiB resource envelope.
- The parent-agent supplied observations—Stage-I shape error dominates, Stage II can improve or worsen geometry, CD-M responds to trajectory changes—are project context, not independently reverified here.

## Verified primary-source evidence

| Source and version | Read depth / locator | Verified mechanism and collision assessment |
|---|---|---|
| **Explorative Inbetweening of Time and Space** (Feng et al., ECCV 2024), arXiv:2403.14611v1. [Full text](https://arxiv.org/html/2403.14611v1); [official repository](https://github.com/HavenFeng/time_reversal). | D2, §3.2, Fig. 4, Algorithm 1, Eqs. 2–3; limitations of fusion in the following noise-reinjection subsection. | TRF uses a frozen video denoiser on forward and backward paths, temporally reverses the backward prediction, and fuses them at each sampling step with frame-dependent weights. It also supports identical endpoint images for loops. This collides directly with a broad claim of training-free temporal reversal fusion. Different endpoint conditioning and RGB-video outputs distinguish its problem from fixed-anchor, fully video-conditioned mesh reconstruction; those differences do not make averaging a new operator. |
| **Motion Prior Distillation in Time Reversal Sampling for Generative Inbetweening** (Jeon et al.), arXiv:2602.12679v1, 2026-02-13. [Full text](https://arxiv.org/html/2602.12679v1). | D2, §3.2 Eqs. 5–10; §4.1 Eqs. 11–12; §4.2 Eqs. 13–19 and Algorithm 1; Appendix A. | The paper explicitly formulates agreement between a denoiser prediction and a temporally flipped backward prediction, and warns that optimizing agreement under incompatible motion priors can worsen motion. Its implemented remedy transfers forward temporal residuals into a backward estimate, avoiding early end-conditioned denoising. This is a particularly close precedent for reversal-consistency guidance and for the warning that lower disagreement need not mean lower error. The candidate has one shared physical anchor and full observed video conditioning, so it does not inherit MPD's two-endpoint ambiguity unchanged. |
| **ActionMesh: Animated 3D Mesh Generation with Temporal 3D Diffusion** (Sabathier et al.), arXiv:2601.16148v1, 2026-01-22. [Full text](https://arxiv.org/html/2601.16148v1). | D2, §3.2 “Masked generation”; §3.3 “Formulation”; Table 2; Appendix B. | Stage I supports known source meshes at selected temporal positions; the image-to-3D anchor may come from any input frame. Stage II predicts displacement between arbitrary source and target framesteps using source vertex queries, normals, source/target time tokens, and sequence latents. Thus moving the same physical anchor to the reversed index and querying a common source mesh are supported problem formulations, not themselves new contributions. Inflated attention already accesses all frames, so SVD's forward-generation bias cannot simply be assumed to transfer. |
| **DeepInv EquivariantDenoiser**, official documentation, retrieved version 0.4.2. [Documentation](https://deepinv.org/api/stubs/deepinv.models.EquivariantDenoiser.html). | D2 for the documented operator; equation immediately under “The denoiser can be turned into an equivariant denoiser”; References entry. | The documentation explicitly gives the group average of inverse-transformed denoiser predictions, full averaging at evaluation, and Monte Carlo approximation. Therefore fixed forward/reverse prediction averaging is the two-element special case of an established equivariance operator when all relevant inputs transform correctly. This source establishes the operator as an executable baseline; it is not a claim of priority for temporal 4D reconstruction. |

No more than 200 words of derived scientific summary per source above; no extended quotations.

## High-risk sources with incomplete access

- **Motion-Residual Conflict-Aware Time Reversal for Generative Inbetweening** (Zhang et al., ICML/PMLR 306, 2026): [publisher record](https://proceedings.mlr.press/v306/zhang26z.html), [publisher-linked PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/zhang26z/zhang26z.pdf). D1 only: publisher search result says it is inference-time sampling that aligns conflicting motion priors. Direct HTML and PDF opens returned internal errors. No detailed mechanism claim or novelty verdict relies on this source.
- **Equivariant Plug-and-Play Image Reconstruction** (Terris et al., CVPR 2024), cited by DeepInv. Attempted [official CVF record](https://openaccess.thecvf.com/content/CVPR2024/html/Terris_Equivariant_Plug-and-Play_Image_Reconstruction_CVPR_2024_paper.html) and PDF; both failed to open. D0/D1 only through the official documentation bibliography. No claim about its proof or experiments is made here.

Search results also exposed other relevant families, but they were not opened as additional primary works under the six-family cap. A future authorized expansion should include sequential bidirectional sampling, temporally reordered video diffusion, and nonrigid reconstruction/registration consistency. Discovery snippets do not establish their mechanisms.

## Operator-level comparison and decisive controls

Let R reverse temporal order, and let c include video conditions, anchor location, time identifiers, masks, source and target query identifiers, and any coupled noise. For a frozen predictor f, fixed reversal TTA is

`f_sym(x,c) = 0.5 * [f(x,c) + R f(Rx,Rc)]`.

**Analysis, not a claim from an experiment:** for a deterministic predictor, an involution R, compatible linear output space, and the complete transformation of conditioning, this construction is exactly equivariant to R. If the proposed “consensus” reduces to this equation at the output, it is fixed TTA. Applying that form to intermediate denoiser/flow predictions changes the sampler, but retains close TRF and group-symmetrization ancestry. Shared source vertices make the average well-defined; they do not change its algebra.

A useful distinction would require evidence that a specific, independently motivated intervention corrects an error that fixed averaging cannot. Replacing constant weights with a tunable schedule, adding a disagreement penalty, or moving the average into a different layer is not sufficient evidence by itself.

Minimum falsification logic, for planning only:

1. **Noise and anchor control.** Reuse the exact anchor geometry/latent, source vertices, normals, and point samples. Move the same physical anchor from 0 to 15. Couple actual noise tensors by reversal where appropriate; the same integer seed alone does not guarantee corresponding per-frame noise or random samples. Reverse source/target identities, frame conditioning, masks and any time-dependent metadata consistently. Distinguish “reorder tokens preserving physical timestamps” from “reverse the time coordinate”; record which symmetry is tested.
2. **Separate stages.** First reverse a fixed Stage-I latent sequence through Stage II to isolate decoder order dependence. Separately test reversal during Stage-I generation. A full-pipeline disagreement cannot identify which stage causes it.
3. **Matched-cost simple alternatives.** Compare forward single-run, reversed single-run, two forward seeds, corresponding-output averaging of those seeds, and fixed forward/reverse TTA. Add step-level group averaging only as a separate compute-matched intervention. Shared source queries are necessary for a fair displacement average; independently meshed Stage-I surfaces do not provide index correspondence.
4. **Demand natural consequence.** On held-out natural clips, test whether reversal discrepancy predicts actual geometry/motion error beyond seed spread. The experiment must retain low-disagreement failures and high-disagreement accurate outputs. Agreement between two incorrect reconstructions is possible.
5. **Avoid an averaging victory caused by motion shrinkage.** Track shape fidelity, trajectory error, and motion amplitude together. A lower CD-M or disagreement accompanied by flattened motion or geometric degradation does not demonstrate corrected temporal bias.
6. **Budget interpretation.** Two sequential frozen passes may fit the same peak memory but cost additional runtime. Gradient-based symmetry guidance may have a different memory footprint; 22 GiB feasibility is unverified.

Practical no-go conditions: no error information beyond seed variation; gains explained by two-forward-run averaging; gains confined to reduced motion; or Stage-I errors shared by both directions remain dominant while the intervention targets only Stage II. These are prospective falsifiers, not observed results.

## Search protocol and provenance

Provider: web.run search index (API/index version not exposed). Search date and cutoff 2026-10-02; no date/domain filters, default ranking, English query strings, response_length=long, no pagination. Primary papers and official repositories/documentation only used as evidence. Non-primary search hits were ignored.

Exactly four search queries:

1. `video diffusion time reversal equivariance bidirectional inference temporal order consistency`
2. `4D reconstruction time reversal test time augmentation deformation equivariance`
3. `diffusion inference equivariance symmetrization group averaging denoiser test time augmentation`
4. `ActionMesh 4D reconstruction time reversal bidirectional deformation temporal consistency`

Queries 1–2 were issued together, then 3–4 together. Follow-up access used only direct opens, finds, and a repository-to-paper citation link; no additional search queries. Citation expansion was limited to the explicit TRF paper link and the DeepInv bibliography. Version-specific arXiv HTML was preferred where available. Failed direct opens are listed above; some alternate-format opens for already counted works also failed.

The second search batch and subsequent primary-access tool responses are captured in the companion JSON. The first search batch remains in the tool transcript but was not separately serialized before this report; this is a provenance gap. No reproducible normalized-output pipeline, independent adjudicator, two-batch saturation, full neural-deformation landscape coverage, or complete recent-paper full-text verification was completed. Therefore this packet cannot certify literature completeness or satisfy a final collision gate.

## Allowed and disallowed conclusions

Allowed: this bounded audit verified close operator/mechanism precedents; fixed reversal TTA and seed averaging are mandatory strong comparators; the same-anchor ActionMesh diagnostic remains an untested empirical hypothesis.

Disallowed: “first time-reversal guidance,” “new group symmetrization,” “novel because mesh-native,” “no prior work,” “reversal disagreement measures true error,” or “consensus improves geometry.” No effectiveness or novelty result was measured here.

