# 4D editing direction audit — 2026-10-02

**Decision:** reject a new-method pitch based on generic contact preservation, motion freezing, part-wise motion transfer, or detecting incompatible edit constraints. Keep one consequential **capability/evaluation question** for a small falsification probe. No method novelty, cross-model effectiveness, or 22GiB feasibility has been established. No GPU, SSH or implementation was performed.

Search budget used: six queries; four papers inspected deeply in relevant sections. Queries, primary URLs, read depth and coverage gaps are in `new-directions-editing-20261002-captures.json`. Cutoff: 2026-10-02. This branch has not measured natural failure prevalence in our current asset population.

## Evidence and closest-work collisions

| Primary source | Exact locator and established evidence | Implication |
|---|---|---|
| [Catalyst4D, CVPR2026](https://arxiv.org/html/2603.12766v1), [CVF record](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Catalyst4D_High-Fidelity_3D-to-4D_Scene_Editing_via_Dynamic_Propagation_CVPR_2026_paper.html) | §1; Anchor-based Motion Guidance, Eqs7–13; §5.4/Fig6. Edited points can fall outside a deformation model's source support. Three-neighbor motion transfer leaks hand motion into torso; structural anchor matching addresses this. §5.2/Fig5 documents unintended edits to other objects. | Generic edited-to-original correspondence, semantic region separation, and local motion transfer are already core contributions. “Cross-model” alone does not distinguish an alternative. |
| [ReConForM, Eurographics2025](https://arxiv.org/html/2502.21207v1) | **§6.3/Eq3/Fig10** already identifies conflicting contact/penetration objectives and exposes their tradeoff. The example cannot reach its feet after morphology changes. §3.1 uses a humanoid template, key vertices and skinning; §7.1 states structural limitations. | Even edit feasibility/conflict feedback is explicit prior art. Merely introducing a feasibility check before a controller is insufficient novelty. |
| [Basset et al., 2020](https://www.sciencedirect.com/science/article/abs/pii/S0097849320300406), DOI `10.1016/j.cag.2020.04.002` | Publisher abstract/introduction; §§3–6 section snippets. Rigging-free shape transfer combines shape, volume and contact/collision terms, including morphology-induced new contacts and continuous sequences. Full paper not deeply inspected. | Reject “training-free contact-preserving shape edits on arbitrary characters” as a new mechanism. A strong predecessor exists outside diffusion literature. |
| [Dynamic-eDiTor, CVPR2026](https://arxiv.org/html/2512.00677v1), [CVF record](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_Dynamic-eDiTor_Training-Free_Text-Driven_4D_Scene_Editing_with_Multimodal_Diffusion_Transformer_CVPR_2026_paper.html) | Qualitative comparison/Fig4 reports hand-motion distortion and incomplete semantic edits. §6 limitation explicitly excludes large geometric/topological changes. Appendix A specifies 30k source-reconstruction and 20k editing iterations, using H100/A6000 hardware. | “Training-free” refers to reused diffusion capability, not zero scene optimization or proven ≤22GiB execution. A paper title is not a deployability guarantee. |
| [Shape-for-Motion, 2025](https://arxiv.org/html/2506.22432v1) | §3.2/Eqs2–3 and §4.2.2/Fig8 treat mesh/Gaussian transfer mismatch and texture displacement; dual propagation is their solution. Appendix D.8/FigD15 shows an edited swan with inconsistent surrounding reflection. | Material appearance propagation and geometry transfer must be baseline controls. Preserving the outside mask exactly is not always correct, but reflection-aware editing needs its own PBR baseline and is not selected here. |

Additional collision signals, **not full-text clearance**: [MotionSplicer](https://ivl.cs.brown.edu/motionsplicer/) already exposes part timing/isolation/freezing/scaling; [Consistent 4D Appearance Editing](https://orca.cardiff.ac.uk/id/eprint/188702/) explicitly preserves trajectories during appearance changes. This makes generic action/identity disentanglement by appearance-only freezing a weak direction.

## Surviving question: do visually successful shape edits still satisfy the source action's obligations?

**Question:** For a substantive identity/shape edit, can a generated 4D asset still execute the same *kinematic interaction task*, after a properly configured correspondence-transfer plus contact-retargeting baseline? Do current appearance/temporal-quality rankings select outputs that violate those obligations?

This is a problem and measurement question, **not a claim that contact retargeting, feasibility checking, or an editing benchmark is new**. Its possible value is discovering a consequential residual failure that survives strong existing solvers. If the residual disappears, abandon this branch as a new-method project.

### Natural task, consequence, and what does not count

Eligible tasks include holding a prop through a motion, maintaining supporting foot contacts, or completing a specified hand-to-body contact event after editing the character's proportions or the interacting object's shape. An apparent success that loses the prop contact or intersects the support cannot be used as the requested animated asset, even if its appearance matches the instruction.

These examples are proposed task families; **their prevalence in current generative outputs has not been measured**. The existing eight ActionBench objects are not automatically a contact-editing benchmark. In particular, source sequences need validated contact events, known object identities and an explicit allowed-motion contract before evaluation. Source defects must be identified before looking at edited model outputs, rather than blamed on editing or removed afterward.

Simple recoloring with all vertices/trajectories frozen is a **negative control**, not main-task success. A shape edit must change an operational geometric attribute, such as limb length, torso clearance or the grasped object's cross-section. Exclude adding/removing limbs in the first probe: that introduces an underdetermined action assignment and exceeds comparable retargeting assumptions. Do not label an impossible request a generator failure.

### Strong simple alternative first

1. Exact source trajectory/rotation copying plus the edited canonical geometry.
2. Standard deformation transfer or three-neighbor displacement transfer, with a stronger region/anchor correspondence baseline where available.
3. Contact-aware retargeting on appropriate rigged cases; provide the necessary rigs, annotations and fair setup rather than handicapping it. For mesh-only cases, include a competent surface-based/contact solver. A basic nonpenetration/IK repair is a useful lower-cost baseline, not the strongest comparator.
4. When constraints conflict, report the admissible contact-versus-motion tradeoff, or return infeasible/unknown. Optimizer nonconvergence is **not a certificate of physical infeasibility**.

Keep geometric correspondence quality separate from edit controllability. If manual/oracle correspondence makes the classical baseline solve the task, the residual is a correspondence/tooling problem; do not rename it a new steering principle. Similarly, a semantic edit that implicitly changes physical properties has no unique correct dynamics without an explicit physical specification.

### Evidence versus hypothesis

**Established by the audited sources:** geometric edits can disrupt propagation; conventional contact-aware transfer and conflict handling already exist; several editing evaluations emphasize semantic appearance and temporal visual consistency.

**Unmeasured hypothesis:** after reasonable correspondence and solver qualification, a nontrivial fraction of naturally requested shape edits still receive good visual-edit scores while violating task-specific contact timing or support constraints. The discrepancy should be larger for interaction-changing shape edits than for frozen-geometry appearance controls.

**Distinct prediction for a probe:** ranking by generic visual edit quality and ranking by the predeclared action contract disagree on feasible geometry edits, while zero-edit/appearance controls preserve the contract. Mere jitter or small surface-distance differences do not satisfy this prediction. If simple contact-aware transfer restores the contracts in nearly every valid case, there is no demonstrated need for a new inference controller.

**Strongest counterargument:** this is established animation retargeting plus application engineering. Complete geometry, known contacts and a rig may already reduce the problem to ordinary IK/constrained optimization. Incomplete generative geometry and missing correspondences do not automatically create a new scientific mechanism; they may just make an existing task noisier.

### Small decisive probe, before method design

- Select six independently validated source actions before seeing model edit results: two support-contact, two self-contact, two object-contact examples. Use realistic target morphologies/objects from existing assets rather than adversarial distortions designed to break a baseline. Save every exclusion and retain the six-source denominator.
- For each source, define two substantive shape edits, plus an appearance-only control. Specify unchanged obligations and allowed adaptation: contact partners/timing, support schedule, and whether root/body movement may adjust. A single exact target trajectory is unnecessary; valid outputs form a set.
- First test trajectory copy, qualified correspondence transfer, and contact/IK alternatives on known mesh sequences. If they solve the task, stop before spending GPU time on a new mechanism. Use known feasible constructions to check evaluator correctness; such synthetic calibration is not natural-prevalence evidence.
- Only then test the same frozen requests through two qualified 4D backbones. A second backbone must genuinely support the input task; an added adapter's edits and costs must be reported. Do not call two seeds of one model “cross-model.”
- Score declared contact-event retention, penetration/support violations, edit attainment and motion outside the allowed adaptation region. Report individual sources, solver convergence and infeasible/unknown requests. Contact geometry alone is not a proof of force closure or physical stability; do not use those labels without a dynamics model and specified physical parameters.
- Compare visual and task rankings descriptively first. No method selection or threshold tuning on the same edited outputs; no pointwise/framewise independence claims. A useful next result is either a documented residual on both backbones or a clear baseline-sufficiency rejection.

**Budget proposal:** CPU-only cached-geometry calibration first, maximum two hours of setup/audit and 60 seconds per simple correspondence/IK case, with a separately reported stronger-solver cap up to five minutes per case. Do not claim a baseline fails if those caps do not numerically qualify it. Initial new inference should be limited to at most four representative requests per qualified backbone, with a declared one-hour wall budget per backbone, before broadening. These are proposed stop budgets, not measured runtimes.

**Memory:** all new 4D-backbone/editor configurations remain unqualified for ≤22GiB. Existing ActionMesh native measurements do not establish its complete semantic-editing pipeline footprint or another model's footprint. ReConForM reports a laptop RTX3060 in §5.3.1, but that does not specify our generated-asset adapter. Cached mesh diagnostics can be CPU-based; an inference method must be separately profiled and stopped above the actual 22GiB limit, without quietly changing resolution or the task.

## What this does and does not support

This is a justified **new question to test**, not an ICML-ready idea. It earns further method work only if consequential natural failures remain after legitimate solver baselines and are not explained by invalid inputs, infeasible requests or unqualified correspondences. The limited audit found substantial direct collisions, so formal novelty status is **not cleared**. No uncertainty-guided surface correction or time-reversal variant is being repackaged here.
