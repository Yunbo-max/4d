# Local action-preserving edit controllability — mathematical candidate B

**Status:** derived diagnostic and candidate operator, not an established method contribution. No search, GPU or implementation in this task. The earlier full operator/safeguard derivation is in `edit-contract-projection-critique-20261002.md`.

## Observable space and contract

Use a common output observable `y=R_m(F_m(z))∈R^n`, where `F_m` is a frozen 4D model under explicitly allowed controls z. `R_m` extracts comparable task geometry across models: declared material/semantic points and times, in one fixed scene coordinate system. Do not compare arbitrary vertex indices or raw latent coordinates across models. Unknown correspondences restrict the claim to observable points; they do not become known by writing R.

An action is a **set**

\[
\mathcal C_a=\{y:c(y)=0,\ g(y)\le0\},
\]

including allowed contact partners/times, support and nonpenetration conditions, and permitted non-target motion changes. Let f(y) be edit features, d the requested local feature change, and let

\[
J_m=D(R_m\circ F_m),\quad A=Dc(y),\quad B=Df(y),\quad G=Dg(y).
\]

Choose a fixed output metric W and edit-feature metric Q from declared physical units and sampling weights, **without GT or model-dependent normalization**. Let U be a W-orthonormal basis of `range J_m`, and N an orthonormal basis of `ker(AU)`. Thus `V_m=UN` has `V_m^T W V_m=I` and its columns span the model's first-order action-preserving directions.

This construction is invariant to invertible latent reparameterization: `z'=h(z)` changes the Jacobian's columns but not its range. For a change of output coordinates `y'=Sy`, transform the metric as `W'=S^{-T}WS^{-1}` and constraints/features consistently; the scores then remain invariant. A raw latent Euclidean norm or a raw-coordinate rank score lacks this guarantee.

## Two different scores: missing direction versus insufficient gain/budget

First consider feasible y with inactive inequalities (or a fixed equality active set). Define the **unbounded linear range residual**

\[
\gamma_m^{\infty}(d)=\min_{a\in\mathbb R^r}\|B V_m a-d\|_Q^2
=\|(I-CC^\dagger)Q^{1/2}d\|_2^2,
\qquad C=Q^{1/2}BV_m.
\]

It tests whether d lies in the model's available first-order edit-feature range while preserving the action. It is **not a realizability claim for a finite edit**: the minimizing displacement can be arbitrarily large, outside any valid linearization. A zero value may require a huge displacement along a poorly conditioned direction.

Define separately the **trust-region residual**

\[
\gamma_m^{\rho}(d)=\min_{\|a\|_2\le\rho}
\|B V_m a-d\|_Q^2
\quad\text{subject to}\quad g(y)+GV_ma\le0.
\]

Under matching inequality conventions, `γ_m^ρ≥γ_m^∞`. The difference measures finite-step budget/conditioning restrictions, rather than absent first-order directions. With active inequalities, the feasible directions form a cone, not generally a nullspace: include them in **both** optimizations, and do not apply the closed-form projector above to unilateral constraints without qualification.

Report the normalized residual `γ/(d^T Qd)` only for nonzero d, the edit gain singular spectrum of C, the declared radius, and numerical-rank tolerances. Do not add an L2 regularizer when interpreting the optimum as a pure attainability score; it changes the question. Regularization can be used for a chosen controller, with its bias reported separately.

## A reference gap requires a nested control space

Compute `γ_ref` with the same metrics, radius, linearized contract and desired edit, but a reference control space that **contains** the model tangent. Then

\[
\Delta_m^{\rho}=\gamma_m^{\rho}-\gamma_{ref}^{\rho}\ge0.
\]

A fully unrestricted observable-space reference is a useful mathematical relaxation. It is not necessarily a valid animated mesh or physical trajectory. A rig/ARAP/surface retargeter is a valuable practical comparator, but unless its control space contains the model tangent, the difference has no guaranteed sign and must not be described as this nonnegative gap.

A **validated finite edited trajectory** from a classical solver can provide an independent feasibility witness. A vanishing linearized reference residual, or a failed solver, cannot respectively prove true feasibility or infeasibility. Physical stability also needs specified dynamics; geometric contact alone is insufficient.

## Concrete candidate controller, with ordinary mathematics

For desired edit velocity v, the equality-preserving model step is

\[
u^*=V_mV_m^TWv
=U[I-(AU)^\dagger(AU)]U^TWv.
\]

Equivalently solve the bounded least-squares subproblem above, map the step to admissible model controls, decode, and relinearize. This is classical constrained projection/SQP; it is not the proposed scientific novelty.

First-order preservation needs differentiable maps, a valid tangent/retraction, a feasible starting state, and an appropriate active set. `Au=0` permits second-order constraint drift. Recompute the full original contract after every decoded step, backtrack on violations, limit cumulative edit/motion drift, and stop at a fixed iteration/compute budget. Contact switching, rank changes and nearest-surface nonsmoothness require rejection or explicit active-set changes. Sampled-frame validation does not guarantee continuous-time contact.

## The scientific question and its falsifiers

**Question:** Are useful, independently feasible identity/shape edits locally inaccessible under action-preserving controls of frozen 4D generators, and can an output-space controllability gap predict actual bounded edit failure across models better than ordinary conditioning, edit magnitude or optimization budget?

This asks whether inference steering is the right intervention, rather than proposing another loss. Distinguish four explanations before interpreting an observed residual:

1. **Unavailable interface:** controls never expose the requested edit. For example, an unchanged cached ActionMesh anchor cannot undergo an anchor-shape edit. This is a trivial task/interface restriction, not a scientific representational failure.
2. **Unresolved task feasibility:** requested contact/action constraints may conflict. Existing retargeting and IK already address much of this problem; no impossibility claim follows from optimization failure.
3. **Local representation restriction:** the full admissible tangent misses a direction accessible to a valid reference. This remains local; nonlinear multi-step paths can invalidate a global claim.
4. **Numerical/computational restriction:** sampled Jacobian bases, truncation or a small trust budget omit useful directions. Increase only a predeclared basis/precision budget to test this explanation; otherwise label the restricted-control measurement honestly.

First test independently feasible edits against qualified motion transfer and contact-aware retargeting. Compare unbounded and bounded residuals, actual edit attainment at a fixed action-violation allowance, and the strongest classical output-space solver. If the solver already produces equally valid assets more cheaply, there is no demonstrated need for the model-space controller. If a basis enlargement or relinearization removes the effect, reject the stronger bottleneck interpretation. Cross-model repetition alone does not establish novelty.

Exact priors remain [Basset et al. 2020, §§3–6](https://www.sciencedirect.com/science/article/abs/pii/S0097849320300406), [ReConForM §§4,6.3](https://arxiv.org/html/2502.21207v1), and [Catalyst4D §5.4](https://arxiv.org/html/2603.12766v1). A possible contribution would be a validated new finding about **task-conditioned controllability and its consequences**, not projection, pseudoinverses, SQP, rank, feasibility testing, or a new weighted energy.

**Resources:** ≤22GiB for full decoder derivatives is unknown. JVP/VJP and low-dimensional probes may avoid a dense Jacobian, but do not by themselves guarantee memory or coverage. Freeze the model's actual admissible interface and a derivative budget before any experiment; report approximation scope and peak memory. No benchmark improvement or method novelty is currently established.
