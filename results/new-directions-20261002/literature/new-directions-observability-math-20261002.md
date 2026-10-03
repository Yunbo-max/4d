# Hypothesis A: shared-bias covariance steering

Date: 2026-10-02. Status: conditional mathematical hypothesis; not implemented, experimentally qualified, calibrated, or novelty-cleared. This uses frozen models and per-instance optimization without updating model weights.

## Variables and observation contract

Let \(a\in\mathbb R^p\) parameterize a small correction to a frozen 4D reconstruction:

\[
X(a)=X_0+Ba.
\]

Use a fixed deformation basis \(B\), preserve the anchor, and fix the camera/world gauge. Define:

- \(r_o(a)\): residuals against actual source observations, such as silhouettes, reliable tracks, or rendered features.
- \(r_s(a)\): residuals against cached generated views.
- \(J_o,J_s\): their Jacobians at the local reference, with predetermined residual scales.

From \(J_o^\top J_o\), take \(U\in\mathbb R^{p\times k}\), the directions weakly constrained by actual observations. This estimates local sensitivity under the chosen residual, camera and visibility assumptions; it does not establish global identifiability or true uncertainty.

## Shared-bias regularizer

Use a model of generated-view residuals with a shared latent 4D bias:

\[
r_s(a)=J_sUb+\epsilon,\qquad
b\sim\mathcal N(0,\Lambda),\quad
\epsilon\sim\mathcal N(0,\sigma^2I),\qquad \sigma>0.
\]

The resulting working covariance is

\[
C_s=\sigma^2I+(J_sU)\Lambda(J_sU)^\top.
\]

All generated descendants of the same source clip share the nuisance bias \(b\); changing random seeds does not make them independent observations.

**The generated views are conditioned on the actual source input. Therefore, adding this quadratic to the real-observation objective is a model-based regularizer, not an exact Bayesian likelihood for additional independent evidence.** The Gaussian nuisance construction motivates its algebra; it does not establish that the residual model is statistically correct. Useful pretrained prior information can improve reconstruction even when no new independent observation has been acquired.

\(\Lambda\), its scalar special case \(\tau^2\), and \(\sigma\) are declared modeling/ambiguity-budget choices. They are not identifiable as true hidden-geometry error covariance from repeated generation seeds. Seed variation measures conditional generator variability and can remain small under shared systematic error.

Freeze \(U,J_s,\Lambda,\sigma\), and hence \(C_s\), during each local solve. The simplest bounded experiment freezes them for the whole correction. If a later protocol recomputes them between local solves, it must explicitly define and validate the resulting outer algorithm; no global descent claim transfers automatically.

## Objective and update

Optimize

\[
\min_a\quad
\frac12\|r_o(a)\|^2+
\frac{\lambda}{2}r_s(a)^\top C_s^{-1}r_s(a)
+\frac{\mu}{2}\|a\|^2,
\qquad \lambda\ge0,\ \mu>0,
\]

subject to

\[
\|r_{o,t}(a)\|^2\le
\|r_{o,t}(0)\|^2+\varepsilon_t
\quad\forall t,\qquad
\|a\|\le\rho.
\]

The per-frame constraints limit harm to actual observed content. They do not guarantee hidden-surface correctness.

At a current feasible iterate, the Gauss–Newton gradient and approximate Hessian are

\[
g=J_o^\top r_o+\lambda J_s^\top C_s^{-1}r_s+\mu a,
\qquad
H=J_o^\top J_o+\lambda J_s^\top C_s^{-1}J_s+\mu I.
\]

Solve the small constrained quadratic in the step \(\delta\), using linearized residuals in the per-frame constraints and an appropriate local trust region. Backtracking must check the actual nonlinear constraints before accepting the update. No model-weight training is involved.

For \(A=J_sU\) and positive definite \(\Lambda\), Woodbury gives

\[
C_s^{-1}=\sigma^{-2}I-
\sigma^{-4}A\bigl(\Lambda^{-1}+\sigma^{-2}A^\top A\bigr)^{-1}A^\top.
\]

## Distinction from inverse-count weighting

The correlation model acts on common ambiguous 4D directions and their different projections, rather than uniformly dividing every generated residual by the view count. In a scalar duplicated-evidence case,

\[
\mathbf1^\top(\sigma^2I+\tau^2\mathbf1\mathbf1^\top)^{-1}\mathbf1
=\frac{n}{\sigma^2+n\tau^2}
\longrightarrow\frac1{\tau^2}.
\]

Thus repeated synthetic support cannot accumulate unlimited working precision along that shared-bias direction. This is a property of the stipulated regularizer, not proof that its precision equals actual posterior precision.

## Acceptance and abstention

Reserve actual observations for an input-only check, and reject a correction that worsens them beyond a fixed tolerance. For a differentiable downstream quantity \(T(a)\), report the local ambiguity sensitivity

\[
s_T=\sqrt{\nabla T^\top U\Lambda U^\top\nabla T}.
\]

If allowed perturbations change the requested occlusion-order decision, abstain from certifying that decision. This is a local sensitivity test, not a confidence interval or a probability-coverage guarantee. Native support/confidence scores are not assumed calibrated. Hidden-surface correctness and probability calibration cannot be established from agreement among synthetic descendants alone.

## Baselines and failure conditions

The mandatory simple baselines are source-only correction, isotropic trust-region regularization, a fixed synthetic-weight cap, and inverse-count weighting. The candidate must show an advantage specifically where generated errors have coherent shared 4D structure. Standard covariance modeling alone is not a novelty claim.

Incorrect cameras, visibility or tracks can corrupt \(U\). Nonlinear and topological errors can fall outside the selected subspace. Covariance floors and ambiguity budgets can suppress useful priors. If synthetic targets are only renders of \(X_0\), this construction reduces to a reconstruction regularizer; it does not add independent evidence. Kill the method claim if simple weight caps perform equally well, or if the hypothesized natural failure is absent.

## Compute and feasibility

With \(p=128\), \(k=32\), and \(m_s=8192\), an explicit FP32 synthetic Jacobian occupies approximately 4 MiB. Woodbury requires a \(32\times32\) solve; the coefficient Hessian is \(128\times128\). The correction can use CPU computation and cached views. Renderer runtime and memory remain unmeasured.

Producing the generated views is a separate resource requirement. Full FreeOrbit4D and PREX are not qualified under the available 22 GB GPU budget. No implementation, GPU run, correctness result, calibration guarantee, or novelty PASS is claimed here.
