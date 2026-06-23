# Model Diagnostic — CellBox NB (06052026)

## 1. Evaluation metric audit

### The all-gene LFC Pearson is biased toward Gaussian

The primary early-stopping metric `lfc_pearson_r` is computed over all 4725 genes.
Only ~0.6% of gene × perturbation pairs have |LFC| > 0.5; the remaining 99.4% are near zero.
Pearson r over all genes therefore measures how well the model fits the **near-zero background**,
not the perturbation signal. This creates a systematic advantage for the Gaussian model, which is
directly trained to minimize MSE in log-CP10K space (= the evaluation metric), while the NB model
optimizes count log-likelihood.

Confirmed by results:

| model | `lfc_pearson_r` (all genes) | `lfc_pearson_r_top20` | `auroc_top20` |
|---|---|---|---|
| Gaussian (no rollout) | 0.238 | 0.497 | 0.788 |
| NB (rollout) | 0.095 | 0.582 | 0.791 |

On the top-20 DEG metrics, the NB model is competitive or better.
`lfc_pearson_r` alone should not be used to compare models with different training objectives.
**Use `lfc_pearson_r_top20_degs` and `auroc_top20_degs` as primary metrics.**

### Pearson r is shift-invariant for scalar shifts but not vector shifts

`pearsonr(mu_pred, mu_gt) ≠ pearsonr(lfc_pred, lfc_gt)` when `mu_control` is a vector.
Subtracting the per-gene control mean changes the Pearson r because it removes a large shared
component (`mu_ctrl` ~ 0.97 Pearson with `mu_gt`), making `lfc_pearson_r` more discriminative.
The old `mu_pearson_r` metric was therefore not a duplicate and not kept for the wrong reason —
but `lfc_pearson_r` is more useful since the signal of interest is the LFC, not absolute expression.

---

## 2. lacI case study: interpreting individual perturbation predictions

### Ground truth is biologically valid

For the lacI KD (5 test cells), the top upregulated genes are exactly the canonical direct targets:

| gene | LFC | z-score |
|---|---|---|
| lacZ | +3.47 | +64.6 |
| lacY | +1.94 | +48.8 |
| lacA | +1.18 | +29.0 |
| lacI | −0.70 | −4.8 |

These are high-confidence effects (z > 25) despite only 5 cells. The dataset contains real signal.
The top-25 downregulated genes (ribosomal proteins, tRNA synthetases) are more ambiguous —
plausible as indirect growth-rate effects, but with lower z-scores (2–13) and no canonical biology.

### The model completely misses the direct biology

The NB model predictions for lacI KD:

| gene | n_reg | lfc_true | lfc_pred |
|---|---|---|---|
| lacZ | 4 | +3.47 | **−0.04** |
| lacY | 4 | +1.94 | **−0.02** |
| lacA | 4 | +1.18 | **−0.03** |

The three biologically meaningful genes are entirely missed. The A-matrix entries
`A[lacZ/lacY/lacA, lacI]` are all positive (+0.81, +0.65, +0.54) when they should be negative
(lacI is a repressor). The model learned a spurious positive co-expression correlation between
lacI and its targets — because **lacI was never perturbed during training** (it is a held-out TF),
so A[·, lacI] was shaped only by observational covariance, not causal signal.

---

## 3. Core failure mode: model collapses to predicting the training marginal mean

### The fixed-point test

A model consistent with its own definition of control steady state should satisfy:

```
predict(x_mean, u=0) = x_mean
```

It does not:

```
|predict(x_mean, u=0) − x_mean|  >  0.1 log-CP10K   for 49% of genes
                                  >  0.5 log-CP10K   for 18% of genes
mean |bias| / x_std  =  1.16σ    (a third of genes >1 ctrl SD off)
```

Rolling the control mean forward with `u=0` diverges immediately and does not return:

```
step 2:  mean|x − x_mean| = 0.260
step 11: mean|x − x_mean| = 0.241   (flat, not decaying)
```

The model's learned attractor and the control mean are different fixed points.

### Prediction variance decomposition

Across 44 test perturbations:

```
Total pred LFC std:                     0.321
  ├─ Constant bias (per-gene mean):     0.320   (99.6% of variance)
  └─ Perturbation-specific residual:    0.029   ( 0.4% of variance)

Genes with exactly constant predictions across all perturbations:  2736 / 4725 (58%)
Genes with near-constant predictions (var < 1e-4):                 3884 / 4725 (82%)
```

The model is predicting the same expression profile under every perturbation.
Less than 1% of prediction variance is perturbation-specific.

### Root cause

The model collapses to `F(X, u) ≈ C`, a constant per-gene vector that ignores both `X` and `u`,
with two compounding mechanisms:

**Direct bias (e.g., phoB).**
`ss_ctrl[i] = exp(ε_i) · sigmoid(b_i)` was learned from the training distribution
(control + perturbed cells mixed), while `x_mean_[i]` was frozen from control-only cells.
For genes globally downregulated in perturbed conditions (hns, ihfa, ihfb), `ss_ctrl ≪ x_mean`.
After step 1 of any rollout, all genes immediately land at `ss_ctrl`, discarding the starting
state X. The predicted LFC is then `ss_ctrl − x_mean`, constant across all perturbations.

**Amplified bias (e.g., fliZ).**
Genes whose regulators have large `ss_ctrl − x_mean` biases inherit those biases amplified
through the A matrix. For fliZ: hns (z_ss = −5.5), ihfb (z_ss = −2.1), ihfa (z_ss = −2.3)
are all far below their control means at step 1. Since A[fliZ, hns/ihfb/ihfa] < 0 (repressors),
the negative z-scores produce a large positive preactivation:

```
Δpreact[fliZ] = (−0.40)(−5.54) + (−0.87)(−2.07) + (−0.57)(−2.28) + ... = +2.50
→ x[fliZ] jumps from 0.98 to 1.54 at step 2, independent of the perturbation
```

Confirmed: fliZ predicted LFC = +1.439 under 38/44 perturbations (identical to 4 decimal places).
phoB predicted LFC = +0.808 under all 44 perturbations (literally constant).

### Code bug contributing to the bias

`cellbox_steady_state_nb.py:36` ignores the data-driven `epsilon_init`:

```python
# NB model (wrong): eps starts at exp(0)=1 for all genes
self.epsilon_ = self.param("epsilon_", zeros, (G,))

# Gaussian model (correct): eps initialized from 2 * control_mean
self.epsilon_ = self.param("epsilon_", _const_init(_eps), (G,))
```

Without a data-driven initialization, the NB model has no anchor pushing `ss_ctrl` toward
`x_mean` at the start of training, making the bias worse.

---

## 4. What this means structurally

The model `F(X, u) ≈ C_train` is the maximum likelihood solution under the NB loss on the
joint training distribution `p(X, u, Z)`. The perturbation effect `E[Z|u] − E[Z]` is a small
signal relative to the baseline variance `Var(Z)`. Gradient descent finds the baseline minimum
first and never escapes it. This is confirmed by the 99.6% / 0.4% variance decomposition.

The model is not learning perturbation effects. It is learning a single "average perturbed cell"
profile per gene, with the NB dispersion absorbing the residual.

---

## 5. Principled remedies

The following strategies are statistically motivated and address the collapse at its source.
They are listed from most to least directly implementable.

### 5.1 Log-likelihood ratio as training objective

Replace the NB log-likelihood with the log-likelihood ratio against a baseline predictor:

$$\mathcal{L}_{\mathrm{LR}} = -\mathbb{E}\big[\log p_{\mathrm{NB}}(Z \mid \ell\, F(X,u)) - \log p_{\mathrm{NB}}(Z \mid \ell\, X_{\mathrm{ctrl}})\big].$$

**Why this is principled:**
- The second term is a control variate: correlated with the first (both large for highly expressed
  genes) but has zero gradient w.r.t. $F$, so subtracting it costs nothing.
- At the degenerate minimum $F = X_{\mathrm{ctrl}}$, $\mathcal{L}_{\mathrm{LR}} = 0$ for all
  samples — the model receives no gradient at the collapse point rather than a spurious one.
- For the NB exponential family with log link, this equals the profile likelihood after profiling
  out the per-gene baseline mean — it is the efficient score for the perturbation effect
  (Bickel et al. semiparametric efficiency theory).

### 5.2 ΔF as a mechanistically-derived treatment effect (pinned control fixed point)

**The problem this addresses.** §3 showed the collapse: the model's control fixed point $F(X,0)$
drifts away from the observed control mean $x_{\mathrm{mean}}$ (fixed-point residual $\sim 1.16\sigma$),
and once that baseline is wrong, $\sim 99.6\%$ of every prediction is a constant per-gene bias and
only $\sim 0.4\%$ is perturbation-specific. The cure is to stop the model from explaining data by
moving the baseline: **pin $F(X,0) = x_{\mathrm{mean}}$** so the baseline is no longer a free
quantity, and the *only* fittable signal left is the perturbation response. When that holds, every
prediction is automatically an additive treatment effect on the frozen baseline,
$F(X,u) = x_{\mathrm{mean}} + \Delta F(X,u)$.

The naive way to get an additive treatment effect — bolt a free-form network $\Delta F(X,u)$ onto
$x_{\mathrm{mean}}$ — is not mechanistic and reintroduces an unconstrained function. Instead, keep
the GRN dynamics and derive $\Delta F$ from them.

The treatment effect is then **not parameterized as a free function**. In the mechanistic model it
is a *derived* quantity: the gap between two fixed points of the same dynamics, sharing one set of
GRN parameters $(A, b, \alpha)$, with the perturbation entering only through the $-p \odot u$ term:

$$F(X,u) = \text{fixed point of } \; x \mapsto \alpha \odot \sigma\big(A\,z(x) + b - p \odot u\big), \qquad z(x) = \frac{x - x_{\mathrm{mean}}}{x_{\mathrm{std}}}, \quad \alpha = e^{\epsilon},$$

$$\Delta F(X,u) = F(X,u) - F(X,0) \qquad \text{(both terms use the same } A, b, \alpha\text{)}.$$

The reparameterization, done mechanistically, is simply: **pin the control fixed point $F(X,0)$ to
the observed $x_{\mathrm{mean}}$ by construction**, and let $\Delta F$ be whatever the dynamics
produce.

**Closed-form constraint on $b$ (the basal Pol recruitment).**
Because the normalization makes $z(x_{\mathrm{mean}}) = 0$, the $A$-term vanishes at control, so
$F(X,0)_i = \alpha_i\, \sigma(b_i)$. Requiring this to equal $x_{\mathrm{mean},i}$ gives a closed
form:

$$b_i = \operatorname{logit}\!\big(x_{\mathrm{mean},i} / \alpha_i\big), \qquad \text{requires } \alpha_i > x_{\mathrm{mean},i}.$$

This is not a heuristic — it is exactly $b_i = \ln P_i$, the basal polymerase recruitment weight,
calibrated so the baseline Pol leakiness reproduces observed control expression. The regulatory
edges $A$ are left free to explain *deviations* from baseline, which is all perturbation data should
inform. (This also explains the Gaussian init: $\alpha = 2 x_{\mathrm{mean}} \Rightarrow
x_{\mathrm{mean}}/\alpha = 0.5 \Rightarrow b = 0$. And it shows why the `epsilon_` bug is fatal:
with $\alpha = e^0 = 1$, the constraint $\alpha_i > x_{\mathrm{mean},i}$ is violated for every gene
with $x_{\mathrm{mean},i} > 1$, so no valid basal $b$ reproduces control — forcing the collapse.)

**What ΔF then is: network comparative statics.**
Linearizing the fixed point in $u$ via the implicit function theorem on
$G(x,u) = \alpha \odot \sigma(A\,z(x) + b - p u) - x = 0$, evaluated at control ($s = b$, $z = 0$):

$$\Delta F \approx -(I - M_0)^{-1}\, \operatorname{diag}\!\big(\alpha \odot \sigma'(b) \odot p\big)\, u, \qquad M_0 = \operatorname{diag}\!\big(\alpha \odot \sigma'(b)\big)\, A\, \operatorname{diag}(1/x_{\mathrm{std}}),$$

with $\sigma'(b) = \sigma(b)\big(1 - \sigma(b)\big)$.

- $\Delta F(X,0) = 0$ by construction — the degenerate minimum is now the correct biological null
  (no perturbation), not the training mean.
- $(I - M_0)^{-1} = I + M_0 + M_0^2 + \dots$ is literal network propagation: the $I$ term is the
  direct knockdown of the targeted gene, $M_0$ the one-hop effect through regulators, $M_0^2$
  two-hop, etc. This is the mechanistic content — $\Delta F$ *is* the GRN response, not a generic
  correction.
- $\sigma'(b)$ is the per-gene responsiveness: genes near their sigmoid midpoint respond, saturated
  genes are buffered. This is exactly the residual degree of freedom perturbation data identifies.

**How ΔF is used (forward pass and loss).**
$\Delta F$ is never computed or added explicitly. The model still predicts a single profile by
running the rollout to its fixed point:

$$\text{prediction: } \mu(X,u) = F(X,u) = \text{rollout of } x \mapsto \alpha \odot \sigma\big(A\,z(x) + b - p \odot u\big) \text{ from } x = x_0,$$

$$\text{loss: } Z \sim \mathrm{NB}\big(\ell \cdot \operatorname{expm1}(F(X,u)),\, \theta\big) \qquad \text{(unchanged from the current head)}.$$

The only change versus the current code is structural: $b$ is pinned to
$\operatorname{logit}(x_{\mathrm{mean}}/\alpha)$ instead of being free. As a consequence of that
pinning, $F$ factors *automatically* as

$$F(X,u) = x_{\mathrm{mean}} + \Delta F(X,u), \qquad \Delta F(X,0) = 0,$$

so the prediction is mechanically an additive treatment effect on top of the frozen control
baseline — even though the forward pass only ever evaluates $F$. The closed-form $\Delta F$
expression above is therefore **not part of training**; it is (i) the proof that $\Delta F(X,0)=0$
and that the collapse minimum is gone, (ii) an interpretation of each prediction as direct +
network-propagated effect, and (iii) an optional fast linear surrogate for $F$ when a full rollout
is not needed.

**Identifiability division of labor.**
At control, $\alpha_i\, \sigma(b_i) = x_{\mathrm{mean},i}$ is one equation in two unknowns
$(\alpha_i, b_i)$: static control data cannot separate the ON-ceiling $\alpha$ from basal
recruitment $b$. The constraint removes this baseline degree of freedom (killing the collapse); the
operating point $\sigma'(b_i)$ — how far $b$ sits from saturation — is then fit by the perturbation
response. Control data fixes the baseline, perturbation data fixes the responsiveness.

**Concrete changes.**
1. Fix the bug: $\alpha_i = e^{\epsilon_i}$ with data-driven init; enforce
   $\alpha_i > x_{\mathrm{mean},i}$, e.g. $\alpha_i = x_{\mathrm{mean},i} + \operatorname{softplus}(\xi_i)$.
2. Constrain $b$: set $b_i = \operatorname{logit}(x_{\mathrm{mean},i} / \alpha_i)$ as a deterministic
   function of $\alpha$ (not a free parameter). If slack is wanted, learn $\delta b_i$ on top and
   regularize toward $0$ — but the unregularized point must be the pinned baseline.
3. Stability: require spectral radius $\rho(M_0) < 1$ (regularize $A$). This guarantees the rollout
   converges to the pinned control fixed point and makes $(I - M_0)^{-1}$ well-defined — it is what
   makes $\Delta F$ exist as comparative statics.

Net effect: the model can no longer explain perturbed cells by shifting the global baseline (frozen
to $x_{\mathrm{mean}}$ mechanistically), so the only fittable signal is the network-propagated
perturbation response. The collapse minimum is removed from the parameterization, not merely
penalized.

Strategies 5.1 and 5.2 are complementary: 5.2 removes the degenerate minimum from the model
parameterization; 5.1 removes it from the loss landscape.

### 5.3 Orthogonalized / doubly-robust loss (double ML for treatment effects)

**Why this matters here.** §5.2 kills the collapse by *pinning* the baseline ($F(X,0)=
x_{\mathrm{mean}}$). But if the baseline is instead *estimated* (e.g. learned per gene, or
shrunk across genes), errors in that estimate leak into the perturbation effect $\Delta F$ and
re-bias it — the same failure mode in softer form. Double ML is the principled fix: build the
training objective so it is *orthogonal* to the baseline, i.e. first-order errors in the baseline
estimate do not bias $\Delta F$ (Neyman orthogonality). This is the insurance you want if you ever
relax the §5.2 pin.

**Real training-time operationalizations** (loss / pseudo-outcome level):
- **R-learner** — Nie & Wager, "Quasi-oracle estimation of heterogeneous treatment effects,"
  *Biometrika* 108(2), 2021: Robinson-style partialling-out — residualize outcome and treatment by
  their conditional means, then regress residual-on-residual. Origin: Robinson, "Root-N-consistent
  semiparametric regression," *Econometrica* 56(4), 1988.
- **DR-learner** — Kennedy, "Towards optimal doubly robust estimation of heterogeneous causal
  effects," arXiv 2020 / *Electronic Journal of Statistics* 2023: regress the AIPW doubly-robust
  pseudo-outcome on covariates.
- **Targeted regularization / Dragonnet** — Shi, Blei & Veitch, "Adapting Neural Networks for the
  Estimation of Treatment Effects," *NeurIPS* 2019; related to TMLE (van der Laan & Rubin, 2006).
  The closest "train a neural net with the efficient influence function" reference.

Theory underneath: Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins,
"Double/debiased machine learning for treatment and structural parameters," *The Econometrics
Journal* 21(1), 2018; and Bickel, Klaassen, Ritov & Wellner, *Efficient and Adaptive Estimation for
Semiparametric Models*, 1993.

**Caveat.** These methods target a low-dimensional (often scalar) CATE with a scalar treatment;
here the effect is a high-dimensional gene×perturbation field. And they orthogonalize the
*estimating equation*, not a network's loss gradient. If you keep the §5.2 pin, this section is
unnecessary — it earns its place only when the baseline becomes a learned nuisance.

### 5.4 Conditional likelihood (sufficient statistics)

In the NB exponential family, the baseline expression can be conditioned out:

$$p(Z \mid Z_{\mathrm{ctrl}}, u) = \frac{p_{\mathrm{NB}}(Z \mid F(X,u))\; p_{\mathrm{NB}}(Z_{\mathrm{ctrl}} \mid X_{\mathrm{ctrl}})}{p(Z + Z_{\mathrm{ctrl}})}.$$

Training on this conditional likelihood eliminates the per-gene baseline as a nuisance,
equivalent to Fisher's exact test / conditional Poisson regression in matched case-control studies.
The resulting estimator is Rao-Blackwellized (minimum variance among unbiased estimators).
