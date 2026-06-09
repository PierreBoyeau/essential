# Predicting transcriptomic responses to unseen perturbations with a steady-state CellBox / NB model

## 1. Problem setup and target

Let genes be indexed by $g \in \{1,\dots,G\}$. A perturbation $p$ is a genetic intervention (here, a transcription-factor knockdown), which we encode as a one-hot vector $u \in \{0,1\}^G$ with $u_k = 1$ when gene $k$ is knocked down; the control ("non-targeting") condition corresponds to $u = 0$.

We observe a cell's transcriptome through integer counts $X_t \in \mathbb{N}^G$ measured at time $t$ after perturbation, and we write $H_t \in \mathbb{R}^G$ for the corresponding *log-normalized* expression vector. Several cells may share the same perturbation $p$.

The quantity of interest is the **expected log-normalized response to a perturbation**, conditioned on the perturbation identity:

$$
\mathbb{E}\!\left[H_t \mid p = p^*\right] \;=\; f_\theta(D_t, p^*),
$$

where $p^*$ is an **unseen** perturbation (not present in the training conditions), and

$$
D_t \;=\; \{(X_t^i, p^i)\}_{i=1}^{n}
$$

is the training set of observed perturbations and their measured effects, **including control observations**. The model $f_\theta$ is fit on $D_t$ and then queried at $p^*$; generalization is to perturbations, not merely to new cells.

**Evaluation.** Performance is measured per held-out perturbation by the Pearson correlation between predicted and ground-truth log-fold-changes (relative to control). The data split holds out $\sim 20\%$ of charted TFs as $\mathcal{D}^{\text{test}}$, the remaining $\sim 80\%$ forming $\mathcal{D}^{\text{train}}$.

## 2. The mechanistic predictor: a steady-state CellBox model

### 2.1 Continuous-time dynamics (CellBox; Yuan et al., 2021)

CellBox models the abundance $[X_i]^{(k)}$ of gene $i$ under perturbation of gene $k$ by the ODE

$$
\frac{d[X_i]^{(k)}}{dt}
= \underbrace{\epsilon_i\, \sigma\!\Big( \textstyle\sum_{j} A_{ij}[X_j]^{(k)} + b_i \Big)}_{\text{production}}
\;-\; \underbrace{\mu_i\,[X_i]^{(k)}}_{\text{degradation}},
$$

with $\sigma$ the sigmoid nonlinearity. The matrix $A$ encodes the gene-regulatory network: $A_{ij}$ is the regulatory effect of gene $j$ on gene $i$, with $A_{ij}\neq 0$ only if $j$ is a plausible regulator of $i$ (a causal mask $M$ can be imposed). The diagonal is fixed to zero. This formulation does **not** model metabolite-mediated TF effects.

### 2.2 Steady-state reduction

We work under a steady-state assumption, $d[X_i]/dt \big|_{X^*} = 0$, which yields the fixed-point relation

$$
X_i^* = \alpha_i\, \sigma\!\big( g^{\phi}_i(X^*) + b_i \big),
$$

i.e. each gene's steady abundance is a saturating function of a regulatory pre-activation $g^\phi_i$ of the whole state. In the present parameterization $g^\phi$ is a linear GRN, $g^\phi_i(X) = \sum_j A_{ij}\,\tilde X_j$, applied to a normalized state $\tilde X$, but $g^\phi$ may also be instantiated as a self-normalized attention model or a generic linear projection.

Concretely, with a standardized predictor
$$
\tilde X = \frac{\log(1+X) - \log(1+\bar X^{\text{ctrl}})}{\,s^{\text{ctrl}}\,}
$$
(mean $\bar X^{\text{ctrl}}$ and s.d. $s^{\text{ctrl}}$ estimated on control cells), the perturbed fixed point solves

$$
X^* \;=\; \epsilon \odot \sigma\!\big( A\,\tilde X^* \;-\; \rho\, u \;+\; b \big),
$$

where $u$ injects the knockdown of the targeted gene with strength $\rho$ (a large negative drive on the perturbed gene's pre-activation), and $\odot$ is the elementwise product.

### 2.3 Prediction by rollout

Given an unperturbed input state $X^{(0)}$ (a sampled control cell) and a target perturbation $u$, the steady state is obtained by **iterating the fixed-point map** to convergence,

$$
X^{(m+1)} \;=\; \epsilon \odot \sigma\!\big( A\,\tilde X^{(m)} - \rho\,u + b \big),
\qquad m = 0,\dots,M-1,
$$

starting from $X^{(0)}$ and running $M$ steps (rollout). The predicted log-normalized response is

$$
H \;=\; f_\theta(D_t, p) \;=\; \log\!\big(1 + c\, X^{(M)}\big),
\qquad c = 10^4,
$$

the standard log-normalization scale. The conditional mean $\mathbb{E}[H_t \mid p^*]$ is estimated by averaging $H$ over a population of control input cells rolled out under $u^*$.

The learnable parameters are
$$
\theta = \big\{\, A,\; (b_i)_{i=1}^G,\; (\epsilon_i)_{i=1}^G,\; \phi,\; \text{(normalization stats)} \,\big\},
$$
with $A$ optionally constrained by the causal mask $M$ and zero diagonal.

## 3. The negative-binomial observation model

To fit $\theta$ directly against raw counts, the deterministic prediction $H$ is coupled to a count likelihood. We model the observed count of gene $g$ in cell $i$ as negative-binomial,

$$
X_{t,g}^{i} \;\sim\; \mathrm{NB}\big(\mu_{t,g}^{i},\, \phi_g\big),
$$

in the mean–dispersion ($\mathrm{NB2}$) parameterization, so that
$$
\mathbb{E}[X_{t,g}^{i}] = \mu_{t,g}^{i},
\qquad
\mathrm{Var}[X_{t,g}^{i}] = \mu_{t,g}^{i} + \frac{(\mu_{t,g}^{i})^2}{\phi_g},
$$
with a per-gene dispersion $\phi_g > 0$ (parameterized as $\phi_g = \exp(\omega_g)$).

The NB mean is set by mapping the model's log-normalized prediction $H^i$ back to a count scale and rescaling by a cell-specific size factor:

$$
\mu_{t,g}^{i} \;=\; l^{i}\,\big(\exp(H_{t,g}^{i}) - 1\big),
$$

where $\exp(H_{t,g}^i)-1$ recovers the normalized expression fraction and $l^i$ is the **size factor** (library scale) of cell $i$. The size factor is itself amortized,

$$
l^{i} = s_\theta(X^{i}),
$$

implemented as a small feed-forward network taking the cell's total observed count and its normalized predicted profile as input (with softplus output to enforce $l^i > 0$).

### 3.1 Training objectives

Two regimes are used:

- **Reconstruction.** The NB negative log-likelihood is evaluated with the per-gene predictions computed from each observed cell's own state, summed over genes and cells. The targeted gene is excluded from the loss via a mask $m^i = \mathbf{1} - u^i$, since its expression is mechanically suppressed by the knockdown:
$$
\mathcal{L}_{\text{reco}}(\theta)
= -\,\frac{1}{nG}\sum_{i=1}^{n}\sum_{g=1}^{G}
m_g^{i}\;\log \mathrm{NB}\big(X_{t,g}^{i}\mid \mu_{t,g}^{i}, \phi_g\big).
$$

- **Rollout.** Starting from control inputs $X^{(0)}$, the perturbed steady state $H = f_\theta(D_t,p)$ is rolled out for $M$ steps and matched to observed perturbed profiles by a masked squared error in log-normalized space:
$$
\mathcal{L}_{\text{roll}}(\theta)
= \frac{1}{n}\sum_{i=1}^{n}\sum_{g=1}^{G}
m_g^{i}\,\big(H_g^{i} - H_{t,g}^{i,\text{obs}}\big)^2 .
$$

In both cases the masking ensures the model is scored on the *propagated* transcriptomic response rather than on the trivially-determined knocked-down gene itself.

## 5. Proposed extension: nonlinear regulatory aggregation via DeepSets

### 5.1 Motivation

The current pre-activation is a **linear** map over regulators:

$$
g^\phi_i(X) = \sum_{j \in \mathrm{pa}(i)} A_{ij}\,\tilde X_j,
$$

where $\mathrm{pa}(i) = \{j : M_{ij} = 1\}$ is the parent set of gene $i$ under causal mask $M$, and $A_{ij}$ is a scalar learned weight per edge. This is equivalent to a message-passing GNN with linear messages and sum aggregation.

Two limitations motivate a richer parameterization:

1. **Linear interactions only.** The effect of regulator $j$ on gene $i$ is proportional to $\tilde X_j$ with a fixed slope $A_{ij}$. Saturation, thresholding, and synergistic effects between regulators are not representable.
2. **No regulator identity in the aggregation function.** Because the aggregation is linear, the model cannot learn that a given TF (say, gene $j^*$) exerts qualitatively different dynamics than a generic regulator with the same expression level.

### 5.2 Gathered regulator representation

Define the **gather index** $R \in \mathbb{Z}^{G \times K}$, where $K = \max_i |\mathrm{pa}(i)|$ (here $K = 17$), by

$$
R_{ik} = \begin{cases} j_k & \text{if } k < |\mathrm{pa}(i)|, \\ 0 & \text{(padding)} \end{cases}
$$

with $j_1 < j_2 < \cdots$ the sorted parent indices of gene $i$. The corresponding validity mask is $M^R_{ik} = \mathbf{1}[k < |\mathrm{pa}(i)|]$.

For a normalized expression vector $\tilde X \in \mathbb{R}^G$, the gathered regulator matrix is

$$
\tilde X^R \in \mathbb{R}^{G \times K}, \qquad \tilde X^R_{ik} = \tilde X_{R_{ik}} \cdot M^R_{ik}.
$$

This is a static topology-encoding step with no learnable parameters; $R$ and $M^R$ are precomputed from $M$ once and reused throughout training.

### 5.3 DeepSets aggregation with regulator embeddings

Each gene is assigned a **learned identity embedding** $e_j \in \mathbb{R}^d$ collected in a matrix $E \in \mathbb{R}^{G \times d}$. The element feature for regulator slot $k$ of target gene $i$ is

$$
f_{ik} = \bigl[\,\tilde X^R_{ik},\; E_{R_{ik}}\,\bigr] \;\in\; \mathbb{R}^{1+d},
$$

which pairs the expression value of the regulator with its identity.

The proposed pre-activation follows the **DeepSets** decomposition (Zaheer et al., NeurIPS 2017):

$$
\boxed{
g^\phi_i(\tilde X) = \rho\!\left(\;\sum_{k=0}^{K-1} \phi(f_{ik})\cdot M^R_{ik}\;\right)
}
$$

where

- $\phi : \mathbb{R}^{1+d} \to \mathbb{R}^{h}$ is a **shared message MLP** (one hidden layer, tanh activation),
- $\rho : \mathbb{R}^{h} \to \mathbb{R}$ is a **shared linear readout**.

The sum is masked so that padding slots ($M^R_{ik} = 0$) contribute zero. The full pre-activation and fixed-point map are otherwise unchanged:

$$
\mathrm{preact}_i = g^\phi_i(\tilde X) - \rho_0\, u_i + b_i, \qquad X^*_i = \epsilon_i\,\sigma(\mathrm{preact}_i).
$$

### 5.4 Relation to the current model

Setting $\phi(f) = (W_\phi^\top e)\, x$ (bilinear, no nonlinearity) and $\rho = \mathrm{id}$ recovers

$$
g^\phi_i(\tilde X) = \sum_{k} (W_\phi^\top E_{R_{ik}})\,\tilde X^R_{ik} = \sum_{j \in \mathrm{pa}(i)} A_{ij}\,\tilde X_j,
$$

i.e., the current linear model with $A_{ij} = W_\phi^\top e_j$. The proposed model is therefore a strict generalization.

### 5.5 Learnable parameters

| Parameter | Shape | Count ($d=16,\,h=16$) |
|---|---|---|
| $E$ — regulator identity embeddings | $(G,\, d)$ | $75{,}600$ |
| $W_\phi,\, b_\phi$ — message net | $(1+d) \times h + h$ | $288$ |
| $W_\rho,\, b_\rho$ — readout | $h + 1$ | $17$ |
| $b$ — per-gene bias | $(G,)$ | $4{,}725$ (unchanged) |

Total new parameters: $\approx 76\mathrm{K}$, compared to $6{,}070$ active entries in the current $A$ matrix. The increase is almost entirely in the embedding table $E$.

### 5.6 Degenerate cases

- **Gene with no regulators** ($|\mathrm{pa}(i)| = 0$, affects $48\%$ of genes here): all slots are masked, the sum is zero, and $g^\phi_i = \rho(0)$, a learned scalar constant. Pre-activation reduces to $\rho(0) + b_i$, which is exactly the bias-only behavior of the current model for unregulated genes.
- **Single regulator** ($|\mathrm{pa}(i)| = 1$): the sum has one term, $g^\phi_i = \rho(\phi(\tilde X_j, e_j))$, a nonlinear function of the sole regulator's expression and identity.

---

## 4. Status and next step

Current predictive power on held-out TF knockdowns ($\mathcal{D}^{\text{test}}$, $20\%$ of charted TFs) is low under the Pearson-$R$(predicted LFC, ground-truth LFC) metric. The intended extension is to incorporate **metabolite-concentration effects on TF activity** into the CellBox model, addressing the modeling gap noted in §2.1.

## Reproducing the experiments

```bash
cd /workspace

# prepare data (re-run if data/ is missing or stale)
python experiments/06052026_cellbox_noise/prepare_data.py \
    --out_dir experiments/06052026_cellbox_noise/data

for CONFIG in experiments/06052026_cellbox_noise/configs/*.py; do
    python src/cellbox/train.py --config=$CONFIG
    python src/cellbox/predict.py --config=$CONFIG
    python src/evaluation/evaluate.py --config=$CONFIG
done
```


CONFIG=/workspace/experiments/06052026_cellbox_noise/configs/base_config.py
CUDA_VISIBLE_DEVICES=0 python src/cellbox/train.py --config=$CONFIG
python src/cellbox/predict.py --config=$CONFIG
python src/evaluation/evaluate.py --config=$CONFIG

CONFIG=/workspace/experiments/06052026_cellbox_noise/configs/nb_model.py
CUDA_VISIBLE_DEVICES=1 python src/cellbox/train.py --config=$CONFIG
python src/cellbox/predict.py --config=$CONFIG
python src/evaluation/evaluate.py --config=$CONFIG

CONFIG=/workspace/experiments/06052026_cellbox_noise/configs/nb_model_deepset.py
CUDA_VISIBLE_DEVICES=2 python src/cellbox/train.py --config=$CONFIG
python src/cellbox/predict.py --config=$CONFIG
python src/evaluation/evaluate.py --config=$CONFIG

CONFIG=/workspace/experiments/06052026_cellbox_noise/configs/no_rollout.py
CUDA_VISIBLE_DEVICES=3 python src/cellbox/train.py --config=$CONFIG
python src/cellbox/predict.py --config=$CONFIG
python src/evaluation/evaluate.py --config=$CONFIG
