"""Flax modules and likelihood for TF → gene-expression NB regression.

Each model's ``__call__(x_tf, lib)`` returns ``(mean, concentration)`` —
the parameters of a per-cell, per-gene Negative-Binomial likelihood.  The
likelihood itself lives outside the model class so eval code that only
needs ``mean`` can skip the NLL branch and so swapping likelihoods later
does not touch the model.
"""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import glorot_normal, zeros
from numpyro.distributions import NegativeBinomial2


def lcp_to_count_mean(lcp_pred, lib, eps: float = 1e-8):
    """log-CP10K → NB-mean (raw-count scale), using the observed library."""
    return jnp.maximum(jnp.expm1(lcp_pred) * lib[:, None] / 1e4, eps)


def nb_nll(mean, concentration, y_raw):
    """Per-(cell, gene) NB NLL. No reduction — the caller picks the mean."""
    return -NegativeBinomial2(mean=mean, concentration=concentration).log_prob(y_raw)


class TFRegression(nn.Module):
    """Linear regression of log-CP10K expression on (normalised) TF expression.

    Predictor (per cell):
        lcp_pred = x_mean + (W ⊙ Amask_tf) @ x_tf + b

    Likelihood (per cell, per gene):
        y_raw | mean, conc  ~  NB2(mean, conc)
        mean = expm1(lcp_pred) · lib / 1e4
        conc = exp(overdispersion_)

    Set ``Amask_tf=None`` for the unconstrained variant (dense W).
    """

    n_genes: int
    n_tfs: int
    x_mean: jnp.ndarray  # (n_genes,)  frozen ctrl log-CP10K mean
    Amask_tf: jnp.ndarray | None = None  # (n_genes, n_tfs)  {0, 1}

    @nn.compact
    def __call__(self, x_tf, lib):
        W = self.param("W", glorot_normal(), (self.n_genes, self.n_tfs))
        b = self.param("b", zeros, (self.n_genes,))
        overdispersion_ = self.param("overdispersion_", zeros, (self.n_genes,))

        x_mean = jax.lax.stop_gradient(jnp.asarray(self.x_mean))
        if self.Amask_tf is not None:
            W = W * jax.lax.stop_gradient(jnp.asarray(self.Amask_tf))
        lcp_pred = x_mean + x_tf @ W.T + b

        mean = lcp_to_count_mean(lcp_pred, lib)
        conc = jnp.exp(overdispersion_)
        return mean, conc


# Floor on the log mean-rate offset ᾱ_g (= log of the control-cell mean CP10K).
# Genes silent in controls have mean rate 0 → ᾱ_g = −∞, so clamp to this value.
ALPHA_FLOOR = math.log(1e-3)


class TFRegressionFixed(nn.Module):
    """Log-link variant of ``TFRegression`` (the corrected mean parameterisation).

    Predictor (per cell):
        lcp_pred = ᾱ + (W ⊙ Amask_tf) @ x_tf + b

    Likelihood (per cell, per gene):
        y_raw | mean, conc  ~  NB2(mean, conc)
        mean = exp(lcp_pred) · lib / 1e4        # log link: mean > 0 always
        conc = exp(overdispersion_)

    Two differences from ``TFRegression``:

    * the inverse link is ``exp`` (not ``expm1``), so the count mean is strictly
      positive for every predictor value and needs no positive-part / ``eps``
      floor;
    * ``x_mean`` is the log mean-rate offset ᾱ_g = ``log`` of the control-cell mean
      CP10K rate (so at ``x_tf = 0, b = 0`` the mean equals the control rate),
      floored at ``ALPHA_FLOOR`` so genes silent in controls stay finite.

    Set ``Amask_tf=None`` for the unconstrained variant (dense W).
    """

    n_genes: int
    n_tfs: int
    x_mean: jnp.ndarray  # (n_genes,)  frozen offset ᾱ_g = log(ctrl-mean CP10K rate)
    Amask_tf: jnp.ndarray | None = None  # (n_genes, n_tfs)  {0, 1}

    @nn.compact
    def __call__(self, x_tf, lib):
        W = self.param("W", glorot_normal(), (self.n_genes, self.n_tfs))
        b = self.param("b", zeros, (self.n_genes,))
        overdispersion_ = self.param("overdispersion_", zeros, (self.n_genes,))

        x_mean = jax.lax.stop_gradient(jnp.maximum(jnp.asarray(self.x_mean), ALPHA_FLOOR))
        if self.Amask_tf is not None:
            W = W * jax.lax.stop_gradient(jnp.asarray(self.Amask_tf))
        lcp_pred = x_mean + x_tf @ W.T + b

        mean = jnp.exp(lcp_pred) * lib[:, None] / 1e4  # log link
        conc = jnp.exp(overdispersion_)
        return mean, conc


class TFRegressionNuisance(nn.Module):
    """``TFRegression`` with an additive linear block for nuisance covariates.

    The input ``x_tf`` is expected to carry the ``n_cov`` nuisance-covariate
    columns appended *after* the ``n_tfs`` TF columns::

        x = [ x_tf (n_tfs) | x_cov (n_cov) ]      # (B, n_tfs + n_cov)

    Packing the covariates onto ``x_tf`` keeps the ``__call__(x_tf, lib)``
    contract, so ``fit`` and every evaluator run unchanged.

    Predictor (per cell):
        lcp_pred = x_mean + (W ⊙ Amask_tf) @ x_tf + C @ x_cov + b

    The covariate block ``C`` (n_genes, n_cov) is always dense — it is never
    masked and never enters the regulon weights ``W``, so it absorbs
    technical/batch variation out of the NB residuals without perturbing the
    z-score diagnostics (which read ``params["W"] · Amask_tf``).  Likelihood is
    identical to ``TFRegression``.

    Set ``Amask_tf=None`` for the unconstrained variant (dense W).
    """

    n_genes: int
    n_tfs: int
    n_cov: int
    x_mean: jnp.ndarray  # (n_genes,)  frozen ctrl log-CP10K mean
    Amask_tf: jnp.ndarray | None = None  # (n_genes, n_tfs)  {0, 1}

    @nn.compact
    def __call__(self, x_tf, lib):
        W = self.param("W", glorot_normal(), (self.n_genes, self.n_tfs))
        C = self.param("C", glorot_normal(), (self.n_genes, self.n_cov))
        b = self.param("b", zeros, (self.n_genes,))
        overdispersion_ = self.param("overdispersion_", zeros, (self.n_genes,))

        x_tf, x_cov = x_tf[:, : self.n_tfs], x_tf[:, self.n_tfs :]
        x_mean = jax.lax.stop_gradient(jnp.asarray(self.x_mean))
        if self.Amask_tf is not None:
            W = W * jax.lax.stop_gradient(jnp.asarray(self.Amask_tf))
        lcp_pred = x_mean + x_tf @ W.T + x_cov @ C.T + b

        mean = lcp_to_count_mean(lcp_pred, lib)
        conc = jnp.exp(overdispersion_)
        return mean, conc


class ConstantMean(nn.Module):
    """Constant-mean NB null model: mean fixed at the control log-CP10K mean.

    Only per-gene NB concentration is trainable, so fitting with the package's
    ``fit`` is equivalent to per-gene MLE of overdispersion at fixed mean —
    exactly the "correct" baseline used in the notebook diagnostics.
    """

    n_genes: int
    x_mean: jnp.ndarray  # (n_genes,)

    @nn.compact
    def __call__(self, x_tf, lib):
        del x_tf  # constant predictor; TFs unused
        overdispersion_ = self.param("overdispersion_", zeros, (self.n_genes,))
        x_mean = jax.lax.stop_gradient(jnp.asarray(self.x_mean))
        mean = lcp_to_count_mean(x_mean[None, :], lib)
        conc = jnp.exp(overdispersion_)
        return mean, conc


class TFCellBox(nn.Module):
    """CellBox-style saturating regression of expression on TF expression.

    Predictor (per gene i, single forward pass — no rollout):
        f_i(x_tf) = α_i · σ( Σ_{j ∈ reg(i)} A_{ij} · x_tf_j + b_i )

    Mean (mode-dependent):
        ``absolute``:  lcp_pred = f(x_tf)
        ``residual``:  lcp_pred = x_mean + f(x_tf) − stop_gradient(f(0))

    The residual mode anchors the prediction at the population control mean and
    treats f's output as the perturbation-induced shift, so the default init
    (α=1, b=0) already places predictions near x_mean.  For TF inputs, f(0) is
    the constant ``α · σ(b)`` — no per-cell control vector needed.

    α is parameterised as ``exp(log_alpha)`` for positivity.  Setting
    ``Amask_tf=None`` gives an unconstrained (dense) A.
    """

    n_genes: int
    n_tfs: int
    x_mean: jnp.ndarray  # (n_genes,)
    Amask_tf: jnp.ndarray | None = None  # (n_genes, n_tfs)  {0, 1}
    mean_mode: str = "residual"  # "residual" | "absolute"

    @nn.compact
    def __call__(self, x_tf, lib):
        A = self.param("W", glorot_normal(), (self.n_genes, self.n_tfs))
        b = self.param("b", zeros, (self.n_genes,))
        log_alpha = self.param("log_alpha", zeros, (self.n_genes,))
        overdispersion_ = self.param("overdispersion_", zeros, (self.n_genes,))

        if self.Amask_tf is not None:
            A = A * jax.lax.stop_gradient(jnp.asarray(self.Amask_tf))
        alpha = jnp.exp(log_alpha)

        f_x = alpha * nn.sigmoid(x_tf @ A.T + b)  # (B, n_genes)

        if self.mean_mode == "residual":
            f_0 = jax.lax.stop_gradient(alpha * nn.sigmoid(b))  # (n_genes,)
            x_mean = jax.lax.stop_gradient(jnp.asarray(self.x_mean))
            lcp_pred = x_mean + f_x - f_0[None, :]
        elif self.mean_mode == "absolute":
            lcp_pred = f_x
        else:
            raise ValueError(f"mean_mode must be 'residual' or 'absolute', got {self.mean_mode!r}")

        mean = lcp_to_count_mean(lcp_pred, lib)
        conc = jnp.exp(overdispersion_)
        return mean, conc


class TFCellBoxNuisance(nn.Module):
    """``TFCellBox`` with an additive linear block for nuisance covariates.

    Covariates are appended to ``x_tf`` exactly as in ``TFRegressionNuisance``::

        x = [ x_tf (n_tfs) | x_cov (n_cov) ]      # (B, n_tfs + n_cov)

    The covariate contribution is linear and additive on the log-CP10K scale —
    it is *not* passed through the saturating nonlinearity, and (like the
    saturating block's ``f(0)`` anchor) stays out of the regulon matrix ``A``
    used by the z-score diagnostics::

        f(x_tf)      = α · σ( (A ⊙ Amask_tf) @ x_tf + b )
        ``absolute``:  lcp_pred = f(x_tf) + C @ x_cov
        ``residual``:  lcp_pred = x_mean + f(x_tf) − stop_gradient(f(0)) + C @ x_cov

    ``f(0)`` uses only the TF block (the constant ``α · σ(b)``), so covariates
    shift the mean additively without changing the perturbation-response shape.
    See ``TFCellBox`` for the rest of the parameterisation.
    """

    n_genes: int
    n_tfs: int
    n_cov: int
    x_mean: jnp.ndarray  # (n_genes,)
    Amask_tf: jnp.ndarray | None = None  # (n_genes, n_tfs)  {0, 1}
    mean_mode: str = "residual"  # "residual" | "absolute"

    @nn.compact
    def __call__(self, x_tf, lib):
        A = self.param("W", glorot_normal(), (self.n_genes, self.n_tfs))
        C = self.param("C", glorot_normal(), (self.n_genes, self.n_cov))
        b = self.param("b", zeros, (self.n_genes,))
        log_alpha = self.param("log_alpha", zeros, (self.n_genes,))
        overdispersion_ = self.param("overdispersion_", zeros, (self.n_genes,))

        x_tf, x_cov = x_tf[:, : self.n_tfs], x_tf[:, self.n_tfs :]
        if self.Amask_tf is not None:
            A = A * jax.lax.stop_gradient(jnp.asarray(self.Amask_tf))
        alpha = jnp.exp(log_alpha)

        f_x = alpha * nn.sigmoid(x_tf @ A.T + b)  # (B, n_genes)
        cov = x_cov @ C.T  # (B, n_genes)

        if self.mean_mode == "residual":
            f_0 = jax.lax.stop_gradient(alpha * nn.sigmoid(b))  # (n_genes,)
            x_mean = jax.lax.stop_gradient(jnp.asarray(self.x_mean))
            lcp_pred = x_mean + f_x - f_0[None, :] + cov
        elif self.mean_mode == "absolute":
            lcp_pred = f_x + cov
        else:
            raise ValueError(f"mean_mode must be 'residual' or 'absolute', got {self.mean_mode!r}")

        mean = lcp_to_count_mean(lcp_pred, lib)
        conc = jnp.exp(overdispersion_)
        return mean, conc
