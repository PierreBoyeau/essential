import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from tqdm import tqdm
from flax.training import train_state
import pandas as pd
from flax.linen.initializers import normal
import scanpy as sc
import diffrax

class SteadyStateForcingModel(nn.Module):
    """Steady-state model with external forcing perturbation.
    
    Model: Σ_j A_{ij} x_j = u_i
    Issue: Treats knockdown as external forcing (unphysical at steady-state)
    """

    n_genes: int
    n_tfs: int
    tf2gene_indicators: jnp.ndarray
    lambda_prior: float

    def setup(self):
        self.Amat_ = self.param(
            'Amat_',
            normal(),
            (self.n_genes, self.n_genes)
        )
        self.bvec_ = self.param(
            'bvec_',
            normal(),
            (self.n_tfs)
        )

    def get_Amat(self):
        # Amat = self.Amat_ * (1 - jnp.eye(self.n_genes))  
        # this was a mistake, we should allow for diagonal terms
        # to capture decay of regulatory elements
        return self.Amat_

    def get_bvec(self):
        return -nn.softplus(self.bvec_)

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> dict:

        indic_times_param = u * self.get_bvec()        
        perturb_contribution = jnp.einsum('gf,nf->ng', self.tf2gene_indicators, indic_times_param)
        A_mat = self.get_Amat()
        conc_contribution = jnp.einsum('gj,nj->ng', A_mat, x)

        # reco_loss = jnp.mean((conc_contribution + perturb_contribution)**2)
        # l1_prior = jnp.mean(jnp.abs(A_mat))        
        # loss = reco_loss + self.lambda_prior * l1_prior
        
        N = x.shape[0]
        reco_loss = jnp.mean((conc_contribution + perturb_contribution)**2)
        lap_density = jnp.abs(A_mat) * self.lambda_prior
        l1_prior = jnp.sum(lap_density) / N
        loss = reco_loss + l1_prior
        return {'loss': loss, 'reco_loss': reco_loss, 'l1_prior': l1_prior}

class DynamicCellboxModel(nn.Module):
    n_genes: int
    n_tfs: int
    tf2gene_indicators: jnp.ndarray
    lambda_prior: float

    def setup(self):
        self.Amat_ = self.param(
            'Amat_',
            normal(),
            (self.n_genes, self.n_genes)
        )
        self.bvec_ = self.param(
            'bvec_',
            normal(),
            (self.n_tfs)
        )

    def get_Amat(self):
        return self.Amat_

    def get_bvec(self):
        return -nn.softplus(self.bvec_)

    def __call__(self, x: jnp.ndarray, u: jnp.ndarray) -> dict:

        A_mat = self.get_Amat()
        bvec = self.get_bvec()
        solver = diffrax.Dopri5()
        saveat = diffrax.SaveAt(t1=True)

        def solve_single(x_i, u_i):
            indic_times_param_i = u_i * bvec
            perturb_i = jnp.einsum('gf,f->g', self.tf2gene_indicators, indic_times_param_i)

            def ode_fn(t, y, args):
                conc_contribution = jnp.einsum('gj,j->g', A_mat, y)
                return conc_contribution + perturb_i

            ode_term = diffrax.ODETerm(ode_fn)
            sol = diffrax.diffeqsolve(ode_term, solver, t0=0.0, t1=1.0, dt0=0.1, y0=x_i, saveat=saveat)
            return sol.ys

        xpred = jax.vmap(solve_single, in_axes=(0, 0))(x, u)
        reco_loss = jnp.mean((xpred - x)**2)
        l1_prior = jnp.mean(jnp.abs(A_mat))
        loss = reco_loss + self.lambda_prior * l1_prior
        return {'loss': loss, 'reco_loss': reco_loss, 'l1_prior': l1_prior}


class ODEstimator:
    def __init__(self, adata: sc.AnnData, model_class=SteadyStateForcingModel, preprocess_mode="normalized_concentration", model_kwargs=None):
        self.adata = adata.copy()
        self.preprocess_mode = preprocess_mode
        self._prepare_data()

        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs["lambda_prior"] = model_kwargs.get("lambda_prior", 1e0)
        
        self.model = model_class(
            n_genes=self.n_genes,
            n_tfs=self.n_tfs,
            tf2gene_indicators=self.tf2gene_indicators,
            **model_kwargs
        )
        self.state = None

    def _prepare_data(self):
        self.adata.X = self.adata.layers["counts"].copy()
        sc.pp.filter_genes(self.adata, min_cells=10)
        sc.pp.normalize_total(self.adata, target_sum=1)
        self.X = self.adata.X.toarray()
        if self.preprocess_mode == "normalized_concentration":
            self.X = self.X / self.X.max(0)
        elif self.preprocess_mode == "concentration":
            pass
        else:
            raise ValueError("Invalid preprocess mode")

        self.tf_list = self.adata.obs["consensus_target"].unique()
        self.tf_list = [t for t in self.tf_list if t in self.adata.var_names]
        self.n_tfs = len(self.tf_list)
        self.n_genes = self.X.shape[1]

        tf2gene_indicators = np.zeros((self.n_genes, self.n_tfs))
        for tf_idx, tf_knockdown in enumerate(self.tf_list):
            gene_idx = self.adata.var_names.get_loc(tf_knockdown)
            tf2gene_indicators[gene_idx, tf_idx] = 1
        self.tf2gene_indicators = jnp.array(tf2gene_indicators)

        U = []
        for cell_idx in range(self.X.shape[0]):
            tf_knockdown = self.adata.obs["consensus_target"].iloc[cell_idx]
            if tf_knockdown in self.tf_list:
                res_ = np.zeros((self.n_tfs,))
                tf_idx = self.tf_list.index(tf_knockdown)
                res_[tf_idx] = 1
                U.append(res_)
            else:
                U.append(np.zeros((self.n_tfs,)))
        self.U = jnp.array(U)

    def fit(self, learning_rate=1e-3, n_iter=5000, early_stopping_patience=20, early_stopping_metric="loss"):
        x_ = jnp.array(self.X)
        u_ = jnp.array(self.U)

        @jax.jit
        def train_step(state):
            def loss_fn(params):
                variables = {'params': params}
                loss_dict = state.apply_fn(variables, x_, u_)
                return loss_dict['loss'], loss_dict
            (loss_val, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)    
            return state, loss_dict

        dummy_x = jnp.zeros((4, self.n_genes))
        dummy_u = jnp.zeros((4, self.n_tfs))
        params = self.model.init(jax.random.PRNGKey(0), dummy_x, dummy_u)['params']

        optimizer = optax.adam(learning_rate=learning_rate)
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer
        )

        loss_history = []
        pbar = tqdm(range(n_iter))
        hasnt_improved_counter = 0
        best_loss = 1e10

        for i in pbar:
            self.state, loss_dict = train_step(self.state)
            loss_history.append(loss_dict)
            pbar.set_postfix(loss=f'{loss_dict["loss"].item()}')
            if loss_dict[early_stopping_metric] < best_loss:
                best_loss = loss_dict[early_stopping_metric]
                hasnt_improved_counter = 0
            else:
                hasnt_improved_counter += 1
            if hasnt_improved_counter > early_stopping_patience:
                break
        
        return pd.DataFrame(loss_history).astype(float)

    def get_interaction_matrix(self, return_square=True, delta=None):
        """Extract the learned gene-gene interaction matrix from the trained model.
        
        This method retrieves the interaction matrix (Amat) that represents regulatory
        relationships between genes. The matrix can be returned in square form or as a
        long-format DataFrame with additional filtering options.
        
        Parameters
        ----------
        return_square : bool, default=True
            If True, returns a square DataFrame with genes as both rows and columns.
            If False, returns a long-format DataFrame with one row per gene pair.
        delta : float, optional
            Threshold for filtering interactions by absolute score. Only used when
            return_square=False. If provided, only interactions with score > delta
            are returned.
        
        Returns
        -------
        pd.DataFrame
            If return_square=True:
                Square DataFrame with shape (n_genes, n_genes), where rows and columns
                are indexed by gene names from adata.var_names. Values represent the
                signed interaction strength.
            If return_square=False:
                Long-format DataFrame with columns:
                - 'target_gene': lowercase target gene name
                - 'regulator_gene': lowercase regulator gene name
                - 'signed_score': signed interaction strength
                - 'score': absolute interaction strength
        
        Raises
        ------
        RuntimeError
            If the model has not been trained yet (state is None). Call .fit() first.
        
        Examples
        --------
        >>> # Get square interaction matrix
        >>> Amat = model.get_interaction_matrix(return_square=True)
        
        >>> # Get filtered long-format interactions
        >>> interactions = model.get_interaction_matrix(return_square=False, delta=0.1)
        """
        if self.state is None:
            raise RuntimeError("Model has not been trained yet. Please call .fit() first.")
        
        processed_Amat = self.model.apply({'params': self.state.params}, method=self.model.get_Amat)
        Amat_ = pd.DataFrame(processed_Amat, index=self.adata.var_names, columns=self.adata.var_names)
        if return_square:
            return Amat_

        Amat_unstack = (
            Amat_
            .unstack()
            .to_frame("signed_score")
            .reset_index()
            # .rename(columns={"level_0": "target_gene", "level_1": "regulator_gene"})
            .rename(columns={"level_0": "regulator_gene", "level_1": "target_gene"})
            .assign(
                target_gene_=lambda x: x["target_gene"],
                regulator_gene_=lambda x: x["regulator_gene"],
                target_gene=lambda x: x["target_gene"].str.lower(),
                regulator_gene=lambda x: x["regulator_gene"].str.lower(),
                score=lambda x: np.abs(x["signed_score"].values),
            )
        )
        if delta is not None:
            Amat_unstack.loc[:, "decision"] = Amat_unstack["score"] > delta
        return Amat_unstack

