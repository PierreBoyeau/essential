import numpy as np
import jax
import jax.numpy as jnp
from .models import MODEL_REGISTRY
import scanpy as sc
import pandas as pd


class ODESimulator:
    def __init__(
        self,
        n_genes,
        n_perturbed,
        sparsity=0.1,
        model_class="steady_state_forcing",
        random_seed=0,
        model_kwargs=None,
    ):
        self.n_genes = n_genes
        self.n_perturbed = n_perturbed
        self.sparsity = sparsity
        self.model_class = model_class
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        model_class_ = MODEL_REGISTRY[model_class]
        tf2gene_indicators = np.zeros((n_genes, n_perturbed))
        for i in range(n_perturbed):
            tf2gene_indicators[i, i] = 1

        np.random.seed(random_seed)

        Amat = np.random.randn(n_genes, n_genes)
        mask_ = np.random.rand(n_genes, n_genes) >= sparsity
        Amat[mask_] = 0.0
        Amat = Amat.astype(np.float32)

        self.model = model_class_(
            n_genes=n_genes,
            n_tfs=n_perturbed,
            tf2gene_indicators=tf2gene_indicators,
            **self.model_kwargs,
        )

        self.params = self.model.init_params(jax.random.PRNGKey(random_seed))
        self.params["Amat_"] = jnp.array(Amat)
        self.Amat = Amat

    def simulate_batch(self, x0: jnp.ndarray, u: jnp.ndarray, t: jnp.ndarray):
        xpred = self.model.apply({"params": self.params}, x0, u, t, method="simulate")
        return xpred

    def simulate(self, n_obs, t, batch_size=1024, fraction_control=0.1):
        if n_obs < batch_size:
            batch_size = n_obs
        n_batches = n_obs // batch_size
        fn_ = jax.jit(
            lambda x0, u, t: self.model.apply({"params": self.params}, x0, u, t, method="simulate")
        )

        x0_ = np.random.random((n_obs, self.n_genes))
        u_1d = np.random.randint(0, self.n_perturbed, (n_obs))
        u_ = np.eye(self.n_perturbed)[u_1d]
        t_ = t * np.ones((n_obs,))

        # Add control cells
        n_control = int(n_obs * fraction_control)
        control_indices = np.random.choice(n_obs, n_control, replace=False)
        u_[control_indices] = 0

        xpred_ = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            x0_batch = jnp.array(x0_[start:end])
            u_batch = jnp.array(u_[start:end])
            t_batch = jnp.array(t_[start:end])
            xpred_batch = fn_(x0_batch, u_batch, t_batch)
            xpred_batch = np.array(xpred_batch)
            xpred_.append(xpred_batch)

        xt = np.concatenate(xpred_, axis=0)
        var_names = ["gene_" + str(i) for i in range(self.n_genes)]
        perturbations_list = np.array(var_names)[u_1d].tolist()
        for i in control_indices:
            perturbations_list[i] = "nontargeting"
        perturbations = np.array(perturbations_list)

        adata_ = sc.AnnData(X=xt, var=pd.DataFrame(index=var_names))
        adata_.layers["counts"] = adata_.X.copy()
        adata_.obs["consensus_target"] = perturbations
        adata_.obs["rt_bc"] = "batch1"
        adata_.obsm["x0"] = x0_
        return adata_
