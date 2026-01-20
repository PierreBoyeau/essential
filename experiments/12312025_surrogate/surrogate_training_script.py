import scanpy as sc
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import plotnine as gg
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import scipy.stats as stats


fitness_df = (
    pd.read_csv("/workspace/data/calvo2020_dcas9fitness/Supp_data2_log2FC.csv")
    .rename(columns={"Unnamed: 0": "spacer"})
    .set_index("spacer")
)
display(fitness_df.head())
FITNESS_COLS = ["T1", "T2", "T3", "T4"]


adata = sc.read_h5ad(
    "/workspace/data/251117_genomescale_CRISPRi/sample_mix_umi200_hvg500_pc25_neighbors10_mindist0.55.h5ad"
)
adata.X = adata.layers["reads"].copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

adata.obs = adata.obs.merge(fitness_df, how="left", left_on="spacer", right_index=True)
adata.obs["spacer_has_fitness_data"] = adata.obs["spacer"].isin(fitness_df.index)
adata.obs["spacer_is_control"] = (
    adata.obs["target"] == "nontargeting"
) | adata.obs.gene.str.startswith("Control")
adata.obs["spacer_is_valid"] = adata.obs["spacer_has_fitness_data"] | adata.obs["spacer_is_control"]
adata = adata[adata.obs["spacer_is_valid"]].copy()

LEVEL = "gene"
# LEVEL = "spacer"

# transcripts
adata.obs[LEVEL] = adata.obs[LEVEL].astype(str)
perturbation_names = adata.obs[LEVEL].unique()

Y = []
for perturbation_name in tqdm(perturbation_names):
    adata_sub = adata[adata.obs[LEVEL] == perturbation_name].copy()
    Y_mean = adata_sub.X.mean(axis=0).A1
    Y.append(Y_mean)
Y = np.array(Y)

pca_ = PCA(n_components=100)
Y_pca = pca_.fit_transform(Y)
control_names = [
    perturbation_name for perturbation_name in perturbation_names if "Control" in perturbation_name
]
is_control = np.isin(perturbation_names, control_names)
Y0 = Y[is_control].mean(0)
Y0_pca = Y_pca[is_control].mean(0)

# gene embeddings
print("processing gene embeddings")
embedding_dir = "/workspace/data/e_coli_llm_embeddings"
llm_embeddings = np.load(os.path.join(embedding_dir, "llm_embeddings.npz"))
gene_order = pd.Series(np.arange(len(llm_embeddings["genes"])), index=llm_embeddings["genes"])
llm_df = pd.DataFrame(llm_embeddings["embeddings"], index=llm_embeddings["genes"])
default_embedding = llm_df.mean(0)
X = []
n_missing = 0
for perturbation_name in tqdm(perturbation_names):
    if perturbation_name in llm_df.index:
        X.append(llm_df.loc[perturbation_name])
    else:
        print(perturbation_name)
        n_missing += 1
        X.append(default_embedding)
X = np.array(X)
print(f"n_missing: {n_missing}")

# fitness data
print("processing fitness data")
W = []
W_default = fitness_df[FITNESS_COLS].mean(0).values
n_missing = 0
for perturbation_name in tqdm(perturbation_names):
    if perturbation_name in fitness_df["gene"].values:
        W.append(
            fitness_df.loc[lambda x: x["gene"] == perturbation_name, FITNESS_COLS].mean(0).values
        )
    else:
        n_missing += 1
        print(perturbation_name)
        W.append(W_default)
W = np.array(W)
print(f"n_missing: {n_missing}")


import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Sequence


class ResNetBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        x = nn.Dense(features=self.features)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        if x.shape != residual.shape:
            residual = nn.Dense(features=self.features)(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        x = x + residual
        x = nn.relu(x)
        return x


class MLP(nn.Module):
    """A ResNet model for regression."""

    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        for dim in self.hidden_dims:
            x = ResNetBlock(features=dim)(x, train=train)
        x = nn.Dense(features=self.output_dim)(x)
        return x


class RegressionPredictor:
    """
    A class to train and evaluate a regression model (X -> Y) using an MLP.
    Supports multidimensional targets.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        validation_split: float = 0.2,
        hidden_dims: Sequence[int] = (4096, 2048),
        learning_rate: float = 1e-3,
        batch_size: int = 1024,
        patience: int = 20,
        seed: int = 42,
    ):
        """
        Initializes the RegressionPredictor.

        Args:
            X: Input features (N, D).
            Y: Target values (N, K) or (N,).
            validation_split: Fraction of data to use for validation.
            hidden_dims: Sequence of hidden dimensions.
            learning_rate: The learning rate for the Adam optimizer.
            batch_size: The batch size for training.
            patience: Early stopping patience (epochs).
            seed: Random seed.
        """
        self.X = X
        self.Y = Y

        # Ensure Y is 2D (N, K)
        if self.Y.ndim == 1:
            self.output_dim = 1
            self.Y = self.Y[:, None]
        else:
            self.output_dim = self.Y.shape[1]

        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

        # Split data
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X, self.Y, test_size=validation_split, random_state=seed
        )

        self.model = MLP(hidden_dims=self.hidden_dims, output_dim=self.output_dim)
        self.tx = optax.adam(self.learning_rate)

        self.params = None
        self.batch_stats = None
        self.opt_state = None

        self.history: List[Dict[str, Any]] = []
        self.best_params = None
        self.best_batch_stats = None

    def _init_model_state(self):
        """Initializes model parameters, batch stats, and optimizer state."""
        dummy_x = self.X_train[:1]

        variables = self.model.init(jax.random.PRNGKey(self.seed), dummy_x, train=False)
        self.params = variables["params"]
        self.batch_stats = variables["batch_stats"]
        self.opt_state = self.tx.init(self.params)

    def fit(self, max_epochs: int = 250):
        """
        Trains the model with early stopping.

        Args:
            max_epochs: The maximum number of epochs to train for.
        """
        if self.params is None:
            self._init_model_state()

        @jax.jit
        def train_step(params, batch_stats, opt_state, x, y):
            def loss_fn(params):
                (y_pred, updates) = self.model.apply(
                    {"params": params, "batch_stats": batch_stats},
                    x,
                    train=True,
                    mutable=["batch_stats"],
                )
                loss = jnp.mean((y_pred - y) ** 2)
                return loss, updates["batch_stats"]

            (loss_val, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, new_opt_state = self.tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_batch_stats, new_opt_state, loss_val

        @jax.jit
        def eval_step(params, batch_stats, x, y):
            y_pred = self.model.apply(
                {"params": params, "batch_stats": batch_stats},
                x,
                train=False,
            )
            return jnp.mean((y_pred - y) ** 2)

        best_val_loss = float("inf")
        patience_counter = 0
        n_samples = self.X_train.shape[0]
        steps_per_epoch = int(np.ceil(n_samples / self.batch_size))

        for epoch in range(max_epochs):
            # Shuffle training data
            perm = np.random.permutation(n_samples)
            X_shuffled = self.X_train[perm]
            Y_shuffled = self.Y_train[perm]

            train_losses = []

            for i in range(steps_per_epoch):
                start = i * self.batch_size
                end = start + self.batch_size
                x_batch = X_shuffled[start:end]
                y_batch = Y_shuffled[start:end]

                if x_batch.shape[0] == 0:
                    continue

                self.params, self.batch_stats, self.opt_state, loss = train_step(
                    self.params, self.batch_stats, self.opt_state, x_batch, y_batch
                )
                train_losses.append(loss)

            mean_train_loss = np.mean(train_losses)

            # Validation
            val_loss = eval_step(self.params, self.batch_stats, self.X_val, self.Y_val)

            self.history.append(
                {"epoch": epoch, "train_loss": float(mean_train_loss), "val_loss": float(val_loss)}
            )

            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_params = self.params
                self.best_batch_stats = self.batch_stats
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {mean_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Restore best weights
        if self.best_params is not None:
            self.params = self.best_params
            self.batch_stats = self.best_batch_stats

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions on new data."""
        if self.params is None:
            raise RuntimeError("Model has not been trained.")

        y_pred = self.model.apply(
            {"params": self.params, "batch_stats": self.batch_stats},
            X,
            train=False,
        )
        return np.array(y_pred)

    def get_history(self) -> pd.DataFrame:
        """Returns the training history as a DataFrame."""
        return pd.DataFrame(self.history)


from sklearn.linear_model import LinearRegression, RidgeCV


HIDDEN_DIMS = (128,)


def experiment(Y, X, W, Y0, n_labeled=500, seed=42, **model_kwargs):
    np.random.seed(seed)
    l_indices = np.random.choice(range(len(Y)), size=n_labeled, replace=False)
    X_l, Y_l, W_l = X[l_indices], Y[l_indices], W[l_indices]
    XW_l = np.concatenate([X_l, W_l], axis=1)

    u_indices = np.setdiff1d(range(len(Y)), l_indices)
    X_u, Y_u, W_u = X[u_indices], Y[u_indices], W[u_indices]
    XW_u = np.concatenate([X_u, W_u], axis=1)

    # train surrogate model
    # g_reg = RegressionPredictor(XW_l, Y_l, hidden_dims=HIDDEN_DIMS, **model_kwargs)
    # g_reg.fit()

    # g_reg = LinearRegression()
    g_reg = RidgeCV()
    g_reg.fit(XW_l, Y_l)

    # prepare training data for target model
    X_all = np.concatenate([X_l, X_u], axis=0)
    Yhat_all = np.concatenate([Y_l, g_reg.predict(XW_u)], axis=0)
    # f_reg = RegressionPredictor(X_all, Yhat_all, hidden_dims=HIDDEN_DIMS, **model_kwargs)
    # f_reg.fit()

    # f_reg = LinearRegression()
    f_reg = RidgeCV()
    f_reg.fit(X_all, Yhat_all)
    Ypred_u = f_reg.predict(X_u)
    err = np.linalg.norm(Y_u - Ypred_u)
    delta_pred = Ypred_u - Y0
    delta_gt = Y_u - Y0
    corrs = stats.pearsonr(delta_pred, delta_gt, axis=0).statistic.mean()

    # naive training baseline
    # f_naive = RegressionPredictor(X_l, Y_l, hidden_dims=HIDDEN_DIMS, **model_kwargs)
    # f_naive.fit()
    # f_naive = LinearRegression()
    f_naive = RidgeCV()
    f_naive.fit(X_l, Y_l)
    Ypred_u_naive = f_naive.predict(X_u)
    err_naive = np.linalg.norm(Y_u - Ypred_u_naive)
    delta_pred_naive = Ypred_u_naive - Y0
    corrs_naive = stats.pearsonr(delta_pred_naive, delta_gt, axis=0).statistic.mean()

    # naive average
    Ypred_u_naive_avg = Y_l.mean(0)
    err_naive_avg = np.linalg.norm(Y_u - Ypred_u_naive_avg)
    corrs_naive_avg = 0.0

    res = pd.DataFrame(
        {
            "err": [err, err_naive, err_naive_avg],  # overall error
            "corr": [corrs, corrs_naive, corrs_naive_avg],  # avg pearson R over perturbations
            "approach": ["PAST", "naive", "naive_avg"],
        }
    ).assign(n_labeled=n_labeled, seed=seed)
    return res

    import time


res_all = pd.DataFrame()
for n_labeled in [100, 500, 1000, 2000]:
    for seed in range(10):
        start = time.time()
        res = experiment(Y, X, W, Y0=Y0, n_labeled=n_labeled, seed=seed)
        # res = experiment(Y_pca, X, W, Y0=Y0_pca, n_labeled=n_labeled, seed=seed)
        res_all = pd.concat([res_all, res])
        end = time.time()
        print(f"Time taken: {end - start:.2f} seconds")