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
