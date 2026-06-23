from essential.gpu_utils import select_best_gpus

select_best_gpus(n_gpus=1)

import numpy as np
import optuna
import scanpy as sc

from essential.configs.base import get_config
from essential.ode import ODEstimator
from essential.utils import evaluate_interactions_on_regulondb

# Load base configuration
config = get_config()
adata = sc.read_h5ad(config.processing.adata_path)
if config.processing.rt_bc != "all":
    adata = adata[adata.obs["rt_bc"] == config.processing.rt_bc].copy()
if config.processing.consolidated_cluster != "all":
    adata = adata[
        adata.obs["consolidated_cluster"] == config.processing.consolidated_cluster
    ].copy()

adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata, target_sum=1)
adata.layers["concentration"] = adata.X.copy()
adata.X = adata.layers["counts"].copy()

adata_ = adata
sc.pp.filter_genes(adata_, min_cells=10)


def objective(trial):
    """
    Objective function for Optuna to optimize.
    A trial is a single run of the model with a specific set of hyperparameters.
    """
    # Define the hyperparameter search space
    lambda_prior = 0.0
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    n_latent = trial.suggest_int("n_latent", 32, 256)
    n_epochs = trial.suggest_int("n_epochs", 1000, 5000)
    model_class = "dynamic_cellboxlowdim2"

    trial_config = config.copy_and_resolve_references()
    trial_config.estimator.model_kwargs.lambda_prior = lambda_prior
    trial_config.estimator.model_kwargs.n_latent = n_latent
    trial_config.estimator.model_class = model_class
    trial_config.training.learning_rate = learning_rate
    trial_config.training.n_epochs = n_epochs

    ode_model = ODEstimator(adata_, **trial_config.estimator.to_dict())
    ode_model.fit(**trial_config.training.to_dict())

    df_ode = ode_model.get_interaction_matrix(return_square=False)
    metrics = evaluate_interactions_on_regulondb(df_ode)
    prauc = metrics["pr_auc"]
    reversed_prauc = metrics["reversed_pr_auc"]
    return np.maximum(prauc, reversed_prauc)


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_params}")


study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///hparam_tuning_lowdim2.db",
    study_name="hparam_tuning_lowdim2",
    load_if_exists=True,
)
study.optimize(
    objective,
    n_trials=25,
    callbacks=[print_callback],
)

print("Best trial:")
trial = study.best_trial

print(f"  Value (maximal PRAUC): {trial.value}")
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
