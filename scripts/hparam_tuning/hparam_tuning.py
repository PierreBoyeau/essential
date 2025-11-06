from essential.gpu_utils import select_best_gpus

select_best_gpus(n_gpus=1)

import numpy as np
import scanpy as sc
import optuna
from essential.ode import ODEstimator
from essential.utils import evaluate_interactions_on_regulondb, load_regulondb_full
from essential.configs.base import get_config
import optax

# trial_name = "hparam_tuning7"
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
ODEstimator.process_data(adata_, latent_obsm_key=config.processing.latent_obsm_key)


def objective(trial):
    """
    Objective function for Optuna to optimize.
    A trial is a single run of the model with a specific set of hyperparameters.
    """
    # Define the hyperparameter search space
    lambda_prior = trial.suggest_float("lambda_prior", 1e-7, 1e0, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    mode = trial.suggest_categorical("mode", ["dynamic", "steady"])
    # n_epochs = trial.suggest_int("n_epochs", 10, 500)
    n_epochs = 100
    model_class = trial.suggest_categorical(
        "model_class",
        [
            "linear",
            "linearhardko",
            "linearhardkozeroorder",
            "linearzeroorder",
            "linearhardmultiplicative",
            "linearmultiplicative",
            "sigmoidhardko",
            "sigmoid2",
        ],
    )

    # Create a copy of config and override with trial hyperparameters
    trial_config = config.copy_and_resolve_references()
    trial_config.estimator.model_kwargs.lambda_prior = lambda_prior
    trial_config.estimator.model_class = model_class
    trial_config.estimator.model_kwargs.mode = mode
    trial_config.training.learning_rate = learning_rate
    trial_config.training.n_epochs = n_epochs

    ode_model = ODEstimator(adata_, **trial_config.estimator.to_dict())
    if trial_config.do_lr_optimization:
        best_params = ode_model.find_best_lr(**trial_config.lr_optimization_params.to_dict())
        optimizer = optax.sgd(**best_params["best_params"])
        trial_config.training.optimizer = optimizer
    ode_model.fit(**trial_config.training.to_dict())

    ref_db = load_regulondb_full(drop_duplicates=True)
    v1 = (
        ode_model.get_results(delta=0.0, ref_db=ref_db, transpose_amat=True)
        .sort_values("score", ascending=False)
        .head(3000)["is_tp"]
        .sum()
    )
    v2 = (
        ode_model.get_results(delta=0.0, ref_db=ref_db, transpose_amat=False)
        .sort_values("score", ascending=False)
        .head(3000)["is_tp"]
        .sum()
    )
    return np.maximum(v1, v2)


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_params}")


study = optuna.create_study(
    direction="maximize",
    storage=f"sqlite:///{trial_name}.db",
    study_name=f"{trial_name}",
    load_if_exists=True,
)
study.optimize(
    objective,
    n_trials=50,
    callbacks=[print_callback],
)

print("Best trial:")
trial = study.best_trial

print(f"  Value (maximal PRAUC): {trial.value}")
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
