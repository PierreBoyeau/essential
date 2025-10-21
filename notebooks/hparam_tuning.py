import scanpy as sc
import optuna
from essential.ode import ODEstimator
from essential.utils import evaluate_interactions_on_regulondb


def objective(trial):
    """
    Objective function for Optuna to optimize.
    A trial is a single run of the model with a specific set of hyperparameters.
    """
    # 1. Define the hyperparameter search space
    lambda_prior = trial.suggest_float("lambda_prior", 1e-7, 1e0, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
    n_epochs = trial.suggest_int("n_epochs", 10, 1000)
    model_class = trial.suggest_categorical(
        "model_class",
        [
            "dynamic_cellbox",
            "dynamic_hardmultiplicative",
            "dynamic_multiplicative",
        ],
    )

    ode_model = ODEstimator(
        adata_,
        expression_type="concentration",
        model_kwargs={"lambda_prior": lambda_prior},
        model_class=model_class,
        subset_treated=True,
    )
    ode_model.fit(
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        # early_stopping_patience=3,
        # early_stopping_metric="loss",
        # log_every_n_steps=1,
        batch_size=8000,
    )

    df_ode = ode_model.get_interaction_matrix(return_square=False)
    metrics = evaluate_interactions_on_regulondb(df_ode)
    prauc = metrics["pr_auc"]
    return prauc


# Load and preprocess data
adata = sc.read_h5ad("/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad")
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata, target_sum=1)
adata.layers["concentration"] = adata.X.copy()
adata.X = adata.layers["counts"].copy()

adata_ = adata
sc.pp.filter_genes(adata_, min_cells=10)


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_params}")


study = optuna.create_study(
    direction="maximize",
    storage="sqlite:///hparam_tuning.db",
    study_name="hparam_tuning",
    load_if_exists=True,
)
study.optimize(
    objective,
    n_trials=10,
    callbacks=[print_callback],
)

print("Best trial:")
trial = study.best_trial

print(f"  Value (maximal PRAUC): {trial.value}")
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
