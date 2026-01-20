from essential.gpu_utils import select_best_gpus

select_best_gpus(n_gpus=1)

import optuna
import json
import numpy as np
import os
import sys
import argparse
from functools import partial

# Local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from regression_predictor import RegressionPredictor
from run_experiment import prepare_data

HIDDEN_DIMS_CHOICES = [64, 128, 256, 512, 1024, 2048]
N_LAYERS_CHOICES = [1, 2, 3, 4]


def objective_g(trial, XW_l, Y_l, XW_u, Y_u, seed=42):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    width = trial.suggest_categorical("width", HIDDEN_DIMS_CHOICES)
    hidden_dims = (width,) * n_layers

    # g_reg maps XW -> Y
    model = RegressionPredictor(XW_l, Y_l, hidden_dims=hidden_dims, seed=seed, patience=10)
    # Using fewer epochs for faster tuning, but enough to converge reasonably
    model.fit(max_epochs=1000)

    # Predict on unlabeled (held-out)
    Y_pred_u = model.predict(XW_u)
    mse = np.mean((Y_pred_u - Y_u) ** 2)
    return mse


def objective_f(trial, X_all, Yhat_all, X_u, Y_u, seed=42):
    n_layers = trial.suggest_int("n_layers", 1, 4)
    width = trial.suggest_categorical("width", HIDDEN_DIMS_CHOICES)
    hidden_dims = (width,) * n_layers

    # f_reg maps X -> Y (using pseudo-labels for training)
    model = RegressionPredictor(X_all, Yhat_all, hidden_dims=hidden_dims, seed=seed, patience=10)
    model.fit(max_epochs=200)

    # Predict on unlabeled (held-out) and compare to ground truth
    Y_pred_u = model.predict(X_u)
    mse = np.mean((Y_pred_u - Y_u) ** 2)
    return mse


def run_tuning(n_labeled=500, n_trials=20, use_pca=False, fitness_cols="all"):
    results = {}

    config_name = "pca" if use_pca else "expression"
    print(f"\n--- Tuning for config: {config_name}, fitness_cols: {fitness_cols} ---")

    # Load data
    Y, X, W, Y0, Y_pca, Y0_pca = prepare_data(fitness_cols=fitness_cols)

    if use_pca:
        Y_target = Y_pca
        Y0_target = Y0_pca
    else:
        Y_target = Y
        Y0_target = Y0

    # Split data (using fixed seed for reproducibility of split)
    np.random.seed(42)
    l_indices = np.random.choice(range(len(Y_target)), size=n_labeled, replace=False)
    X_l, Y_l, W_l = X[l_indices], Y_target[l_indices], W[l_indices]
    XW_l = np.concatenate([X_l, W_l], axis=1)

    u_indices = np.setdiff1d(range(len(Y_target)), l_indices)
    X_u, Y_u, W_u = X[u_indices], Y_target[u_indices], W[u_indices]
    XW_u = np.concatenate([X_u, W_u], axis=1)

    # Optimize g_reg
    print("Optimizing g_reg...")
    study_g = optuna.create_study(direction="minimize")
    study_g.optimize(
        partial(objective_g, XW_l=XW_l, Y_l=Y_l, XW_u=XW_u, Y_u=Y_u), n_trials=n_trials
    )

    best_params_g = study_g.best_params
    best_hidden_dims_g = [best_params_g["width"]] * best_params_g["n_layers"]
    print(f"Best g_reg params: {best_params_g} -> {best_hidden_dims_g}")

    # Train best g_reg to get pseudo-labels
    g_reg_best = RegressionPredictor(XW_l, Y_l, hidden_dims=best_hidden_dims_g, seed=42)
    g_reg_best.fit(max_epochs=250)

    # Prepare data for f_reg
    X_all = np.concatenate([X_l, X_u], axis=0)
    Yhat_u = g_reg_best.predict(XW_u)
    Yhat_all = np.concatenate([Y_l, Yhat_u], axis=0)

    # Optimize f_reg
    print("Optimizing f_reg...")
    study_f = optuna.create_study(direction="minimize")
    study_f.optimize(
        partial(objective_f, X_all=X_all, Yhat_all=Yhat_all, X_u=X_u, Y_u=Y_u),
        n_trials=n_trials,
    )

    best_params_f = study_f.best_params
    best_hidden_dims_f = [best_params_f["width"]] * best_params_f["n_layers"]
    print(f"Best f_reg params: {best_params_f} -> {best_hidden_dims_f}")

    results[config_name] = {
        "g_reg": {"hidden_dims": best_hidden_dims_g},
        "f_reg": {"hidden_dims": best_hidden_dims_f},
    }

    # Save results
    output_filename = f"hyperparameters_{config_name}_{fitness_cols}.json"
    output_path = os.path.join(current_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved hyperparameters to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for Optuna")
    parser.add_argument(
        "--use_pca",
        action="store_true",
        help="Use PCA components for Y instead of raw transcriptomics.",
    )
    parser.add_argument(
        "--fitness_cols",
        type=str,
        default="all",
        help="Fitness columns to use: 'all' or specific column name (e.g. 'T1').",
    )
    args = parser.parse_args()

    run_tuning(
        n_labeled=500,
        n_trials=args.n_trials,
        use_pca=args.use_pca,
        fitness_cols=args.fitness_cols,
    )
