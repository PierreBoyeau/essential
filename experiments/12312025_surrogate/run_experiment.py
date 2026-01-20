import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
import time
import argparse
import json
import hashlib

# Local imports
import sys
import os

# Add current directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from regression_predictor import RegressionPredictor
from data_resources import (
    load_fitness_data,
    load_crispri_data,
    load_llm_embeddings,
)

FITNESS_COLS_ALL = ["T1", "T2", "T3", "T4"]
HIDDEN_DIMS = (128,)
LEVEL = "gene"  # or "spacer"


def prepare_data(fitness_cols="all"):
    """
    Loads and prepares the data for the experiment.

    Args:
        fitness_cols: "all" to use all fitness columns, or a specific column name (e.g., "T1").

    Returns:
        tuple: (Y, X, W, Y0, Y_pca, Y0_pca)
            Y: Transcriptomic profiles (mean per perturbation)
            X: Gene embeddings (LLM)
            W: Fitness data
            Y0: Control mean transcriptomics
            Y_pca: PCA reduced transcriptomics
            Y0_pca: Control mean PCA transcriptomics
    """
    if fitness_cols == "all":
        selected_fitness_cols = FITNESS_COLS_ALL
    else:
        if fitness_cols not in FITNESS_COLS_ALL:
            raise ValueError(
                f"Invalid fitness col: {fitness_cols}. Must be one of {FITNESS_COLS_ALL} or 'all'"
            )
        selected_fitness_cols = [fitness_cols]

    print("Loading data...")
    fitness_df = load_fitness_data()
    adata = load_crispri_data()

    # Merge fitness data
    adata.obs = adata.obs.merge(fitness_df, how="left", left_on="spacer", right_index=True)
    adata.obs["spacer_has_fitness_data"] = adata.obs["spacer"].isin(fitness_df.index)
    adata.obs["spacer_is_control"] = (
        adata.obs["target"] == "nontargeting"
    ) | adata.obs.gene.str.startswith("Control")
    adata.obs["spacer_is_valid"] = (
        adata.obs["spacer_has_fitness_data"] | adata.obs["spacer_is_control"]
    )
    adata = adata[adata.obs["spacer_is_valid"]].copy()

    # Transcripts processing
    print("Processing transcripts...")
    adata.obs[LEVEL] = adata.obs[LEVEL].astype(str)
    perturbation_names = adata.obs[LEVEL].unique()

    Y = []
    for perturbation_name in tqdm(perturbation_names, desc="Extracting Y"):
        adata_sub = adata[adata.obs[LEVEL] == perturbation_name].copy()
        Y_mean = np.asarray(adata_sub.X.mean(axis=0)).flatten()
        Y.append(Y_mean)
    Y = np.array(Y)

    # PCA on Y
    pca_ = PCA(n_components=100)
    Y_pca = pca_.fit_transform(Y)

    control_names = [
        perturbation_name
        for perturbation_name in perturbation_names
        if "Control" in perturbation_name
    ]
    is_control = np.isin(perturbation_names, control_names)
    Y0 = Y[is_control].mean(0)
    Y0_pca = Y_pca[is_control].mean(0)

    # Gene embeddings (X)
    print("Processing gene embeddings (X)...")
    llm_df = load_llm_embeddings()
    default_embedding = llm_df.mean(0)
    X = []
    n_missing_x = 0

    for perturbation_name in tqdm(perturbation_names, desc="Extracting X"):
        if perturbation_name in llm_df.index:
            X.append(llm_df.loc[perturbation_name])
        else:
            # print(perturbation_name)
            n_missing_x += 1
            X.append(default_embedding)
    X = np.array(X)
    print(f"n_missing embeddings: {n_missing_x}")

    # Fitness data (W)
    print(f"Processing fitness data (W) using cols: {selected_fitness_cols}...")
    # Pre-calculate gene-level fitness to speed up extraction
    gene_fitness = fitness_df.groupby("gene")[selected_fitness_cols].mean()
    W = []
    W_default = fitness_df[selected_fitness_cols].mean(0).values
    n_missing_w = 0

    for perturbation_name in tqdm(perturbation_names, desc="Extracting W"):
        if perturbation_name in gene_fitness.index:
            W.append(gene_fitness.loc[perturbation_name].values)
        else:
            n_missing_w += 1
            # print(perturbation_name)
            W.append(W_default)
    W = np.array(W)
    print(f"n_missing fitness: {n_missing_w}")

    return Y, X, W, Y0, Y_pca, Y0_pca


def experiment(
    Y,
    X,
    W,
    Y0,
    n_labeled=500,
    seed=42,
    model_type="ridge",
    g_reg_params=None,
    f_reg_params=None,
    **model_kwargs,
):
    """
    Runs a single experiment comparing approaches.
    """
    if g_reg_params is None:
        g_reg_params = {}
    if f_reg_params is None:
        f_reg_params = {}

    def get_hidden_dims(params):
        return tuple(params.get("hidden_dims", HIDDEN_DIMS))

    np.random.seed(seed)
    l_indices = np.random.choice(range(len(Y)), size=n_labeled, replace=False)
    X_l, Y_l, W_l = X[l_indices], Y[l_indices], W[l_indices]
    XW_l = np.concatenate([X_l, W_l], axis=1)

    u_indices = np.setdiff1d(range(len(Y)), l_indices)
    X_u, Y_u, W_u = X[u_indices], Y[u_indices], W[u_indices]
    XW_u = np.concatenate([X_u, W_u], axis=1)

    # train surrogate model
    if model_type == "ridge":
        g_reg = RidgeCV()
        g_reg.fit(XW_l, Y_l)
    elif model_type == "mlp":
        g_reg = RegressionPredictor(
            XW_l, Y_l, hidden_dims=get_hidden_dims(g_reg_params), **model_kwargs
        )
        g_reg.fit()
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    X_all = np.concatenate([X_l, X_u], axis=0)
    Yhat_all = np.concatenate([Y_l, g_reg.predict(XW_u)], axis=0)
    # f_reg = RegressionPredictor(X_all, Yhat_all, hidden_dims=HIDDEN_DIMS, **model_kwargs)
    # f_reg.fit()

    if model_type == "ridge":
        f_reg = RidgeCV()
        f_reg.fit(X_all, Yhat_all)
    elif model_type == "mlp":
        f_reg = RegressionPredictor(
            X_all, Yhat_all, hidden_dims=get_hidden_dims(f_reg_params), **model_kwargs
        )
        f_reg.fit()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    Ypred_u = f_reg.predict(X_u)
    err = np.sqrt(np.mean((Y_u - Ypred_u) ** 2))
    delta_pred = Ypred_u - Y0
    delta_gt = Y_u - Y0
    corrs = stats.pearsonr(delta_pred, delta_gt, axis=0).statistic.mean()

    # naive training baseline
    # f_naive = RegressionPredictor(X_l, Y_l, hidden_dims=HIDDEN_DIMS, **model_kwargs)
    # f_naive.fit()

    if model_type == "ridge":
        f_naive = RidgeCV()
        f_naive.fit(X_l, Y_l)
    elif model_type == "mlp":
        f_naive = RegressionPredictor(
            X_l, Y_l, hidden_dims=get_hidden_dims(f_reg_params), **model_kwargs
        )
        f_naive.fit()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    Ypred_u_naive = f_naive.predict(X_u)
    err_naive = np.sqrt(np.mean((Y_u - Ypred_u_naive) ** 2))
    delta_pred_naive = Ypred_u_naive - Y0
    corrs_naive = stats.pearsonr(delta_pred_naive, delta_gt, axis=0).statistic.mean()

    # naive average
    Ypred_u_naive_avg = Y_l.mean(0)
    err_naive_avg = np.sqrt(np.mean((Y_u - Ypred_u_naive_avg) ** 2))
    corrs_naive_avg = 0.0

    res = pd.DataFrame(
        {
            "err": [err, err_naive, err_naive_avg],  # overall error
            "corr": [corrs, corrs_naive, corrs_naive_avg],  # avg pearson R over perturbations
            "approach": ["PAST", "naive", "naive_avg"],
        }
    ).assign(n_labeled=n_labeled, seed=seed)
    return res


def main():
    parser = argparse.ArgumentParser(description="Run surrogate experiment.")
    parser.add_argument(
        "--n_labeled",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000],
        help="List of labeled data sizes.",
    )
    parser.add_argument("--seed", type=int, nargs="+", default=range(10), help="List of seeds.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="ridge",
        choices=["ridge", "mlp"],
        help="Model type to use (ridge or mlp).",
    )
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

    # Load hyperparameters
    config_key = "pca" if args.use_pca else "expression"
    hyperparams_filename = f"hyperparameters_{config_key}_{args.fitness_cols}.json"
    hyperparams_path = os.path.join(current_dir, hyperparams_filename)
    g_reg_params = {}
    f_reg_params = {}

    # if os.path.exists(hyperparams_path) and args.model_type == "mlp":
    #     try:
    #         with open(hyperparams_path, "r") as f:
    #             all_params = json.load(f)
    #         if config_key in all_params:
    #             print(f"Loading hyperparameters for {config_key} config from {hyperparams_path}.")
    #             g_reg_params = all_params[config_key].get("g_reg", {})
    #             f_reg_params = all_params[config_key].get("f_reg", {})
    #         else:
    #             print(
    #                 f"Warning: Config {config_key} not found in {hyperparams_path}. Using defaults."
    #             )
    #     except Exception as e:
    #         print(f"Error loading hyperparameters: {e}. Using defaults.")
    # elif args.model_type == "mlp":
    #     print(f"Hyperparameters file not found: {hyperparams_path}. Using defaults.")

    Y, X, W, Y0, Y_pca, Y0_pca = prepare_data(fitness_cols=args.fitness_cols)

    if args.use_pca:
        print("Using PCA components for Y.")
        Y_target = Y_pca
        Y0_target = Y0_pca
    else:
        print("Using raw transcriptomics for Y.")
        Y_target = Y
        Y0_target = Y0

    res_all = pd.DataFrame()
    labeled_sizes = args.n_labeled
    seeds = args.seed

    print("Starting experiment loop...")
    for n_labeled in labeled_sizes:
        print(f"n_labeled: {n_labeled}")
        for seed in seeds:
            res = experiment(
                Y_target,
                X,
                W,
                Y0=Y0_target,
                n_labeled=n_labeled,
                seed=seed,
                model_type=args.model_type,
                g_reg_params=g_reg_params,
                f_reg_params=f_reg_params,
            )
            res_all = pd.concat([res_all, res])

    print(res_all)

    args_dict = vars(args)
    processed_args = {
        f"arg_{k}": (str(v) if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)) else v)
        for k, v in args_dict.items()
    }
    res_all = res_all.assign(**processed_args)

    args_json = json.dumps(args_dict, sort_keys=True)
    args_hash = hashlib.sha256(args_json.encode()).hexdigest()
    save_dir = os.path.join(current_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, f"results_{args_hash}.csv")
    res_all.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
