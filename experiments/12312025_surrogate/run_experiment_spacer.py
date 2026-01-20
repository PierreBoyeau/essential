from essential.gpu_utils import select_best_gpus

select_best_gpus(n_gpus=1)

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
import argparse
import json
import hashlib
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
    one_hot_encode,
)

FITNESS_COLS_ALL = ["T1", "T2", "T3", "T4"]
HIDDEN_DIMS = (128,)
LEVEL = "spacer"


def prepare_data(fitness_cols="all"):
    """
    Loads and prepares the data for the experiment at the spacer level.

    Args:
        fitness_cols: "all" to use all fitness columns, or a specific column name (e.g., "T1").

    Returns:
        tuple: (Y, X, W, Y0, Y_pca, Y0_pca, perturbation_to_gene_map)
            Y: Transcriptomic profiles (mean per perturbation)
            X: Gene embeddings (LLM) + One-hot spacer
            W: Fitness data
            Y0: Control mean transcriptomics
            Y_pca: PCA reduced transcriptomics
            Y0_pca: Control mean PCA transcriptomics
            perturbation_to_gene_map: Map from spacer name to gene name
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

    print("Processing transcripts...")
    adata.obs[LEVEL] = adata.obs[LEVEL].astype(str)
    perturbation_names = adata.obs[LEVEL].unique()

    perturbation_to_gene_map = {}
    for perturbation_name in perturbation_names:
        gene_name = adata.obs.loc[lambda x: x[LEVEL] == perturbation_name, "gene"].values[0]
        perturbation_to_gene_map[perturbation_name] = gene_name

    Y = []
    for perturbation_name in tqdm(perturbation_names, desc="Extracting Y"):
        adata_sub = adata[adata.obs[LEVEL] == perturbation_name].copy()
        Y_mean = np.asarray(adata_sub.X.mean(axis=0)).flatten()
        Y.append(Y_mean)
    Y = np.array(Y)

    # PCA on Y
    pca_ = PCA(n_components=100)
    Y_pca = pca_.fit_transform(Y)

    control_names = []
    for perturbation_name in perturbation_names:
        try:
            if "Control" in perturbation_to_gene_map[perturbation_name]:
                control_names.append(perturbation_name)
        except:
            pass
    is_control = np.isin(perturbation_names, control_names)
    Y0 = Y[is_control].mean(0)
    Y0_pca = Y_pca[is_control].mean(0)

    # Gene embeddings (X)
    print("Processing gene embeddings (X)...")
    gene_embeddings = load_llm_embeddings()
    default_embedding = gene_embeddings.mean(0)
    X = []
    n_missing_x = 0

    for perturbation_name in tqdm(perturbation_names, desc="Extracting X"):
        gene_name = perturbation_to_gene_map[perturbation_name]
        if gene_name in gene_embeddings.index:
            gene_emb = gene_embeddings.loc[gene_name]
        else:
            n_missing_x += 1
            gene_emb = default_embedding
        spacer_encoded = one_hot_encode(perturbation_name).flatten()
        X.append(np.concatenate([gene_emb, spacer_encoded]))
    X = np.array(X)
    # print(f"n_missing embeddings: {n_missing_x}")

    selected_fitness_cols = FITNESS_COLS_ALL if fitness_cols == "all" else [fitness_cols]
    spacer_fitness = fitness_df[selected_fitness_cols]
    W = []
    W_default = fitness_df[selected_fitness_cols].mean(0).values
    n_missing_w = 0

    for perturbation_name in tqdm(perturbation_names, desc="Extracting W"):
        if perturbation_name in spacer_fitness.index:
            W.append(spacer_fitness.loc[perturbation_name].values)
        else:
            n_missing_w += 1
            W.append(W_default)
    W = np.array(W)
    print(f"n_missing fitness: {n_missing_w}")

    return Y, X, W, Y0, Y_pca, Y0_pca, perturbation_to_gene_map, perturbation_names


def experiment(
    Y,
    X,
    W,
    Y0,
    perturbation_to_gene_map,
    perturbation_names,
    n_labeled=500,
    seed=42,
    model_type="ridge",
    g_reg_params=None,
    f_reg_params=None,
    **model_kwargs,
):
    """
    Runs a single experiment comparing approaches.
    n_labeled: Number of GENES to include in the labeled set.
    """
    if g_reg_params is None:
        g_reg_params = {}
    if f_reg_params is None:
        f_reg_params = {}

    def get_hidden_dims(params):
        return tuple(params.get("hidden_dims", HIDDEN_DIMS))

    np.random.seed(seed)

    # Split based on genes
    unique_genes = np.unique(list(perturbation_to_gene_map.values()))
    # Check if n_labeled is feasible
    if n_labeled > len(unique_genes):
        print(
            f"Warning: n_labeled ({n_labeled}) > total genes ({len(unique_genes)}). Using all genes."
        )
        labeled_genes = unique_genes
    else:
        labeled_genes = np.random.choice(unique_genes, size=n_labeled, replace=False)

    l_indices = []
    u_indices = []
    for idx, perturbation_name in enumerate(perturbation_names):
        gene_name = perturbation_to_gene_map[perturbation_name]
        if gene_name in labeled_genes:
            l_indices.append(idx)
        else:
            u_indices.append(idx)
    l_indices = np.array(l_indices)
    u_indices = np.array(u_indices)

    X_l, Y_l, W_l = X[l_indices], Y[l_indices], W[l_indices]
    XW_l = np.concatenate([X_l, W_l], axis=1)

    X_u, Y_u, W_u = X[u_indices], Y[u_indices], W[u_indices]
    XW_u = np.concatenate([X_u, W_u], axis=1)

    # train surrogate model
    if model_type == "ridge":
        g_reg = RidgeCV()
        g_reg.fit(XW_l, Y_l)
    elif model_type == "mlp":
        g_reg = RegressionPredictor(
            XW_l, Y_l, hidden_dims=get_hidden_dims(g_reg_params), seed=seed, **model_kwargs
        )
        g_reg.fit(max_epochs=1000)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    X_all = np.concatenate([X_l, X_u], axis=0)
    Yhat_all = np.concatenate([Y_l, g_reg.predict(XW_u)], axis=0)

    if model_type == "ridge":
        f_reg = RidgeCV()
        f_reg.fit(X_all, Yhat_all)
    elif model_type == "mlp":
        f_reg = RegressionPredictor(
            X_all, Yhat_all, hidden_dims=get_hidden_dims(f_reg_params), seed=seed, **model_kwargs
        )
        f_reg.fit(max_epochs=1000)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    Ypred_u = f_reg.predict(X_u)
    err = np.sqrt(np.mean((Y_u - Ypred_u) ** 2))
    delta_pred = Ypred_u - Y0
    delta_gt = Y_u - Y0
    corrs = stats.pearsonr(delta_pred, delta_gt, axis=0).statistic

    if model_type == "ridge":
        f_naive = RidgeCV()
        f_naive.fit(X_l, Y_l)
    elif model_type == "mlp":
        f_naive = RegressionPredictor(
            X_l, Y_l, hidden_dims=get_hidden_dims(f_reg_params), seed=seed, **model_kwargs
        )
        f_naive.fit(max_epochs=1000)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    Ypred_u_naive = f_naive.predict(X_u)
    err_naive = np.sqrt(np.mean((Y_u - Ypred_u_naive) ** 2))
    delta_pred_naive = Ypred_u_naive - Y0
    corrs_naive = stats.pearsonr(delta_pred_naive, delta_gt, axis=0).statistic

    # naive average
    Ypred_u_naive_avg = Y_l.mean(0)
    err_naive_avg = np.sqrt(np.mean((Y_u - Ypred_u_naive_avg) ** 2))
    corrs_naive_avg = 0.0

    res = pd.DataFrame(
        {
            "err": [err, err_naive, err_naive_avg],  # overall error
            "corr_mean": [
                corrs.mean(),
                corrs_naive.mean(),
                0.0,
            ],  # avg pearson R over perturbations
            "corr_q0.75": [np.quantile(corrs, 0.75), np.quantile(corrs_naive, 0.75), 0.0],
            "corr_q0.80": [np.quantile(corrs, 0.80), np.quantile(corrs_naive, 0.80), 0.0],
            "corr_q0.90": [np.quantile(corrs, 0.90), np.quantile(corrs_naive, 0.90), 0.0],
            "corr_q0.95": [np.quantile(corrs, 0.95), np.quantile(corrs_naive, 0.95), 0.0],
            "approach": ["PAST", "naive", "naive_avg"],
        }
    ).assign(n_labeled=n_labeled, seed=seed)
    return res


def main():
    parser = argparse.ArgumentParser(description="Run surrogate experiment (spacer level).")
    parser.add_argument(
        "--n_labeled",
        type=int,
        nargs="+",
        default=[100, 300, 500],
        help="List of labeled data sizes (number of genes).",
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

    Y, X, W, Y0, Y_pca, Y0_pca, perturbation_to_gene_map, perturbation_names = prepare_data(
        fitness_cols=args.fitness_cols
    )

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
        print(f"n_labeled (genes): {n_labeled}")
        for seed in seeds:
            res = experiment(
                Y_target,
                X,
                W,
                Y0=Y0_target,
                perturbation_to_gene_map=perturbation_to_gene_map,
                perturbation_names=perturbation_names,
                n_labeled=n_labeled,
                seed=seed,
                model_type=args.model_type,
                g_reg_params=g_reg_params,
                f_reg_params=f_reg_params,
                patience=10,
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
    save_dir = os.path.join(current_dir, "results_spacer")
    os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, f"results_spacer_{args_hash}.csv")
    res_all.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
