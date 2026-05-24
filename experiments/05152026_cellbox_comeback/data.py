from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from essential.data import load_regulondb_full


def _as_dense(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _read_genes(path_or_paths):
    paths = path_or_paths if isinstance(path_or_paths, (list, tuple)) else [path_or_paths]
    genes = set()
    for p in paths:
        genes.update(g.strip().lower() for g in Path(p).read_text().splitlines() if g.strip())
    return genes


def normalize_counts(adata, norm_cfg):
    """Apply the configured count-normalization scheme in place on ``adata.X``."""
    adata.X = adata.layers["reads"].copy()
    method = norm_cfg.method

    if method == "none":
        return

    sc.pp.normalize_total(adata, target_sum=norm_cfg.target_sum)
    if method == "library":
        return

    if method == "log1p":
        sc.pp.log1p(adata)
        return

    if method == "quantile_clip":
        X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        upper = np.percentile(X, norm_cfg.clip_percentile, axis=0)
        upper = np.where(upper > 0, upper, 1.0)
        adata.X = np.clip(X, 0, upper) / upper
        return

    raise ValueError(f"Unknown normalization method: {method!r}")


def compute_amask(adata, ref_db):
    """Binary (n_genes, n_genes) regulator->target mask from a reference DB."""
    var_lower = adata.var_names.str.lower()
    name_to_idx = pd.Series(np.arange(len(var_lower)), index=var_lower)
    n = len(adata.var_names)
    Amask = np.zeros((n, n), dtype=np.float32)

    pairs = ref_db[["target_gene", "regulator_gene"]].copy()
    pairs["t_idx"] = pairs["target_gene"].map(name_to_idx)
    pairs["r_idx"] = pairs["regulator_gene"].map(name_to_idx)
    pairs = pairs.dropna(subset=["t_idx", "r_idx"])
    Amask[pairs["t_idx"].astype(int).values, pairs["r_idx"].astype(int).values] = 1.0

    gene_is_regulator = Amask.sum(0) != 0
    gene_is_target = Amask.sum(1) != 0
    return Amask, gene_is_regulator, gene_is_target


def prepare_data(config):
    """Load, filter, normalize, select features, split.

    Returns ``(adata_train, adata_test, adata_control, Amask)``.
    """
    adata = sc.read_h5ad(config.adata_path)
    adata.obs["library_size"] = adata.layers["reads"].sum(1).A1
    adata = adata[adata.obs["library_size"] > config.min_library_size].copy()
    if config.experiment_subset != "all":
        adata = adata[adata.obs["experiment"] == config.experiment_subset].copy()
    adata = adata[adata.obs[config.perturbation_col].notna()].copy()

    normalize_counts(adata, config.normalization)

    ref_db = load_regulondb_full()
    ref_db = ref_db.loc[lambda x: x["ri_type"].str.startswith("TF")]
    _, gene_is_regulator, gene_is_target = compute_amask(adata, ref_db)
    adata = adata[:, gene_is_regulator | gene_is_target].copy()
    Amask, _, _ = compute_amask(adata, ref_db)

    paths = [config.split.train_targets_path]
    if config.split.train_extra_targets_path:
        paths.append(config.split.train_extra_targets_path)
    train_targets = _read_genes(paths)
    train_targets.add(config.control_key.lower())
    test_targets = _read_genes(config.split.test_targets_path)
    pert = adata.obs[config.perturbation_col].str.lower()
    adata_train = adata[pert.isin(train_targets)].copy()
    adata_test = adata[pert.isin(test_targets)].copy()
    adata_control = adata[pert == config.control_key.lower()].copy()

    var_names_set = set(adata.var_names)
    test_targets = adata_test.obs[config.perturbation_col].astype(str).unique()
    offenders = [t for t in test_targets if t not in var_names_set]
    if offenders:
        print(f"Dropping {len(offenders)} test target(s) not in var_names: {offenders}")
        adata_test = adata_test[adata_test.obs[config.perturbation_col].isin(var_names_set)].copy()

    print(f"{adata_train.n_obs} train cells, {adata_test.n_obs} test cells")
    return adata_train, adata_test, adata_control, Amask
