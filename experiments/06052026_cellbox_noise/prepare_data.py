"""Prepare train/test splits for the 06052026_cellbox_noise experiment.

Splits TFs into 5 folds (same logic as 05152026_cellbox_comeback); fold <seed>
is used as the test fold.  Outputs, all written to out_dir/data/:

    adata_tf_train_<seed>.h5ad  -- TF train cells + controls
    adata_tf_test_<seed>.h5ad   -- TF test cells
    adata_control.h5ad          -- control cells
    adata_train_<seed>.h5ad     -- TF train cells + all non-TF perturbations + controls
    amask.npy                   -- (n_genes, n_genes) float32 TF→target adjacency mask

All h5ad files carry the following layers computed from the "reads" layer:
    counts        raw counts
    library       library-size normalised (target_sum=1e4)
    log1p         log1p of library-normalised
    quantile_clip per-gene 99th-percentile clipping of library-normalised

adata.X is set to the counts layer.

Usage:
    python prepare_data.py [--seed 0] [--out_dir .]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.insert(0, "/workspace/src")
from essential.data import load_regulondb_full


def _add_all_layers(adata, target_sum=1e4, clip_percentile=99.0):
    reads = adata.layers["reads"]
    raw = reads.toarray() if sparse.issparse(reads) else np.asarray(reads, dtype=np.float32)

    adata.layers["counts"] = raw.copy()

    lib = raw / (raw.sum(1, keepdims=True) / target_sum + 1e-12)
    adata.layers["library"] = lib.astype(np.float32)

    adata.layers["log1p"] = np.log1p(lib).astype(np.float32)

    upper = np.percentile(lib, clip_percentile, axis=0)
    upper = np.where(upper > 0, upper, 1.0)
    adata.layers["quantile_clip"] = (np.clip(lib, 0, upper) / upper).astype(np.float32)

    adata.X = adata.layers["counts"]


def _compute_amask(adata, ref_db):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        default="/workspace/experiments/06052026_cellbox_noise/data",
    )
    parser.add_argument(
        "--adata_path",
        default="/workspace/data/de122_lce75/adata_de122_lce75_merged.h5ad",
    )
    parser.add_argument("--perturbation_col", default="target")
    parser.add_argument("--control_key", default="nontargeting")
    parser.add_argument("--min_library_size", type=float, default=1e3)
    parser.add_argument("--experiment_subset", default="lce75")
    parser.add_argument("--target_sum", type=float, default=1e4)
    parser.add_argument("--clip_percentile", type=float, default=99.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load and filter ──────────────────────────────────────────────────────
    adata = sc.read_h5ad(args.adata_path)
    adata.obs["library_size"] = adata.layers["reads"].sum(1).A1
    adata = adata[adata.obs["library_size"] > args.min_library_size].copy()
    if args.experiment_subset != "all":
        adata = adata[adata.obs["experiment"] == args.experiment_subset].copy()
    adata = adata[adata.obs[args.perturbation_col].notna()].copy()

    # ── restrict to TF/target genes ──────────────────────────────────────────
    ref_db = load_regulondb_full()
    ref_db = ref_db.loc[lambda x: x["ri_type"].str.startswith("TF")]
    Amask, gene_is_regulator, gene_is_target = _compute_amask(adata, ref_db)
    adata.var_names = adata.var_names.str.lower()
    adata.obs[args.perturbation_col] = adata.obs[args.perturbation_col].str.lower()
    print(f"{adata.n_vars} genes (all genes kept, no TF/target filtering)")

    # ── compute all normalisation layers (after gene subsetting) ────────────
    _add_all_layers(adata, target_sum=args.target_sum, clip_percentile=args.clip_percentile)

    # ── save Amask (recomputed on the gene-subsetted adata) ──────────────────
    Amask_sub, _, _ = _compute_amask(adata, ref_db)
    amask_path = out_dir / "amask.npy"
    np.save(amask_path, Amask_sub)
    print(f"amask saved → {amask_path}  shape={Amask_sub.shape}")

    # ── TF 5-fold split (fold <seed> = test) ────────────────────────────────
    all_tfs = np.array(sorted(ref_db["regulator_gene"].str.lower().unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_tfs)
    folds = np.array_split(all_tfs, 5)
    tf_test = set(folds[args.seed % 5])
    tf_train = set(t for i, fold in enumerate(folds) if i != args.seed % 5 for t in fold)

    # ── per-cell group membership ────────────────────────────────────────────
    pert_lower = adata.obs[args.perturbation_col].str.lower()
    obs_targets = set(pert_lower.unique()) - {args.control_key.lower()}
    non_tfs = obs_targets - set(all_tfs)

    is_control = pert_lower == args.control_key.lower()
    is_tf_train = pert_lower.isin(tf_train)
    is_tf_test = pert_lower.isin(tf_test)
    is_non_tf = pert_lower.isin(non_tfs)

    s = args.seed
    splits = {
        f"adata_tf_train_{s}.h5ad": adata[is_tf_train | is_control],
        f"adata_tf_test_{s}.h5ad": adata[is_tf_test],
        "adata_control.h5ad": adata[is_control],
        f"adata_train_{s}.h5ad": adata[is_tf_train | is_non_tf | is_control],
    }

    for fname, a in splits.items():
        path = out_dir / fname
        a.write_h5ad(path)
        print(f"{fname}: {a.n_obs} cells → {path}")


if __name__ == "__main__":
    main()
