"""Generate 5-fold TF cross-validation splits and a non-TF leaf list.

Writes to experiments/05152026_cellbox_comeback/splits/:
    tf_train_0.txt .. tf_train_4.txt  -- 80% of TFs per fold
    tf_test_0.txt  .. tf_test_4.txt   -- remaining 20% of TFs per fold
    non_tfs.txt                        -- perturbation targets in the data that are not TFs

Usage:
    python generate_splits.py [--seed SEED] [--out_dir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc

sys.path.insert(0, "/workspace/src")
from essential.data import load_regulondb_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        default="/workspace/experiments/05152026_cellbox_comeback/splits",
    )
    parser.add_argument(
        "--adata_path",
        default="/workspace/data/de122_lce75/adata_de122_lce75_merged.h5ad",
    )
    parser.add_argument("--perturbation_col", default="target")
    parser.add_argument("--control_key", default="nontargeting")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_db = load_regulondb_full()
    ref_db = ref_db.loc[lambda x: x["ri_type"].str.startswith("TF")]

    tfs = sorted(ref_db["regulator_gene"].str.lower().unique())

    # non-TFs = perturbation targets present in the data that are not TFs
    # (drop the control and any missing values before differencing).
    adata = sc.read_h5ad(args.adata_path)
    obs_targets = adata.obs[args.perturbation_col].dropna().astype(str).str.lower()
    obs_targets = set(obs_targets.unique()) - {args.control_key.lower()}
    non_tfs = sorted(obs_targets - set(tfs))

    rng = np.random.default_rng(args.seed)
    tfs = np.array(tfs)
    rng.shuffle(tfs)

    n_folds = 5
    folds = np.array_split(tfs, n_folds)

    for i in range(n_folds):
        test = sorted(folds[i].tolist())
        train = sorted(g for j, fold in enumerate(folds) if j != i for g in fold.tolist())
        (out_dir / f"tf_train_{i}.txt").write_text("\n".join(train))
        (out_dir / f"tf_test_{i}.txt").write_text("\n".join(test))
        print(f"fold {i}: {len(train)} train TFs, {len(test)} test TFs")

    (out_dir / "non_tfs.txt").write_text("\n".join(non_tfs))
    print(f"non_tfs.txt: {len(non_tfs)} genes")
    print(f"Splits written to {out_dir}")


if __name__ == "__main__":
    main()
