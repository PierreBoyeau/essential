"""Generate 5-fold TF cross-validation splits and a non-TF leaf list.

Writes to experiments/05152026_cellbox_comeback/splits/:
    tf_train_0.txt .. tf_train_4.txt  -- 80% of TFs per fold
    tf_test_0.txt  .. tf_test_4.txt   -- remaining 20% of TFs per fold
    non_tfs.txt                        -- genes that are targets but not regulators

Usage:
    python generate_splits.py [--seed SEED] [--out_dir DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/src")
from essential.data import load_regulondb_full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out_dir",
        default="/workspace/experiments/05152026_cellbox_comeback/splits",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_db = load_regulondb_full()
    ref_db = ref_db.loc[lambda x: x["ri_type"].str.startswith("TF")]

    tfs = sorted(ref_db["regulator_gene"].str.lower().unique())
    targets = set(ref_db["target_gene"].str.lower().unique())
    non_tfs = sorted(targets - set(tfs))

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
