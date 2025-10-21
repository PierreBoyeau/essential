import os
import json
import argparse
import numpy as np

from essential.utils import evaluate_interactions_on_regulondb, load_results


def validate_single_graph(filename):
    df_ode = load_results(filename)
    metrics_res = evaluate_interactions_on_regulondb(df_ode)
    return metrics_res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--folder", type=str, required=True, help="Root folder containing run subfolders"
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.folder)
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue

        npz_path = os.path.join(run_dir, "Amat.npz")
        if not os.path.exists(npz_path):
            continue

        metrics = validate_single_graph(npz_path)
        out_path = os.path.join(run_dir, "metrics.json")
        with open(out_path, "w") as f:
            json.dump(
                metrics, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else o
            )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
