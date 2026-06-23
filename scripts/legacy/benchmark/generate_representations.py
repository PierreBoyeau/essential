import argparse
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.metabolic_models import get_model
from src.essential.legacy.benchmark.methods import METHOD_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--out_kernel", required=True)
    parser.add_argument("--out_expectations", required=True)
    parser.add_argument("--out_distances", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache_fluxes", default=None)
    args = parser.parse_args()

    with open(args.config_json) as f:
        config = json.load(f)

    method_name = config["method"]
    model_type = config["model_type"]
    method_params = config["params"]

    target_genes = pd.read_csv(config["genes_csv"])["gene"].tolist()
    model = get_model(model_type)

    method_params["n_jobs"] = args.threads
    method_params["cache_file"] = args.cache_fluxes

    method = METHOD_REGISTRY[method_name](**method_params)
    method.fit(model, target_genes, model_type=model_type)

    os.makedirs(os.path.dirname(args.out_kernel), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_expectations), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_distances), exist_ok=True)
    # method.get_kernel().to_csv(args.out_kernel)
    method.get_kernel().to_pickle(args.out_kernel)
    method.get_expectations().to_csv(args.out_expectations)
    method.get_distance().to_pickle(args.out_distances)


if __name__ == "__main__":
    main()
