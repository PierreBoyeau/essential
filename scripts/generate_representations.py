import argparse
import pandas as pd
import json
import os
import sys

# Ensure src can be imported if script is run from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.metabolic_models import get_model
from src.methods import METHOD_REGISTRY

def main():
    parser = argparse.ArgumentParser(description="Generate metabolic representations (kernels & expectations).")
    parser.add_argument("--method", type=str, required=True, choices=["fba_moma", "gene_graph"], help="Method to use.")
    parser.add_argument("--model_type", type=str, required=True, help="Model configuration identifier from MODEL_REGISTRY.")
    parser.add_argument("--genes_csv", type=str, required=True, help="Path to CSV containing target genes.")
    parser.add_argument("--config_json", type=str, required=True, help="Path to JSON file containing method kwargs.")
    parser.add_argument("--out_kernel", type=str, required=True, help="Path to save the kernel CSV.")
    parser.add_argument("--out_expectations", type=str, required=True, help="Path to save the expectations CSV.")
    
    # Specific options for FBA/MOMA caching and threading managed by Snakemake
    parser.add_argument("--threads", type=int, default=1, help="Number of threads (used by joblib in FBA/MOMA).")
    parser.add_argument("--cache_fluxes", type=str, default=None, help="Cache file for FBA/MOMA fluxes.")
    
    args = parser.parse_args()

    # 1. Load configuration kwargs
    with open(args.config_json, 'r') as f:
        kwargs = json.load(f)

    # 2. Load data and model
    print(f"Loading genes from {args.genes_csv}...")
    genes_df = pd.read_csv(args.genes_csv)
    # Assume the CSV has an 'id' or 'gene' column; adjust if needed based on data format
    if 'id' in genes_df.columns:
        target_genes = genes_df['id'].tolist()
    elif 'gene' in genes_df.columns:
        target_genes = genes_df['gene'].tolist()
    else:
        # Fallback to index if genes are the index
        target_genes = genes_df.index.tolist()
        
    print(f"Loading COBRA model type: {args.model_type}...")
    model = get_model(args.model_type)

    # 3. Instantiate the correct method dynamically using the registry
    if args.method not in METHOD_REGISTRY:
        raise ValueError(f"Unsupported method: {args.method}. Available: {list(METHOD_REGISTRY.keys())}")
        
    # Standardize Snakemake orchestrator arguments into kwargs
    kwargs["n_jobs"] = args.threads
    kwargs["cache_file"] = args.cache_fluxes

    print(f"Initializing {args.method} with kwargs: {kwargs}...")
    MethodClass = METHOD_REGISTRY[args.method]
    
    # We pass all kwargs. The methods must accept **kwargs in their init or explicitly list them.
    method = MethodClass(**kwargs)

    # 4. Fit and generate outputs
    print(f"Fitting {args.method} for {len(target_genes)} target genes...")
    method.fit(model, target_genes)

    print("Saving outputs...")
    os.makedirs(os.path.dirname(args.out_kernel), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_expectations), exist_ok=True)
    
    kernel_df = method.get_kernel()
    kernel_df.to_csv(args.out_kernel)
    
    expectations_df = method.get_expectations()
    expectations_df.to_csv(args.out_expectations)
    
    print("Done!")

if __name__ == "__main__":
    main()
