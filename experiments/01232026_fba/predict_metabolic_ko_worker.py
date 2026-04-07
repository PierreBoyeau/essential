import argparse
import json
import os

import numpy as np
import optlang
import pandas as pd
from cobra.flux_analysis import moma, pfba
from cobra_models import get_model_components_df, load_ecoli_rich_medium_model
from tqdm import tqdm


def run_worker(input_csv, output_dir, job_id):
    """
    Worker function to process a chunk of genes.
    """
    print(f"Worker {job_id} starting...")

    # Load input genes
    genes_df = pd.read_csv(input_csv)
    # Ensure 'id' is the index if it's not already
    if "id" in genes_df.columns:
        genes_df = genes_df.set_index("id")

    # Load model and compute WT solution
    print(f"Worker {job_id}: Loading model...")
    model = load_ecoli_rich_medium_model()
    model.solver = "glpk"

    print(f"Worker {job_id}: Computing WT solution...")
    sol_wt_fba = model.optimize()
    growth_wt = sol_wt_fba.objective_value
    sol_wt = pfba(model)

    # Containers for results
    # Use dicts to ensure gene names are preserved as keys (index)
    fba_fluxes = {}
    fba_growth_ratios = []
    moma_fluxes = {}

    # Process genes serially
    print(f"Worker {job_id}: Processing {len(genes_df)} genes...")
    for gene_id in tqdm(genes_df.index, desc=f"Worker {job_id}"):
        gene_name = genes_df.loc[gene_id, "name"]

        with model:
            try:
                # Handle cases where gene ID might not be in the model
                if gene_id not in model.genes:
                    print(f"Worker {job_id}: Gene {gene_id} ({gene_name}) not found in model.")
                    fba_growth_ratios.append({"gene_name": gene_name, "growth_ratio": np.nan})
                    continue

                model.genes.get_by_id(gene_id).knock_out()

                # 1. Run FBA
                model.solver = "glpk"
                sol_fba = model.optimize()

                if sol_fba.status == "optimal":
                    fba_growth = sol_fba.objective_value
                    # Store fluxes with gene_name as key
                    fba_fluxes[gene_name] = sol_fba.fluxes
                    fba_growth_ratios.append(
                        {
                            "gene_name": gene_name,
                            "growth_ratio": fba_growth / growth_wt,
                            "growth": fba_growth,
                            "growth_wt": growth_wt,
                        }
                    )
                else:
                    fba_growth_ratios.append(
                        {
                            "gene_name": gene_name,
                            "growth_ratio": np.nan,
                            "growth": np.nan,
                            "growth_wt": growth_wt,
                        }
                    )

                # 2. Run MOMA
                # Using linear=True (GLPK) for stability
                # Passing the FULL solution object as required
                sol_moma = moma(model, solution=sol_wt, linear=True)

                if sol_moma.status == "optimal":
                    moma_fluxes[gene_name] = sol_moma.fluxes

            except Exception as e:
                print(f"Worker {job_id}: Error processing {gene_name} ({gene_id}): {e}")
                fba_growth_ratios.append({"gene_name": gene_name, "growth_ratio": np.nan})

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    print(f"Worker {job_id}: Saving results...")

    if fba_growth_ratios:
        pd.DataFrame(fba_growth_ratios).set_index("gene_name").to_csv(
            os.path.join(output_dir, f"fba_growth_ratios_{job_id}.csv")
        )

    if fba_fluxes:
        # orient='index' makes keys (gene names) the rows
        pd.DataFrame.from_dict(fba_fluxes, orient="index").to_csv(
            os.path.join(output_dir, f"fba_fluxes_{job_id}.csv")
        )

    if moma_fluxes:
        pd.DataFrame.from_dict(moma_fluxes, orient="index").to_csv(
            os.path.join(output_dir, f"moma_fluxes_{job_id}.csv")
        )

    sol_wt.fluxes.to_csv(os.path.join(output_dir, f"wt_fluxes_{job_id}.csv"))
    print(f"Worker {job_id} finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker script for metabolic KO prediction.")
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to the input genes chunk CSV."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the outputs."
    )
    parser.add_argument("--job_id", type=str, required=True, help="Job ID for file naming.")

    args = parser.parse_args()

    run_worker(args.input_csv, args.output_dir, args.job_id)
