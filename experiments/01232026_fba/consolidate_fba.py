import pandas as pd
import glob
import os
import argparse


def consolidate_results(output_dir):
    """
    Consolidates partial results from parallel workers.
    """
    print(f"Consolidating results in {output_dir}...")

    # 1. FBA Growth Ratios
    growth_files = glob.glob(os.path.join(output_dir, "fba_growth_ratios_*.csv"))
    if growth_files:
        print(f"Found {len(growth_files)} growth ratio files.")
        dfs = []
        for f in growth_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if dfs:
            full_growth_df = pd.concat(dfs, ignore_index=False)
            if "gene_name" in full_growth_df.columns:
                full_growth_df = full_growth_df.set_index("gene_name")

            output_path = os.path.join(output_dir, "fba_growth_ratios.csv")
            full_growth_df.to_csv(output_path)
            print(f"Saved consolidated growth ratios to {output_path}")

            # # Cleanup
            # for f in growth_files:
            #     os.remove(f)
    else:
        print("No growth ratio files found.")

    # 2. FBA Fluxes
    flux_files = glob.glob(os.path.join(output_dir, "fba_fluxes_*.csv"))
    if flux_files:
        print(f"Found {len(flux_files)} FBA flux files.")
        dfs = []
        for f in flux_files:
            try:
                dfs.append(
                    pd.read_csv(f, index_col=0)
                )  # Assuming first col is index (original index from worker)
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if dfs:
            full_flux_df = pd.concat(dfs, ignore_index=False)
            output_path = os.path.join(output_dir, "fba_fluxes.csv")
            full_flux_df.to_csv(output_path)
            print(f"Saved consolidated FBA fluxes to {output_path}")

            # # Cleanup
            # for f in flux_files:
            #     os.remove(f)
    else:
        print("No FBA flux files found.")

    # 3. MOMA Fluxes
    moma_files = glob.glob(os.path.join(output_dir, "moma_fluxes_*.csv"))
    if moma_files:
        print(f"Found {len(moma_files)} MOMA flux files.")
        dfs = []
        worker_ids = []
        for f in moma_files:
            try:
                dfs.append(pd.read_csv(f, index_col=0))
                worker_ids.append(
                    pd.DataFrame({"worker_id": os.path.basename(f)}, index=dfs[-1].index)
                )
            except Exception as e:
                print(f"Error reading {f}: {e}")

        if dfs:
            full_moma_df = pd.concat(dfs, ignore_index=False)
            output_path = os.path.join(output_dir, "moma_fluxes.csv")
            full_moma_df.to_csv(output_path)
            print(f"Saved consolidated MOMA fluxes to {output_path}")

            full_worker_ids = pd.concat(worker_ids, ignore_index=False)
            full_worker_ids.to_csv(os.path.join(output_dir, "worker_ids.csv"))

            # # Cleanup
            # for f in moma_files:
            #     os.remove(f)
    else:
        print("No MOMA flux files found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate simulation results.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory containing partial results."
    )

    args = parser.parse_args()

    consolidate_results(args.output_dir)
