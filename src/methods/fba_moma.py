import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List, Optional
import cobra
from cobra.flux_analysis import moma, pfba
from tqdm import tqdm

from .base import MetabolicRepresentationMethod
from src.data.metabolic_models import get_model


def _simulate_ko_chunk(gene_chunk: List[str], model_type: str):
    """Standalone function that runs in a separate process to avoid pickling issues."""
    model = get_model(model_type)
    model.solver = "glpk"

    # Compute WT solution
    sol_wt_fba = model.optimize()
    sol_wt = pfba(model)

    moma_fluxes = {}

    for gene_name in gene_chunk:
        with model:
            try:
                # Handle cases where gene ID might not be in the model
                if gene_name in model.genes:
                    gene_obj = model.genes.get_by_id(gene_name)
                else:
                    query_res = model.genes.query(gene_name, attribute="name")
                    if query_res:
                        gene_obj = query_res[0]
                    else:
                        continue

                gene_obj.knock_out()

                # Run MOMA using linear=True (GLPK) for stability
                sol_moma = moma(model, solution=sol_wt, linear=True)

                if sol_moma.status == "optimal":
                    moma_fluxes[gene_name] = sol_moma.fluxes
            except Exception as e:
                pass

    return moma_fluxes


class FBAMOMAMethod(MetabolicRepresentationMethod):
    """
    FBA/MOMA metabolic representation method.
    Uses native multiprocessing (joblib) to parallelize gene knockouts.
    """

    def __init__(self, n_jobs: int = 1, cache_file: Optional[str] = None, **kwargs):
        """
        Args:
            n_jobs: Number of CPU cores to use for parallel execution.
            cache_file: Path to a CSV file to cache/load flux results.
        """
        self.n_jobs = n_jobs
        self.cache_file = cache_file
        self.model_type = None
        self.fluxes = None
        self.kernel = None
        self.distance_df = None

    def fit(
        self, model: cobra.Model, genes: List[str], model_type: str = "ecoli_rich_medium", **kwargs
    ):
        """
        Fit the method by running FBA/MOMA knockouts and computing the Hamming kernel.

        Args:
            model: COBRApy metabolic model (unused directly in worker to avoid pickling, but kept for signature matching).
            genes: Target list of genes to knockout.
            model_type: Model type identifier for workers to re-instantiate the model.
        """
        print(self.n_jobs)
        self.model_type = model_type

        if self.cache_file and os.path.exists(self.cache_file):
            self.fluxes = pd.read_csv(self.cache_file, index_col=0)
        else:
            chunks = np.array_split(genes, self.n_jobs)
            gen = Parallel(n_jobs=self.n_jobs, return_as="generator")(
                delayed(_simulate_ko_chunk)(chunk, self.model_type) for chunk in chunks
            )
            results = list(tqdm(gen, total=len(chunks), desc="Computing FBA/MOMA fluxes"))

            # Consolidate results in memory
            valid_results = [r for r in results if r]
            if valid_results:
                self.fluxes = pd.concat(
                    [pd.DataFrame.from_dict(r, orient="index") for r in valid_results]
                )
            else:
                self.fluxes = pd.DataFrame()

            if self.cache_file and not self.fluxes.empty:
                os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
                self.fluxes.to_csv(self.cache_file)

        if self.fluxes.empty:
            self.kernel = pd.DataFrame()
            self.distance_df = pd.DataFrame()
            return self

        # Compute Hamming distance-based kernel
        # Binarize fluxes
        flux_df_bin = (self.fluxes.abs() >= 1e-6).astype(float)

        # Fast Hamming distance computation for binary matrices
        # D_ij = sum(x_i != x_j) = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
        X = flux_df_bin.values
        X_norm = np.sum(X, axis=1)
        D = X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)
        D = np.clip(D, 0, None)  # Handle numerical precision issues

        self.distance_df = pd.DataFrame(D, index=flux_df_bin.index, columns=flux_df_bin.index)

        # Convert distance to similarity for the kernel
        # Using a simple exponential kernel over Hamming distance
        gamma = 1.0 / (np.mean(D) + 1e-8) if np.mean(D) > 0 else 1.0
        similarity = np.exp(-gamma * D)

        self.kernel = pd.DataFrame(similarity, index=flux_df_bin.index, columns=flux_df_bin.index)

        return self

    def get_kernel(self) -> pd.DataFrame:
        """Return the G x G symmetric similarity matrix."""
        if self.kernel is None:
            raise ValueError("Must call fit() before get_kernel()")
        return self.kernel

    def get_expectations(self) -> pd.DataFrame:
        """
        Return a DataFrame of expected gene-gene interactions.
        For FBA/MOMA, we return pairs sorted by lowest Hamming distance (highest similarity).
        """
        # TODO: verify logic
        if self.distance_df is None:
            raise ValueError("Must call fit() before get_expectations()")

        if self.distance_df.empty:
            return pd.DataFrame(columns=["gene1", "gene2", "distance"])

        # Extract upper triangle without diagonal
        mask = np.triu(np.ones(self.distance_df.shape), k=1).astype(bool)

        # Unstack and filter
        stacked = self.distance_df.where(mask).stack().reset_index()
        stacked.columns = ["gene1", "gene2", "distance"]

        # Sort by lowest distance
        expectations = stacked.sort_values("distance", ascending=True).reset_index(drop=True)
        return expectations
