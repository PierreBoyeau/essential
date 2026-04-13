import os
from typing import List, Optional

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import pfba
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from .base import MetabolicRepresentationMethod


def dict_to_description(card_dict: dict) -> str:
    """
    Converts a dictionary of attributes into a structured text description.
    Assumes values are lists of strings or single strings.
    """
    lines = []
    for key, value in card_dict.items():
        if isinstance(value, list):
            val_str = ", ".join(str(v) for v in value)
        else:
            val_str = str(value)
        # Format key as Title Case (e.g. "substrates" -> "Substrates")
        key_formatted = key.replace("_", " ").title()
        lines.append(f"{key_formatted}: {val_str}")

    return "\n".join(lines)


class LLMEmbeddingMethod(MetabolicRepresentationMethod):
    """
    Computes Euclidean distance and RBF kernel from LLM-generated embeddings
    of gene metabolic functions (substrates and products).
    """

    def __init__(
        self,
        model_name: str = "embeddinggemma",
        flux_threshold: float = 1e-6,
        cache_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            model_name: The name of the Ollama embedding model.
            flux_threshold: Minimum flux to consider a reaction active.
            cache_file: Optional path to cache embeddings. If it ends with .csv, it will be saved as .pkl.
        """
        self.model_name = model_name
        self.flux_threshold = flux_threshold

        if cache_file and cache_file.endswith(".csv"):
            self.cache_file = cache_file.replace(".csv", ".pkl")
        else:
            self.cache_file = cache_file

        self.kernel = None
        self.distance_df = None

    def _generate_gene_cards(self, model: cobra.Model) -> dict:
        """
        Run FBA to extract active substrates and products per gene.
        """
        print("Running FBA to get fluxes...")
        try:
            model.solver = "glpk"
        except Exception:
            pass

        try:
            sol = pfba(model)
            fluxes = sol.fluxes
        except Exception:
            print("pFBA failed, falling back to standard FBA...")
            sol = model.optimize()
            fluxes = sol.fluxes

        print(f"Generating gene cards for {len(model.genes)} genes...")
        gene_cards = {}
        for gene in tqdm(model.genes, desc="Processing genes"):
            subs = set()
            prods = set()

            for rxn in gene.reactions:
                flux = fluxes.get(rxn.id, 0.0)

                if abs(flux) < self.flux_threshold:
                    continue

                forward = flux > 0
                reverse = flux < 0

                rxn_reactants = [m.name for m in rxn.reactants]
                rxn_products = [m.name for m in rxn.products]

                if forward:
                    subs.update(rxn_reactants)
                    prods.update(rxn_products)
                elif reverse:
                    subs.update(rxn_products)
                    prods.update(rxn_reactants)

            # Skip genes that have no active substrates and products
            if not subs and not prods:
                continue

            gene_name = gene.name if gene.name else gene.id
            gene_cards[gene_name] = {"substrates": list(subs), "products": list(prods)}

        return gene_cards

    def _generate_embeddings(self, gene_cards: dict) -> pd.DataFrame:
        """
        Generate embeddings for each gene using the specified Ollama model.
        """
        import ollama  # Import here to avoid requiring it if cached

        gene_names = list(gene_cards.keys())
        embeddings = []

        print(
            f"Generating embeddings using Ollama model '{self.model_name}' for {len(gene_names)} genes..."
        )
        for gene_name in tqdm(gene_names, desc="Embedding genes"):
            card = gene_cards[gene_name]

            # Omit the gene name (if present) to force reliance on metabolic function
            card_without_name = {k: v for k, v in card.items() if k != "name"}
            description = dict_to_description(card_without_name)

            try:
                response = ollama.embeddings(model=self.model_name, prompt=description)
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"\nError generating embedding for {gene_name}: {e}")
                print(
                    "Make sure Ollama is running and the model is pulled ('ollama serve' and 'ollama pull <model_name>')"
                )
                raise e

        print("Converting embeddings to DataFrame...")
        embeddings_matrix = np.array(embeddings)
        return pd.DataFrame(embeddings_matrix, index=gene_names)

    def fit(self, model: cobra.Model, genes: List[str], **kwargs):
        """
        Fit the method by generating embeddings and computing the distance/kernel.

        Args:
            model: COBRApy metabolic model.
            genes: Target list of genes to retain in the final representations.
        """
        self.target_genes = genes

        embeddings_df = None
        if self.cache_file and os.path.exists(self.cache_file):
            print(f"Loading cached embeddings from {self.cache_file}")
            embeddings_df = pd.read_pickle(self.cache_file)

        if embeddings_df is None:
            gene_cards = self._generate_gene_cards(model)
            embeddings_df = self._generate_embeddings(gene_cards)

            if self.cache_file:
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                embeddings_df.to_pickle(self.cache_file)
                print(f"Successfully saved embeddings to {self.cache_file}")

        # Compute Distance Matrix (Euclidean)
        # Drop rows where there are no embeddings (if any genes were skipped)
        valid_genes = [g for g in self.target_genes if g in embeddings_df.index]
        filtered_embeddings = embeddings_df.loc[valid_genes]

        if len(valid_genes) > 1:
            dists = pdist(filtered_embeddings.values, metric="euclidean")
            D = squareform(dists)
        elif len(valid_genes) == 1:
            D = np.array([[0.0]])
        else:
            D = np.array([])

        self.distance_df = pd.DataFrame(D, index=valid_genes, columns=valid_genes)

        # Compute Kernel Matrix (RBF/Exponential)
        if len(valid_genes) > 0:
            gamma = 1.0 / (np.mean(D) + 1e-8) if np.mean(D) > 0 else 1.0
            similarity = np.exp(-gamma * D)
        else:
            similarity = np.array([])

        self.kernel = pd.DataFrame(similarity, index=valid_genes, columns=valid_genes)

        return self

    def get_kernel(self) -> pd.DataFrame:
        """Return the G x G symmetric similarity matrix."""
        if self.kernel is None:
            raise ValueError("Must call fit() before get_kernel()")
        return self.kernel

    def get_distance(self) -> pd.DataFrame:
        """Return the G x G symmetric distance matrix."""
        if self.distance_df is None:
            raise ValueError("Must call fit() before get_distance()")
        return self.distance_df

    def get_expectations(self) -> pd.DataFrame:
        """
        Return a DataFrame of expected gene-gene interactions.
        For embeddings, we return pairs sorted by lowest Euclidean distance.
        """
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
