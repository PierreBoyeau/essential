from typing import List, Optional

import cobra
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from .base import MetabolicRepresentationMethod

DEFAULT_CURRENCY = [
    "atp",
    "adp",
    "nad",
    "nadh",
    "h2o",
    "h",
    "coa",
    "pi",
    "ppi",
    "nadp",
    "nadph",
    "amp",
    "co2",
]


class GeneGraphMethod(MetabolicRepresentationMethod):
    """
    Computes diffusion kernel-based similarity on a degree-weighted gene graph.
    Nodes are genes. Edge weight w(g, k) = sum_{m in shared} 1/deg(m),
    where 'shared' are metabolites produced by one and consumed by the other.
    Currency metabolites are excluded.
    The kernel K = exp(-beta * L_norm) where L_norm is the normalized graph Laplacian.
    """

    def __init__(
        self, currency_metabolites: Optional[List[str]] = None, beta: float = 1.0, **kwargs
    ):
        """
        Args:
            currency_metabolites: List of metabolite IDs to exclude from the graph.
            beta: Diffusion length scale parameter (default: 1.0).
        """
        self.currency_metabolites = set(
            currency_metabolites if currency_metabolites is not None else DEFAULT_CURRENCY
        )
        if beta == "None":
            beta = None
        self.beta = beta

        self.model = None
        self.genes = []
        self.target_genes = []
        self.kernel_matrix = None
        self._kernel_df = None
        self.gene_degrees = None

        self.metabolite_to_idx = {}
        self.idx_to_metabolite = {}
        self.indicator_products = None
        self.indicator_substrates = None

    def fit(self, model: cobra.Model, genes: List[str], **kwargs):
        """
        Fit the method by computing the diffusion kernel on the metabolic graph.

        Args:
            model: COBRApy metabolic model.
            genes: Target list of genes to retain in the final representations.
        """
        self.model = model
        self.target_genes = genes

        # 1. Build metabolic graph properties
        self._construct_simplified_graph()
        self._build_gene_info()

        # 2. Compute kernel
        self._compute_kernel()

        # 3. Filter to target genes
        self._format_kernel_df()

        return self

    def _construct_simplified_graph(self):
        """Builds directed graphs of metabolites excluding currency metabolites."""
        valid_metabolites = [
            m.id
            for m in self.model.metabolites
            if m.id.rsplit("_", 1)[0] not in self.currency_metabolites
        ]
        self.metabolite_to_idx = {m: i for i, m in enumerate(valid_metabolites)}
        self.idx_to_metabolite = {i: m for i, m in enumerate(valid_metabolites)}

    def _build_gene_info(self):
        """Extracts products and substrates for each gene, and builds indicator matrices
        indicating products and substrates associated with each gene."""
        self.genes = [g.name if g.name else g.id for g in self.model.genes]
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}
        num_genes = len(self.genes)
        num_metabolites = len(self.metabolite_to_idx)

        prod_rows, prod_cols = [], []
        sub_rows, sub_cols = [], []

        for i, gene in enumerate(self.model.genes):
            subs = set()
            prods = set()

            for rxn in gene.reactions:
                forward = rxn.upper_bound > 0
                reverse = rxn.lower_bound < 0

                rxn_reactants = [m.id for m in rxn.reactants if m.id in self.metabolite_to_idx]
                rxn_products = [m.id for m in rxn.products if m.id in self.metabolite_to_idx]

                if forward:
                    subs.update(rxn_reactants)
                    prods.update(rxn_products)
                if reverse:
                    subs.update(rxn_products)
                    prods.update(rxn_reactants)

            sub_indices = [self.metabolite_to_idx[m] for m in subs]
            prod_indices = [self.metabolite_to_idx[m] for m in prods]

            for idx in sub_indices:
                sub_rows.append(i)
                sub_cols.append(idx)

            for idx in prod_indices:
                prod_rows.append(i)
                prod_cols.append(idx)

        self.indicator_substrates = csr_matrix(
            (np.ones(len(sub_rows), dtype=float), (sub_rows, sub_cols)),
            shape=(num_genes, num_metabolites),
        )
        self.indicator_products = csr_matrix(
            (np.ones(len(prod_rows), dtype=float), (prod_rows, prod_cols)),
            shape=(num_genes, num_metabolites),
        )

    def _compute_kernel(self):
        """
        Computes the gene graph and its diffusion kernel.

        To compute the edge weight between gene g and gene k:
        w(g, k) = sum_{m \in (P_g \cap S_k) \cup (P_k \cap S_g)} 1 / deg(m)

        This is computed efficiently using sparse matrix operations and the
        inclusion-exclusion principle: |A \cup B| = |A| + |B| - |A \cap B|

        1. |A| = P * D_inv * S^T
        2. |B| = S * D_inv * P^T = (P * D_inv * S^T)^T
        3. |A \cap B| = (P \circ S) * D_inv * (P \circ S)^T
        """
        num_genes = len(self.genes)
        if num_genes == 0:
            self.kernel_matrix = np.zeros((0, 0))
            return

        valid_metabolites = [m for m in self.model.metabolites if m.id in self.metabolite_to_idx]
        num_metabolites = len(self.metabolite_to_idx)

        inv_degrees = np.zeros(num_metabolites)
        for m in valid_metabolites:
            idx = self.metabolite_to_idx[m.id]
            deg = len(m.reactions)
            if deg > 0:
                inv_degrees[idx] = 1.0 / deg

        D_inv = sp.diags(inv_degrees)

        P = self.indicator_products
        S = self.indicator_substrates

        # 1. |A|: sum_{m in (P_g \cap S_k)} 1 / deg(m)
        W_dir = P.dot(D_inv).dot(S.T)

        # 3. |A \cap B|: Metabolites that BOTH genes simultaneously produce AND consume.
        B = P.multiply(S)
        W_intersect = B.dot(D_inv).dot(B.T)

        # 4. Final union via inclusion-exclusion
        W = W_dir + W_dir.T - W_intersect

        # Remove self-loops
        W = W.tolil()
        W.setdiag(0)
        W = W.tocsr()

        self.adjacency_matrix = W

        # Compute gene degrees
        d_g = np.array(W.sum(axis=1)).flatten()
        self.gene_degrees = d_g

        with np.errstate(divide="ignore"):
            d_inv_sqrt = 1.0 / np.sqrt(d_g)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        D_g_inv_sqrt = sp.diags(d_inv_sqrt)

        transition_matrix = D_g_inv_sqrt.dot(W).dot(D_g_inv_sqrt)

        L_norm = sp.eye(num_genes) - transition_matrix
        L_norm_dense = L_norm.toarray()

        if self.beta is None:
            eigenvalues = scipy.linalg.eigvalsh(L_norm_dense)
            positive_eigenvalues = eigenvalues[eigenvalues > 1e-8]
            if len(positive_eigenvalues) > 0:
                lambda_2 = positive_eigenvalues[0]
                self.beta = 1.0 / lambda_2
            else:
                self.beta = 1.0
            print(f"Computed beta: {self.beta}")
        self.kernel_matrix = scipy.linalg.expm(-self.beta * L_norm_dense)

    def _format_kernel_df(self):
        """Formats the kernel matrix into a DataFrame and filters to target genes."""
        full_df = pd.DataFrame(self.kernel_matrix, index=self.genes, columns=self.genes)
        connected_mask = self.gene_degrees > 0
        connected_genes = np.array(self.genes)[connected_mask]
        is_valid_df = lambda g: (g in connected_genes) and (g in full_df.index)
        valid_targets = [g for g in self.target_genes if is_valid_df(g)]
        self._kernel_df = full_df.loc[valid_targets, valid_targets]

    def get_kernel(self) -> pd.DataFrame:
        """Return the G x G symmetric similarity matrix."""
        if self._kernel_df is None:
            raise ValueError("Must call fit() before get_kernel()")
        return self._kernel_df

    def get_distance(self) -> pd.DataFrame:
        """Return the G x G symmetric distance matrix."""
        if self._kernel_df is None:
            raise ValueError("Must call fit() before get_distance()")

        K = self._kernel_df.values
        k_diag = np.diag(K)
        dist_squared = k_diag[:, None] + k_diag[None, :] - 2 * K
        dist_squared = np.clip(dist_squared, 0, None)
        D = np.sqrt(dist_squared)

        return pd.DataFrame(D, index=self._kernel_df.index, columns=self._kernel_df.columns)

    def get_expectations(self) -> pd.DataFrame:
        """
        Return a DataFrame of expected gene-gene interactions.
        For the gene graph, this returns all non-diagonal pairs sorted by similarity.
        """
        # TODO: verify logic
        if self._kernel_df is None:
            raise ValueError("Must call fit() before get_expectations()")

        kernel_df = self._kernel_df

        # Extract upper triangle without diagonal
        mask = np.triu(np.ones(kernel_df.shape), k=1).astype(bool)

        # Unstack and filter
        stacked = kernel_df.where(mask).stack().reset_index()
        stacked.columns = ["gene1", "gene2", "similarity"]

        # Sort by highest similarity
        expectations = stacked.sort_values("similarity", ascending=False).reset_index(drop=True)
        return expectations
