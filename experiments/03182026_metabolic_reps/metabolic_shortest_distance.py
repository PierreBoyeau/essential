import cobra
import numpy as np
import pandas as pd
from metabolic_base import MetabolicNetworkBase
from scipy.sparse.csgraph import shortest_path


class MetabolicShortestDistance(MetabolicNetworkBase):
    """
    Computes directed shortest distances d(g -> k) between pairs of genes in a metabolic network.

    The metabolic directed graph G=(V,E) consists of metabolites as nodes.
    An edge (m_a, m_b) exists if m_a is a substrate and m_b is a product of any reaction.
    Currency metabolites can be excluded from V.

    For a gene g, its products P_g and substrates S_g are defined as the union
    of all products/substrates of all reactions it is associated with via GPR rules.
    Reaction reversibility is considered (bounds).

    d(g -> k) = min_{p in P_g, s in S_k} d_G(p, s)
    """

    def __init__(self, model: cobra.Model, currency_metabolites: list[str]):
        """
        Initialize the calculator.

        Args:
            model: A COBRApy model instance.
            currency_metabolites: List of metabolite IDs to exclude from the graph.
        """
        super().__init__(model, currency_metabolites)

        # Cache for APSP
        self._apsp_matrix = None

    def compute_distance(self, g: str, k: str) -> float:
        """
        Computes the shortest directed distance from gene g to gene k.
        Returns np.inf if no path exists.
        """
        if g not in self.gene_products or k not in self.gene_substrates:
            raise ValueError(f"Gene not found: {g} or {k}")

        P_g = self.gene_products[g]
        S_k = self.gene_substrates[k]

        if not P_g or not S_k:
            return np.inf

        if self._apsp_matrix is not None:
            sub_D = self._apsp_matrix[np.ix_(P_g, S_k)]
            return np.min(sub_D)

        # Single-source shortest path from all nodes in P_g
        D = shortest_path(self.adj_matrix, directed=True, unweighted=True, indices=P_g)
        if D.ndim == 1:
            D = D[np.newaxis, :]
        sub_D = D[:, S_k]
        return np.min(sub_D)

    def compute_all_distances(self) -> pd.DataFrame:
        """
        Computes shortest distances between all pairs of genes simultaneously using optimized numpy ops.
        Returns a DataFrame with columns: ['gene_start', 'gene_stop', 'directed_distance']
        """
        if self._apsp_matrix is None:
            self._apsp_matrix = shortest_path(self.adj_matrix, directed=True, unweighted=True)

        D = self._apsp_matrix
        num_genes = len(self.genes)
        num_metabolites = D.shape[1]

        # gene_to_node[i, j] is the min distance from any product of gene i to metabolite j
        gene_to_node = np.full((num_genes, num_metabolites), np.inf)

        for i, g in enumerate(self.genes):
            P_g = self.gene_products[g]
            if P_g:
                gene_to_node[i, :] = np.min(D[P_g, :], axis=0)

        # dist_matrix[i, j] is the min distance from gene_to_node[i, :] to any substrate of gene j
        dist_matrix = np.full((num_genes, num_genes), np.inf)

        for j, k in enumerate(self.genes):
            S_k = self.gene_substrates[k]
            if S_k:
                dist_matrix[:, j] = np.min(gene_to_node[:, S_k], axis=1)

        # Flatten the matrix to create DataFrame
        genes_array = np.array(self.genes)
        start_genes = np.repeat(genes_array, num_genes)
        stop_genes = np.tile(genes_array, num_genes)
        distances = dist_matrix.flatten()

        df = pd.DataFrame(
            {"gene_start": start_genes, "gene_stop": stop_genes, "directed_distance": distances}
        )

        return df
