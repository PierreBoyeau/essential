import cobra
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from metabolic_base import MetabolicNetworkBase


class MetabolicGeneGraphKernel(MetabolicNetworkBase):
    """
    Computes diffusion kernel-based similarity on a degree-weighted gene graph.

    Nodes are genes. Edge weight w(g, k) = sum_{m in shared} 1/deg(m),
    where 'shared' are metabolites produced by one and consumed by the other.
    Currency metabolites are excluded.

    The kernel K = exp(-beta * L_norm) where L_norm is the normalized graph Laplacian of this gene graph.
    """

    def __init__(
        self, model: cobra.Model, currency_metabolites: list[str] = None, beta: float = 1.0
    ):
        """
        Initialize the calculator.

        Args:
            model: A COBRApy model instance.
            currency_metabolites: List of metabolite IDs to exclude from the graph.
            beta: Diffusion length scale parameter (default: 1.0).
        """
        super().__init__(model, currency_metabolites)
        self.beta = beta
        self.kernel_matrix = None
        self.gene_degree = None
        self.connected_genes = None

        self._compute_kernel()

    def _compute_kernel(self):
        """Computes the gene graph and its diffusion kernel."""
        num_genes = len(self.genes)
        if num_genes == 0:
            self.kernel_matrix = np.zeros((0, 0))
            return

        # Metabolite Degrees
        valid_metabolites = [m for m in self.model.metabolites if m.id in self.metabolite_to_idx]

        # Build array of 1/deg(m)
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

        # W_dir = P * D_inv * S^T
        W_dir = P.dot(D_inv).dot(S.T)

        # B = P \circ S (element-wise multiplication, essentially logical AND because they are 0/1)
        B = P.multiply(S)

        # W_intersect = B * D_inv * B^T
        W_intersect = B.dot(D_inv).dot(B.T)

        # W = W_dir + W_dir^T - W_intersect
        W = W_dir + W_dir.T - W_intersect

        # Remove self-loops
        W = W.tolil()
        W.setdiag(0)
        W = W.tocsr()

        # Compute gene degrees
        d_g = np.array(W.sum(axis=1)).flatten()
        self.gene_degree = d_g
        connected_mask = d_g > 0
        self.connected_genes = [g for g, m in zip(self.genes, connected_mask) if m]

        with np.errstate(divide="ignore"):
            d_inv_sqrt = 1.0 / np.sqrt(d_g)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        D_g_inv_sqrt = sp.diags(d_inv_sqrt)

        transition_matrix = D_g_inv_sqrt.dot(W).dot(D_g_inv_sqrt)

        L_norm = sp.eye(num_genes) - transition_matrix
        L_norm_dense = L_norm.toarray()

        self.kernel_matrix = scipy.linalg.expm(-self.beta * L_norm_dense)

    def compute_similarity(self, g: str, k: str) -> float:
        """
        Returns the diffusion kernel similarity between gene g and gene k.
        """
        if g not in self.gene_to_idx or k not in self.gene_to_idx:
            raise ValueError(f"Gene not found: {g} or {k}")

        i = self.gene_to_idx[g]
        j = self.gene_to_idx[k]

        if self.kernel_matrix is None:
            return 0.0

        return self.kernel_matrix[i, j]

    def compute_all_similarities(self, connected_only: bool = False) -> pd.DataFrame:
        """
        Returns all similarities as a DataFrame.
        Since kernel_matrix is already gene-by-gene, simply flatten it.
        """
        if self.kernel_matrix is None:
            self._compute_kernel()

        num_genes = len(self.genes)

        genes_array = np.array(self.genes)
        start_genes = np.repeat(genes_array, num_genes)
        stop_genes = np.tile(genes_array, num_genes)
        similarities = self.kernel_matrix.flatten()

        df = pd.DataFrame(
            {
                "gene_start": start_genes,
                "gene_stop": stop_genes,
                "diffusion_similarity": similarities,
            }
        )

        if connected_only:
            df = df.loc[
                df["gene_start"].isin(self.connected_genes)
                & df["gene_stop"].isin(self.connected_genes)
            ]

        return df
