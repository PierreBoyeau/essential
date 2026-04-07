import cobra
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp
from metabolic_base import MetabolicNetworkBase


class MetabolicDiffusionKernel(MetabolicNetworkBase):
    """
    Computes diffusion kernel-based similarity between pairs of genes in a metabolic network.

    The kernel K = exp(-beta * L_norm) where L_norm is the normalized graph Laplacian of the
    symmetric metabolite graph (currency metabolites excluded).

    The similarity between gene g and gene k is computed as the mean of the kernel values
    over all pairs of metabolites (m, m') where m is associated with g and m' is associated with k.
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

        self._compute_kernel()

    def _compute_kernel(self):
        """Computes the normalized Laplacian and the diffusion kernel matrix."""
        A = self.adj_matrix_symmetric
        n = A.shape[0]

        if n == 0:
            self.kernel_matrix = np.zeros((0, 0))
            return

        # Compute degrees
        degrees = np.array(A.sum(axis=1)).flatten()

        # Avoid division by zero for isolated nodes
        with np.errstate(divide="ignore"):
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        # D^{-1/2}
        D_inv_sqrt = sp.diags(d_inv_sqrt)

        # Normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
        # A is sparse, so D_inv_sqrt @ A @ D_inv_sqrt is efficient
        pseudo_transition_matrix = D_inv_sqrt.dot(A).dot(D_inv_sqrt)

        # I - pseudo_transition_matrix
        L_norm = sp.eye(n) - pseudo_transition_matrix
        # Convert to dense for matrix exponential since expm requires dense arrays
        L_norm_dense = L_norm.toarray()

        if self.beta == None:
            # Compute eigenvalues for the symmetric matrix in ascending order
            eigenvalues = scipy.linalg.eigvalsh(L_norm_dense)

            # Isolate strictly positive eigenvalues to account for disconnected components
            positive_eigenvalues = eigenvalues[eigenvalues > 1e-8]

            if len(positive_eigenvalues) > 0:
                lambda_2 = positive_eigenvalues[0]
                self.beta = 1.0 / lambda_2
            else:
                self.beta = 1.0
            print(f"Computed beta: {self.beta}")

        # K = exp(-beta * L_norm)
        self.kernel_matrix = scipy.linalg.expm(-self.beta * L_norm_dense)

    def compute_similarity(self, g: str, k: str) -> float:
        """
        Computes the symmetrized diffusion kernel similarity between gene g and gene k.
        Returns 0.0 if either gene lacks associated products/substrates.
        """
        if g not in self.gene_products or k not in self.gene_substrates:
            raise ValueError(f"Gene not found: {g} or {k}")

        P_g = self.gene_products[g]
        S_k = self.gene_substrates[k]
        P_k = self.gene_products[k]
        S_g = self.gene_substrates[g]

        s_g_to_k = 0.0
        if P_g and S_k:
            s_g_to_k = np.mean(self.kernel_matrix[np.ix_(P_g, S_k)])

        s_k_to_g = 0.0
        if P_k and S_g:
            s_k_to_g = np.mean(self.kernel_matrix[np.ix_(P_k, S_g)])

        return 0.5 * (s_g_to_k + s_k_to_g)

    def compute_all_similarities(self) -> pd.DataFrame:
        """
        Computes similarities between all pairs of genes simultaneously using matrix operations.
        Returns a DataFrame with columns: ['gene_start', 'gene_stop', 'diffusion_similarity']
        """
        if self.kernel_matrix is None:
            self._compute_kernel()

        num_genes = len(self.genes)

        # Row-normalize the indicator matrices so that each row sums to 1
        # This will compute the mean when doing P_tilde @ K @ S_tilde.T
        def row_normalize(mat):
            row_sums = np.array(mat.sum(axis=1)).flatten()
            with np.errstate(divide="ignore"):
                row_inv = 1.0 / row_sums
            row_inv[np.isinf(row_inv)] = 0.0
            return sp.diags(row_inv).dot(mat)

        P_tilde = row_normalize(self.indicator_products)
        S_tilde = row_normalize(self.indicator_substrates)

        # S_dir = P_tilde @ K @ S_tilde^T
        # K is dense, so the result will be dense
        # sparse @ dense is much faster than dense @ sparse, so we do (S_tilde @ (P_tilde @ K)^T)^T
        temp = P_tilde.dot(self.kernel_matrix)  # temp is dense (num_genes, num_metabolites)

        # S_tilde @ temp.T gives (num_genes, num_genes). We then transpose to get the right order.
        S_dir = S_tilde.dot(temp.T).T

        S_dir = np.asarray(S_dir)

        # Symmetrize at the gene level
        S_sym = 0.5 * (S_dir + S_dir.T)

        # Flatten
        genes_array = np.array(self.genes)
        start_genes = np.repeat(genes_array, num_genes)
        stop_genes = np.tile(genes_array, num_genes)
        similarities = S_sym.flatten()

        df = pd.DataFrame(
            {
                "gene_start": start_genes,
                "gene_stop": stop_genes,
                "diffusion_similarity": similarities,
            }
        )

        return df
