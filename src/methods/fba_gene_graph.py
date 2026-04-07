from typing import List, Optional

import cobra
import numpy as np
from cobra.flux_analysis import pfba
from scipy.sparse import csr_matrix

from .gene_graph import GeneGraphMethod


class FBAGeneGraphMethod(GeneGraphMethod):
    """
    Hybrid between GeneGraph and FBA/MOMA.
    Computes diffusion kernel-based similarity on a degree-weighted gene graph,
    where graph edges are determined by active fluxes from WT FBA rather than
    reaction bounds. Reactions with very small absolute fluxes are excluded.
    """

    def __init__(
        self,
        currency_metabolites: Optional[List[str]] = None,
        beta: float = 1.0,
        flux_threshold: float = 1e-6,
        **kwargs,
    ):
        """
        Args:
            currency_metabolites: List of metabolite IDs to exclude from the graph.
            beta: Diffusion length scale parameter (default: 1.0).
            flux_threshold: Minimum absolute flux required to consider a reaction active.
        """
        super().__init__(currency_metabolites=currency_metabolites, beta=beta, **kwargs)
        self.flux_threshold = flux_threshold
        self.fluxes = None

    def fit(self, model: cobra.Model, genes: List[str], **kwargs):
        """
        Fit the method by running WT FBA and computing the diffusion kernel
        on the active metabolic graph.
        """
        self.model = model
        self.target_genes = genes

        # Run FBA to get fluxes
        # Ensure solver is stable
        try:
            model.solver = "glpk"
        except Exception:
            pass

        try:
            sol_wt = pfba(model)
            self.fluxes = sol_wt.fluxes
        except Exception:
            # Fallback to standard FBA if pFBA fails
            sol_wt = model.optimize()
            self.fluxes = sol_wt.fluxes

        # 1. Build metabolic graph properties
        self._construct_simplified_graph()
        self._build_gene_info()

        # 2. Compute kernel
        self._compute_kernel()

        # 3. Filter to target genes
        self._format_kernel_df()

        return self

    def _build_gene_info(self):
        """
        Extracts products and substrates for each gene using actual flux directions
        from FBA. Excludes reactions with flux below threshold.
        """
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
                flux = self.fluxes.get(rxn.id, 0.0)

                if abs(flux) < self.flux_threshold:
                    continue

                forward = flux > 0
                reverse = flux < 0

                rxn_reactants = [m.id for m in rxn.reactants if m.id in self.metabolite_to_idx]
                rxn_products = [m.id for m in rxn.products if m.id in self.metabolite_to_idx]

                if forward:
                    subs.update(rxn_reactants)
                    prods.update(rxn_products)
                elif reverse:
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
