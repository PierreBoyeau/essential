import cobra
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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


class MetabolicNetworkBase:
    """
    Base class for computing relationships between genes in a metabolic network.

    The metabolic graph G=(V,E) consists of metabolites as nodes.
    An edge (m_a, m_b) exists if m_a is a substrate and m_b is a product of any reaction.
    Currency metabolites can be excluded from V.

    For a gene g, its products P_g and substrates S_g are defined as the union
    of all products/substrates of all reactions it is associated with via GPR rules.
    Reaction reversibility is considered (bounds).
    """

    def __init__(self, model: cobra.Model, currency_metabolites: list[str] = None):
        """
        Initialize the base calculator.

        Args:
            model: A COBRApy model instance.
            currency_metabolites: List of metabolite IDs to exclude from the graph.
        """
        self.model = model
        if currency_metabolites is None:
            currency_metabolites = DEFAULT_CURRENCY
        self.currency_metabolites = set(currency_metabolites)

        # Internal graph representations
        self.metabolite_to_idx = {}
        self.idx_to_metabolite = {}
        self.adj_matrix = None
        self.adj_matrix_symmetric = None

        # Gene metadata
        self.genes = []
        self.gene_to_idx = {}
        self.gene_products = {}  # gene_id -> list of metabolite indices
        self.gene_substrates = {}  # gene_id -> list of metabolite indices
        self.gene_metabolites = (
            {}
        )  # gene_id -> list of metabolite indices (union of products and substrates)

        # Indicator matrices (sparse, shape: num_genes x num_metabolites)
        self.indicator_products = None
        self.indicator_substrates = None
        self.indicator_metabolites = None

        # Build states
        self._construct_simplified_graph()
        self._build_gene_info()

    def _construct_simplified_graph(self):
        """Builds directed and symmetric undirected graphs of metabolites excluding currency metabolites."""
        valid_metabolites = [
            m.id
            for m in self.model.metabolites
            if m.id.rsplit("_", 1)[0] not in self.currency_metabolites
        ]
        self.metabolite_to_idx = {m: i for i, m in enumerate(valid_metabolites)}
        self.idx_to_metabolite = {i: m for i, m in enumerate(valid_metabolites)}

        rows = []
        cols = []

        for rxn in self.model.reactions:
            forward = rxn.upper_bound > 0
            reverse = rxn.lower_bound < 0

            # Map valid metabolites to their integer indices
            reactants = [
                self.metabolite_to_idx[m.id]
                for m in rxn.reactants
                if m.id in self.metabolite_to_idx
            ]
            products = [
                self.metabolite_to_idx[m.id] for m in rxn.products if m.id in self.metabolite_to_idx
            ]

            if forward:
                for r in reactants:
                    for p in products:
                        rows.append(r)
                        cols.append(p)
            if reverse:
                for p in products:
                    for r in reactants:
                        rows.append(p)
                        cols.append(r)

        n = len(valid_metabolites)

        if len(rows) > 0:
            data = np.ones(len(rows), dtype=bool)
            self.adj_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))

            # Symmetric matrix: A_{ij} = 1 if (i, j) in E or (j, i) in E
            sym_data = np.ones(len(rows) * 2, dtype=bool)
            sym_rows = rows + cols
            sym_cols = cols + rows
            self.adj_matrix_symmetric = csr_matrix((sym_data, (sym_rows, sym_cols)), shape=(n, n))
            # ensure boolean, since multiple edges could be accumulated
            self.adj_matrix_symmetric.data = np.ones_like(
                self.adj_matrix_symmetric.data, dtype=bool
            )
        else:
            self.adj_matrix = csr_matrix((n, n), dtype=bool)
            self.adj_matrix_symmetric = csr_matrix((n, n), dtype=bool)

    def _build_gene_info(self):
        """Extracts P_g, S_g, and M_g for each gene in the model, and builds indicator matrices."""
        self.genes = [g.name if g.name else g.id for g in self.model.genes]
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}
        num_genes = len(self.genes)
        num_metabolites = len(self.metabolite_to_idx)

        # Lists to construct sparse indicator matrices
        prod_rows, prod_cols = [], []
        sub_rows, sub_cols = [], []
        met_rows, met_cols = [], []

        for i, gene in enumerate(self.model.genes):
            gene_key = gene.name if gene.name else gene.id
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
            met_indices = list(set(sub_indices + prod_indices))

            self.gene_substrates[gene_key] = sub_indices
            self.gene_products[gene_key] = prod_indices
            self.gene_metabolites[gene_key] = met_indices

            for idx in sub_indices:
                sub_rows.append(i)
                sub_cols.append(idx)

            for idx in prod_indices:
                prod_rows.append(i)
                prod_cols.append(idx)

            for idx in met_indices:
                met_rows.append(i)
                met_cols.append(idx)

        self.indicator_substrates = csr_matrix(
            (np.ones(len(sub_rows), dtype=float), (sub_rows, sub_cols)),
            shape=(num_genes, num_metabolites),
        )
        self.indicator_products = csr_matrix(
            (np.ones(len(prod_rows), dtype=float), (prod_rows, prod_cols)),
            shape=(num_genes, num_metabolites),
        )
        self.indicator_metabolites = csr_matrix(
            (np.ones(len(met_rows), dtype=float), (met_rows, met_cols)),
            shape=(num_genes, num_metabolites),
        )
