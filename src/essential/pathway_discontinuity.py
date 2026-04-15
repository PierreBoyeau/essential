"""
Pathway discontinuity: graph helpers and MMD-based scores on gene pairs.

Genes are nodes in the metabolic graph. Edge equivalence on the operational graph
is computed in ``fit`` / ``compute_graph_equivalences`` once scores exist;
``compute_pair_score`` returns only a scalar MMD statistic or p-value.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData

from essential.equivalence_results import EquivalenceResults
from essential.stats import MMDTestJax

GenePair = tuple[str, str]  # Gene pair; use ``_normalize_gene_pair`` for a canonical key.


def global_sigma_median_heuristic(
    Z: np.ndarray,
    *,
    max_n: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    RBF bandwidth: sqrt(median squared pairwise distance) on up to ``max_n`` points,
    matching the legacy MMD target pipeline (subset avoids O(N^2) cost).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {Z.shape}")
    n = Z.shape[0]
    if n < 2:
        return 1.0
    if rng is None:
        rng = np.random.default_rng()
    if n > max_n:
        idx = rng.choice(n, size=max_n, replace=False)
        Z_sigma = Z[idx]
    else:
        Z_sigma = Z

    dists = ((Z_sigma[:, None, :] - Z_sigma[None, :, :]) ** 2).sum(-1)
    upper_tri = dists[np.triu_indices_from(dists, k=1)]
    median_sq_dist = float(np.median(upper_tri))
    return float(np.sqrt(median_sq_dist) if median_sq_dist > 0 else 1.0)


def _normalize_gene_pair(g1: str, g2: str) -> GenePair:
    a, b = sorted((g1, g2))
    return (a, b)


class PathwayDiscontinuity:
    """
    Statistics along a metabolic graph G = (V, E) with expression in ``adata``.

    Nodes are genes (or reactions); edges encode metabolic adjacency, including
    convergent reactions with multiple substrates from independent pathways.
    ``MultiGraph`` / edge-keyed graphs are not handled; use a simple ``networkx.Graph``
    or ``networkx.DiGraph``.
    """

    def __init__(
        self,
        adata: AnnData,
        representation_obsm_key: str,
        metabolic_graph: nx.Graph | nx.DiGraph,
        global_sigma: float | None = None,
        *,
        perturbation_obs_key: str = "perturbation_class",
        sigma_heuristic_max_n: int = 2000,
        sigma_heuristic_rng: np.random.Generator | None = None,
    ) -> None:
        self.adata = adata
        self.metabolic_graph = metabolic_graph
        self.representation_obsm_key = representation_obsm_key
        self.perturbation_obs_key = perturbation_obs_key
        if global_sigma is None:
            Z = np.asarray(adata.obsm[representation_obsm_key], dtype=np.float64)
            global_sigma = global_sigma_median_heuristic(
                Z, max_n=sigma_heuristic_max_n, rng=sigma_heuristic_rng
            )
        self.global_sigma = float(global_sigma)
        self._mmd = MMDTestJax(sigma=self.global_sigma, max_n=500)

    def _extract_adjacent_pairs(self, g: nx.Graph | nx.DiGraph) -> set[frozenset[str]]:
        """
        All unordered pairs of gene names directly connected by an edge in G.
        For directed graphs the undirected adjacency is used, so direction encodes
        pathway flow but does not restrict which pairs are scored.
        """
        ug = g.to_undirected() if g.is_directed() else g
        return {frozenset((str(u), str(v))) for u, v in ug.edges()}

    def compute_pair_score(
        self,
        g1: str,
        g2: str,
        *,
        mode: Literal["mmd_stat", "mmd_pvalue"] = "mmd_stat",
        **kwargs: Any,
    ) -> float:
        obs = self.adata.obs[self.perturbation_obs_key]
        X = self.adata.obsm[self.representation_obsm_key][obs.values == g1].astype(np.float32)
        Y = self.adata.obsm[self.representation_obsm_key][obs.values == g2].astype(np.float32)
        if mode == "mmd_stat":
            return self._mmd.compute_mmd(X, Y)
        elif mode == "mmd_pvalue":
            return -np.log10(self._mmd.test(X, Y, **kwargs) + 1e-12)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def compute_graph_equivalences(
        self,
        edge_dissimilarity: Mapping[frozenset[str], float],
        *,
        threshold: float,
    ) -> dict[int, list[str]]:
        """
        Build the operational graph from the metabolic graph, keeping only edges
        whose dissimilarity score is below ``threshold``, then return connected
        components as equivalence classes.

        Nodes with no retained neighbor form singleton classes.
        """
        g = self.metabolic_graph
        ug = g.to_undirected() if g.is_directed() else g

        op_graph: nx.Graph = nx.Graph()
        op_graph.add_nodes_from(str(n) for n in ug.nodes())

        for u, v in ug.edges():
            gene1, gene2 = str(u), str(v)
            dissimilarity_score = edge_dissimilarity.get(frozenset((gene1, gene2)))
            if dissimilarity_score is not None and dissimilarity_score <= threshold:
                op_graph.add_edge(gene1, gene2)

        return {
            cid: sorted(component)
            for cid, component in enumerate(nx.connected_components(op_graph))
        }

    def fit(
        self,
        metabolic_graph: nx.Graph | nx.DiGraph | None = None,
        threshold: float | None = None,
        **kwargs,
    ) -> EquivalenceResults:
        if metabolic_graph is not None:
            self.metabolic_graph = metabolic_graph

        g = self.metabolic_graph
        adjacent_pairs = self._extract_adjacent_pairs(g)

        edge_dissimilarity: dict[frozenset[str], float] = {}
        records = []

        for pair in adjacent_pairs:
            gene1, gene2 = tuple(pair)
            score = self.compute_pair_score(gene1, gene2, **kwargs)
            edge_dissimilarity[pair] = score
            records.append({"g1": gene1, "g2": gene2, "score": score})

        classes = self.compute_graph_equivalences(edge_dissimilarity, threshold=threshold)
        return EquivalenceResults(
            edge_equivalence=classes,
            gene_pair_scores=pd.DataFrame(records, columns=["g1", "g2", "score"]),
            metabolic_graph=g,
        )
