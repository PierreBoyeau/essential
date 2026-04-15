"""
Pathway discontinuity: graph helpers and (planned) MMD-based scores on gene pairs.

Edge equivalence on the operational graph is computed in ``fit`` / ``compute_graph_equivalences``
once scores exist; ``compute_pair_score`` returns only a scalar MMD statistic or p-value.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping

import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData

from essential.equivalence_results import EquivalenceResults
from essential.stats import MMDTestJax

Edge = tuple[str, str]  # Undirected edge as a sorted 2-tuple (lexicographic on node ids).
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


def _normalize_edge(u: Any, v: Any) -> Edge:
    a, b = sorted((str(u), str(v)))
    return (a, b)


def _normalize_gene_pair(g1: str, g2: str) -> GenePair:
    a, b = sorted((g1, g2))
    return (a, b)


class PathwayDiscontinuity:
    """
    Statistics along a metabolic graph G = (V, E) with expression in ``adata``.

    ``MultiGraph`` / edge-keyed graphs are not handled yet; use a simple ``networkx.Graph``.
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


    def _compute_line_graph(self, g: nx.Graph | nx.DiGraph) -> nx.Graph:
        """
        Line graph L(G): nodes are edges of G; two nodes are adjacent if the corresponding
        edges share an endpoint in G. Always operates on the undirected version so that all
        edges meeting at a metabolite are treated as consecutive, regardless of reaction direction.
        """
        return nx.line_graph(g.to_undirected() if g.is_directed() else g)

    def _extract_consecutive_pairs(self, g: nx.Graph) -> set[frozenset[str]]:
        """
        All unordered pairs of gene names whose edges share a vertex in G (consecutive edges).
        Equivalent to the edge set of the line graph, with each node mapped to its gene name.
        """
        lg = self._compute_line_graph(g)
        pairs: set[frozenset[str]] = set()
        for n1, n2 in lg.edges():
            gene1 = self._get_edge_gene(g, n1[0], n1[1])
            gene2 = self._get_edge_gene(g, n2[0], n2[1])
            pairs.add(frozenset((gene1, gene2)))
        return pairs

    def _get_edge_gene(self, g: nx.Graph, u: str, v: str) -> str:
        """Helper to get gene name from an edge, falling back to string representation of the edge."""
        edge_data = g.get_edge_data(u, v)
        if edge_data and "name" in edge_data:
            return edge_data["name"]
        elif edge_data and "gene" in edge_data:
            return edge_data["gene"]
        return str(_normalize_edge(u, v))

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
        threshold,
    ) -> dict[int, list[str]]:
        """
        Build the operational graph from the line graph of the metabolic graph, keeping only
        edges whose dissimilarity score is below ``threshold``, then return connected components
        as equivalence classes.

        Nodes with no retained neighbor form singleton classes.
        """
        g = self.metabolic_graph
        lg = self._compute_line_graph(g)

        op_graph: nx.Graph = nx.Graph()
        # populate nodes
        for node in lg.nodes():
            op_graph.add_node(self._get_edge_gene(g, node[0], node[1]))

        # populate edges iff the 
        for n1, n2 in lg.edges():
            gene1 = self._get_edge_gene(g, n1[0], n1[1])
            gene2 = self._get_edge_gene(g, n2[0], n2[1])
            dissimilarity_score = edge_dissimilarity.get(frozenset((gene1, gene2)))
            if dissimilarity_score <= threshold:
                op_graph.add_edge(gene1, gene2)

        return {
            cid: sorted(component)
            for cid, component in enumerate(nx.connected_components(op_graph))
        }

    def fit(self, metabolic_graph: nx.Graph | nx.DiGraph | None = None, threshold: float | None = None, **kwargs) -> EquivalenceResults:
        if metabolic_graph is not None:
            self.metabolic_graph = metabolic_graph

        g = self.metabolic_graph
        consecutive_pairs = self._extract_consecutive_pairs(g)

        edge_dissimilarity: dict[frozenset[str], float] = {}
        records = []

        for pair in consecutive_pairs:
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
