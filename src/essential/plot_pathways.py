"""
Publication-quality metabolic pathway visualization with equivalence-class coloring.

Plots the original KEGG metabolic graph (compound nodes, gene-annotated edges)
with edges colored by the equivalence classes predicted by PathwayDiscontinuity.
Designed for chain-like / Markov-chain topologies common in KEGG modules.

Usage
-----
>>> from plot_pathway_results import plot_pathway_results
>>> fig, ax = plot_pathway_results(g, results.edge_equivalence)

# or pass the EquivalenceResults object directly:
>>> fig, ax = plot_pathway_results(g, results)

# with pair-score annotations on edges:
>>> fig, ax = plot_pathway_results(g, results, pair_scores=results.gene_pair_scores)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ── colour palette ──────────────────────────────────────────────────
# Curated qualitative palette – high contrast, colour-blind safe,
# based on Tol-Bright (P. Tol, 2021).
_CLASS_COLORS = [
    "#4477AA",  # blue
    "#EE6677",  # rose
    "#228833",  # green
    "#CCBB44",  # olive / yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#EE8866",  # orange
    "#44BB99",  # teal
]


# ── layout ──────────────────────────────────────────────────────────


def _dag_layout(
    g: nx.DiGraph,
    x_sep: float = 1.6,
    y_sep: float = 1.0,
) -> dict:
    """
    Topological-sort layout for DAG-like graphs.

    Nodes are arranged left-to-right by longest-path depth; siblings within
    a layer are stacked vertically.  Falls back to ``kamada_kawai_layout``
    for graphs with cycles.
    """
    try:
        topo = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        return nx.kamada_kawai_layout(g)

    depth: dict = {}
    for node in topo:
        preds = list(g.predecessors(node))
        depth[node] = 0 if not preds else max(depth[p] for p in preds) + 1

    layers: dict[int, list] = defaultdict(list)
    for node in topo:
        layers[depth[node]].append(node)

    pos = {}
    for d, members in layers.items():
        n = len(members)
        for i, node in enumerate(members):
            y = -(i - (n - 1) / 2) * y_sep
            pos[node] = np.array([d * x_sep, y])
    return pos


# ── helpers ─────────────────────────────────────────────────────────


def _resolve_equivalence(
    equivalence_classes,
) -> dict[int, list[str]]:
    """Accept either a raw dict or an EquivalenceResults object."""
    if isinstance(equivalence_classes, dict):
        return equivalence_classes
    # duck-type: EquivalenceResults has .edge_equivalence
    if hasattr(equivalence_classes, "edge_equivalence"):
        return equivalence_classes.edge_equivalence
    raise TypeError(
        "equivalence_classes must be a dict or EquivalenceResults, "
        f"got {type(equivalence_classes).__name__}"
    )


def _build_score_lookup(
    pair_scores: pd.DataFrame | None,
) -> dict[frozenset[str], float]:
    """Build a frozenset -> score lookup from the gene_pair_scores DataFrame."""
    if pair_scores is None or pair_scores.empty:
        return {}
    out: dict[frozenset[str], float] = {}
    for _, row in pair_scores.iterrows():
        out[frozenset((row["g1"], row["g2"]))] = float(row["score"])
    return out


# ── main plotting function ──────────────────────────────────────────


def plot_pathway_results(
    g: nx.DiGraph,
    equivalence_classes: Union[dict[int, list[str]], "EquivalenceResults"],
    *,
    pair_scores: pd.DataFrame | None = None,
    # geometry
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
    x_sep: float = 1.6,
    y_sep: float = 1.0,
    # nodes
    node_size: float = 40,
    node_color: str = "#2d2d2d",
    node_edge_color: str = "white",
    node_linewidth: float = 0.5,
    show_metabolite_labels: bool = False,
    metabolite_font_size: float = 5.5,
    # edges
    edge_width: float = 3.2,
    singleton_color: str = "#c0c0c0",
    singleton_alpha: float = 0.55,
    arrow_style: str = "-|>",
    arrow_mutation: float = 14,
    connectionstyle: str = "arc3,rad=0.07",
    # labels
    font_size: float = 7.5,
    label_bbox_alpha: float = 0.88,
    show_scores: bool = False,
    score_fmt: str = ".2f",
    score_font_size: float = 5.5,
    # legend / title
    show_title: bool = True,
    show_legend: bool = True,
    legend_fontsize: float = 7,
    title_fontsize: float = 9,
    # output
    save_path: str | Path | None = None,
    show: bool = False,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a KEGG metabolic graph with edges coloured by equivalence class.

    Parameters
    ----------
    g : nx.DiGraph
        The *metabolic* graph returned by ``kegg_module_to_graph``.
        Nodes are compound IDs; edges carry ``gene`` / ``reaction`` attrs.
    equivalence_classes : dict or EquivalenceResults
        ``{class_id: [gene, ...]}``, or an ``EquivalenceResults`` object
        (its ``.edge_equivalence`` is used automatically).
    pair_scores : pd.DataFrame, optional
        DataFrame with columns ``g1``, ``g2``, ``score``.  When provided
        together with ``show_scores=True``, the MMD statistic is printed
        as a small annotation at intermediate compound nodes.
    show_metabolite_labels : bool
        If True, render compound names next to each node.
    show_scores : bool
        If True and ``pair_scores`` is provided, annotate intermediate
        nodes with pairwise MMD statistics.
    save_path : str or Path, optional
        Save figure to this path (PDF / SVG / PNG ...).

    Returns
    -------
    fig, ax
    """
    equiv = _resolve_equivalence(equivalence_classes)

    # ── reverse map: gene -> class_id ────────────────────────────────
    gene_to_class: dict[str, int] = {}
    for cid, genes in equiv.items():
        for gene in genes:
            gene_to_class[gene] = cid

    # ── class -> colour ──────────────────────────────────────────────
    unique_classes = sorted(equiv.keys())
    multi_classes = [c for c in unique_classes if len(equiv[c]) > 1]
    single_classes = [c for c in unique_classes if len(equiv[c]) == 1]

    class_color: dict[int, str] = {}
    for i, cid in enumerate(multi_classes):
        class_color[cid] = _CLASS_COLORS[i % len(_CLASS_COLORS)]
    for cid in single_classes:
        class_color[cid] = singleton_color

    # ── layout ──────────────────────────────────────────────────────
    pos = _dag_layout(g, x_sep=x_sep, y_sep=y_sep)

    # ── figure ──────────────────────────────────────────────────────
    if ax is None:
        if figsize is None:
            xs = [pos[n][0] for n in pos]
            ys = [pos[n][1] for n in pos]
            span_x = (max(xs) - min(xs)) if xs else 1
            span_y = (max(ys) - min(ys)) if ys else 1
            w = max(5.5, span_x * 1.5 + 2.2)
            h = max(3.5, span_y * 1.5 + 2.2)
            figsize = (min(w, 18), min(h, 12))
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.get_figure()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── score lookup ────────────────────────────────────────────────
    score_lookup = _build_score_lookup(pair_scores) if show_scores else {}

    # ── draw edges ──────────────────────────────────────────────────
    for u, v, data in g.edges(data=True):
        gene = (data.get("gene") or "").strip()
        cid = gene_to_class.get(gene)
        is_singleton = cid in single_classes if cid is not None else True
        color = class_color.get(cid, singleton_color) if cid is not None else singleton_color
        alpha = singleton_alpha if is_singleton else 1.0

        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops=dict(
                arrowstyle=arrow_style,
                color=color,
                lw=edge_width,
                alpha=alpha,
                mutation_scale=arrow_mutation,
                connectionstyle=connectionstyle,
                shrinkA=np.sqrt(node_size) * 0.55,
                shrinkB=np.sqrt(node_size) * 0.55,
            ),
        )

    # ── edge labels (gene symbols, italic per biology convention) ───
    for u, v, data in g.edges(data=True):
        gene = (data.get("gene") or "").strip()
        if not gene:
            continue

        x_mid = (pos[u][0] + pos[v][0]) / 2
        y_mid = (pos[u][1] + pos[v][1]) / 2

        cid = gene_to_class.get(gene)
        is_singleton = cid in single_classes if cid is not None else True
        label_c = class_color.get(cid, "#888888") if cid is not None else "#888888"
        if is_singleton:
            label_c = "#888888"

        ax.text(
            x_mid,
            y_mid,
            gene,
            fontsize=font_size,
            fontstyle="italic",
            fontweight="bold",
            fontfamily="sans-serif",
            color=label_c,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.12",
                fc="white",
                ec="none",
                alpha=label_bbox_alpha,
            ),
            zorder=5,
        )

    # ── optional score annotations at intermediate nodes ────────────
    if show_scores and score_lookup:
        scored_nodes: set = set()
        for u, v, d1 in g.edges(data=True):
            g1 = (d1.get("gene") or "").strip()
            if not g1:
                continue
            for _, w, d2 in g.out_edges(v, data=True):
                g2 = (d2.get("gene") or "").strip()
                if g2 and g2 != g1 and v not in scored_nodes:
                    key = frozenset((g1, g2))
                    if key in score_lookup:
                        scored_nodes.add(v)
                        ax.text(
                            pos[v][0],
                            pos[v][1] - 0.22,
                            f"{score_lookup[key]:{score_fmt}}",
                            fontsize=score_font_size,
                            color="#555555",
                            ha="center",
                            va="top",
                            bbox=dict(
                                boxstyle="round,pad=0.08",
                                fc="#f5f5f5",
                                ec="#dddddd",
                                alpha=0.8,
                                lw=0.4,
                            ),
                            zorder=4,
                        )

    # ── draw nodes (small, understated) ─────────────────────────────
    node_collection = nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=node_color,
        edgecolors=node_edge_color,
        linewidths=node_linewidth,
    )
    if node_collection is not None:
        node_collection.set_zorder(6)

    # ── optional metabolite labels ──────────────────────────────────
    if show_metabolite_labels:
        for node, ndata in g.nodes(data=True):
            label = ndata.get("name", node)
            x, y = pos[node]
            ax.text(
                x,
                y - 0.22,
                label,
                fontsize=metabolite_font_size,
                color="#777777",
                ha="center",
                va="top",
                zorder=3,
            )

    # ── title ───────────────────────────────────────────────────────
    if show_title:
        module_name = g.graph.get("name", "")
        module_id = g.graph.get("module_id", "")
        if module_id and module_name:
            title = f"{module_id}  —  {module_name}"
        else:
            title = module_name or module_id or ""
        if title:
            ax.set_title(
                title,
                fontsize=title_fontsize,
                fontweight="bold",
                pad=14,
                color="#1a1a1a",
            )

    # ── legend ──────────────────────────────────────────────────────
    if show_legend and multi_classes:
        handles = []
        for cid in multi_classes:
            genes_str = ", ".join(sorted(equiv[cid]))
            handles.append(
                mpatches.Patch(
                    facecolor=class_color[cid],
                    edgecolor="none",
                    label=genes_str,
                )
            )
        if single_classes:
            handles.append(
                mpatches.Patch(
                    facecolor=singleton_color,
                    edgecolor="none",
                    alpha=singleton_alpha,
                    label="singletons",
                )
            )
        ax.legend(
            handles=handles,
            title="Equivalence classes",
            title_fontproperties={"weight": "bold", "size": legend_fontsize},
            loc="lower right",
            fontsize=legend_fontsize,
            frameon=True,
            framealpha=0.92,
            edgecolor="#e0e0e0",
            fancybox=True,
            borderpad=0.8,
            handlelength=1.4,
            handleheight=0.9,
        )

    # ── axis limits with padding ────────────────────────────────────
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    if xs and ys:
        pad_x = (max(xs) - min(xs)) * 0.12 + 0.6
        pad_y = (max(ys) - min(ys)) * 0.15 + 0.6
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

    ax.set_axis_off()
    fig.tight_layout(pad=0.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()

    return fig, ax
