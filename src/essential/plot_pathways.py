"""
Publication-quality metabolic pathway visualization with equivalence-class coloring.

Plots the original KEGG metabolic graph (compound nodes, gene-annotated edges)
with edges colored by the equivalence classes predicted by PathwayDiscontinuity.
Designed for chain-like / Markov-chain topologies common in KEGG modules.

Usage
-----
>>> from plot_pathways import plot_pathway_results
>>> fig, ax, info = plot_pathway_results(g, results.edge_equivalence)

# or pass the EquivalenceResults object directly:
>>> fig, ax, info = plot_pathway_results(g, results)

# with pair-score annotations on edges:
>>> fig, ax, info = plot_pathway_results(g, results, pair_scores=results.gene_pair_scores)

# inspect per-gene colors used in the plot:
>>> info["gene_color_mapping"]   # {gene_symbol: "#rrggbb", ...}
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from essential.equivalence_results import EquivalenceResults

# ── colour palettes ─────────────────────────────────────────────────
# Unified qualitative palette for all scored equivalence classes.
# Based on Tol-Bright (P. Tol, 2021) with reduced saturation.
_PALETTE = [
    "#6A9EC2",  # muted blue
    "#E05C5C",  # muted red
    "#5A9E5A",  # muted green
    "#C4AA44",  # muted olive / yellow
    "#72BBCC",  # muted cyan
    "#A05080",  # muted purple
    "#D08860",  # muted orange
    "#55A88A",  # muted teal
]

# Fixed grey for the control / no-phenotype class.
_CONTROL_COLOR = "#BCBCBC"

# Color for genes that have no observations in the dataset (not evaluable).
_UNSCORED_COLOR = "#AAAAAA"


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

    Returns
    -------
    pos : dict
        ``{node: np.array([x, y])}`` in data coordinates.
    """
    try:
        topo = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        pos = nx.kamada_kawai_layout(g)
        return pos

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


def _resolve_equivalence(equivalence_classes) -> dict[int, list[str]]:
    """Accept either a raw dict or an EquivalenceResults object."""
    if isinstance(equivalence_classes, dict):
        return equivalence_classes
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


# ── main plotting functions ──────────────────────────────────────────
def _edge_caption(d):
    gene = (d.get("gene") or "").strip()
    rxn = (d.get("reaction") or "").strip()
    if gene and rxn:
        return f"{gene}\n{rxn}"
    return gene or rxn or ""


def plot_metabolic_pathway(g, figsize=(14, 14), dpi=500, seed=42, node_size=400):
    """Draw a KEGG pathway graph: compound names on nodes, gene + reaction on edges."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    n_nodes = max(g.number_of_nodes(), 1)
    pos = nx.spring_layout(g, seed=seed, k=1.8 / n_nodes**0.5)

    node_labels = {n: str(d.get("name", n)) for n, d in g.nodes(data=True)}
    edge_labels = {(u, v): _edge_caption(d) for u, v, d in g.edges(data=True)}

    nx.draw(
        g,
        pos,
        labels=node_labels,
        with_labels=True,
        font_size=7,
        node_size=node_size,
        node_color="lightsteelblue",
        edgecolors="steelblue",
        linewidths=0.6,
        width=0.8,
        arrowsize=12,
        node_shape="o",
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        font_size=5,
        ax=ax,
        rotate=False,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
    )
    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def plot_pathway_results(
    g: nx.MultiDiGraph,
    equivalence_classes: Union[dict[int, list[str]], "EquivalenceResults"],
    *,
    pair_scores: pd.DataFrame | None = None,
    # ── geometry ────────────────────────────────────────────────────
    figsize: tuple[float, float] | None = None,
    dpi: int = 300,
    x_sep: float = 1.6,
    y_sep: float = 1.0,
    inter_arrow_distance: float = 0.0,
    # ── metabolites (nodes) ─────────────────────────────────────────
    node_mode: Literal["none", "labels"] = "none",
    metabolite_font_size: float = 5.5,
    metabolite_label_offset: tuple[float, float] = (0.0, -0.22),
    # ── edges ───────────────────────────────────────────────────────
    edge_width: float = 3.2,
    arrow_style: str = "-|>",
    arrow_mutation: float = 14,
    connectionstyle: str = "arc3,rad=0.0",
    skip_edge_rad: float = 0.3,
    # ── edge labels ─────────────────────────────────────────────────
    font_size: float = 7.5,
    label_offset: tuple[float, float] = (0.0, 0.0),
    isoenzyme_spacing: float = 0.15,
    # ── score annotations ───────────────────────────────────────────
    show_scores: bool = False,
    score_fmt: str = ".2f",
    score_font_size: float = 5.5,
    # ── legend / title ──────────────────────────────────────────────
    show_title: bool = True,
    show_legend: bool = True,
    legend_fontsize: float = 7,
    title_fontsize: float = 9,
    # ── output ──────────────────────────────────────────────────────
    save_path: str | Path | None = None,
    show: bool = False,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes, dict]:
    """
    Plot a KEGG metabolic graph with edges coloured by equivalence class.

    Parameters
    ----------
    g : nx.MultiDiGraph
        The metabolic graph returned by ``kegg_module_to_graph``.
        Nodes are compound IDs.  Each edge represents one gene (isoenzymes
        of the same reaction appear as parallel edges).  This function reads:

        - Node attributes: ``name`` (used when ``node_mode="labels"``; falls back to the node id)
        - Edge attributes: ``gene`` (gene symbol, may be ``None`` for unmapped reactions)
    equivalence_classes : dict or EquivalenceResults
        ``{class_id: [gene, ...]}``, mapping each equivalence class to the
        list of gene symbols it contains.  An ``EquivalenceResults`` object
        can be passed directly; its ``.edge_equivalence`` attribute is used.
    pair_scores : pd.DataFrame, optional
        DataFrame with columns ``g1``, ``g2``, ``score``.  When provided
        together with ``show_scores=True``, the MMD statistic is printed
        as a small annotation at intermediate compound nodes.

    figsize : (float, float), optional
        Figure size in inches ``(width, height)``.  When *None* (default),
        the figure is sized so that ``x_sep`` and ``y_sep`` translate
        directly to inches of inter-node spacing (1 data-unit = 1 inch
        inside the axes).  Providing an explicit ``figsize`` overrides this
        and the spacing becomes relative to the canvas size.
    dpi : int
        Resolution in dots per inch (default 300).
    x_sep : float
        Horizontal distance in data-units (= inches when ``figsize`` is
        auto) between successive layout columns.  Increase to spread the
        graph horizontally.
    y_sep : float
        Vertical distance in data-units (= inches when ``figsize`` is auto)
        between nodes in the same column.  Increase to spread the graph
        vertically.

    inter_arrow_distance : float
        Vertical offset in data-units (= inches when ``figsize`` is auto)
        used to separate multiple arrows that enter or leave the same node.
        This reduces overplotting when a node has several incoming/outgoing
        edges.  Set to 0 to disable.

    node_mode : {"none", "labels"}
        How to render metabolite nodes.  ``"none"`` draws no nodes at all.
        ``"labels"`` draws the metabolite name (``node["name"]`` or node ID)
        at each node position.
    metabolite_font_size : float
        Font size for compound / metabolite labels.
    metabolite_label_offset : (float, float)
        ``(dx, dy)`` offset in data units applied to every metabolite label
        relative to its node position.  Default ``(0.0, -0.22)`` places the
        label just below the node.

    edge_width : float
        Line width of arrow edges.
    arrow_style : str
        Matplotlib arrowstyle string (default ``"-|>"``).
    arrow_mutation : float
        ``mutation_scale`` controlling arrowhead size.
    connectionstyle : str
        Matplotlib connectionstyle string.  Default ``"arc3,rad=0.0"``
        produces straight arrows.  Use e.g. ``"arc3,rad=0.15"`` for curved.

    font_size : float
        Font size for gene-symbol labels on edges.
    label_offset : (float, float)
        ``(dx, dy)`` offset in data units applied to every edge label
        relative to the midpoint of the edge.  Use this to nudge labels
        away from overlapping arrows, e.g. ``(0.0, 0.15)``.
    isoenzyme_spacing : float
        Perpendicular separation in data-units (= inches when ``figsize``
        is auto) between parallel isoenzyme edges that share the same
        ``(substrate, product)`` node pair.  Each isoenzyme arrow and its
        label are shifted symmetrically along the unit vector orthogonal
        to the edge direction, so the separation is correct regardless of
        the edge's orientation in the layout.  Set to 0 to disable.

    show_scores : bool
        If *True* and ``pair_scores`` is provided, annotate each intermediate
        compound node with the pairwise MMD score of the flanking gene pair.
    score_fmt : str
        Python format spec for score values (default ``".2f"``).
    score_font_size : float
        Font size for score annotations.

    show_title : bool
        If *True*, render a title from the graph's ``module_id`` / ``name``
        attributes.
    show_legend : bool
        If *True*, draw a legend mapping colours to equivalence classes.
        Omitted when there are no multi-gene classes and singletons are not
        colored individually.
    legend_fontsize : float
        Font size for legend entries and title.
    title_fontsize : float
        Font size for the figure title.

    save_path : str or Path, optional
        Save the figure to this path; format inferred from the extension.
    show : bool
        If *True*, call ``plt.show()`` (skipped on non-interactive backends).
    ax : plt.Axes, optional
        Axes to draw into.  When *None* a new figure is created.

    Returns
    -------
    fig : plt.Figure
    ax  : plt.Axes
    plot_info : dict
        Metadata about the colors used in the plot.  Keys:

        ``"color_mapping"`` : dict[str, str]
            Maps every gene symbol that appears on an edge to its hex color
            as rendered in the plot.  All scored equivalence classes share
            a unified qualitative palette; control class is grey.
    """
    equiv = _resolve_equivalence(equivalence_classes)
    unscored_genes_set: set[str] = (
        equivalence_classes.unscored_genes
        if hasattr(equivalence_classes, "unscored_genes")
        else set()
    )

    # ── reverse map: gene -> class_id ────────────────────────────────
    gene_to_class: dict[str, int] = {}
    for cid, genes in equiv.items():
        for gene in genes:
            gene_to_class[gene] = cid

    # ── class -> colour ──────────────────────────────────────────────
    control_node: str | None = getattr(equivalence_classes, "control_node", None)
    control_class: int | None = gene_to_class.get(control_node) if control_node else None

    unique_classes = sorted(equiv.keys())
    unscored_classes = [
        c for c in unique_classes if len(equiv[c]) == 1 and equiv[c][0] in unscored_genes_set
    ]
    unscored_class_set = set(unscored_classes)
    scored_classes = [
        c for c in unique_classes if c not in unscored_class_set and c != control_class
    ]

    class_color: dict[int, str] = {}
    for i, cid in enumerate(scored_classes):
        class_color[cid] = _PALETTE[i % len(_PALETTE)]
    for cid in unscored_classes:
        class_color[cid] = _UNSCORED_COLOR
    if control_class is not None:
        class_color[control_class] = _CONTROL_COLOR

    # ── layout ──────────────────────────────────────────────────────
    pos = _dag_layout(g, x_sep=x_sep, y_sep=y_sep)

    # ── axis data-limits (computed once, reused for figsize and ax.*lim) ──
    # Padding is half a step so nodes never touch the axes edge.
    xs_vals = [p[0] for p in pos.values()]
    ys_vals = [p[1] for p in pos.values()]
    x_min = min(xs_vals) if xs_vals else 0.0
    x_max = max(xs_vals) if xs_vals else 0.0
    y_min = min(ys_vals) if ys_vals else 0.0
    y_max = max(ys_vals) if ys_vals else 0.0
    xlim = (x_min - x_sep * 0.5, x_max + x_sep * 0.5)
    ylim = (y_min - y_sep * 0.5, y_max + y_sep * 0.5)

    # ── figure ──────────────────────────────────────────────────────
    # Design principle: x_sep / y_sep are treated as inches-per-step so that
    # 1 data-unit == 1 inch inside the axes.  figsize = axes content + fixed
    # inch margins.  Margins are independent of graph size, so font sizes
    # (always in absolute points) are never rescaled.  tight_layout is
    # intentionally not used.
    #
    # Fixed margins in inches:
    _ml = 0.20  # left
    _mr = 0.20  # right
    _mb = 0.20  # bottom
    _mt = 0.50 if show_title else 0.20  # top (extra headroom for title text)

    if ax is None:
        if figsize is None:
            axes_w = xlim[1] - xlim[0]  # data units == inches
            axes_h = ylim[1] - ylim[0]
            figsize = (
                max(2.0, axes_w + _ml + _mr),
                max(1.5, axes_h + _mb + _mt),
            )
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # Anchor axes at the fixed inch margins so the inch-per-unit ratio is
        # preserved and font sizes stay accurate regardless of figure size.
        fig.subplots_adjust(
            left=_ml / figsize[0],
            right=1.0 - _mr / figsize[0],
            bottom=_mb / figsize[1],
            top=1.0 - _mt / figsize[1],
        )
    else:
        fig = ax.get_figure()

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── score lookup ────────────────────────────────────────────────
    score_lookup = _build_score_lookup(pair_scores) if show_scores else {}

    # ── column index per node (used to detect skip edges) ───────────
    # pos[n][0] == col * x_sep, so dividing and rounding recovers the column.
    node_col: dict = {n: round(pos[n][0] / x_sep) if x_sep > 0 else 0 for n in g.nodes()}

    # ── inter-arrow offsets (separate edges sharing endpoints) ──────
    # Offsets are in data units (== inches when figsize is auto).
    out_offset_by_edge: dict[tuple, float] = {}
    in_offset_by_edge: dict[tuple, float] = {}
    if inter_arrow_distance > 0:
        for n in g.nodes():
            outs = list(g.out_edges(n))
            if outs:
                outs_sorted = sorted(outs, key=lambda e: float(pos[e[1]][1]))
                m = len(outs_sorted)
                for i, (_, v) in enumerate(outs_sorted):
                    out_offset_by_edge[(n, v)] = (i - (m - 1) / 2) * inter_arrow_distance

            ins = list(g.in_edges(n))
            if ins:
                ins_sorted = sorted(ins, key=lambda e: float(pos[e[0]][1]))
                m = len(ins_sorted)
                for i, (u, _) in enumerate(ins_sorted):
                    in_offset_by_edge[(u, n)] = (i - (m - 1) / 2) * inter_arrow_distance

    # ── isoenzyme perpendicular offsets ─────────────────────────────
    # For each (u, v) node pair that has > 1 edge (isoenzymes), compute
    # a 2D shift along the unit perpendicular to the edge direction so
    # parallel arrows and their labels fan out symmetrically.
    # Keyed by (u, v, edge_key); zero vector for non-isoenzyme edges.
    perp_offset_by_edge: dict[tuple, np.ndarray] = {}
    iso_groups: dict[tuple, list] = defaultdict(list)
    for u, v, k in g.edges(keys=True):
        iso_groups[(u, v)].append(k)
    for (u, v), keys in iso_groups.items():
        n_iso = len(keys)
        d = np.array(pos[v], dtype=float) - np.array(pos[u], dtype=float)
        norm = float(np.linalg.norm(d))
        perp = np.array([-d[1], d[0]]) / norm if norm > 1e-9 else np.array([0.0, 1.0])
        for i, k in enumerate(keys):
            scalar = (i - (n_iso - 1) / 2) * isoenzyme_spacing
            perp_offset_by_edge[(u, v, k)] = perp * scalar

    # ── draw edges ──────────────────────────────────────────────────
    _NEUTRAL_ARROW = "#888888"
    for u, v, k, data in g.edges(keys=True, data=True):
        gene = (data.get("gene") or "").strip()
        cid = gene_to_class.get(gene) if gene else None
        if cid is not None:
            color = class_color[cid]
            alpha = 1.0
        else:
            color = _NEUTRAL_ARROW
            alpha = 1.0

        col_dist = node_col.get(v, 0) - node_col.get(u, 0)
        if col_dist > 1:
            edge_cs = f"arc3,rad={skip_edge_rad * (col_dist - 1):.2f}"
        else:
            edge_cs = connectionstyle

        perp_off = perp_offset_by_edge.get((u, v, k), np.zeros(2))
        p0 = np.array(pos[u], dtype=float) + perp_off
        p1 = np.array(pos[v], dtype=float) + perp_off
        edge_off = 0.5 * (out_offset_by_edge.get((u, v), 0.0) + in_offset_by_edge.get((u, v), 0.0))
        p0[1] += edge_off
        p1[1] += edge_off

        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops=dict[str, Any](
                arrowstyle=arrow_style,
                color=color,
                lw=edge_width,
                alpha=alpha,
                mutation_scale=arrow_mutation,
                connectionstyle=edge_cs,
                shrinkA=0.0,
                shrinkB=0.0,
            ),
        )

    # ── edge labels (gene symbols, italic per biology convention) ───
    dx, dy = label_offset
    for u, v, k, data in g.edges(keys=True, data=True):
        gene = (data.get("gene") or "").strip()
        if not gene:
            continue

        perp_off = perp_offset_by_edge.get((u, v, k), np.zeros(2))
        p0 = np.array(pos[u], dtype=float) + perp_off
        p1 = np.array(pos[v], dtype=float) + perp_off
        edge_off = 0.5 * (out_offset_by_edge.get((u, v), 0.0) + in_offset_by_edge.get((u, v), 0.0))
        p0[1] += edge_off
        p1[1] += edge_off

        x_mid = (p0[0] + p1[0]) / 2 + dx
        y_mid = (p0[1] + p1[1]) / 2 + dy

        cid = gene_to_class.get(gene)
        label_c = class_color.get(cid, _CONTROL_COLOR) if cid is not None else _CONTROL_COLOR
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

    # ── metabolite labels (optional) ────────────────────────────────
    if node_mode == "labels":
        mdx, mdy = metabolite_label_offset
        for node, ndata in g.nodes(data=True):
            label = ndata.get("name", node)
            x, y = pos[node]
            ax.text(
                x + mdx,
                y + mdy,
                label,
                fontsize=metabolite_font_size,
                color="#444444",
                ha="center",
                va="top",
                zorder=3,
            )
    elif node_mode != "none":
        raise ValueError(f"node_mode must be 'none' or 'labels', got {node_mode!r}")

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
    has_legend_entries = bool(scored_classes or control_class is not None or unscored_classes)
    if show_legend and has_legend_entries:
        handles = []
        for cid in scored_classes:
            handles.append(
                mpatches.Patch(
                    facecolor=class_color[cid], edgecolor="none", label=_class_label(cid)
                )
            )
        if control_class is not None:
            handles.append(
                mpatches.Patch(facecolor=_CONTROL_COLOR, edgecolor="none", label="no phenotype")
            )
        if unscored_classes:
            handles.append(
                mpatches.Patch(
                    facecolor=_UNSCORED_COLOR,
                    edgecolor="none",
                    label="not observed",
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

    # ── axis limits ─────────────────────────────────────────────────
    # Use the pre-computed limits (half-step padding) so the scale stays
    # consistent with the figsize calculated above.
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()

    # ── build plot_info ─────────────────────────────────────────────
    def _class_label(cid: int) -> str:
        if cid == control_class:
            return "no phenotype"
        return ", ".join(sorted(equiv[cid]))

    # gene_color_mapping: every gene that appears on an edge -> its rendered hex color.
    gene_color_mapping: dict[str, str] = {}
    for _, _, data in g.edges(data=True):
        gene = (data.get("gene") or "").strip()
        if not gene:
            continue
        cid = gene_to_class.get(gene)
        gene_color_mapping[gene] = (
            class_color.get(cid, _CONTROL_COLOR) if cid is not None else _CONTROL_COLOR
        )

    # gene_to_class_name: each gene -> human-readable equivalence class label (sorted gene list).
    gene_to_class_name: dict[str, str] = {
        gene: _class_label(cid) for gene, cid in gene_to_class.items() if gene != control_node
    }

    # class_color_mapping: equivalence class label -> color.
    class_color_mapping: dict[str, str] = {
        _class_label(cid): color for cid, color in class_color.items()
    }

    plot_info: dict = {
        "gene_color_mapping": gene_color_mapping,
        "gene_to_class": gene_to_class_name,
        "class_color_mapping": class_color_mapping,
    }

    # ── save / show ─────────────────────────────────────────────────
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show and plt.get_backend().lower() != "agg":
        plt.show()

    return fig, ax, plot_info
