"""
Metabolic pathway visualization.

Layout strategy: trunk-aware orthogonal placement.
  1. Find the longest path (trunk) → horizontal spine at y = 0.
  2. Branches alternate above / below the trunk, placed horizontally
     at y = ±1, ±2, … with vertical departure edges from the branch point.
  3. Sub-branches recurse the same logic within their y-band.

Design: "gray scaffold, colored fractures"
  – Largest equivalence class → warm gray.
  – All other classes → distinct pastel accents.
  – Metabolite nodes: colored to match edge class; neutral when ambiguous.
  – Gene names: 7 pt sans-serif along each edge.
  – All edges uniform thickness, orthogonal routing (90° only).
  – Optional open chevrons for reaction direction (``show_chevrons`` in ``path_plot``).
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# Design tokens
# ═══════════════════════════════════════════════════════════════════════════════

_GRAY_EDGE = "#BCBCBC"
_GRAY_LABEL = "#8C8C8C"
_NODE_NEUTRAL = "#BCBCBC"  # neutral dot when incident classes differ
_WHITE = "#FFFFFF"

_PALETTE = [
    "#D4A5A0",  # dusty rose
    "#9DC5B4",  # sage
    "#9BB5CC",  # powder blue
    "#CCBA8A",  # sand
    "#B5A0C4",  # lilac
    "#C4ADA0",  # warm taupe
    "#A0C4B8",  # mint
    "#C4B5A0",  # wheat
]

"""
Design Constants / Scale Parameters:
  _EDGE_WIDTH     : Uniform stroke width of all path edges.
  _NODE_RADIUS    : Radius of metabolite dots, in inches (absolute size).
  _FONT_SIZE      : Font size in exact pt.
  _LABEL_PAD      : Perpendicular offset for edge labels, in inches.
  _EDGE_X_SCALE   : Horizontal edge length between nodes, in inches.
  _EDGE_Y_SCALE   : Vertical edge length between nodes, in inches.
  _CHEVRON_SIZE   : Half-length of each chevron arm, in inches.
  _CHEVRON_POS    : Position of chevron along the segment (0 = start, 1 = end).
  _FIG_DPI        : Default figure canvas DPI (display and raster save).
"""
_EDGE_WIDTH = 1.5
_NODE_RADIUS = 0.02
_FONT_SIZE = 7.0
_LABEL_PAD = 0.05
_EDGE_X_SCALE = 0.25
_EDGE_Y_SCALE = 0.25
_CHEVRON_SIZE = 0.018
_CHEVRON_POS = 0.62
# Subtle under-stroke so the chevron separates cleanly from pale edge colors.
_CHEVRON_OUTLINE_EXTRA_LW = 0.2
_CHEVRON_OUTLINE_DARKEN = 0.22
_FIG_DPI = 600


# ═══════════════════════════════════════════════════════════════════════════════
# Font
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for c in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if c in available:
            return c
    return "sans-serif"


def _darken(hx: str, amt: float) -> str:
    r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
    return f"#{int(r*(1-amt)):02x}{int(g*(1-amt)):02x}{int(b*(1-amt)):02x}"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph utilities
# ═══════════════════════════════════════════════════════════════════════════════


def _build_adj(
    edges: list[tuple[str, str, str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (forward adj, reverse adj) ignoring gene labels."""
    fwd: dict[str, list[str]] = defaultdict(list)
    rev: dict[str, list[str]] = defaultdict(list)
    for src, tgt, _ in edges:
        fwd[src].append(tgt)
        rev[tgt].append(src)
    return dict(fwd), dict(rev)


def _find_longest_path(
    nodes: Sequence[str],
    edges: list[tuple[str, str, str]],
) -> list[str]:
    """
    Longest path in a DAG via dynamic programming.
    Returns the ordered list of nodes on the longest path.
    """
    fwd, rev = _build_adj(edges)

    # topological sort (Kahn's)
    in_deg = defaultdict(int)
    for n in nodes:
        in_deg[n] = 0
    for _, tgt, _ in edges:
        in_deg[tgt] += 1

    queue = [n for n in nodes if in_deg[n] == 0]
    topo = []
    while queue:
        # pick alphabetically for determinism
        queue.sort()
        n = queue.pop(0)
        topo.append(n)
        for nb in fwd.get(n, []):
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)

    # DP: longest path ending at each node
    dist: dict[str, int] = {n: 0 for n in nodes}
    parent: dict[str, str | None] = {n: None for n in nodes}
    for n in topo:
        for nb in fwd.get(n, []):
            if dist[n] + 1 > dist[nb]:
                dist[nb] = dist[n] + 1
                parent[nb] = n

    # backtrack from the node with max dist
    end = max(nodes, key=lambda n: dist[n])
    path = []
    cur: str | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _collect_subtree(
    root: str,
    edges: list[tuple[str, str, str]],
    exclude: set[str],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """BFS from root, collecting all reachable nodes/edges not in exclude."""
    fwd: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for s, t, g in edges:
        fwd[s].append((t, g))

    visited = {root}
    queue = [root]
    sub_nodes = [root]
    sub_edges: list[tuple[str, str, str]] = []

    while queue:
        n = queue.pop(0)
        for nb, gene in fwd.get(n, []):
            if nb not in visited and nb not in exclude:
                visited.add(nb)
                sub_nodes.append(nb)
                sub_edges.append((n, nb, gene))
                queue.append(nb)
    return sub_nodes, sub_edges


# ═══════════════════════════════════════════════════════════════════════════════
# Node color assignment
# ═══════════════════════════════════════════════════════════════════════════════


def _assign_node_colors(
    nodes: Sequence[str],
    edges: list[tuple[str, str, str]],
    gene_to_edge_color: dict[str, str],
) -> dict[str, str]:
    """
    Determine each node's dot color from incident edges.

    Rule:
      - Collect the set of edge-class colors for all edges touching
        the node (incoming + outgoing).
      - If every incident edge shares a single color → use that color.
      - If the node sits at a boundary between classes → neutral gray.
      - Isolated nodes (no incident edges) → neutral gray.
    """
    node_colors_incident: dict[str, set[str]] = {n: set() for n in nodes}
    for src, tgt, gene in edges:
        c = gene_to_edge_color.get(gene, _GRAY_EDGE)
        node_colors_incident[src].add(c)
        node_colors_incident[tgt].add(c)

    result: dict[str, str] = {}
    for n in nodes:
        colors = node_colors_incident[n]
        if len(colors) == 1:
            result[n] = next(iter(colors))
        else:
            result[n] = _NODE_NEUTRAL
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Trunk-aware layout
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_layout(
    nodes: Sequence[str],
    edges: list[tuple[str, str, str]],
    edge_x_scale: float,
    edge_y_scale: float,
) -> dict[str, tuple[float, float]]:
    """
    Trunk-aware orthogonal layout.

    1. Longest path → horizontal trunk at y = 0.
    2. Branches off the trunk alternate above/below.
    3. Sub-branches recurse within their y-band.
    """
    positions: dict[str, tuple[float, float]] = {}

    def _layout_spine(
        spine_nodes: list[str],
        spine_edges: list[tuple[str, str, str]],
        all_edges: list[tuple[str, str, str]],
        start_col: int,
        y_base: float,
        branch_dir: int,  # +1 = next branch goes up, -1 = down
        depth: int,
    ) -> int:
        """
        Place a spine (longest path of a sub-DAG) horizontally,
        then recurse into its branches.

        Returns the next available column.
        """
        trunk = _find_longest_path(spine_nodes, spine_edges)
        trunk_set = set(trunk)

        # place trunk nodes
        trunk_col = {}
        for i, n in enumerate(trunk):
            col = start_col + i
            positions[n] = (col * edge_x_scale, y_base)
            trunk_col[n] = col
        next_col = start_col + len(trunk)

        # identify branch departure edges
        # (edges from a trunk node to a non-trunk node)
        fwd: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for s, t, g in spine_edges:
            fwd[s].append((t, g))

        direction = branch_dir
        y_step = edge_y_scale * (0.8 + 0.2 * depth)

        for trunk_node in trunk:
            branch_targets = [(t, g) for t, g in fwd.get(trunk_node, []) if t not in trunk_set]
            for bt, bg in branch_targets:
                sub_nodes, sub_edges = _collect_subtree(
                    bt,
                    spine_edges,
                    exclude=trunk_set,
                )

                if not sub_nodes:
                    continue

                branch_y = y_base + direction * y_step

                full_sub_edges = [(trunk_node, bt, bg)] + sub_edges
                full_sub_nodes = [trunk_node] + sub_nodes

                sub_spine = _find_longest_path(sub_nodes, sub_edges)

                branch_start_col = trunk_col[trunk_node] + 1
                sub_trunk_set = set(sub_spine)

                for j, sn in enumerate(sub_spine):
                    c = branch_start_col + j
                    positions[sn] = (c * edge_x_scale, branch_y)
                    trunk_col[sn] = c

                sub_fwd: dict[str, list[tuple[str, str]]] = defaultdict(list)
                for s, t, g in sub_edges:
                    sub_fwd[s].append((t, g))

                sub_direction = direction
                for sn in sub_spine:
                    sub_branch_targets = [
                        (t, g)
                        for t, g in sub_fwd.get(sn, [])
                        if t not in sub_trunk_set and t not in positions
                    ]
                    for sbt, sbg in sub_branch_targets:
                        sub2_nodes, sub2_edges = _collect_subtree(
                            sbt,
                            sub_edges,
                            exclude=sub_trunk_set,
                        )
                        sub2_y = branch_y + sub_direction * y_step
                        sub2_col = trunk_col[sn] + 1
                        sub2_spine = _find_longest_path(sub2_nodes, sub2_edges)
                        for k, s2n in enumerate(sub2_spine):
                            positions[s2n] = (
                                (sub2_col + k) * edge_x_scale,
                                sub2_y,
                            )
                        sub_direction *= -1

                direction *= -1

        return next_col

    _layout_spine(
        list(nodes),
        list(edges),
        list(edges),
        start_col=0,
        y_base=0.0,
        branch_dir=1,
        depth=0,
    )

    max_col = max((p[0] for p in positions.values()), default=0) / edge_x_scale
    for n in nodes:
        if n not in positions:
            max_col += 1
            positions[n] = (max_col * edge_x_scale, 0)

    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# Orthogonal edge routing
# ═══════════════════════════════════════════════════════════════════════════════


def _route_edge(
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> list[tuple[float, float]]:
    """
    Route from p0 to p1 with at most two 90° bends.

    Strategy:
      - Same row → straight horizontal.
      - Same col → straight vertical.
      - Otherwise → vertical-first L-bend (depart vertically, then horizontal).
    """
    x0, y0 = p0
    x1, y1 = p1

    if abs(y0 - y1) < 1e-6:
        return [p0, p1]
    if abs(x0 - x1) < 1e-6:
        return [p0, p1]

    return [(x0, y0), (x0, y1), (x1, y1)]


# ═══════════════════════════════════════════════════════════════════════════════
# Direction chevrons
# ═══════════════════════════════════════════════════════════════════════════════


def _draw_chevron(
    ax: plt.Axes,
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
    color: str,
    size: float = _CHEVRON_SIZE,
    position: float = _CHEVRON_POS,
    linewidth: float = 0.8,
) -> None:
    """
    Draw a small open chevron (>) on a straight segment to indicate
    reaction direction.

    The chevron is placed at *position* (0–1) along the segment.
    Two short arms open backward relative to the direction of travel.

    Parameters
    ----------
    seg_start, seg_end : (x, y)
        Endpoints of the straight segment (horizontal or vertical).
    color : str
        Stroke color; matches the edge.
    size : float
        Half-length of each chevron arm in inches.
    position : float
        Fractional position along the segment (0 = start, 1 = end).
    linewidth : float
        Stroke width of the chevron lines.
    """
    x0, y0 = seg_start
    x1, y1 = seg_end
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)

    if length < 1e-9:
        return

    # unit direction and perpendicular
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux  # 90° CCW

    # tip of the chevron
    tx = x0 + position * dx
    ty = y0 + position * dy

    # two arm endpoints: step backward along direction, outward along normal
    arm1 = (tx - size * ux + size * nx, ty - size * uy + size * ny)
    arm2 = (tx - size * ux - size * nx, ty - size * uy - size * ny)

    xs = [arm1[0], tx, arm2[0]]
    ys = [arm1[1], ty, arm2[1]]
    ax.plot(
        xs,
        ys,
        color=_darken(color, _CHEVRON_OUTLINE_DARKEN),
        linewidth=linewidth + _CHEVRON_OUTLINE_EXTRA_LW,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Color assignment
# ═══════════════════════════════════════════════════════════════════════════════


def _assign_colors(
    equivalence_classes: Sequence[set[str]],
    class_colors: Sequence[str] | None = None,
) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}

    if class_colors is None:
        sorted_cls = sorted(equivalence_classes, key=len, reverse=True)
        boring = sorted_cls[0]
        others = sorted_cls[1:]

        for g in boring:
            mapping[g] = (_GRAY_EDGE, _GRAY_LABEL)

        for i, cls in enumerate(others):
            c = _PALETTE[i % len(_PALETTE)]
            lc = _darken(c, 0.25)
            for g in cls:
                mapping[g] = (c, lc)
    else:
        for cls, c in zip(equivalence_classes, class_colors):
            lc = _darken(c, 0.25)
            for g in cls:
                mapping[g] = (c, lc)

    return mapping


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def path_plot(
    nodes: Sequence[str],
    edges: list[tuple[str, str, str]],
    equivalence_classes: Sequence[set[str]],
    *,
    class_colors: Sequence[str] | None = None,
    edge_width: float = _EDGE_WIDTH,
    node_radius: float = _NODE_RADIUS,
    font_size: float = _FONT_SIZE,
    label_pad: float = _LABEL_PAD,
    chevron_size: float = _CHEVRON_SIZE,
    chevron_pos: float = _CHEVRON_POS,
    edge_x_scale: float = _EDGE_X_SCALE,
    edge_y_scale: float = _EDGE_Y_SCALE,
    save_path: str | Path | None = None,
    dpi: int = _FIG_DPI,
    show: bool = False,
    show_chevrons: bool = True,
) -> plt.Figure:
    """
    Draw a metabolic pathway graph with equivalence-class coloring.

    Parameters
    ----------
    nodes : list[str]
        Metabolite node identifiers (not displayed).
    edges : list of (source, target, gene_name)
        Directed edges; each represents a gene/reaction.
    equivalence_classes : list[set[str]]
        Partition of gene names.  Largest class → gray;
        others → pastel accent colors.
    edge_width : float
        Stroke width of all path edges.
    node_radius : float
        Radius of metabolite dots in inches.
    font_size : float
        Font size in pt for gene labels.
    label_pad : float
        Perpendicular offset for edge labels in inches.
    chevron_size : float
        Half-length of each chevron arm in inches.
    chevron_pos : float
        Position of chevron along segment (0 = start, 1 = end).
    edge_x_scale : float
        Horizontal edge length in inches.
    edge_y_scale : float
        Vertical edge length in inches.
    dpi : int
        Figure canvas DPI (interactive display and raster ``savefig``).
    show_chevrons : bool
        If True, draw direction chevrons on edges. Set False for undirected pathways.
    """
    font_family = _resolve_font()
    pos = _compute_layout(nodes, edges, edge_x_scale, edge_y_scale)
    color_map = _assign_colors(equivalence_classes, class_colors)

    # build gene → edge color lookup for node coloring
    gene_to_edge_color: dict[str, str] = {gene: ec for gene, (ec, _lc) in color_map.items()}
    node_color_map = _assign_node_colors(nodes, edges, gene_to_edge_color)

    # figure sizing
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    x_lo, x_hi = min(all_x), max(all_x)
    y_lo, y_hi = min(all_y), max(all_y)

    pad_x = edge_x_scale * 0.55
    pad_y = edge_y_scale * 0.55
    data_w = (x_hi - x_lo) + 2 * pad_x
    data_h = (y_hi - y_lo) + 2 * pad_y

    fig_w = max(data_w, 1.0)
    fig_h = max(data_h, 1.0)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(_WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(_WHITE)
    ax.set_xlim(x_lo - pad_x, x_lo - pad_x + fig_w)
    ax.set_ylim(y_lo - pad_y, y_lo - pad_y + fig_h)
    ax.axis("off")

    # ── draw edges + chevrons ────────────────────────────────────────────
    for src, tgt, gene in edges:
        p0, p1 = pos[src], pos[tgt]
        wp = _route_edge(p0, p1)
        ec, lc = color_map.get(gene, (_GRAY_EDGE, _GRAY_LABEL))

        ax.plot(
            [w[0] for w in wp],
            [w[1] for w in wp],
            color=ec,
            linewidth=edge_width,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )

        # find the longest segment for label + chevron placement
        lengths = [
            math.hypot(wp[i + 1][0] - wp[i][0], wp[i + 1][1] - wp[i][1]) for i in range(len(wp) - 1)
        ]
        best_seg = int(np.argmax(lengths))

        # ── chevron on the longest segment ───────────────────────────
        if show_chevrons:
            _draw_chevron(
                ax,
                wp[best_seg],
                wp[best_seg + 1],
                color=ec,
                size=chevron_size,
                position=chevron_pos,
            )

        # ── label on the longest segment ─────────────────────────────
        mx = (wp[best_seg][0] + wp[best_seg + 1][0]) / 2
        my = (wp[best_seg][1] + wp[best_seg + 1][1]) / 2
        sdx = wp[best_seg + 1][0] - wp[best_seg][0]
        sdy = wp[best_seg + 1][1] - wp[best_seg][1]

        if abs(sdy) > abs(sdx):
            # vertical segment → label to the right
            lx, ly = mx + label_pad, my
            ha, va = "left", "center"
        else:
            # horizontal segment → label above
            lx, ly = mx, my + label_pad
            ha, va = "center", "bottom"

        ax.text(
            lx,
            ly,
            gene,
            fontsize=font_size,
            fontfamily=font_family,
            color=lc,
            ha=ha,
            va=va,
            zorder=4,
        )

    # ── draw nodes ───────────────────────────────────────────────────────
    for n in nodes:
        x, y = pos[n]
        nc = node_color_map[n]
        circle = plt.Circle(
            (x, y),
            radius=node_radius,
            facecolor=nc,
            edgecolor=nc,
            linewidth=0,
            zorder=5,
        )
        ax.add_patch(circle)

    # ── legend ───────────────────────────────────────────────────────────
    if class_colors is None:
        sorted_cls = sorted(equivalence_classes, key=len, reverse=True)
        others = sorted_cls[1:]

        handles = [
            mlines.Line2D(
                [0], [0], color=_GRAY_EDGE, lw=edge_width, solid_capstyle="round", label="expected"
            ),
        ]
        for i, cls in enumerate(others):
            c = _PALETTE[i % len(_PALETTE)]
            lab = ", ".join(sorted(cls)) if len(cls) <= 3 else f"{len(cls)} genes"
            handles.append(
                mlines.Line2D([0], [0], color=c, lw=edge_width, solid_capstyle="round", label=lab)
            )
    else:
        handles = []
        for cls, c in zip(equivalence_classes, class_colors):
            lab = ", ".join(sorted(cls)) if len(cls) <= 3 else f"{len(cls)} genes"
            handles.append(
                mlines.Line2D([0], [0], color=c, lw=edge_width, solid_capstyle="round", label=lab)
            )

    # leg = ax.legend(
    #     handles=handles,
    #     loc="lower right", bbox_to_anchor=(1.02, -0.02),
    #     ncol=min(len(handles), 3),
    #     frameon=False, fontsize=5.5,
    #     handlelength=1.2, columnspacing=0.8,
    #     prop={"family": font_family, "size": 5.5},
    # )
    # for t in leg.get_texts():
    #     t.set_color("#777777")

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor=_WHITE,
        )
    if show:
        plt.show()
    return fig
