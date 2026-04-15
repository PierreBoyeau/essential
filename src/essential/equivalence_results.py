from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import pandas as pd

from essential.plot_temp import (
    _CHEVRON_POS,
    _CHEVRON_SIZE,
    _EDGE_WIDTH,
    _EDGE_X_SCALE,
    _EDGE_Y_SCALE,
    _FIG_DPI,
    _FONT_SIZE,
    _LABEL_PAD,
    _NODE_RADIUS,
    path_plot,
)


@dataclass
class EquivalenceResults:
    """
    Canonical storage: ``edge_equivalence`` maps a component id to the list of genes in that class.

    Use ``gene_to_class_id`` for the reverse map.
    ``metabolic_graph`` is stored from ``PathwayDiscontinuity.fit()`` so that ``plot()``
    requires no graph argument.
    """

    edge_equivalence: dict[int, list[str]]
    gene_pair_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    metabolic_graph: Optional[nx.Graph | nx.DiGraph] = field(default=None, repr=False)

    @property
    def gene_to_class_id(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for cid, genes in self.edge_equivalence.items():
            for gene in genes:
                out[gene] = cid
        return out

    def plot(
        self,
        directed_graph: nx.DiGraph | None = None,
        *,
        edge_width: float = _EDGE_WIDTH,
        node_radius: float = _NODE_RADIUS,
        font_size: float = _FONT_SIZE,
        label_pad: float = _LABEL_PAD,
        chevron_size: float = _CHEVRON_SIZE,
        chevron_pos: float = _CHEVRON_POS,
        edge_x_scale: float = _EDGE_X_SCALE,
        edge_y_scale: float = _EDGE_Y_SCALE,
        dpi: int = _FIG_DPI,
        save_path: str | Path | None = None,
        show: bool = False,
        show_chevrons: bool = True,
    ):
        """
        Plot the equivalence classes overlaid on the metabolic pathway.

        Parameters
        ----------
        directed_graph :
            The metabolic graph used for layout. If ``None``,
            falls back to the graph stored at fit time (``self.metabolic_graph``).
        edge_width :
            Stroke width of all path edges.
        node_radius :
            Radius of metabolite dots in inches.
        font_size :
            Font size in pt for gene labels.
        label_pad :
            Perpendicular offset for edge labels in inches.
        chevron_size :
            Half-length of each chevron arm in inches.
        chevron_pos :
            Position of chevron along segment (0 = start, 1 = end).
        edge_x_scale :
            Horizontal edge length in inches.
        edge_y_scale :
            Vertical edge length in inches.
        dpi :
            Output resolution for raster images.
        save_path :
            If provided, save the figure to this path.
        show :
            If True, call ``plt.show()``.
        show_chevrons :
            If False, omit edge direction chevrons (e.g. undirected ``Graph``).
        """
        g = directed_graph if directed_graph is not None else self.metabolic_graph
        if g is None:
            raise ValueError("No graph available. Pass directed_graph or fit() first.")
        nodes = [str(n) for n in g.nodes()]
        edges = [
            (str(u), str(v), data.get("name") or data.get("gene") or f"{u}-{v}")
            for u, v, data in g.edges(data=True)
        ]
        equivalence_classes = [set(genes) for genes in self.edge_equivalence.values()]

        return path_plot(
            nodes,
            edges,
            equivalence_classes,
            edge_width=edge_width,
            node_radius=node_radius,
            font_size=font_size,
            label_pad=label_pad,
            chevron_size=chevron_size,
            chevron_pos=chevron_pos,
            edge_x_scale=edge_x_scale,
            edge_y_scale=edge_y_scale,
            dpi=dpi,
            save_path=save_path,
            show=show,
            show_chevrons=show_chevrons,
        )
