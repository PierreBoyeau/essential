"""
KEGG REST API helpers: fetch pathway lists and convert KGML to NetworkX graphs.

Public API
----------
list_kegg_pathways(organism)
    Return all pathway IDs and titles for a KEGG organism code.

kegg_pathway_to_graph(pathway_id, organism, backbone=False)
    Fetch a single KEGG pathway and return a ``DiGraph`` where nodes are
    compound KEGG IDs and directed edges are enzymatic reactions annotated
    with the catalyzing gene.

Notes
-----
- KEGG API terms: academic use only; ≤ 3 requests/second.
- KGML is fetched from ``https://rest.kegg.jp/get/{pathway_id}/kgml``.
- Responses are cached on disk under ``cache_dir`` (default ``~/.cache/kegg``)
  to avoid redundant network calls.
- KEGG pathway maps already exclude currency metabolites (ATP, NADH, …);
  only structural metabolites that define pathway topology appear as compound
  nodes in KGML.
- Most reactions in a pathway map have one substrate and one product.
  A small number are genuinely multi-product or multi-substrate (e.g. aldolase
  splits fructose-1,6-bisphosphate into DHAP and G3P).  The Cartesian product
  of substrates × products is used; each resulting edge is a valid metabolic
  transformation catalyzed by the annotated gene.
- When multiple genes (isoenzymes) catalyze the same reaction, each gene
  yields a separate edge.  Because ``nx.DiGraph`` allows only one edge per
  (u, v) pair, the last gene written wins for a given substrate-product pair.
  Use ``return_multigraph=True`` to get an ``nx.MultiDiGraph`` that preserves
  all isoenzymes.

Chain-connectivity backbone extraction (``backbone=True``)
----------------------------------------------------------
Metabolic reactions in KGML often involve multiple substrates and products.
The Cartesian product of substrates × products creates edges that, while
biochemically valid, obscure the main metabolic flow.  For example, an
acyl transfer reaction::

    acyl-ACP + lipid-intermediate  →  ACP + acylated-intermediate

generates four edges, but only one (lipid-intermediate → acylated-intermediate)
traces the pathway backbone.  The other three involve auxiliary compounds
(the acyl donor entering from outside the pathway, the released carrier
leaving to nowhere within it).

The ``backbone=True`` option applies three sequential post-processing steps
to produce a clean graph suitable for ``PathwayDiscontinuity``:

**Step 1: Chain-connectivity filtering.**

Retains only "chain" compounds using a graph-connectivity heuristic that
requires no external biochemical knowledge beyond what KEGG already encodes
in the pathway's reaction topology:

1. **Collect connectivity sets** across all reactions in the pathway:

   - ``produced``: compounds appearing as a ``<product>`` in any reaction.
   - ``consumed``: compounds appearing as a ``<substrate>`` in any reaction.

2. **Filter multi-substrate reactions.**  A substrate is "on-chain" if it
   appears in ``produced`` — meaning some upstream reaction manufactures
   it within this pathway.  Substrates absent from ``produced`` enter from
   outside (cofactors, donors) and are classified as auxiliary.

3. **Filter multi-product reactions.**  A product is "on-chain" if it
   appears in ``consumed`` — meaning some downstream reaction uses it.
   Products absent from ``consumed`` leave to nowhere (released groups,
   by-products) and are classified as auxiliary.

4. **Fallback.**  If the filter would remove *all* substrates or *all*
   products of a reaction (e.g. at the pathway entry point where no
   substrate is produced internally), no filtering is applied to that
   side — all compounds are retained.

5. **Build edges** only between the retained (chain) substrates and
   products.

Single-substrate / single-product reactions (1→1) are never filtered.

**Step 2: Consecutive-edge merging.**

When the same gene catalyzes two or more sequential reactions whose edges
form a directed path (A → B → C), and the intermediate node B has no other
connections from other genes, the path is collapsed into a single edge
(A → C, removing B).  This handles enzymes that perform repeated additions
(e.g. waaA in LPS biosynthesis performing two KDO additions:
C04919 → C06024 → C06025 becomes C04919 → C06025).

If an intermediate is shared with other genes (a metabolic junction), no
merging occurs — the edges are left for greedy deduplication in step 3.

**Step 3: Gene deduplication.**

After chain-connectivity filtering, a gene may still label multiple edges.
Two cases cause this:

- *Same reaction, stoichiometric artifact.*  A multi-substrate reaction
  where multiple on-chain substrates pass the filter (e.g. lpxB in
  eco00540: both C04652 and C04824 are on-chain, creating a shortcut
  edge alongside the real sequential path).

- *Different reactions.*  The same gene catalyzes distinct KEGG reactions
  within one pathway (e.g. lpxL catalyzing R05146 on the main chain and
  R12193 on a variant branch; or thiG at a convergence of three
  independent branches in thiamine biosynthesis).

For ``PathwayDiscontinuity``, each gene must map to exactly one edge so
that the line graph produces well-defined consecutive gene pairs.  The
deduplication heuristic is:

1. Collect all genes that label more than one edge.
2. Process them greedily, least-ambiguous first (fewest edges).
3. For each gene, try keeping each of its edges as the sole survivor.
   Score each candidate by the number of nodes in the largest weakly
   connected component of the graph after removing the other edges.
4. Keep the edge with the highest score.  Mark it with
   ``pruned_alternatives = k`` (the number of removed sibling edges)
   so downstream analysis knows a choice was made.

**Step 4: Largest weakly connected component.**

After deduplication, disconnected fragments (variant branches, boundary
reactions with no topological context) may remain.  Only the largest
weakly connected component is retained.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
import requests

_KEGG_BASE = "https://rest.kegg.jp"
_REQUEST_DELAY = 0.35  # KEGG requests ≤ 3 req/s


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _fetch(url: str, cache_dir: Optional[Path]) -> str:
    """GET ``url``, optionally reading/writing a plain-text cache."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = url.replace(_KEGG_BASE, "").lstrip("/").replace("/", "_")
        cache_file = cache_dir / cache_key
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(_REQUEST_DELAY)
    text = resp.text

    if cache_dir is not None:
        cache_file.write_text(text, encoding="utf-8")

    return text


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def list_kegg_pathways(
    organism: str = "eco",
    *,
    cache_dir: Optional[Path] = Path.home() / ".cache" / "kegg",
) -> list[dict[str, str]]:
    """
    Return all KEGG pathways for ``organism`` as a list of dicts with keys
    ``pathway_id`` and ``title``.

    Parameters
    ----------
    organism :
        Three- or four-letter KEGG organism code.  Examples: ``"eco"``
        (*E. coli* K-12), ``"hsa"`` (*Homo sapiens*), ``"mmu"`` (*Mus
        musculus*).
    cache_dir :
        Directory for caching raw API responses.  Pass ``None`` to disable.
    """
    text = _fetch(f"{_KEGG_BASE}/list/pathway/{organism}", cache_dir)
    records = []
    for line in text.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            records.append({"pathway_id": parts[0], "title": parts[1]})
    return records


def kegg_pathway_to_graph(
    pathway_id: str,
    organism: str = "eco",
    *,
    backbone: bool = False,
    return_multigraph: bool = False,
    cache_dir: Optional[Path] = Path.home() / ".cache" / "kegg",
) -> nx.DiGraph | nx.MultiDiGraph:
    """
    Fetch a KEGG pathway and return a directed metabolic graph.

    Nodes are compound KEGG IDs (e.g., ``"cpd:C00022"``).  Node attribute
    ``name`` holds the human-readable compound name from KGML.

    Each edge represents one enzymatic reaction step and carries:

    - ``gene``      : gene symbol from the KGML graphics element (e.g., ``"pykF"``).
    - ``gene_id``   : KEGG gene ID (e.g., ``"eco:b1676"``).
    - ``reaction``  : KEGG reaction ID (e.g., ``"rn:R00200"``).
    - ``reversible``: ``True`` if KGML marks the reaction as reversible.

    When ``backbone=True``, the following additional edge attribute may
    appear:

    - ``pruned_alternatives``: number of sibling edges removed during
      gene deduplication.  Present only on edges where a choice was made.

    Reversible reactions produce edges in both directions.

    Parameters
    ----------
    pathway_id :
        Pathway ID in any of: ``"eco00010"``, ``"00010"``, or
        ``"path:eco00010"``.  When only a number is given, ``organism``
        is prepended.
    organism :
        KEGG organism code used when ``pathway_id`` contains no prefix.
    backbone :
        If ``True``, apply the four-step backbone extraction pipeline:
        chain-connectivity filtering, consecutive-edge merging, gene
        deduplication, and largest weakly connected component.  See module
        docstring for details.
    return_multigraph :
        If ``True``, return an ``nx.MultiDiGraph`` so that multiple
        isoenzymes catalyzing the same (substrate, product) pair are all
        preserved as separate edges.
    cache_dir :
        Directory for caching KGML responses.  Pass ``None`` to disable.

    Returns
    -------
    nx.DiGraph or nx.MultiDiGraph
    """
    pid = _normalize_pathway_id(pathway_id, organism)
    kgml_text = _fetch(f"{_KEGG_BASE}/get/{pid}/kgml", cache_dir)
    return _kgml_to_graph(kgml_text, multigraph=return_multigraph, backbone=backbone)


# ---------------------------------------------------------------------------
# Internal: backbone pipeline
# ---------------------------------------------------------------------------


def _compute_chain_sets(
    parsed_reactions: list[dict],
) -> tuple[set[str], set[str]]:
    """
    Across all reactions in the pathway, collect:

    - ``produced`` : compounds appearing as a product in any reaction.
    - ``consumed`` : compounds appearing as a substrate in any reaction.

    These sets are used by ``_filter_to_backbone`` to distinguish chain
    compounds from auxiliary ones.
    """
    produced: set[str] = set()
    consumed: set[str] = set()
    for rxn in parsed_reactions:
        produced.update(rxn["products"])
        consumed.update(rxn["substrates"])
    return produced, consumed


def _filter_to_backbone(
    substrates: list[str],
    products: list[str],
    produced: set[str],
    consumed: set[str],
) -> tuple[list[str], list[str]]:
    """
    For a single reaction, retain only chain compounds.

    A substrate is on-chain if it is produced by some other reaction in the
    pathway (it has an upstream connection).  A product is on-chain if it is
    consumed by some other reaction (it has a downstream connection).

    If the filter would eliminate *every* substrate or *every* product
    (e.g. at a pathway boundary), no filtering is applied to that side.
    Single-substrate or single-product sides are never filtered.
    """
    filtered_substrates = substrates
    filtered_products = products

    if len(substrates) > 1:
        chain = [s for s in substrates if s in produced]
        if chain:
            filtered_substrates = chain

    if len(products) > 1:
        chain = [p for p in products if p in consumed]
        if chain:
            filtered_products = chain

    return filtered_substrates, filtered_products


def _merge_consecutive_gene_edges(g: nx.DiGraph) -> set[str]:
    """
    Merge consecutive edges labeled with the same gene into a single edge.

    When the same gene catalyzes two (or more) sequential reactions whose
    edges form a directed path A → B → C in the graph, and the intermediate
    node B has **no other incident edges** from other genes, the path is
    collapsed into a single edge A → C.

    This handles the pattern where one enzyme catalyzes repeated additions
    (e.g. waaA performing two sequential KDO additions in LPS biosynthesis:
    C04919 → C06024 → C06025 becomes C04919 → C06025).

    If an intermediate node is shared with other genes (it's a junction),
    no merging occurs — the gene edges are left for greedy deduplication.

    The merged edge carries the first reaction's data, plus:
    - ``merged_intermediates``: list of removed intermediate node IDs.
    - ``merged_reactions``: list of all reaction IDs that were collapsed.

    Parameters
    ----------
    g : nx.DiGraph
        Modified **in place**.

    Returns
    -------
    set[str]
        Gene names whose edges were merged.
    """
    gene_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for u, v, data in g.edges(data=True):
        gene_edges[data.get("gene", "")].append((u, v))

    merged_genes: set[str] = set()

    for gene, edges in gene_edges.items():
        if len(edges) <= 1:
            continue

        # Build a mini digraph of just this gene's edges
        edge_graph = nx.DiGraph()
        for u, v in edges:
            edge_graph.add_edge(u, v)

        # Find directed chains: walk from sources (in-degree 0 in edge_graph)
        for start in [n for n in edge_graph.nodes() if edge_graph.in_degree(n) == 0]:
            chain = [start]
            current = start
            while edge_graph.out_degree(current) == 1:
                nxt = list(edge_graph.successors(current))[0]
                # Stop if the next node has fan-in within this gene's edges
                if edge_graph.in_degree(nxt) > 1:
                    chain.append(nxt)
                    break
                chain.append(nxt)
                current = nxt

            if len(chain) < 3:
                # Need at least A → B → C (3 nodes, 2 edges) to have
                # an intermediate worth merging
                continue

            # Check that every intermediate has no other gene's edges
            intermediates = chain[1:-1]
            safe_to_merge = True
            for node in intermediates:
                other_edges = [
                    (u, v) for u, v, d in g.in_edges(node, data=True) if d.get("gene") != gene
                ] + [(u, v) for u, v, d in g.out_edges(node, data=True) if d.get("gene") != gene]
                if other_edges:
                    safe_to_merge = False
                    break

            if not safe_to_merge:
                continue

            # Merge: collect data from all edges in the chain
            first, last = chain[0], chain[-1]
            merged_rxns = []
            for i in range(len(chain) - 1):
                merged_rxns.append(g[chain[i]][chain[i + 1]].get("reaction", ""))

            # Use first edge's data as base
            new_data = g[chain[0]][chain[1]].copy()
            new_data["merged_intermediates"] = list(intermediates)
            new_data["merged_reactions"] = merged_rxns

            # Remove all chain edges
            for i in range(len(chain) - 1):
                if g.has_edge(chain[i], chain[i + 1]):
                    g.remove_edge(chain[i], chain[i + 1])

            # Remove orphaned intermediate nodes
            for node in intermediates:
                if g.in_degree(node) == 0 and g.out_degree(node) == 0:
                    g.remove_node(node)

            # Add spanning edge
            g.add_edge(first, last, **new_data)
            merged_genes.add(gene)

    return merged_genes


def _deduplicate_genes(g: nx.DiGraph) -> set[str]:
    """
    Ensure each gene name labels exactly one edge in ``g`` (modified in place).

    For genes appearing on multiple edges, greedily keep the edge whose
    retention (with all sibling edges of that gene removed) maximizes the
    number of nodes in the largest weakly connected component.

    Genes with fewer duplicate edges are processed first so that their
    (less ambiguous) resolution informs the connectivity landscape for
    the more ambiguous genes that follow.

    The surviving edge is annotated with ``pruned_alternatives = k``
    where *k* is the number of removed sibling edges.

    Parameters
    ----------
    g : nx.DiGraph
        Modified **in place**.

    Returns
    -------
    set[str]
        Gene names that underwent deduplication.
    """
    # --- build gene → list of (u, v) edges --------------------------------
    gene_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for u, v, data in g.edges(data=True):
        gene_edges[data.get("gene", "")].append((u, v))

    multi_genes = {gene: edges for gene, edges in gene_edges.items() if len(edges) > 1}
    if not multi_genes:
        return set()

    # --- greedy: process least-ambiguous genes first ----------------------
    processing_order = sorted(multi_genes, key=lambda gene: len(multi_genes[gene]))

    pruned_genes: set[str] = set()

    for gene in processing_order:
        # re-collect edges: earlier rounds may have altered the graph
        current_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("gene") == gene]
        if len(current_edges) <= 1:
            continue

        best_edge: tuple[str, str] | None = None
        best_score = -1

        for candidate in current_edges:
            # temporarily remove all sibling edges, keep only candidate
            removed_data: list[tuple[str, str, dict]] = []
            for sibling in current_edges:
                if sibling != candidate:
                    removed_data.append((sibling[0], sibling[1], g[sibling[0]][sibling[1]].copy()))
                    g.remove_edge(*sibling)

            # score: size of largest weakly connected component
            score = len(max(nx.weakly_connected_components(g), key=len))

            # restore removed edges
            for u, v, data in removed_data:
                g.add_edge(u, v, **data)

            if score > best_score:
                best_score = score
                best_edge = candidate

        # permanently remove all but best
        n_removed = 0
        for edge in current_edges:
            if edge != best_edge:
                g.remove_edge(*edge)
                n_removed += 1

        # annotate the survivor
        g[best_edge[0]][best_edge[1]]["pruned_alternatives"] = n_removed
        pruned_genes.add(gene)

    # --- clean up orphan nodes (no incident edges) ------------------------
    orphans = [n for n in g.nodes() if g.in_degree(n) == 0 and g.out_degree(n) == 0]
    g.remove_nodes_from(orphans)

    return pruned_genes


def _largest_wcc(g: nx.DiGraph) -> nx.DiGraph:
    """
    Return the subgraph induced by the largest weakly connected component.

    If the graph is already connected, returns it unchanged.
    """
    if nx.is_weakly_connected(g):
        return g
    largest = max(nx.weakly_connected_components(g), key=len)
    return g.subgraph(largest).copy()


# ---------------------------------------------------------------------------
# Internal: KGML parsing
# ---------------------------------------------------------------------------


def _normalize_pathway_id(pathway_id: str, organism: str) -> str:
    """Return a bare pathway ID like ``eco00010``."""
    pid = pathway_id.strip()
    if pid.startswith("path:"):
        pid = pid[len("path:") :]
    if pid.isdigit():
        pid = organism + pid.zfill(5)
    return pid


def _kgml_to_graph(
    kgml_text: str,
    *,
    multigraph: bool = False,
    backbone: bool = False,
) -> nx.DiGraph | nx.MultiDiGraph:
    """
    Parse KGML XML text and build a metabolic (Multi)DiGraph.

    When ``backbone=True``, the graph is post-processed through:
    1. Chain-connectivity filtering (auxiliary compound removal).
    2. Consecutive-edge merging (same gene, sequential reactions).
    3. Gene deduplication (one edge per gene name).
    4. Largest weakly connected component extraction.

    KGML structure assumed
    ----------------------
    - ``<entry type="gene">``    : enzyme encoded by a gene; its ``reaction``
      attribute lists the reaction IDs it catalyzes.
    - ``<entry type="compound">`` : metabolite node.
    - ``<reaction>``             : connects substrates to products;
      ``type="reversible"|"irreversible"``.
    """
    root = ET.fromstring(kgml_text)

    # --- 1. Index all entries by numeric KGML id ---------------------------
    entries: dict[str, dict] = {}
    for entry in root.findall("entry"):
        eid = entry.get("id", "")
        etype = entry.get("type", "")
        name = entry.get("name", "")
        reaction_attr = entry.get("reaction", "")

        graphics = entry.find("graphics")
        short_name = ""
        if graphics is not None:
            raw = graphics.get("name", "")
            # KGML truncates long names with "..."; take only the first symbol
            short_name = raw.split(",")[0].replace("...", "").strip()

        entries[eid] = {
            "type": etype,
            "name": name,
            "short_name": short_name,
            "reaction_ids": set(reaction_attr.split()) if reaction_attr else set(),
        }

    # --- 2. Map reaction ID → list of gene entries -------------------------
    reaction_to_genes: dict[str, list[dict]] = {}
    for entry in entries.values():
        if entry["type"] != "gene":
            continue
        for rxn_id in entry["reaction_ids"]:
            reaction_to_genes.setdefault(rxn_id, []).append(entry)

    # --- 3. Map compound KEGG ID → human-readable name ---------------------
    compound_names: dict[str, str] = {}
    for entry in entries.values():
        if entry["type"] == "compound":
            for cid in entry["name"].split():
                compound_names[cid] = entry["short_name"] or cid

    # --- 4. Collect parsed reactions ---------------------------------------
    parsed_reactions: list[dict] = []
    for rxn_elem in root.findall("reaction"):
        rxn_id = rxn_elem.get("name", "")
        reversible = rxn_elem.get("type") == "reversible"

        substrates = [s.get("name") for s in rxn_elem.findall("substrate") if s.get("name")]
        products = [p.get("name") for p in rxn_elem.findall("product") if p.get("name")]
        gene_entries = reaction_to_genes.get(rxn_id, [])

        # Fall back to the reaction ID itself when no gene is annotated
        # (e.g. spontaneous reactions).
        if not gene_entries:
            gene_entries = [{"short_name": rxn_id, "name": "", "reaction_ids": set()}]

        parsed_reactions.append(
            {
                "rxn_id": rxn_id,
                "reversible": reversible,
                "substrates": substrates,
                "products": products,
                "gene_entries": gene_entries,
            }
        )

    # --- 5. If backbone mode, compute chain-connectivity sets --------------
    if backbone:
        produced, consumed = _compute_chain_sets(parsed_reactions)

    # --- 6. Build graph ----------------------------------------------------
    g: nx.DiGraph | nx.MultiDiGraph = nx.MultiDiGraph() if multigraph else nx.DiGraph()

    for rxn in parsed_reactions:
        substrates = rxn["substrates"]
        products = rxn["products"]

        if backbone:
            substrates, products = _filter_to_backbone(
                substrates,
                products,
                produced,
                consumed,
            )

        for substrate in substrates:
            for product in products:
                for ge in rxn["gene_entries"]:
                    gene_ids = ge["name"].split()
                    gene_id = gene_ids[0] if gene_ids else ""
                    edge_data = {
                        "gene": ge["short_name"] or gene_id or rxn["rxn_id"],
                        "gene_id": gene_id,
                        "reaction": rxn["rxn_id"],
                        "reversible": rxn["reversible"],
                    }
                    _ensure_compound_node(g, substrate, compound_names)
                    _ensure_compound_node(g, product, compound_names)
                    g.add_edge(substrate, product, **edge_data)
                    if rxn["reversible"]:
                        g.add_edge(product, substrate, **edge_data)

    # --- 7. Backbone post-processing ----------------------------------------
    #   a) Merge consecutive same-gene edges (e.g. waaA acting twice)
    #   b) Greedy deduplication for remaining multi-edge genes
    #   c) Extract largest weakly connected component
    if backbone and isinstance(g, nx.DiGraph) and not isinstance(g, nx.MultiDiGraph):
        _merge_consecutive_gene_edges(g)
        _deduplicate_genes(g)
        g = _largest_wcc(g)

    return g


def _ensure_compound_node(
    g: nx.DiGraph | nx.MultiDiGraph,
    cpd_id: str,
    compound_names: dict[str, str],
) -> None:
    if cpd_id not in g:
        g.add_node(cpd_id, name=compound_names.get(cpd_id, cpd_id))
