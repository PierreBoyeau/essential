"""
KEGG module helpers.

KEGG modules are curated linear-ish biosynthetic / catabolic units with
explicit input and output compounds — much cleaner than KEGG pathway
maps for extracting linear metabolic chains.

Public API
----------
list_kegg_modules(organism, pathway_only=True)
    Return all KEGG modules for an organism.  When ``pathway_only`` is
    True, restrict to modules whose CLASS begins with ``"Pathway
    modules"`` (excludes Structural complexes, Functional sets, and
    Signature modules, which are not metabolic routes).

kegg_module_to_graph(module_id, organism)
    Parse a KEGG module entry into a ``networkx.DiGraph`` whose edges
    are reactions annotated with KO / EC / gene metadata, and whose
    graph-level attributes carry the curated source/target compounds.

Joining reactions to genes
--------------------------
Two separate REST entries are used together:

- The **organism-specific** entry (e.g. ``md:eco_M00545``) supplies
  the REACTION, COMPOUND, NAME, DEFINITION, CLASS, and GENE blocks.
  Its ORTHOLOGY block does *not* carry ``[RN:...]`` tags.

- The **universal** module entry (e.g. ``md:M00545``) supplies the
  ORTHOLOGY block, whose lines do carry ``[RN:Rxxxxx ...]`` tags and
  are the authoritative KO → reaction mapping.

Deriving KO sets from the universal ORTHOLOGY block instead of
positional alignment with the DEFINITION tokens is the key fix for
branching modules: a single DEFINITION token can expand to multiple
parallel REACTION lines, which breaks any index-based pairing.

Notes
-----
- KEGG REST cache: this module reuses ``_fetch`` from
  ``kegg_pathways`` so all responses are cached on disk under
  ``~/.cache/kegg`` by default.
- No filtering of currency metabolites, no edge merging, no gene
  deduplication: the graph reflects the module entry as-is.  Reversible
  reactions (``<=>``) emit two directed edges.
"""

from __future__ import annotations

import re
from pathlib import Path
from pprint import pprint
from typing import Optional

import networkx as nx

from essential.kegg_pathways import _KEGG_BASE, _fetch

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def list_kegg_modules(
    organism: str = "eco",
    *,
    pathway_only: bool = True,
    cache_dir: Optional[Path] = Path.home() / ".cache" / "kegg",
) -> list[dict[str, str]]:
    """
    Return all KEGG modules for ``organism``.

    Parameters
    ----------
    organism :
        KEGG organism code (e.g. ``"eco"``, ``"hsa"``).
    pathway_only :
        If True, fetch each module entry to read its CLASS field and
        keep only modules whose class begins with ``"Pathway modules"``.
        This issues one cached fetch per module on first run; subsequent
        calls are served entirely from cache.
    cache_dir :
        Directory for caching KEGG REST responses.  Pass ``None`` to
        disable caching.

    Returns
    -------
    list of dict
        Each entry has keys ``module_id`` (organism-prefixed, e.g.
        ``"eco_M00018"``), ``title``, and ``class`` (empty string when
        ``pathway_only=False``).
    """
    link_text = _fetch(f"{_KEGG_BASE}/link/{organism}/module", cache_dir)
    module_ids: set[str] = set()
    for line in link_text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        mid = parts[0]
        if mid.startswith("md:"):
            mid = mid[len("md:") :]
        module_ids.add(mid)

    list_text = _fetch(f"{_KEGG_BASE}/list/module", cache_dir)
    title_by_bare: dict[str, str] = {}
    for line in list_text.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        bare = parts[0]
        if bare.startswith("md:"):
            bare = bare[len("md:") :]
        title_by_bare[bare] = parts[1]

    records: list[dict[str, str]] = []
    for mid in sorted(module_ids):
        bare = mid.split("_", 1)[1] if "_" in mid else mid
        records.append(
            {
                "module_id": mid,
                "title": title_by_bare.get(bare, ""),
                "class": "",
            }
        )

    if not pathway_only:
        return records

    kept: list[dict[str, str]] = []
    for rec in records:
        flat = _fetch(f"{_KEGG_BASE}/get/md:{rec['module_id']}", cache_dir)
        cls = _parse_flat_blocks(flat).get("CLASS", "").strip()
        rec["class"] = cls
        if cls.startswith("Pathway modules"):
            kept.append(rec)
    return kept


def kegg_module_to_graph(
    module_id: str,
    organism: str = "eco",
    *,
    cache_dir: Optional[Path] = Path.home() / ".cache" / "kegg",
    print_info: bool = False,
    tag_bidirectional: bool = False,
) -> nx.DiGraph:
    """
    Parse a KEGG module into a directed reaction graph.

    Parameters
    ----------
    module_id :
        Module ID in any of the forms ``"M00018"``, ``"eco_M00018"``,
        or ``"md:eco_M00018"``.  When only the bare ``"M-number"`` is
        given, ``organism`` is prepended.
    organism :
        KEGG organism code used both for prefixing ``module_id`` and
        for resolving KO → gene mappings.
    cache_dir :
        Directory for caching KEGG REST responses.  Pass ``None`` to
        disable caching.
    tag_bidirectional :
        If True, reversible reactions will be emitted as two directed edges (substrate->product and product->substrate).
        If False (default), emit only substrate->product edges.

    Returns
    -------
    networkx.DiGraph
        Nodes are compound KEGG IDs (``"C00031"`` etc.) with attribute
        ``name`` (human-readable compound name from the COMPOUND block).

        Edges are one per ``(substrate, product)`` pair of each reaction
        in the module's REACTION block.  Reversible reactions emit two
        directed edges if tag_bidirectional=True.  Edge attributes:

        - ``reaction``  : R-number
        - ``kos``       : list of K-numbers catalyzing the reaction
        - ``ec``        : list of EC numbers
        - ``genes``     : list of organism gene symbols (may be empty)
        - ``gene_ids``  : list of organism KEGG gene IDs
        - ``gene``      : primary gene symbol (``genes[0]`` or ``None``)
        - ``gene_id``   : primary gene id (``gene_ids[0]`` or ``None``)
        - ``reversible``: bool

        Graph-level attributes (``g.graph``):

        - ``module_id``, ``name``, ``definition``, ``class``
        - ``source``, ``target`` : curated endpoint compound names
          parsed from the module title (``"..., X => Y"``); ``None``
          when the title carries no such suffix.
    """
    org_pid = _normalize_module_id(module_id, organism)  # e.g. "eco_M00545"
    bare_mid = _bare_module_id(org_pid)  # e.g. "M00545"

    # Organism-specific entry: REACTION, COMPOUND, GENE, NAME, CLASS, DEFINITION.
    organism_flat = _fetch(f"{_KEGG_BASE}/get/md:{org_pid}", cache_dir)
    organism_blocks = _parse_flat_blocks(organism_flat)
    # from pprint import pprint

    # Universal entry: ORTHOLOGY block carries [RN:...] tags that the
    # organism-specific entry omits.  This is the only reliable source
    # for KO → R-number mapping that is independent of module topology.
    universal_flat = _fetch(f"{_KEGG_BASE}/get/md:{bare_mid}", cache_dir)
    universal_blocks = _parse_flat_blocks(universal_flat)

    name = organism_blocks.get("NAME", "").strip()
    definition = organism_blocks.get("DEFINITION", "").strip()
    cls = organism_blocks.get("CLASS", "").strip()

    compounds = _parse_compound_block(organism_blocks.get("COMPOUND", ""))
    reaction_steps = _parse_reaction_block(organism_blocks.get("REACTION", ""))
    ko_to_gene = _parse_gene_block(organism_blocks.get("GENE", ""))

    # rxn_to_kos from the universal ORTHOLOGY block — topology-independent.
    rxn_to_kos: dict[str, list[str]] = _parse_orthology_block(universal_blocks.get("ORTHOLOGY", ""))

    g = nx.DiGraph()
    g.graph["module_id"] = org_pid
    g.graph["name"] = name
    g.graph["definition"] = definition
    g.graph["class"] = cls
    src, tgt = _parse_endpoints_from_name(name)
    g.graph["source"] = src
    g.graph["target"] = tgt

    for cid, cname in compounds.items():
        g.add_node(cid, name=cname)

    for step in reaction_steps:
        # Union KOs across all R-numbers in this step (comma-listed
        # alternatives share the same KO set in practice, but we union
        # to be safe).
        kos_seen: set[str] = set()
        kos: list[str] = []
        for rxn_id in step["rxn_ids"]:
            for ko in rxn_to_kos.get(rxn_id, []):
                if ko not in kos_seen:
                    kos_seen.add(ko)
                    kos.append(ko)

        unique_gene_mappings: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for ko in kos:
            for sym, gid in ko_to_gene.get(ko, []):
                if (sym, gid) not in seen:
                    seen.add((sym, gid))
                    unique_gene_mappings.append((sym, gid))
        genes = [p[0] for p in unique_gene_mappings]
        gene_ids = [p[1] for p in unique_gene_mappings]

        for rxn_id in step["rxn_ids"]:
            edge_data = {
                "reaction": rxn_id,
                "kos": kos,
                "genes": genes,
                "gene_ids": gene_ids,
                "gene": genes[0] if genes else None,  # we only extract the first gene mapping
                "gene_id": gene_ids[0] if gene_ids else None,
                "reversible": step["reversible"],
            }
            for s in step["substrates"]:
                for p in step["products"]:
                    if s not in g:
                        g.add_node(s, name=compounds.get(s, s))
                    if p not in g:
                        g.add_node(p, name=compounds.get(p, p))
                    g.add_edge(s, p, **edge_data)
                    if tag_bidirectional and step["reversible"]:
                        g.add_edge(p, s, **edge_data)

    if print_info:
        pprint(organism_blocks)
        pprint(ko_to_gene)
        print(f"{module_id}:", g.graph["name"])
        print("source ->", g.graph["source"], "|| target ->", g.graph["target"])
        print(f"nodes={g.number_of_nodes()} edges={g.number_of_edges()}")

    return g


# ---------------------------------------------------------------------------
# Internal: ID normalization
# ---------------------------------------------------------------------------


def _normalize_module_id(module_id: str, organism: str) -> str:
    """Return an organism-prefixed module ID like ``"eco_M00018"``."""
    pid = module_id.strip()
    if pid.startswith("md:"):
        pid = pid[len("md:") :]
    if re.fullmatch(r"M\d{5}", pid):
        pid = f"{organism}_{pid}"
    return pid


def _bare_module_id(org_pid: str) -> str:
    """
    Strip the organism prefix from an organism-prefixed module ID.

    ``"eco_M00018"`` → ``"M00018"``
    ``"M00018"``     → ``"M00018"``
    """
    if "_" in org_pid:
        return org_pid.split("_", 1)[1]
    return org_pid


# ---------------------------------------------------------------------------
# Internal: KEGG flat-file parsing
# ---------------------------------------------------------------------------


_FIELD_WIDTH = 12


def _parse_flat_blocks(text: str) -> dict[str, str]:
    """
    Parse a KEGG flat-file entry into ``{field_name: body}``.

    KEGG flat files use a fixed-width layout: field names occupy
    columns 1–12 (left-justified, space-padded); values start at
    column 13.  Continuation lines have 12 leading spaces.  The entry
    is terminated by a ``///`` line.
    """
    blocks: dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        if line.startswith("///"):
            break
        if not line:
            continue
        head = line[:_FIELD_WIDTH]
        body = line[_FIELD_WIDTH:]
        if head.strip():
            current = head.strip()
            blocks.setdefault(current, []).append(body)
        elif current is not None:
            blocks[current].append(body)
    return {k: "\n".join(v) for k, v in blocks.items()}


def _parse_compound_block(text: str) -> dict[str, str]:
    """Return ``{C-number: name}`` from a module's COMPOUND block."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(C\d{5})\s+(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _parse_reaction_block(text: str) -> list[dict]:
    """
    Parse the REACTION block into one entry per line.

    Lines have the form::

        R01786  C00267 -> C00668
        R01773,R01775  C00441 <=> C00263

    Comma-separated R-numbers on a single line denote alternative
    reaction IDs for the *same* step; they share one entry's
    ``rxn_ids`` list.

    Returns dicts with keys ``rxn_ids``, ``substrates``, ``products``,
    ``reversible``.
    """
    out: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^((?:R\d{5}[,\s]*)+)\s+(.+)$", line)
        if not m:
            continue
        rxn_ids = re.findall(r"R\d{5}", m.group(1))
        eq = m.group(2)

        if "<=>" in eq:
            lhs, rhs = eq.split("<=>", 1)
            reversible = True
        elif "->" in eq:
            lhs, rhs = eq.split("->", 1)
            reversible = False
        else:
            continue

        out.append(
            {
                "rxn_ids": rxn_ids,
                "substrates": re.findall(r"C\d{5}", lhs),
                "products": re.findall(r"C\d{5}", rhs),
                "reversible": reversible,
            }
        )
    return out


def _parse_orthology_block(text: str) -> dict[str, list[str]]:
    """
    Parse a **universal** module ORTHOLOGY block into
    ``{R-number: [K-numbers]}``.

    Lines in the universal module entry have the form::

        K05708  hcaE; 3-phenylpropanoate ... [RN:R06783] [EC:1.14.12.19]
        K00626  acetyl-CoA acyltransferase ... [RN:R00238 R00927]

    Multi-subunit enzyme complexes join K-numbers with ``+`` in the
    first whitespace-delimited token::

        K05708+K05709+K05710+K00529  3-phenylpropionate dioxygenase ... [RN:R06783]

    All K-numbers from such a complex are mapped to the same reaction(s).

    The ``[RN:...]`` tag is present only in universal module entries
    (e.g. ``md:M00545``), not in organism-specific ones
    (e.g. ``md:eco_M00545``).  Always call this on the universal flat
    file.

    Returns
    -------
    dict
        ``{R-number: [K-number, ...]}`` ordered by first appearance of
        each KO in the block.
    """
    rxn_to_kos: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # The KO field is the first whitespace-delimited token.
        # Enzyme complexes use '+'-joined K-numbers (e.g. K05708+K05709+...).
        first_token = line.split()[0]
        kos = re.findall(r"K\d{5}", first_token)
        if not kos:
            continue
        rn_match = re.search(r"\[RN:([^\]]+)\]", line)
        if not rn_match:
            continue
        for rxn_id in re.findall(r"R\d{5}", rn_match.group(1)):
            lst = rxn_to_kos.setdefault(rxn_id, [])
            for ko in kos:
                if ko not in lst:
                    lst.append(ko)
    return rxn_to_kos


def _parse_gene_block(text: str) -> dict[str, list[tuple[str, str]]]:
    """
    Parse an organism-specific module's GENE block into a
    ``KO → [(gene_symbol, gene_id), ...]`` map.

    Lines have the shape::

        b0002  thrA; fused aspartate kinase ... [KO:K12524]

    A gene may carry multiple K-numbers in ``[KO:K00928 K12525]``;
    each is registered separately.
    """
    ko_to_gene: dict[str, list[tuple[str, str]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^(\S+)\s+(.+?)\s*\[KO:([^\]]+)\]\s*$", line)
        if not m:
            continue
        gid, desc, ko_field = m.group(1), m.group(2), m.group(3)
        symbol = desc.split(";", 1)[0].strip()
        for ko in re.findall(r"K\d{5}", ko_field):
            ko_to_gene.setdefault(ko, []).append((symbol, gid))
    return ko_to_gene


def _parse_endpoints_from_name(name: str) -> tuple[Optional[str], Optional[str]]:
    """
    Pull the curated source/target compounds out of a module title.

    Module titles end with a ``"..., A => B"`` clause, sometimes
    chained as ``"A => B => C"``.  We take the first compound and the
    last compound of the chain after the first comma.

    Returns ``(source, target)`` when matchable, else ``(None, None)``.
    """
    if "," not in name:
        return None, None
    suffix = name.split(",", 1)[1]
    if "=>" not in suffix:
        return None, None
    parts = [p.strip() for p in suffix.split("=>")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None, None
    return parts[0], parts[-1]


def metabolic_to_operational_graph(
    metabolic_graph: nx.DiGraph,
) -> nx.DiGraph:
    """
    Project a metabolic DiGraph (metabolite nodes, gene-annotated edges)
    onto a gene-level operational graph suitable for PathwayDiscontinuity.

    For each metabolite node M, genes on edges producing M (M is target)
    are connected to genes on edges consuming M (M is source).  A gene
    appearing on multiple metabolic edges (e.g. waaA in a convergent
    pathway) collapses to a single node.  A gene catalyzing consecutive
    steps (A->B->C) produces no self-loop; it connects to its upstream
    and downstream neighbors only.

    Edge attributes on the operational graph:
        metabolites : set of intermediate metabolite IDs linking the pair
    """
    op = nx.DiGraph()

    for metabolite in metabolic_graph.nodes():
        # genes that produce this metabolite (on in-edges)
        producers: set[str] = set()
        for u, _, data in metabolic_graph.in_edges(metabolite, data=True):
            for gene in data.get("genes", []):
                if gene:
                    producers.add(gene)

        # genes that consume this metabolite (on out-edges)
        consumers: set[str] = set()
        for _, v, data in metabolic_graph.out_edges(metabolite, data=True):
            for gene in data.get("genes", []):
                if gene:
                    consumers.add(gene)

        for prod in producers:
            for cons in consumers:
                if prod == cons:
                    continue
                if op.has_edge(prod, cons):
                    op[prod][cons]["metabolites"].add(metabolite)
                else:
                    op.add_edge(prod, cons, metabolites={metabolite})

    # carry over module-level metadata
    op.graph.update(metabolic_graph.graph)

    return op
