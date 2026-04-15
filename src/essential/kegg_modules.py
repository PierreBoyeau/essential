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
Organism-specific module entries (e.g. ``md:eco_M00018``) carry two
parallel blocks that together pin down the reaction → gene mapping
locally, with no extra REST calls:

- ``DEFINITION`` is a space-separated logic expression of K-numbers,
  one top-level token per reaction step, in the same order as the
  REACTION block.  Parentheses group isoenzymes; ``+`` joins complex
  components; ``-`` marks optional units.  We extract the set of
  K-numbers per step and ignore the boolean structure.
- ``GENE`` lists every organism gene present in the module along with
  its ``[KO:Kxxxxx]`` annotation.

Pairing the i-th DEFINITION token with the i-th REACTION line gives,
for each reaction, the KO set that catalyzes it; the GENE block then
resolves each KO to one or more organism genes.  No ORTHOLOGY block
or KEGG reaction entry is needed.

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
    # KEGG REST no longer supports ``list/module/{org}``; obtain the
    # organism's modules via the ``link`` endpoint, then join titles
    # from the universal ``list/module``.
    link_text = _fetch(f"{_KEGG_BASE}/link/{organism}/module", cache_dir)
    module_ids: set[str] = set()
    for line in link_text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        mid = parts[0]
        if mid.startswith("md:"):
            mid = mid[len("md:"):]
        module_ids.add(mid)

    list_text = _fetch(f"{_KEGG_BASE}/list/module", cache_dir)
    title_by_bare: dict[str, str] = {}
    for line in list_text.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        bare = parts[0]
        if bare.startswith("md:"):
            bare = bare[len("md:"):]
        title_by_bare[bare] = parts[1]

    records: list[dict[str, str]] = []
    for mid in sorted(module_ids):
        bare = mid.split("_", 1)[1] if "_" in mid else mid
        records.append({
            "module_id": mid,
            "title": title_by_bare.get(bare, ""),
            "class": "",
        })

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

    Returns
    -------
    networkx.DiGraph
        Nodes are compound KEGG IDs (``"C00031"`` etc.) with attribute
        ``name`` (human-readable compound name from the COMPOUND block).

        Edges are one per ``(substrate, product)`` pair of each reaction
        in the module's REACTION block.  Reversible reactions emit two
        directed edges.  Edge attributes:

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
    pid = _normalize_module_id(module_id, organism)
    flat = _fetch(f"{_KEGG_BASE}/get/md:{pid}", cache_dir)

    blocks = _parse_flat_blocks(flat)
    name = blocks.get("NAME", "").strip()
    definition = blocks.get("DEFINITION", "").strip()
    cls = blocks.get("CLASS", "").strip()

    compounds = _parse_compound_block(blocks.get("COMPOUND", ""))
    reaction_steps = _parse_reaction_block(blocks.get("REACTION", ""))
    definition_steps = _parse_definition(definition)
    ko_to_gene = _parse_gene_block(blocks.get("GENE", ""))

    g = nx.DiGraph()
    g.graph["module_id"] = pid
    g.graph["name"] = name
    g.graph["definition"] = definition
    g.graph["class"] = cls
    src, tgt = _parse_endpoints_from_name(name)
    g.graph["source"] = src
    g.graph["target"] = tgt

    for cid, cname in compounds.items():
        g.add_node(cid, name=cname)

    for step_idx, step in enumerate(reaction_steps):
        kos = (
            definition_steps[step_idx]
            if step_idx < len(definition_steps)
            else []
        )

        gene_pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for ko in kos:
            for sym, gid in ko_to_gene.get(ko, []):
                if (sym, gid) not in seen:
                    seen.add((sym, gid))
                    gene_pairs.append((sym, gid))

        genes = [p[0] for p in gene_pairs]
        gene_ids = [p[1] for p in gene_pairs]

        for rxn_id in step["rxn_ids"]:
            edge_data = {
                "reaction": rxn_id,
                "kos": list(kos),
                "genes": genes,
                "gene_ids": gene_ids,
                "gene": genes[0] if genes else None,
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
                    if step["reversible"]:
                        g.add_edge(p, s, **edge_data)

    return g


# ---------------------------------------------------------------------------
# Internal: ID normalization
# ---------------------------------------------------------------------------


def _normalize_module_id(module_id: str, organism: str) -> str:
    """Return an organism-prefixed module ID like ``"eco_M00018"``."""
    pid = module_id.strip()
    if pid.startswith("md:"):
        pid = pid[len("md:"):]
    if re.fullmatch(r"M\d{5}", pid):
        pid = f"{organism}_{pid}"
    return pid


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
    Parse the REACTION block into one entry per line, preserving the
    in-block order so each step can be paired index-wise with the
    corresponding token of the DEFINITION expression.

    Lines have the form::

        R01786  C00267 -> C00668
        R01773,R01775  C00441 <=> C00263

    Comma-separated R-numbers on a single line denote alternative
    reaction IDs for the *same* step (catalyzed by the same KOs); they
    share one entry's ``rxn_ids`` list.

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

        out.append({
            "rxn_ids": rxn_ids,
            "substrates": re.findall(r"C\d{5}", lhs),
            "products": re.findall(r"C\d{5}", rhs),
            "reversible": reversible,
        })
    return out


def _parse_definition(text: str) -> list[list[str]]:
    """
    Split a module DEFINITION expression into one KO list per
    top-level token.

    Top-level tokens are space-separated, but spaces inside parentheses
    do not split.  Each token may be a single K-number, a parenthesized
    comma-separated alternation of isoenzymes, or a ``+``/``-`` joined
    complex expression — we ignore the boolean structure and simply
    extract every K-number in the token.

    Token order matches the REACTION block line order.
    """
    text = text.strip()
    tokens: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch.isspace() and depth == 0:
            if cur:
                tokens.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    return [re.findall(r"K\d{5}", tok) for tok in tokens]


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
