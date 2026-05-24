"""
iML1515 BiGG model loaders.

Model: /workspace/experiments/05142026_metabolome/iML1515.json
  1877 metabolites  |  2712 reactions  |  1516 genes

Design notes
------------
bigg_metabolite_df
    Index = full BiGG id including compartment suffix (e.g. '10fthf_c').
    Column `base_id` strips the trailing '_c' / '_e' / '_p' — use this to
    join with Rapp metabolomics data (mmc5 Abbrev after stripping adducts).

bigg_reaction_df  ("edge" table)
    Index = reaction id (e.g. 'PFK').  One row per reaction.

bigg_gene_df
    Index = gene *name* (e.g. 'arcA').  b-number is a column.
    Gene names are unique in iML1515 so this index is safe.

bigg_gene_reaction_df  (long format)
    One row per (gene, reaction) association, parsed from gene_reaction_rule.
    Use this to join genes to reactions or to filter reactions by gene.

stoichiometry_wide()
    metabolite (full id) × reaction sparse DataFrame; values are
    stoichiometric coefficients (negative = consumed, positive = produced).

stoichiometry_long()
    Long-format tidy table: metabolite_id, reaction_id, coefficient.
    Easiest to filter/join.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).parent / "iML1515.json"


def _load_raw() -> dict:
    with open(MODEL_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metabolites
# ---------------------------------------------------------------------------


def bigg_metabolite_df() -> pd.DataFrame:
    """
    One row per compartment-specific metabolite.

    Index: full BiGG id (e.g. '10fthf_c')
    Columns:
      base_id      BiGG id without compartment — matches Rapp mmc10 Abbreviation
                   and mmc5 Abbrev (after adduct-stripping)
      name         human-readable name
      compartment  'c' | 'e' | 'p'
      charge       formal charge (int, may be NaN)
      formula      molecular formula string
      biocyc       first BioCyc cross-ref (str or NaN)
      kegg         first KEGG compound ID (str or NaN)
      chebi        first ChEBI ID (str or NaN)
    """
    raw = _load_raw()
    rows = []
    for m in raw["metabolites"]:
        ann = m.get("annotation", {})
        rows.append(
            {
                "id": m["id"],
                "base_id": m["id"].rsplit("_", 1)[0],
                "name": m.get("name", ""),
                "compartment": m.get("compartment", ""),
                "charge": m.get("charge", np.nan),
                "formula": m.get("formula", ""),
                "biocyc": ann.get("biocyc", [None])[0],
                "kegg": ann.get("kegg.compound", [None])[0],
                "chebi": ann.get("chebi", [None])[0],
            }
        )
    df = pd.DataFrame(rows).set_index("id")
    df["charge"] = pd.to_numeric(df["charge"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Reactions  (edges)
# ---------------------------------------------------------------------------


def bigg_reaction_df() -> pd.DataFrame:
    """
    One row per reaction.

    Index: reaction id (e.g. 'PFK')
    Columns:
      name                 human-readable name
      subsystem            metabolic subsystem string
      lower_bound          flux lower bound (mmol/gDW/h)
      upper_bound          flux upper bound
      reversible           True if lower_bound < 0
      gene_reaction_rule   raw GPR string (b-numbers)
      n_genes              number of genes in rule (0 = spontaneous/exchange)
      metabolite_ids       tuple of metabolite ids involved
      kegg_reaction        first KEGG reaction ID (str or NaN)
      ec_number            first EC number (str or NaN)
    """
    raw = _load_raw()
    rows = []
    for r in raw["reactions"]:
        ann = r.get("annotation", {})
        gpr = r.get("gene_reaction_rule", "")
        b_nums = re.findall(r"b\d+", gpr)
        rows.append(
            {
                "id": r["id"],
                "name": r.get("name", ""),
                "subsystem": r.get("subsystem", ""),
                "lower_bound": r["lower_bound"],
                "upper_bound": r["upper_bound"],
                "reversible": r["lower_bound"] < 0,
                "gene_reaction_rule": gpr,
                "n_genes": len(b_nums),
                "metabolite_ids": tuple(r["metabolites"].keys()),
                "kegg_reaction": ann.get("kegg.reaction", [None])[0],
                "ec_number": ann.get("ec-code", [None])[0],
            }
        )
    return pd.DataFrame(rows).set_index("id")


# ---------------------------------------------------------------------------
# Genes
# ---------------------------------------------------------------------------


def bigg_gene_df() -> pd.DataFrame:
    """
    One row per gene, indexed by gene *name* (e.g. 'arcA').

    Gene names are unique in iJO1366, so this index is unambiguous.
    b-numbers are kept as column `b_number` for cross-referencing.

    Columns:
      b_number    Blattner number (e.g. 'b4401')
      ecogene     EcoGene ID (str or NaN)
      ncbi_gene   NCBI Gene ID (str or NaN)
      uniprot     UniProt AC (str or NaN)
    """
    raw = _load_raw()
    rows = []
    for g in raw["genes"]:
        ann = g.get("annotation", {})
        rows.append(
            {
                "name": g["name"],
                "b_number": g["id"],
                "ecogene": ann.get("ecogene", [None])[0],
                "ncbi_gene": ann.get("ncbigene", [None])[0],
                "uniprot": ann.get("uniprot", [None])[0],
            }
        )
    return pd.DataFrame(rows).set_index("name")


def bigg_gene_reaction_df() -> pd.DataFrame:
    """
    Long-format gene → reaction association table.

    Parsed from each reaction's gene_reaction_rule (b-number form),
    then mapped to gene names.  Genes connected by 'and' (same complex)
    and 'or' (isozymes) are both expanded — use `gene_reaction_rule` on the
    reaction table to reconstruct the logic.

    Columns:
      gene_name   gene name (e.g. 'arcA')  — primary join key to bigg_gene_df
      b_number    Blattner number
      reaction_id reaction id               — primary join key to bigg_reaction_df
      subsystem   subsystem of the reaction (convenience copy)
    """
    raw = _load_raw()

    # Build b_number → gene_name map
    b2name = {g["id"]: g["name"] for g in raw["genes"]}

    rows = []
    for r in raw["reactions"]:
        gpr = r.get("gene_reaction_rule", "")
        b_nums = re.findall(r"b\d+", gpr)
        for b in b_nums:
            rows.append(
                {
                    "gene_name": b2name.get(b, b),
                    "b_number": b,
                    "reaction_id": r["id"],
                    "subsystem": r.get("subsystem", ""),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stoichiometry matrix
# ---------------------------------------------------------------------------


def stoichiometry_long() -> pd.DataFrame:
    """
    Tidy long-format stoichiometry table.

    Columns:
      metabolite_id   full BiGG id including compartment (e.g. '10fthf_c')
      base_id         compartment-stripped id — for joining with Rapp data
      reaction_id     reaction id
      coefficient     stoichiometric coefficient
                      negative = substrate (consumed)
                      positive = product (produced)
    """
    raw = _load_raw()
    rows = []
    for r in raw["reactions"]:
        rxn_id = r["id"]
        for met_id, coeff in r["metabolites"].items():
            rows.append(
                {
                    "metabolite_id": met_id,
                    "base_id": met_id.rsplit("_", 1)[0],
                    "reaction_id": rxn_id,
                    "coefficient": coeff,
                }
            )
    return pd.DataFrame(rows)


def stoichiometry_wide() -> pd.DataFrame:
    """
    Metabolite × reaction stoichiometry matrix (sparse-friendly).

    Index:   full metabolite id (e.g. '10fthf_c')  — 1805 rows
    Columns: reaction id                             — 2583 cols
    Values:  stoichiometric coefficient (0 where absent)

    Most entries are 0.  For large downstream computations, convert with:
        S = stoichiometry_wide()
        S_sparse = S.astype(pd.SparseDtype(float, fill_value=0))
    """
    long = stoichiometry_long()
    wide = long.pivot_table(
        index="metabolite_id",
        columns="reaction_id",
        values="coefficient",
        fill_value=0,
        aggfunc="sum",  # handles duplicate entries if any
    )
    wide.columns.name = None
    wide.index.name = "metabolite_id"
    return wide


# ---------------------------------------------------------------------------
# Cross-reference helpers
# ---------------------------------------------------------------------------


def rapp_to_bigg_index(abbrev_series: pd.Series) -> pd.Series:
    """
    Map a Rapp mmc5 Abbrev series to iJO1366 base_ids.

    Rapp Abbrevs encode adduct and isobaric ambiguity, e.g.:
      'ppal[M+H]+'        → 'ppal'
      'ac-gcald[M+H]+'    → 'ac-gcald'  (isobaric pair; kept joined)

    Returns a Series aligned to the input, with adduct stripped.
    Isobaric entries (joined by '-') are left joined — they map to multiple
    metabolites; use `split('-')` and explode if you need one row per
    metabolite.
    """
    return abbrev_series.str.replace(r"\[M[^\]]*\][^-\s]*$", "", regex=True).str.strip()
