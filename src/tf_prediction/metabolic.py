"""iML1515 / BiGG metabolic-model utilities for TF-perturbation GT labels.

Mirrors ``regulondb.py``: load a reference resource once, expose lookups for
downstream evaluation code. Public surface:

    MetabolicModel(json_path, currency_mets=DEFAULT_CURRENCY)
        .has_enzyme(gene)                -> bool
        .enzyme_substrates(gene)         -> set[str]   # compartment-stripped BiGG IDs
        .enzyme_products(gene)           -> set[str]
        .enzyme_downstream(gene, depth)  -> set[str]

    build_gt_labels(model, sensor_df, perturbations, tf_names, *, depth=2) -> GTLabels

All ``gene`` arguments are lowercase gene symbols (matching the TF axis used
by ``build_tf_mask``). Reversible reactions (``lower_bound < 0``) contribute
both directions for substrate/product role assignment. The currency-metabolite
filter applies only inside the downstream BFS, not to direct substrate/product
sets — DnaA senses ATP, and we still want to count ATP as a substrate of an
enzyme even though it's a currency metabolite for graph traversal.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

DEFAULT_CURRENCY: frozenset[str] = frozenset(
    {
        "h",
        "h2o",
        "atp",
        "adp",
        "amp",
        "pi",
        "ppi",
        "nad",
        "nadh",
        "nadp",
        "nadph",
        "fad",
        "fadh2",
        "co2",
        "o2",
        "coa",
        "nh4",
        "hco3",
        "k",
        "na1",
        "mg2",
        "ca2",
        "fe2",
        "fe3",
        "mn2",
        "zn2",
        "cu2",
        "ni2",
        "cl",
        "so4",
        "mobd",
    }
)

_BNUMBER = re.compile(r"\bb\d{4}\b")


def _strip_compartment(bigg_id: str) -> str:
    return bigg_id.rsplit("_", 1)[0]


class MetabolicModel:
    """Parsed iML1515 indexed by lowercase gene symbol."""

    def __init__(self, json_path, currency_mets=DEFAULT_CURRENCY):
        with Path(json_path).open() as f:
            model = json.load(f)
        self._currency = frozenset(currency_mets)

        self._bnumber_to_gene: dict[str, str] = {
            g["id"]: (g.get("name") or g["id"]).lower() for g in model["genes"]
        }

        self._rxn_subs: dict[str, set[str]] = {}
        self._rxn_prods: dict[str, set[str]] = {}
        self._gene_to_rxns: dict[str, set[str]] = defaultdict(set)
        self._met_to_downstream: dict[str, set[str]] = defaultdict(set)

        for rxn in model["reactions"]:
            rid = rxn["id"]
            reversible = rxn.get("lower_bound", 0.0) < 0.0
            subs, prods = set(), set()
            for met_id, coef in rxn["metabolites"].items():
                base = _strip_compartment(met_id)
                if coef < 0:
                    subs.add(base)
                    if reversible:
                        prods.add(base)
                else:
                    prods.add(base)
                    if reversible:
                        subs.add(base)
            self._rxn_subs[rid] = subs
            self._rxn_prods[rid] = prods

            for bnum in _BNUMBER.findall(rxn.get("gene_reaction_rule") or ""):
                gene = self._bnumber_to_gene.get(bnum)
                if gene:
                    self._gene_to_rxns[gene].add(rid)

            for s in subs - self._currency:
                for p in prods - self._currency:
                    if p != s:
                        self._met_to_downstream[s].add(p)

    def has_enzyme(self, gene: str) -> bool:
        return gene in self._gene_to_rxns

    def enzyme_reactions(self, gene: str) -> set[str]:
        return set(self._gene_to_rxns.get(gene, ()))

    def enzyme_substrates(self, gene: str) -> set[str]:
        out: set[str] = set()
        for rid in self._gene_to_rxns.get(gene, ()):
            out |= self._rxn_subs[rid]
        return out

    def enzyme_products(self, gene: str) -> set[str]:
        out: set[str] = set()
        for rid in self._gene_to_rxns.get(gene, ()):
            out |= self._rxn_prods[rid]
        return out

    def enzyme_downstream(self, gene: str, depth: int = 2) -> set[str]:
        seeds = self.enzyme_products(gene) - self._currency
        if not seeds or depth <= 0:
            return set()
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(depth):
            nxt: set[str] = set()
            for m in frontier:
                nxt |= self._met_to_downstream.get(m, set())
            nxt -= visited
            if not nxt:
                break
            visited |= nxt
            frontier = nxt
        return visited


class GTLabels(NamedTuple):
    """Ground-truth TF labels for each perturbation, under three definitions."""

    Y_substrate: np.ndarray  # (n_perts, n_tfs) bool
    Y_substrate_or_product: np.ndarray  # (n_perts, n_tfs) bool
    Y_downstream: np.ndarray  # (n_perts, n_tfs) bool
    coverage: np.ndarray  # (n_perts, n_tfs) bool — eval where True
    perturbations: list[str]  # lowercase gene symbols
    tf_names: list[str]  # lowercase gene symbols


def build_gt_labels(
    model: MetabolicModel,
    sensor_df: pd.DataFrame,
    perturbations,
    tf_names,
    *,
    depth: int = 2,
) -> GTLabels:
    """Build (n_perts, n_tfs) GT label matrices for PR-curve evaluation.

    ``sensor_df`` must have columns ``Transcription factor``, ``bigg_id``, and
    ``resolved``. Rows with ``resolved=False`` are ignored. ``perturbations``
    and ``tf_names`` are lowercase gene symbols defining the axes.

    The ``coverage`` mask is True only where (a) the perturbation has at least
    one reaction in iML1515 and (b) the TF has at least one effector resolved
    to a BiGG ID. PR curves should be computed on coverage=True cells only —
    xenobiotic-sensing TFs and non-metabolic perturbations would otherwise
    bias the eval by silently contributing zero labels.
    """
    resolved = sensor_df[sensor_df["resolved"]]
    tf_effectors: dict[str, set[str]] = defaultdict(set)
    for tf, bigg in zip(resolved["Transcription factor"], resolved["bigg_id"]):
        tf_effectors[str(tf).lower()].add(str(bigg))

    n_perts, n_tfs = len(perturbations), len(tf_names)
    Y_sub = np.zeros((n_perts, n_tfs), dtype=bool)
    Y_sop = np.zeros((n_perts, n_tfs), dtype=bool)
    Y_down = np.zeros((n_perts, n_tfs), dtype=bool)
    pert_in_model = np.zeros(n_perts, dtype=bool)
    tf_has_effector = np.array([bool(tf_effectors.get(t)) for t in tf_names])

    for i, p in enumerate(perturbations):
        if not model.has_enzyme(p):
            continue
        pert_in_model[i] = True
        subs = model.enzyme_substrates(p)
        prods = model.enzyme_products(p)
        sop = subs | prods
        down = model.enzyme_downstream(p, depth=depth)
        for j, t in enumerate(tf_names):
            eff = tf_effectors.get(t)
            if not eff:
                continue
            Y_sub[i, j] = not eff.isdisjoint(subs)
            Y_sop[i, j] = not eff.isdisjoint(sop)
            Y_down[i, j] = not eff.isdisjoint(down)

    coverage = pert_in_model[:, None] & tf_has_effector[None, :]
    return GTLabels(
        Y_substrate=Y_sub,
        Y_substrate_or_product=Y_sop,
        Y_downstream=Y_down,
        coverage=coverage,
        perturbations=list(perturbations),
        tf_names=list(tf_names),
    )
