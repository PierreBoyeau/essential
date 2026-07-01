"""Build a regulator-target mask from a RegulonDB-style edge list."""

import re

import numpy as np
import pandas as pd

REGULONDB_SYNONYM_TSV = "/workspace/data/RegulonDB/GeneProductAllIdentifiersSet.tsv"


def load_synonym_rows(path: str = REGULONDB_SYNONYM_TSV) -> pd.DataFrame:
    """Parse RegulonDB's ``GeneProductAllIdentifiersSet.tsv`` into a per-row
    alias set.

    Returns
    -------
    pandas.DataFrame
        Columns:
          ``gene_name`` : str       -- canonical (lowercased) gene name
          ``aliases``   : set[str]  -- all lowercased aliases from that row
                                       (gene name, gene synonyms, product name,
                                       product synonyms), stripped of HTML tags
                                       and of pure database-ID tokens
                                       (``b#``, ``EG#``, ``ECK#``, ``g#-#``).
    """
    df = pd.read_csv(path, sep="\t", skiprows=28)
    df.columns = [c.split(")", 1)[-1].strip() for c in df.columns]

    def _toks(v):
        if not isinstance(v, str):
            return set()
        out = set()
        for t in re.split(r"[,;]", v):
            t = re.sub(r"<[^>]+>", "", t).strip().lower()
            if t and not re.fullmatch(r"(eg|g|b|eck)\d+|g\d+-\d+", t):
                out.add(t)
        return out

    rows = []
    for _, r in df.iterrows():
        gn = str(r["geneName"]).strip().lower()
        aliases = (
            {gn}
            | _toks(r.get("geneSynonyms"))
            | _toks(r.get("productName"))
            | _toks(r.get("productSynonyms"))
        )
        rows.append({"gene_name": gn, "aliases": aliases})
    return pd.DataFrame(rows)


def reconcile_names(names, synonym_rows, reference, *, verbose: bool = True):
    """Reconcile a list of names against a canonical reference via RegulonDB
    synonyms.

    For each name ``n`` in ``names`` (compared lowercased):
      1. if ``n`` is in ``reference``, keep it;
      2. else, if any of ``n``'s row-mates in ``synonym_rows`` is in
         ``reference``, return that row-mate (ties broken alphabetically);
      3. else, return ``n`` unchanged and record it as unresolved.

    Parameters
    ----------
    names : iterable[str]
        Names to reconcile. NaN / non-string entries pass through untouched.
    synonym_rows : pandas.DataFrame
        Output of :func:`load_synonym_rows` (columns ``gene_name``, ``aliases``).
    reference : iterable[str]
        The canonical name set to reconcile TO (e.g. ``adata.var_names``).
    verbose : bool
        Print a summary and list renamed / unresolved entries.

    Returns
    -------
    pandas.Series
        Same length as ``names``, index preserved if ``names`` is a Series.
    """
    ref = set(reference)

    alias_to_siblings: dict[str, set[str]] = {}
    for aliases in synonym_rows["aliases"]:
        for a in aliases:
            alias_to_siblings.setdefault(a, set()).update(aliases)

    names_list = list(names)
    resolved: list = []
    renamed: list[tuple[str, str]] = []
    unresolved: list[str] = []

    for orig in names_list:
        if not isinstance(orig, str):
            resolved.append(orig)
            continue
        n = orig.lower()
        if n in ref:
            resolved.append(n)
            continue
        hits = alias_to_siblings.get(n, set()) & ref
        if hits:
            new = sorted(hits)[0]
            resolved.append(new)
            renamed.append((n, new))
        else:
            resolved.append(n)
            unresolved.append(n)

    if verbose:
        n_total = len(resolved)
        n_kept = n_total - len(renamed) - len(unresolved)
        print(
            f"reconcile_names: {n_total} in | {n_kept} already in reference | "
            f"{len(renamed)} renamed | {len(unresolved)} unresolved"
        )
        if renamed:
            uniq = sorted(set(renamed))
            head = ", ".join(f"{o}->{v}" for o, v in uniq[:10])
            more = "" if len(uniq) <= 10 else f" (+{len(uniq) - 10} more)"
            print(f"  renamed: {head}{more}")
        if unresolved:
            uniq = sorted(set(unresolved))
            head = ", ".join(uniq[:20])
            more = "" if len(uniq) <= 20 else f" (+{len(uniq) - 20} more)"
            print(f"  unresolved: {head}{more}")

    if isinstance(names, pd.Series):
        return pd.Series(resolved, index=names.index, name=names.name)
    return pd.Series(resolved)


def build_tf_mask(var_names, ref_db) -> dict:
    """Build a (n_genes, n_tfs) regulator mask.

    Parameters
    ----------
    var_names : iterable[str]
        Gene symbols defining the (n_genes,) axis. Matching is case-sensitive;
        the caller is responsible for case folding (e.g. ``adata.var_names.str.lower()``).
    ref_db : pandas.DataFrame
        Edge list with columns ``regulator_gene`` and ``target_gene``.  Any
        upstream filtering (e.g. ``ri_type.startswith("TF")``) is the caller's
        responsibility.

    Returns
    -------
    dict
        ``tf_genes``  : sorted list[str] of TF symbols present in ``var_names``
        ``tf_cols``   : (n_tfs,) int64 — indices into ``var_names``
        ``Amask_tf``  : (n_genes, n_tfs) float32 — ``1`` iff TF k regulates gene i
    """
    var_names = list(var_names)
    gene_idx = {g: i for i, g in enumerate(var_names)}

    candidate_tfs = set(ref_db["regulator_gene"].unique())
    tf_genes = sorted(g for g in candidate_tfs if g in gene_idx)
    tf_idx = {g: k for k, g in enumerate(tf_genes)}
    tf_cols = np.array([gene_idx[g] for g in tf_genes], dtype=np.int64)

    n_genes, n_tfs = len(var_names), len(tf_genes)
    Amask = np.zeros((n_genes, n_tfs), dtype=np.float32)
    for r, t in zip(ref_db["regulator_gene"], ref_db["target_gene"]):
        if t in gene_idx and r in tf_idx:
            Amask[gene_idx[t], tf_idx[r]] = 1.0

    return {"tf_genes": tf_genes, "tf_cols": tf_cols, "Amask_tf": Amask}
