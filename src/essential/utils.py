import pandas as pd
import numpy as np
import json

import sklearn.metrics as metrics
from skimage.filters import threshold_otsu


COLUMNS_TO_KEEP = [
    "tf_promoter",
    "target_gene",
    "regulator_gene",
    "is_evidence",
    "confidenceLevel",
]

PATH_TO_REGULONDB = "/workspace/data/RegulonDB/RISet.tsv"


def _parse_gene_group(gene_group):
    """
    Parse a gene group string into individual gene names.

    Rules:
    - If 0 or 1 uppercase letter: single gene (e.g., 'nadE', 'sdhX', 'uof')
    - If 2+ consecutive uppercase letters at end: gene set (e.g., 'gspCDEF' → gspC, gspD, gspE, gspF)

    Parameters
    ----------
    gene_group : str
        A gene or gene group identifier.

    Returns
    -------
    list of str
        List of individual gene names.
    """
    if not gene_group:
        return []
    import re

    match = re.search(r"([A-Z]{2,})$", gene_group)

    if match:
        uppercase_suffix = match.group(1)
        prefix = gene_group[: match.start()]
        return [prefix + letter for letter in uppercase_suffix]
    else:
        return [gene_group]


def _parse_target_genes(target_tu_or_gene):
    """
    Extract gene list from targetTuOrGene column.

    Format: 'RDBECOLITUCXXXXX:gene1-genegroup2-gene3'
    where genegroups may be compact notation like 'sdhCDAB' → [sdhC, sdhD, sdhA, sdhB]

    Parameters
    ----------
    target_tu_or_gene : str
        Target TU or gene identifier from RegulonDB.

    Returns
    -------
    list of str
        List of individual gene names.
    """
    if pd.isna(target_tu_or_gene):
        return []

    if ":" not in target_tu_or_gene:
        return []

    gene_part = target_tu_or_gene.split(":", 1)[1]
    gene_groups = gene_part.split("-")
    all_genes = []
    for group in gene_groups:
        all_genes.extend(_parse_gene_group(group))

    return all_genes


def load_regulondb():
    """
    Load and preprocess RegulonDB TF-promoter interactions.

    Reads `PATH_TO_REGULONDB` TSV (skipping header rows), filters to
    `riType == 'tf-promoter'`, and constructs columns: `tf_promoter`
    (<regulator>_<firstGene>), `target_gene`, `regulator_gene`, and
    `is_evidence`.

    Returns
    -------
    pandas.DataFrame
        Columns: `tf_promoter`, `target_gene`, `regulator_gene`,
        `is_evidence`, `confidenceLevel`.
    """
    ref_db = pd.read_csv(PATH_TO_REGULONDB, skiprows=44, sep="\t")
    ref_db.columns = ref_db.columns.str.replace(r"^\d+\)", "", regex=True)
    ref_db = (
        ref_db.loc[lambda x: x["riType"].isin(["tf-promoter", "tf-gene"])]
        .assign(
            tf_promoter=lambda x: x["regulatorName"].str.lower() + "_" + x["firstGene"].str.lower(),
            target_gene=lambda x: x["firstGene"].str.lower(),
            regulator_gene=lambda x: x["regulatorName"].str.lower(),
            is_evidence=True,
        )
        .loc[:, COLUMNS_TO_KEEP]
    )
    return ref_db


def load_regulondb_full():
    """
    Load and preprocess RegulonDB regulatory interactions with expanded gene targets.

    Unlike `load_regulondb()`, this function parses the `targetTuOrGene` column
    to extract all individual genes from transcription units, creating one row
    per regulator-target gene pair. This properly handles polycistronic operons
    and compact gene notation (e.g., 'gspCDEF' → gspC, gspD, gspE, gspF).

    Includes all regulator types: transcription factors, sRNAs, and compounds.

    Returns
    -------
    pandas.DataFrame
        Columns: `tf_promoter`, `target_gene`, `regulator_gene`,
        `is_evidence`, `confidenceLevel`.
    """
    ref_db = pd.read_csv(PATH_TO_REGULONDB, skiprows=44, sep="\t")
    ref_db.columns = ref_db.columns.str.replace(r"^\d+\)", "", regex=True)

    # Filter for all regulator types
    valid_ri_types = [
        "tf-promoter",
        "tf-gene",
        "tf-tu",
        "srna-promoter",
        "srna-gene",
        "srna-tu",
        "compound-promoter",
        "compound-gene",
        "compound-tu",
    ]

    ref_db = (
        ref_db.loc[lambda x: x["riType"].isin(valid_ri_types)]
        .assign(
            target_genes=lambda x: x["targetTuOrGene"].apply(_parse_target_genes),
            regulator_gene=lambda x: x["regulatorName"].str.lower(),
            # regulator_gene=lambda x: x["regulatorName"],
        )
        .explode("target_genes")  # Create one row per target gene
        .rename(columns={"target_genes": "target_gene"})
        .assign(
            target_gene=lambda x: x["target_gene"].str.lower(),
            # target_gene=lambda x: x["target_gene"],
            tf_promoter=lambda x: x["regulator_gene"] + "_" + x["target_gene"],
            is_evidence=True,
        )
        .loc[:, COLUMNS_TO_KEEP]
        .dropna(subset=["target_gene"])  # Remove rows with no valid target genes
    )

    return ref_db


def load_results(filename):
    """
    Load model scores from a NumPy `.npz` file and return a long-form edge table.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to a `.npz` file with keys `matrix` (square score matrix) and
        `genes`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns `target_gene`, `regulator_gene`, and `score`.
    """
    saved = np.load(filename, allow_pickle=True)
    mat = saved["matrix"]
    genes = saved["genes"]
    df = (
        pd.DataFrame(mat, index=genes, columns=genes)
        .unstack()
        .to_frame("score")
        .reset_index()
        .rename(
            columns={
                "level_0": "target_gene",
                "level_1": "regulator_gene",
            }
        )
        .assign(
            target_gene=lambda x: x["target_gene"].str.lower(),
            regulator_gene=lambda x: x["regulator_gene"].str.lower(),
        )
    )
    return df


def compute_metrics(
    results_df,
    gt_col="is_evidence",
    score_col="score",
    decision_cols=None,
):
    """
    Compute precision–recall metrics and optional decision-threshold metrics.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Must contain columns `gt_col` and `score_col`. If `decision_cols` is
        provided, each corresponding column should be boolean decisions (0/1).
    gt_col : str, default "is_evidence"
        Name of the ground-truth boolean column.
    score_col : str, default "score"
        Name of the predicted score column.
    decision_cols : list[str] or None, default None
        Optional list of decision columns. For each column name, the suffix
        after the first underscore is used to label metrics (e.g.,
        `decision_0p90` -> type `0p90`).

    Returns
    -------
    dict
        Dictionary containing `pr_auc`, `ap`, `ntotal` and, if `decision_cols`
        is given, `precision_*`, `recall_*`, `ndetections_*`, and
        `ntruepos_*`.
    """
    base_metrics = compute_pr_metrics(
        y_true=results_df[gt_col],
        y_score=results_df[score_col],
    )
    metrics_res = {
        "pr_auc": base_metrics["auc"],
        "ap": base_metrics["ap"],
        "ntotal": results_df.shape[0],
    }

    if decision_cols is not None:
        for decision_col in decision_cols:
            decision_type = decision_col.split("_")[1]
            decision_metrics = compute_pr_metrics(
                y_true=results_df[gt_col],
                y_score=results_df[score_col],
                y_decision=results_df[decision_col],
            )
            metrics_res[f"precision_{decision_type}"] = decision_metrics.get("precision", 0.0)
            metrics_res[f"recall_{decision_type}"] = decision_metrics.get("recall", 0.0)
            metrics_res[f"ndetections_{decision_type}"] = (results_df[decision_col]).sum()
            metrics_res[f"ntruepos_{decision_type}"] = (
                results_df[gt_col] & results_df[decision_col]
            ).sum()
    return metrics_res


def compute_pr_metrics(y_true, y_score, y_decision=None, return_pandas=False):
    """
    Compute precision–recall summary metrics, optionally at a decision threshold.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels (0/1).
    y_score : array-like of shape (n_samples,)
        Continuous scores; larger implies more likely positive.
    y_decision : array-like of shape (n_samples,), optional
        Binary decisions (0/1). If provided, include precision and recall at
        this operating point.

    Returns
    -------
    dict
        Keys:
        - 'auc': area under the precision–recall curve
        - 'ap': average precision
        - 'precision' and 'recall' if y_decision is provided
    """
    if y_decision is not None:
        decision_precision = metrics.precision_score(y_true, y_decision, zero_division=0)
        decision_recall = metrics.recall_score(y_true, y_decision, zero_division=0)

    prec, rec, _ = metrics.precision_recall_curve(y_true, y_score)
    result = {
        "auc": metrics.auc(rec, prec),
        "ap": metrics.average_precision_score(y_true, y_score),
    }
    if y_decision is not None:
        result.update({"precision": decision_precision, "recall": decision_recall})
    if return_pandas:
        return pd.Series(result)
    return result


def load_kegg_pathways():
    path = "/workspace/data/KEGG/eco_pathways.json"
    with open(path, "r") as f:
        records = json.load(f)
    df = pd.DataFrame(records)

    def first_pathway(p):
        if isinstance(p, dict) and p:
            return next(iter(p.values()))
        return "N/A"

    df["pathway1"] = df["pathways"].apply(first_pathway)
    df["target_gene"] = df["query_gene"].str.lower()
    return df


def evaluate_interactions_on_regulondb(df_ode):
    """
    Evaluate a dataframe of predicted interactions against RegulonDB.
    """
    ref_db = load_regulondb()
    valid_regulators = ref_db["regulator_gene"].unique()
    df_ode_reversed = df_ode.rename(
        columns={"target_gene": "regulator_gene", "regulator_gene": "target_gene"}
    )

    def metrics_for_df(df):
        merged_ = df.merge(ref_db, on=["target_gene", "regulator_gene"], how="left").assign(
            is_evidence=lambda x: x["is_evidence"].fillna(False),
            score_abs=lambda x: np.abs(x["score"]),
        )
        merged_ = merged_.loc[lambda x: x["regulator_gene"].isin(valid_regulators)]
        otsu_thresh = threshold_otsu(merged_["score_abs"].values)
        merged_["decision_otsu"] = merged_["score_abs"].values > otsu_thresh
        merged_["decision_1e-3"] = merged_["score_abs"].values > 1e-3
        merged_["decision_1e-1"] = merged_["score_abs"].values > 1e-1
        merged_["decision_5e-1"] = merged_["score_abs"].values > 5e-1
        metrics_res = compute_metrics(
            merged_,
            score_col="score",
            gt_col="is_evidence",
            decision_cols=["decision_otsu", "decision_1e-3", "decision_1e-1", "decision_5e-1"],
        )
        return metrics_res

    metrics_res = metrics_for_df(df_ode)
    metrics_res_reversed = metrics_for_df(df_ode_reversed)
    metrics_res_reversed = {f"reversed_{k}": v for k, v in metrics_res_reversed.items()}
    metrics_res.update(metrics_res_reversed)
    return metrics_res
