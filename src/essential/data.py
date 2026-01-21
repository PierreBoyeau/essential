import pandas as pd
import numpy as np
import os
import scanpy as sc
import json

# Paths used in experiments - keeping them centralized here
FITNESS_DATA_PATH = "/workspace/data/calvo2020_dcas9fitness/Supp_data2_log2FC.csv"
CRISPRI_DATA_PATH = "/workspace/data/251117_genomescale_CRISPRi/sample_mix_umi200_hvg500_pc25_neighbors10_mindist0.55.h5ad"
LLM_EMBEDDINGS_DIR = "/workspace/data/e_coli_llm_embeddings"
PATH_TO_REGULONDB = "/workspace/data/RegulonDB/RISet.tsv"
PATH_TO_KEGG = "/workspace/data/KEGG/eco_pathways.json"

COLUMNS_TO_KEEP_REGULONDB = [
    "ri_id",
    "ri_type",
    "tf_promoter",
    "target_gene",
    "regulator_gene",
    "is_evidence",
    "confidenceLevel",
]

# --- From data_resources.py ---


def load_fitness_data(path: str = FITNESS_DATA_PATH) -> pd.DataFrame:
    """
    Loads the fitness data from the CSV file.

    Args:
        path: Path to the fitness data CSV.

    Returns:
        pd.DataFrame: The fitness data with 'spacer' as the index.
    """
    fitness_df = pd.read_csv(path).rename(columns={"Unnamed: 0": "spacer"}).set_index("spacer")
    return fitness_df


def load_crispri_data(path: str = CRISPRI_DATA_PATH) -> sc.AnnData:
    """
    Loads and normalizes the CRISPRi AnnData object.

    Args:
        path: Path to the .h5ad file.

    Returns:
        sc.AnnData: The normalized AnnData object.
    """
    adata = sc.read_h5ad(path)
    adata.X = adata.layers["reads"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


def load_llm_embeddings(directory: str = LLM_EMBEDDINGS_DIR) -> pd.DataFrame:
    """
    Loads LLM embeddings for genes.

    Args:
        directory: Directory containing the 'llm_embeddings.npz' file.

    Returns:
        pd.DataFrame: DataFrame of embeddings indexed by gene name.
    """
    llm_embeddings = np.load(os.path.join(directory, "llm_embeddings.npz"))
    # gene_order = pd.Series(np.arange(len(llm_embeddings["genes"])), index=llm_embeddings["genes"])
    llm_df = pd.DataFrame(llm_embeddings["embeddings"], index=llm_embeddings["genes"])
    return llm_df


def one_hot_encode(sequence: str) -> np.ndarray:
    """
    One-hot encodes a DNA sequence.

    Args:
        sequence: DNA sequence string (e.g., "ACGT").

    Returns:
        np.ndarray: One-hot encoded sequence flattened.
    """
    mapping = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0], "T": [0, 0, 0, 1]}
    seq_upper = sequence.upper()
    encoded = np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq_upper])
    return encoded


# --- From legacy/ode/utils.py ---


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
            is_evidence_weak=lambda x: x["confidenceLevel"] == "W",
            is_evidence_strong=lambda x: x["confidenceLevel"] == "S",
            is_evidence_confirmed=lambda x: x["confidenceLevel"] == "C",
        )
        .loc[:, COLUMNS_TO_KEEP_REGULONDB]
    )
    return ref_db


def load_regulondb_full(drop_duplicates=True):
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
            ri_id=lambda x: x["riId"],
            ri_type=lambda x: x["riType"],
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
        .loc[:, COLUMNS_TO_KEEP_REGULONDB]
        .dropna(subset=["target_gene"])  # Remove rows with no valid target genes
    )
    if drop_duplicates:
        confidence_order = ["W", "S", "C"]
        ref_db["confidenceLevel"] = pd.Categorical(
            ref_db["confidenceLevel"], categories=confidence_order, ordered=True
        )
        ref_db = ref_db.sort_values("confidenceLevel").drop_duplicates(
            subset=["target_gene", "regulator_gene"], keep="last"
        )
    ref_db = ref_db.assign(
        is_evidence_weak=lambda x: x["confidenceLevel"] == "W",
        is_evidence_strong=lambda x: x["confidenceLevel"] == "S",
        is_evidence_confirmed=lambda x: x["confidenceLevel"] == "C",
    )
    return ref_db


def load_kegg_pathways(path: str = PATH_TO_KEGG):
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

