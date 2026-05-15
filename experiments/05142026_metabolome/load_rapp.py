"""
Data loaders for the Rapp 2026 E. coli CRISPRi metabolomics dataset.
All files are read from /workspace/data/Rapp_2026/.
See README.md for a full description of each file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/workspace/data/Rapp_2026")


def load_sgrna_library() -> pd.DataFrame:
    """
    Table S1 (mmc2): sgRNA library.

    Returns one row per gene with columns:
      Gene, sgRNA Nr., b-Nr., base pairing region, Oligo sequence
    """
    return pd.read_excel(DATA_DIR / "mmc2.xlsx", header=0)


def load_growth_curves() -> tuple[pd.DataFrame, np.ndarray]:
    """
    Table S2 (mmc3): OD600 growth curves.

    The raw sheet has a two-row header; this function resolves it into clean
    column names.

    Returns
    -------
    curves : DataFrame
        Metadata columns: Replicate (int), RXN_Nr (int), Gene, Guide, b_Nr, Plate.
        OD columns named by time as float hours (181 values, 0–30 h, ~10 min spacing).
    time_h : ndarray, shape (181,)
        Time points in hours.
    """
    raw = pd.read_excel(DATA_DIR / "mmc3.xlsx", header=None)
    time_h: np.ndarray = raw.iloc[1, 6:].values.astype(float)

    meta_cols = ["Replicate", "RXN_Nr", "Gene", "Guide", "b_Nr", "Plate"]
    data = raw.iloc[2:].copy()
    data.columns = meta_cols + time_h.tolist()
    data = data.reset_index(drop=True)

    data["Replicate"] = data["Replicate"].astype(int)
    data["RXN_Nr"] = data["RXN_Nr"].astype("Int64")  # nullable; NaN for control rows
    data[time_h.tolist()] = data[time_h.tolist()].astype(float)

    return data, time_h


def load_endpoint_od() -> pd.DataFrame:
    """
    Table S3 (mmc4): Endpoint OD600 at harvest.

    Columns: Gene, b_Nr, OD, Plate, Well, Replicate, Sample_ID.
    Sample_ID matches column names in the metabolomics matrix (mmc5).
    """
    df = pd.read_excel(DATA_DIR / "mmc4.xlsx", header=0)
    df.columns = ["Gene", "b_Nr", "OD", "Plate", "Well", "Replicate", "Sample_ID"]
    return df


def load_metabolomics_matrix() -> pd.DataFrame:
    """
    Table S4 (mmc5): Full metabolomics fold-change matrix.

    Shape: 1880 ions × 3030 columns.
    First four columns are metadata (Abbrev, Metabolite, Mass, KEGG).
    Remaining 3026 columns are samples named {gene}_{Rn}_{msAVxxx}_{Bn}.
    Values are fold-changes relative to WT.
    """
    df = pd.read_excel(DATA_DIR / "mmc5.xlsx", header=0)
    df = df.rename(
        columns={
            df.columns[0]: "Abbrev",
            df.columns[1]: "Metabolite",
            df.columns[2]: "Mass",
            df.columns[3]: "KEGG",
        }
    )
    return df


def load_significant_hits_annotated() -> pd.DataFrame:
    """
    Table S5 (mmc6): Significant metabolite hits with biological annotation.

    Shape: 1385 rows × 29 cols.
    Key columns: Gene, Metabolite, Metabolite Abbreviation, Mean_FC, R1_FC, R2_FC,
    Reactant, Subsystem, FBA, Substrate Abb, Product Abb, Operon effect,
    Pathways, Position, Upstream Accumulation, Downstream Accumulation.
    """
    return pd.read_excel(DATA_DIR / "mmc6.xlsx", sheet_name="TableS5", header=0)


def load_significant_hits_ms2() -> pd.DataFrame:
    """
    Table S6 (mmc7): Significant hits with full LC-MS/MS spectral evidence and QC.

    Shape: 1256 rows × 54 cols.
    Extends Table S5 with QC flags, SMILES, retention times, MS2 fragment lists
    at three collision energies (CE10, CE20, CE40).
    """
    return pd.read_excel(DATA_DIR / "mmc7.xlsx", sheet_name="Table_S6", header=0)


def load_significant_fold_changes() -> pd.DataFrame:
    """
    Table S7 (mmc8): Significant metabolite fold-changes per gene knockdown.

    Shape: 9462 rows × 10 cols.
    One row per (gene, metabolite, ionisation mode).
    Columns: Gene, Metabolite, Mass, Mode, Mean_FC, R1_FC, R2_FC,
             Mean_Int, R1_Int, R2_Int.
    """
    df = pd.read_excel(DATA_DIR / "mmc8.xlsx", header=0)
    df.columns = [
        "Gene",
        "Metabolite",
        "Mass",
        "Mode",
        "Mean_FC",
        "R1_FC",
        "R2_FC",
        "Mean_Int",
        "R1_Int",
        "R2_Int",
    ]
    return df


def load_nonannotated_features() -> pd.DataFrame:
    """
    Table S8 (mmc9): Non-annotated iML1515 m/z features (SIRIUS output).

    Shape: 52 rows × 27 cols.
    Columns include Polarity, ConfidenceScoreExact, ConfidenceScoreApproximate,
    CSI:FingerIDScore, ZodiacScore, SiriusScore.
    """
    return pd.read_excel(DATA_DIR / "mmc9.xlsx", header=None, skiprows=2)


def load_metabolite_reference() -> pd.DataFrame:
    """
    Table S9 (mmc10): Metabolite reference table.

    Shape: 802 rows × 6 cols.
    Columns: Abbreviation, BIGG, Metabolite, KEGG, Monoisotopic_Mass, Formula.
    """
    df = pd.read_excel(DATA_DIR / "mmc10.xlsx", header=0)
    df.columns = ["Abbreviation", "BIGG", "Metabolite", "KEGG", "Monoisotopic_Mass", "Formula"]
    return df


def load_gene_substrates() -> pd.DataFrame:
    """
    Table S10 (mmc11): Gene → substrate metabolites from iML1515.

    Shape: 7683 rows × 9 cols.
    Columns: gene, Gene_Number, Abbreviation, Metabolite, Subsystem,
             KEGG, BIGG, Monoisotopic_Mass, Formula.
    """
    df = pd.read_excel(DATA_DIR / "mmc11.xlsx", header=0)
    df.columns = [
        "gene",
        "Gene_Number",
        "Abbreviation",
        "Metabolite",
        "Subsystem",
        "KEGG",
        "BIGG",
        "Monoisotopic_Mass",
        "Formula",
    ]
    return df


def load_pathway_positions() -> pd.DataFrame:
    """
    Table S11 (mmc12): Gene positions within metabolic pathways.

    Shape: 3340 rows × 11 cols.
    Columns: Position, Pathway_Abbrev, Pathway, GeneID, GeneAccession, GeneName,
             ReactionId, ReactionEC, EnzymaticActivity, Evidence, Hierarchy.
    """
    df = pd.read_excel(DATA_DIR / "mmc12.xlsx", header=0)
    df.columns = [
        "Position",
        "Pathway_Abbrev",
        "Pathway",
        "GeneID",
        "GeneAccession",
        "GeneName",
        "ReactionId",
        "ReactionEC",
        "EnzymaticActivity",
        "Evidence",
        "Hierarchy",
    ]
    return df


def load_pathway_metabolites() -> pd.DataFrame:
    """
    Table S12 (mmc13): Extended pathway-metabolite mapping.

    Shape: 11851 rows × 18 cols.
    Extends Table S11 with the metabolites at each pathway position
    (substrates/products), enabling upstream/downstream accumulation analysis.
    Extra columns over S11: metAbb, metAbNames, subSys, KEGGID, BIGG,
    mass, mass_13C, NeutralFormula.
    """
    df = pd.read_excel(DATA_DIR / "mmc13.xlsx", header=0)
    df.columns = [
        "Position",
        "Pathway_Abbrev",
        "Pathway",
        "GeneID",
        "GeneAccession",
        "GeneName",
        "ReactionId",
        "ReactionEC",
        "EnzymaticActivity",
        "Evidence",
        "metAbb",
        "metAbNames",
        "subSys",
        "KEGGID",
        "BIGG",
        "mass",
        "mass_13C",
        "NeutralFormula",
    ]
    return df
