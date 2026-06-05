import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import anndata as ad
import bbknn
import harmonypy as hm
import helper_functions as hf
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ssp
import seaborn as sns
from pathways import pathway_colors, pathways
from plotting import calculate_umap, cdf, plot_umap, plot_umap_layered_pathways
from scipy.sparse import issparse

index_mapping = {
    "ATAGAGAG": "ACTCTAGG",
    "AGAGGATA": "TCTTACGC",
    "CTCCTTAC": "CTTAATAG",
    "TATGCAGT": "ATAGCCTT",
    "TACTCCTT": "TCGCATAA",
    "AGGCTTAG": "TTACCTCC",
    "ATTAGACG": "CAGTTATG",
    "CGGAGAGA": "CCTTTACT",
    "CTAGTCGA": "GACGATTA",
    "AGCTAGAA": "GAGACGGA",
}

swapped_index_mapping = {
    "ACTCTAGG": "ATAGAGAG",
    "TCTTACGC": "AGAGGATA",
    "CTTAATAG": "CTCCTTAC",
    "ATAGCCTT": "TATGCAGT",
    "TCGCATAA": "TACTCCTT",
    "TTACCTCC": "AGGCTTAG",
    "CAGTTATG": "ATTAGACG",
    "CCTTTACT": "CGGAGAGA",
    "GACGATTA": "CTAGTCGA",
    "GAGACGGA": "AGCTAGAA",
}


def unify_index(cell_barcode, index_map):
    cell_index = cell_barcode[-8:]
    try:
        return cell_barcode[:-8] + index_map[cell_index]
    except KeyError:
        return cell_barcode[:-8] + cell_index


hf.set_plot_defaults(fontsize=12)

crispri_library_info = pd.read_csv(
    "/Users/james/HMS Dropbox/James Taggart/crispri_library_design/bikard_lab_2020NAR_allgene/2020bikard_allgene_library_info.csv"
)
crispri_library_info["all_genes_targeted"].fillna("nontargeting", inplace=True)


# Mapping from genotyping i5 index → cDNA i5 index (lce75 only)
def rc(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


_geno_cdna_pairs = [
    ("ATAGAGAG", "AGGGACTG"),
    ("AGAGGATA", "ATCCGCAT"),
    ("CTCCTTAC", "ATGACTTG"),
    ("TATGCAGT", "ATGCATAT"),
    ("TACTCCTT", "ATTTCCAT"),
    ("AGGCTTAG", "CAACGCAG"),
    ("ATTAGACG", "CAATTCTC"),
    ("CGGAGAGA", "CACAAGTA"),
    ("CTAGTCGA", "CCATCCAC"),
    ("AGCTAGAA", "CGTGTACA"),
    ("ACTCTAGG", "CTAACGCC"),
    ("TCTTACGC", "CTTTATCC"),
    ("CTTAATAG", "GAGAAACC"),
    ("ATAGCCTT", "GAGTGTAC"),
    ("TAAGGCTC", "GATGGTTA"),
    ("TCGCATAA", "GCCAACAT"),
    ("GTAACGTT", "GCGTGCAA"),
    ("ACTGAGTT", "GCTCGTAG"),
    ("ATACTCTT", "GGAAGTCC"),
    ("GACAACTT", "GGACTGGA"),
    ("GAACGATT", "GGTAGCCA"),
    ("AAGATTGT", "GTAATCTG"),
    ("AATCGGGT", "GTACGCTT"),
    ("TTGAGGGT", "GTCCACTA"),
    ("AAGCTTCT", "GTTGTCCG"),
    ("TATTGCCT", "TAATCCAT"),
]
geno_i5_to_cdna_i5 = {cdna: rc(geno) for cdna, geno in _geno_cdna_pairs}


def translate_geno_barcode(bc):
    """Replace genotyping i5 suffix with the corresponding cDNA i5."""
    i5 = bc[-8:]
    return bc[:-8] + geno_i5_to_cdna_i5.get(i5, i5)


colony_genotypes = pd.read_csv("genotyping/cell_barcode_consensus_thresh10.tsv", sep="\t")
colony_genotypes["corrected_cell_barcode"] = colony_genotypes["cell_barcode"].map(
    translate_geno_barcode
)

# ── Gene→pathway mapping (built once, shared across all datasets) ─────────────
gene_to_pathway = {}
for pathway_name, genes in pathways.items():
    for gene in genes:
        if gene not in gene_to_pathway:
            gene_to_pathway[gene] = pathway_name

# ── Dataset definitions ───────────────────────────────────────────────────────
# Set USE_HARMONY=True to load Harmony rt_bc-corrected UMAPs produced by
# preprocess_h5ad_harmony.py; False loads the original UMAP h5ads.
USE_HARMONY = True

_HARMONY_SUFFIX = "harmony_rtbc_hvg1000_pc70_neighbors10_mindist0.55.h5ad"

datasets = [
    {
        "name": "lce75",
        "h5ad": (
            f"/Users/james/Dropbox (HMS)/ecoli_essential_gene_crispri/data/sequencing/"
            f"260309_LCE75_genomescale_ezrdm_glu/processed_h5ad/lce75_{_HARMONY_SUFFIX}"
            if USE_HARMONY
            else "/Users/james/Dropbox (HMS)/ecoli_essential_gene_crispri/data/sequencing/"
            "260309_LCE75_genomescale_ezrdm_glu/processed_h5ad/"
            "260309_lce75_genomescale_ezrdm_glu_umap.h5ad"
        ),
        "use_colony_genotyping": True,
    },
    {
        "name": "DE122",
        "h5ad": (
            f"/Users/james/HMS Dropbox/James Taggart/ecoli_essential_gene_crispri/"
            f"data/sequencing/251117_genomescale_CRISPRi_biorep1/processed_h5ad/DE122_{_HARMONY_SUFFIX}"
            if USE_HARMONY
            else "/Users/james/HMS Dropbox/James Taggart/ecoli_essential_gene_crispri/data/sequencing/"
            "251117_genomescale_CRISPRi_biorep1/processed_h5ad/"
            "Nov2025_DE122_genomescale_EZRDM_Glu_newpipeline_umap.h5ad"
        ),
        "use_colony_genotyping": False,
    },
]

adatas = {}

for ds in datasets:
    name = ds["name"]
    print(f"\n=== Processing {name} ===")

    adata = ad.read_h5ad(ds["h5ad"])

    adata.uns["min_tot"] = 10000
    keep = adata.obs["n_counts"] >= adata.uns["min_tot"]
    adata = adata[keep].copy()

    # Barcode extraction
    bc1 = []
    bc2 = []
    bc3 = []
    bc4 = []
    indexes = []
    for bc in adata.obs.index:
        bc1.append(bc[0:6])
        bc2.append(bc[6:12])
        bc3.append(bc[13:19])
        bc4.append(bc[19:25])
        indexes.append(bc[26:35])
    adata.obs["bc1"] = bc1
    adata.obs["bc2"] = bc2
    adata.obs["bc3"] = bc3
    adata.obs["bc4"] = bc4
    adata.obs["indexes"] = indexes
    adata.obs["log2_counts"] = np.log2(adata.obs["n_counts"])
    adata.obs["log10_counts"] = np.log10(adata.obs["n_counts"])

    # Genotyping: colony remapping for lce75; built-in target column for DE122
    if ds["use_colony_genotyping"]:
        colony_genotype_dict = colony_genotypes.set_index("corrected_cell_barcode")[
            "consensus_barcode"
        ].to_dict()
        spacer_to_gene = crispri_library_info.set_index("spacer")["all_genes_targeted"].to_dict()
        adata.obs["spacer"] = adata.obs.index.map(colony_genotype_dict)
        adata.obs["target"] = adata.obs["spacer"].map(spacer_to_gene)

    adata.obs["target"] = adata.obs["target"].astype("category")
    adata.obs["rt_bc"] = adata.obs["rt_bc"].astype("category")
    adata.obs["target_pathway"] = (
        adata.obs["target"].astype(object).map(gene_to_pathway).fillna("other")
    )
    adata.obs["pathway_color"] = (
        adata.obs["target_pathway"].map(pathway_colors).fillna(pathway_colors["other"])
    )

    # Output directories
    os.makedirs(f"figures/{name}", exist_ok=True)
    os.makedirs(f"figures/{name}/pathway_umaps", exist_ok=True)

    # ── UMAPs ─────────────────────────────────────────────────────────────────
    plot_umap(
        adata,
        outbase=f"{name}/log10_counts",
        colorvals="log10_counts",
        legend_loc="right",
        figsize=(10, 10),
        s=10,
    )
    plot_umap(
        adata,
        outbase=f"{name}/rt_bc",
        colorvals="rt_bc",
        legend_loc="right margin",
        figsize=(12, 10),
        s=10,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    sc.pl.umap(adata, color="target", ax=ax, show=False, legend_loc=None, s=6, title="Target gene")
    plt.tight_layout()
    plt.savefig(f"figures/{name}/umap_target.png", dpi=300)
    plt.close()

    plot_umap_layered_pathways(
        adata,
        outbase=f"{name}/umap_pathways",
        gene_column="target",
        legend_loc="right margin",
        figsize=(20, 14),
        legend_ncol=3,
    )

    # ── Per-pathway UMAPs ──────────────────────────────────────────────────────
    # print(f"  Generating per-pathway UMAPs for {name}...")
    # for pathway in pathways:
    #     plot_umap_layered_pathways(
    #         adata,
    #         outbase=f'{name}/pathway_umaps/{pathway}',
    #         pathways_to_plot=pathway,
    #         pathway_column='target_pathway',
    #         color_column='pathway_color',
    #         gene_column='target',
    #         color_by_gene=True,
    #         background_size=3,
    #         background_alpha=0.3,
    #         foreground_size=22,
    #         foreground_alpha=0.9,
    #         figsize=(12, 10)
    #     )

    # ── Unknown-function gene UMAPs (y-ome) ───────────────────────────────────
    # Gene list from Ghatak et al. 2019 NAR (doi:10.1093/nar/gkz030), Table S1.
    # Includes all E. coli genes of unknown/poorly characterized function, not
    # just those starting with 'y'. Split into chunks so each plot remains legible.
    _genes_per_plot = 20
    _min_cells_unk = 5

    _yome_df = pd.read_csv("S1 y-ome Genes.tsv", sep="\t")
    # primary_name may contain comma-separated aliases; take the first token
    _yome_df["primary_name"] = (
        _yome_df["primary_name"].astype(str).str.split(",").str[0].str.strip()
    )
    _yome_gene_set = set(_yome_df.loc[_yome_df["yome"] == "yes", "primary_name"])

    target_counts = adata.obs["target"].astype(str).value_counts()
    y_genes = sorted(
        [
            g
            for g in target_counts.index
            if g in _yome_gene_set and target_counts[g] >= _min_cells_unk
        ]
    )

    os.makedirs(f"figures/{name}/unknown_function_umaps", exist_ok=True)
    n_chunks = int(np.ceil(len(y_genes) / _genes_per_plot))
    print(
        f"  Generating unknown-function UMAPs for {name}: "
        f"{len(y_genes)} y-genes → {n_chunks} plots..."
    )

    for chunk_idx in range(n_chunks):
        chunk_genes = y_genes[chunk_idx * _genes_per_plot : (chunk_idx + 1) * _genes_per_plot]
        adata.obs["_unk_pathway"] = np.where(
            adata.obs["target"].astype(str).isin(chunk_genes), "unknown_function", "other"
        )
        plot_umap_layered_pathways(
            adata,
            outbase=f"{name}/unknown_function_umaps/y_genes_{chunk_idx+1:02d}_of_{n_chunks:02d}",
            pathways_to_plot="unknown_function",
            pathway_column="_unk_pathway",
            color_column="pathway_color",
            gene_column="target",
            color_by_gene=True,
            background_size=3,
            background_alpha=0.3,
            foreground_size=22,
            foreground_alpha=0.9,
            figsize=(12, 10),
            title=f"{name}: y-genes (plot {chunk_idx+1}/{n_chunks})",
        )

    # ── No-information gene UMAPs ──────────────────────────────────────────────
    # Ghatak et al. 2019 NAR Table S3: genes with truly no functional information.
    # Stricter subset of the y-ome; small enough to need only a few plots.
    # Uses top_target_unthresholded (unfiltered best-guess assignment) to maximise
    # cell counts for these rare targets.
    _noinf_df = pd.read_csv("S3 No Information.tsv", sep="\t")
    _noinf_gene_set = set(_noinf_df["ecocyc_primary_name"].astype(str).str.strip())

    _noinf_target_col = "top_target_unthresholded"
    _noinf_counts = adata.obs[_noinf_target_col].astype(str).value_counts()
    noinf_genes = sorted(
        [
            g
            for g in _noinf_counts.index
            if g in _noinf_gene_set and _noinf_counts[g] >= _min_cells_unk
        ]
    )

    os.makedirs(f"figures/{name}/no_information_umaps", exist_ok=True)
    n_chunks_noinf = int(np.ceil(len(noinf_genes) / _genes_per_plot))
    print(
        f"  Generating no-information UMAPs for {name}: "
        f"{len(noinf_genes)} genes → {n_chunks_noinf} plots..."
    )

    for chunk_idx in range(n_chunks_noinf):
        chunk_genes = noinf_genes[chunk_idx * _genes_per_plot : (chunk_idx + 1) * _genes_per_plot]
        adata.obs["_unk_pathway"] = np.where(
            adata.obs[_noinf_target_col].astype(str).isin(chunk_genes), "no_information", "other"
        )
        plot_umap_layered_pathways(
            adata,
            outbase=f"{name}/no_information_umaps/genes_{chunk_idx+1:02d}_of_{n_chunks_noinf:02d}",
            pathways_to_plot="no_information",
            pathway_column="_unk_pathway",
            color_column="pathway_color",
            gene_column=_noinf_target_col,
            color_by_gene=True,
            background_size=3,
            background_alpha=0.3,
            foreground_size=22,
            foreground_alpha=0.9,
            figsize=(12, 10),
            title=f"{name}: no-information genes (plot {chunk_idx+1}/{n_chunks_noinf})"
            f"\n[colored by top_target_unthresholded]",
        )

    adatas[name] = adata
    print(f"  Done: {name}")

# ── Cross-dataset Harmony integration ─────────────────────────────────────────
# Concatenate lce75 + DE122, run joint normalization → HVG → PCA → Harmony,
# then generate UMAP coloured by (a) dataset identity and (b) pathway.
# Loads the non-harmony _umap.h5ads so .raw.X = log10(1+CP10K) is available.
print("\n=== Building cross-dataset Harmony-integrated UMAP ===")

_integ_raw_paths = {
    "lce75": (
        "/Users/james/Dropbox (HMS)/ecoli_essential_gene_crispri/data/sequencing/"
        "260309_LCE75_genomescale_ezrdm_glu/processed_h5ad/"
        "260309_lce75_genomescale_ezrdm_glu_umap.h5ad"
    ),
    "DE122": (
        "/Users/james/HMS Dropbox/James Taggart/ecoli_essential_gene_crispri/"
        "data/sequencing/251117_genomescale_CRISPRi_biorep1/processed_h5ad/"
        "Nov2025_DE122_genomescale_EZRDM_Glu_newpipeline_umap.h5ad"
    ),
}

os.makedirs("figures/integrated", exist_ok=True)

_parts_integ = []
for _iname, _ipath in _integ_raw_paths.items():
    print(f"  Loading {_iname}...")
    _ia = ad.read_h5ad(_ipath)
    _ia = _ia[_ia.obs["n_counts"] >= 10_000].copy()
    print(f"    {_ia.n_obs:,} cells after count filter")

    if _iname == "lce75":
        _cdict = colony_genotypes.set_index("corrected_cell_barcode")["consensus_barcode"].to_dict()
        _s2g = crispri_library_info.set_index("spacer")["all_genes_targeted"].to_dict()
        _ia.obs["spacer"] = _ia.obs.index.map(_cdict)
        _ia.obs["target"] = _ia.obs["spacer"].map(_s2g)

    _ia.obs["target"] = _ia.obs["target"].astype(str).fillna("nontargeting")
    _ia.obs["target_pathway"] = _ia.obs["target"].map(gene_to_pathway).fillna("other")
    _ia.obs["pathway_color"] = (
        _ia.obs["target_pathway"].map(pathway_colors).fillna(pathway_colors["other"])
    )
    _ia.obs["dataset"] = _iname

    # Extract log-normalised counts from .raw (set by calculate_umap before scaling).
    # Fall back to re-normalising .X if .raw was not stored.
    if _ia.raw is not None:
        _raw_X = _ia.raw.X.toarray() if issparse(_ia.raw.X) else np.asarray(_ia.raw.X)
        _ipart = ad.AnnData(X=_raw_X, obs=_ia.obs.copy(), var=_ia.raw.var.copy())
    else:
        sc.pp.normalize_total(_ia, target_sum=1e4)
        sc.pp.log1p(_ia)
        _ipart = _ia.copy()

    _parts_integ.append(_ipart)

# Inner join on genes present in both datasets
_combined = ad.concat(_parts_integ, join="inner")
print(f"  Combined: {_combined.n_obs:,} cells × {_combined.n_vars:,} genes")

# Joint HVG selection on log-normalised data (seurat flavor suits log-space input)
sc.pp.highly_variable_genes(_combined, n_top_genes=1000, batch_key="dataset", flavor="seurat")

# PCA on HVG subset
_combined_hvg = _combined[:, _combined.var["highly_variable"]].copy()
sc.pp.scale(_combined_hvg, max_value=10)
sc.tl.pca(_combined_hvg, n_comps=70)

# Harmony batch correction: 'dataset' as the batch key
print("  Running Harmony...")
_ho = hm.run_harmony(_combined_hvg.obsm["X_pca"], _combined_hvg.obs, "dataset", max_iter_harmony=20)
_combined_hvg.obsm["X_pca_harmony"] = _ho.Z_corr

# Neighbors + UMAP from Harmony embedding
sc.pp.neighbors(_combined_hvg, use_rep="X_pca_harmony", n_neighbors=10)
sc.tl.umap(_combined_hvg, min_dist=0.55, spread=1.1, random_state=1, maxiter=2000)

# Transfer UMAP coordinates back to the full-gene object
_combined.obsm["X_umap"] = _combined_hvg.obsm["X_umap"]

# ── Plot A: coloured by dataset (integration QC) ──────────────────────────────
_ds_colors = {"lce75": "#1f77b4", "DE122": "#ff7f0e"}
fig, ax = plt.subplots(figsize=(10, 10))
for _ds in ["lce75", "DE122"]:
    _mask = _combined.obs["dataset"] == _ds
    _pts = _combined.obsm["X_umap"][_mask.values]
    ax.scatter(
        _pts[:, 0],
        _pts[:, 1],
        c=_ds_colors[_ds],
        s=5,
        alpha=0.4,
        label=f"{_ds} (n={_mask.sum():,})",
        rasterized=True,
    )
ax.set_xlabel("UMAP 1", fontsize=12)
ax.set_ylabel("UMAP 2", fontsize=12)
ax.set_title("Integrated UMAP: lce75 + DE122\n(Harmony correction on dataset)", fontsize=12)
ax.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig("figures/integrated/umap_dataset.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: figures/integrated/umap_dataset.png")

# ── Plot B: all-pathways UMAP (both datasets combined) ────────────────────────
plot_umap_layered_pathways(
    _combined,
    outbase="integrated/umap_pathways",
    gene_column="target",
    pathway_column="target_pathway",
    color_column="pathway_color",
    legend_loc="right margin",
    figsize=(20, 14),
    legend_ncol=3,
    title="Integrated UMAP: lce75 + DE122 (all pathways)",
)
print("  Saved: figures/integrated/umap_pathways.png")

# ── Plots C & D: per-dataset contribution to the integrated embedding ──────────
for _ds in ["lce75", "DE122"]:
    _mask = _combined.obs["dataset"] == _ds
    _subset = _combined[_mask].copy()
    plot_umap_layered_pathways(
        _subset,
        outbase=f"integrated/umap_pathways_{_ds}",
        gene_column="target",
        pathway_column="target_pathway",
        color_column="pathway_color",
        legend_loc="right margin",
        figsize=(20, 14),
        legend_ncol=3,
        title=f"Integrated UMAP: {_ds} cells only (all pathways)",
    )
    print(f"  Saved: figures/integrated/umap_pathways_{_ds}.png")

print("  Done: integrated UMAP")

# ── Side-by-side pathway comparison (biorep1 vs lce75) ────────────────────────
adata_biorep1 = ad.read_h5ad(
    "/Users/james/HMS Dropbox/James Taggart/ecoli_essential_gene_crispri/data/sequencing/251117_genomescale_CRISPRi_biorep1/processed_h5ad/sample_mix_umi200_hvg500_pc25_neighbors10_mindist0.55_leidenres0.8.h5ad"
)

if "target" in adata_biorep1.obs.columns:
    target_col_biorep1 = "target"
elif "consensus_target" in adata_biorep1.obs.columns:
    target_col_biorep1 = "consensus_target"
else:
    raise ValueError("biorep1 h5ad has no 'target' or 'consensus_target' column")

adata_biorep1.obs["target_pathway"] = (
    adata_biorep1.obs[target_col_biorep1].astype(object).map(gene_to_pathway).fillna("other")
)
adata_biorep1.obs["pathway_color"] = (
    adata_biorep1.obs["target_pathway"].map(pathway_colors).fillna(pathway_colors["other"])
)

_focus_pathways = [
    "LPS_transport",
    "dNTP_biosynthesis",
    "phosphate_transport_pst",
    "ribosomal_proteins_30S",
    "ribosomal_proteins_50S",
    "tRNA_synthetases",
    "LPS_lipid_A_biosynthesis",
]
fig, axes = plt.subplots(1, 2, figsize=(24, 10))
plot_umap_layered_pathways(
    adata_biorep1,
    pathway_column="target_pathway",
    color_column="pathway_color",
    gene_column=target_col_biorep1,
    legend_loc="right margin",
    ax=axes[0],
    title="biorep1",
    pathways_to_plot=_focus_pathways,
)
plot_umap_layered_pathways(
    adatas["lce75"],
    pathway_column="target_pathway",
    color_column="pathway_color",
    gene_column="target",
    legend_loc="right margin",
    ax=axes[1],
    title="lce75",
    pathways_to_plot=_focus_pathways,
)
plt.tight_layout()
plt.savefig("figures/umap_pathways_comparison.png", format="png", bbox_inches="tight", dpi=300)
plt.close()

# breakpoint()
# plot_umap(adata, outbase=f'leiden_{str(leidenres)}', legend_loc='right', figsize=(10,10))
