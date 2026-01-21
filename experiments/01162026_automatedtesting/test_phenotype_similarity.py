import pandas as pd
import plotnine as gg
import scanpy as sc
from tqdm import tqdm
import os
import scipy.stats as st
import numpy as np
from statsmodels.stats.multitest import multipletests
import json

FIG_DIR = "/workspace/experiments/01162026_automatedtesting/outputs/phenotype_similarity/test"
os.makedirs(FIG_DIR, exist_ok=True)

gene_pair_to_description = pd.read_csv(
    "/workspace/experiments/01162026_automatedtesting/outputs/phenotype_similarity/prior/kegg_edges_consolidated.csv",
    sep="\t",
    index_col="gene_pair",
)
unique_kegg_relationships = gene_pair_to_description.index.unique()


adata = sc.read_h5ad(
    "/workspace/data/251117_genomescale_CRISPRi/sample_mix_umi200_hvg500_pc25_neighbors10_mindist0.55.scvi.h5ad"
)
adata_case = sc.read_h5ad("/workspace/data/251117_genomescale_CRISPRi/adata_case.annotated.h5ad")

gene_clusters_case = adata_case.obs.groupby("gene")["annotated_leiden_case"].apply(
    lambda x: x.value_counts().idxmax()
)

# assess whether genes colocalize in different places
transcript_variation = []
for gene_pair_info in tqdm(unique_kegg_relationships):
    pathway_description = gene_pair_to_description.loc[
        gene_pair_info, "pathway_description_consolidated"
    ]
    interaction_annotations = gene_pair_to_description.loc[
        gene_pair_info, "interaction_annotations"
    ]
    gene1, gene2 = gene_pair_info.split("_")
    assign1 = gene_clusters_case.get(gene1, "unknown")
    assign2 = gene_clusters_case.get(gene2, "unknown")
    n_obs_1 = adata_case.obs.loc[lambda x: x["gene"] == gene1, "annotated_leiden_case"].shape[0]
    n_obs_2 = adata_case.obs.loc[lambda x: x["gene"] == gene2, "annotated_leiden_case"].shape[0]
    if (assign1 == "unknown") and (assign2 == "unknown"):
        variation = "both_control_like"
    elif (assign1 == "unknown") or (assign2 == "unknown"):
        variation = "one_control_like"
    elif assign1 == assign2:
        variation = "same_case_cluster"
    else:
        variation = "different_case_cluster"
    transcript_variation.append(
        {
            "gene_pair": gene_pair_info,
            "variation": variation,
            "transcript_cluster_1": assign1,
            "transcript_cluster_2": assign2,
            "n_obs_1": n_obs_1,
            "n_obs_2": n_obs_2,
            "pathway_description": pathway_description,
            "interaction_annotations": interaction_annotations,
        }
    )
transcript_variation = pd.DataFrame(transcript_variation)


# identify surprising gene pairs
surprising_transcript_variation = (
    transcript_variation.loc[lambda x: x["variation"] == "different_case_cluster"]
    .loc[lambda x: x["n_obs_1"] >= 3]
    .loc[lambda x: x["n_obs_2"] >= 3]
)

for gene_pair_info in tqdm(surprising_transcript_variation["gene_pair"]):
    results_path_ = os.path.join(FIG_DIR, gene_pair_info)
    os.makedirs(results_path_, exist_ok=True)

    gene1, gene2 = gene_pair_info.split("_")
    figure_path = os.path.join(results_path_, f"{gene1}_{gene2}.png")
    plot_df_case = adata_case.obs.copy()
    plot_df_case_subset = plot_df_case[plot_df_case["gene"].isin([gene1, gene2])].copy()
    plot_df_case_subset["gene"] = plot_df_case_subset["gene"].astype(str)
    fig = (
        gg.ggplot(plot_df_case, gg.aes(x="transcript_case_UMAP1", y="transcript_case_UMAP2"))
        + gg.geom_point(alpha=0.5)
        + gg.geom_point(
            plot_df_case_subset,
            gg.aes(x="transcript_case_UMAP1", y="transcript_case_UMAP2", color="gene"),
            alpha=1,
        )
        + gg.labs(
            title=pathway_description,
        )
    )
    fig.save(figure_path)

    # data for report
    adata_subset = adata_case[adata_case.obs["gene"].isin([gene1, gene2])].copy()
    X1 = adata_subset[adata_subset.obs["gene"] == gene1].layers["cp10k"].toarray()
    X2 = adata_subset[adata_subset.obs["gene"] == gene2].layers["cp10k"].toarray()
    de_results = st.ks_2samp(X1, X2)
    padjs = multipletests(de_results.pvalue, method="fdr_bh")[1]
    de_results = (
        pd.DataFrame(
            {
                "gene1": gene1,
                "gene2": gene2,
                "pval": de_results.pvalue,
                "padj": padjs,
                "sign": np.where(X1.mean(0) > X2.mean(0), "up", "down"),
                "gene": adata_subset.var_names.values,
            }
        )
        .sort_values("padj")
        .head(25)
    )
    top_de_genes = [f"{row.gene} ({row.sign}regulated)" for _, row in de_results.iterrows()]
    pathway_description = surprising_transcript_variation.loc[
        lambda x: x["gene_pair"] == gene_pair_info, "pathway_description"
    ].values[0]
    transcript_cluster_1 = surprising_transcript_variation.loc[
        lambda x: x["gene_pair"] == gene_pair_info, "transcript_cluster_1"
    ].values[0]
    transcript_cluster_2 = surprising_transcript_variation.loc[
        lambda x: x["gene_pair"] == gene_pair_info, "transcript_cluster_2"
    ].values[0]
    interaction_annotations = surprising_transcript_variation.loc[
        lambda x: x["gene_pair"] == gene_pair_info, "interaction_annotations"
    ].values[0]

    report_data = {
        "gene1": gene1,
        "gene2": gene2,
        "pathway_description": pathway_description,
        "interaction_annotation": interaction_annotations,
        "top_de_genes": top_de_genes,
        "transcript_cluster_1": transcript_cluster_1,
        "transcript_cluster_2": transcript_cluster_2,
    }
    report_file = os.path.join(results_path_, "report.json")
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=4)
