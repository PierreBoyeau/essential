# %%
import json
import os

import numpy as np
import pandas as pd
import plotnine as gg
import scanpy as sc
import scipy.stats as st
from stats_utils import MMDTestJax
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

FIG_DIR = "/workspace/experiments/01162026_automatedtesting/outputs/phenotype_similarity/outputs_presentation"
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


# %%
class PhenotypeSimilarityAnalyzer:
    def __init__(
        self,
        adata: sc.AnnData,
        adata_case: sc.AnnData,
        hypothesis_description: pd.DataFrame,
        output_dir: str,
    ):
        self.adata = adata

        self.adata_case = adata_case
        self.gene_clusters_case = adata_case.obs.groupby("gene")["annotated_leiden_case"].apply(
            lambda x: x.value_counts().idxmax()
        )

        self.hypothesis_description = hypothesis_description
        self.output_dir = output_dir

        self.mmd_test = MMDTestJax(kernel_type="rbf", sigma=5)
        self.all_results_dict = {}

    @property
    def all_results(self):
        res = pd.DataFrame(list(self.all_results_dict.values()))
        pvals = res["mmd_pvalue"]
        padjs = multipletests(pvals, method="fdr_bh")[1]
        res["mmd_padj"] = padjs
        return res

    def test_hypothesis(self, hypothesis_name):
        pathway_description = self.hypothesis_description.loc[
            hypothesis_name, "pathway_description_consolidated"
        ]
        interaction_annotations = self.hypothesis_description.loc[
            hypothesis_name, "interaction_annotations"
        ]
        gene1, gene2 = hypothesis_name.split("_")
        assign1 = self.gene_clusters_case.get(gene1, "unknown")
        assign2 = self.gene_clusters_case.get(gene2, "unknown")
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

        adata_subset = self.adata[self.adata.obs["gene"].isin([gene1, gene2])]
        adata1 = adata_subset[adata_subset.obs["gene"] == gene1]
        adata2 = adata_subset[adata_subset.obs["gene"] == gene2]

        X1 = adata1.layers["cp10k"].toarray()
        X2 = adata2.layers["cp10k"].toarray()
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

        X1_zrep = adata1.obsm["X_scVI"]
        X2_zrep = adata2.obsm["X_scVI"]
        mmd_pvalue = self.mmd_test.test(X1_zrep, X2_zrep)

        results = {
            "gene_pair": hypothesis_name,
            "gene1": gene1,
            "gene2": gene2,
            "variation": variation,
            "transcript_cluster_1": assign1,
            "transcript_cluster_2": assign2,
            "n_obs_1": n_obs_1,
            "n_obs_2": n_obs_2,
            "pathway_description": pathway_description,
            "interaction_annotations": interaction_annotations,
            "top_de_genes": top_de_genes,
            "mmd_pvalue": mmd_pvalue,
        }
        self.all_results_dict[hypothesis_name] = results
        return results

    def is_surprising(self, result: dict):
        return (
            (result["variation"] == "different_case_cluster")
            and (result["n_obs_1"] >= 3)
            and (result["n_obs_2"] >= 3)
        )

    def generate_report(self, result: dict):
        results_path = os.path.join(self.output_dir, result["gene_pair"])
        os.makedirs(results_path, exist_ok=True)

        gene1, gene2 = result["gene1"], result["gene2"]
        pathway_description = result["pathway_description"]

        plot_df_case = self.adata_case.obs.copy()
        plot_df_case_subset = self.adata_case.obs[
            self.adata_case.obs["gene"].isin([gene1, gene2])
        ].copy()
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
        fig.save(os.path.join(results_path, "phenotype_similarity.png"))

        report_file = os.path.join(results_path, "report.json")
        with open(report_file, "w") as f:
            json.dump(result, f, indent=4)


# %%
analyzer = PhenotypeSimilarityAnalyzer(
    adata, adata_case, gene_pair_to_description, output_dir=FIG_DIR
)
transcript_variation = []
for gene_pair_info in tqdm(unique_kegg_relationships):
    results = analyzer.test_hypothesis(gene_pair_info)
    if analyzer.is_surprising(results):
        analyzer.generate_report(results)

# %%
#
results_df = pd.DataFrame(
    list(analyzer.all_results_dict.values()), index=analyzer.all_results_dict.keys()
)
results_df.to_csv(os.path.join(FIG_DIR, "results.csv"))
# %%
FIG_DIR
# %%
