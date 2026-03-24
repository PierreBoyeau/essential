# %%
import pandas as pd
import scanpy as sc
import json
import numpy as np
import os
import plotnine as gg
from tqdm import tqdm
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
from stats_utils import MMDTestJax

# %%
SAVE_DIR = "/workspace/experiments/01162026_automatedtesting/outputs/operon/test"

hypothesis_list = pd.read_csv(
    "/workspace/experiments/01162026_automatedtesting/outputs/operon/gene_pairs_operon.csv"
)

with open(
    "/workspace/experiments/01162026_automatedtesting/outputs/operon/gene_to_tus.json", "r"
) as f:
    gene_to_tus = json.load(f)

adata = sc.read_h5ad(
    "/workspace/data/251117_genomescale_CRISPRi/sample_mix_umi200_hvg500_pc25_neighbors10_mindist0.55.scvi.h5ad"
)
adata_case = sc.read_h5ad("/workspace/data/251117_genomescale_CRISPRi/adata_case.annotated.h5ad")


# %%
class OperonAnalyzer:
    def __init__(
        self,
        adata: sc.AnnData,
        adata_case: sc.AnnData,
        gene_to_tus: dict,
        output_dir: str,
    ):
        adata.obs["annotated_leiden_case"] = "control-like"
        adata.obs.loc[adata_case.obs.index, "annotated_leiden_case"] = adata_case.obs[
            "annotated_leiden_case"
        ]

        self.adata = adata
        self.adata_case = adata_case
        self.gene_to_tus = gene_to_tus
        self.gene_clusters_case = adata.obs.groupby("gene")["annotated_leiden_case"].apply(
            lambda x: x.value_counts().idxmax()
        )
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

    def test_operon(self, gene_pair: str):
        gene1, gene2 = gene_pair.split("_")

        transcript_cluster_1 = self.gene_clusters_case.get(gene1, "unknown")
        transcript_cluster_2 = self.gene_clusters_case.get(gene2, "unknown")

        produce_unexpected_order = (transcript_cluster_1 == "control-like") and (
            transcript_cluster_2 != "control-like"
        )
        produce_two_different_phenotypes = (
            (transcript_cluster_1 != transcript_cluster_2)
            and (transcript_cluster_2 != "control-like")
            and (transcript_cluster_1 != "control-like")
        )
        if produce_unexpected_order:
            variation = "unexpected_order"
        elif produce_two_different_phenotypes:
            variation = "two_different_phenotypes"
        else:
            variation = "unsurprising_variation"

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
            "gene_pair": gene_pair,
            "gene1": gene1,
            "gene2": gene2,
            "variation": variation,
            "transcript_cluster_1": transcript_cluster_1,
            "transcript_cluster_2": transcript_cluster_2,
            "top_de_genes": top_de_genes,
            "mmd_pvalue": mmd_pvalue,
        }
        results = self._add_gene_context(results)
        self.all_results_dict[gene_pair] = results
        return results

    def _add_gene_context(self, result: dict):
        gene1, gene2 = result["gene1"], result["gene2"]
        result["gene1_known_transcription_units"] = self.gene_to_tus.get(gene1, [])
        result["gene2_known_transcription_units"] = self.gene_to_tus.get(gene2, [])
        return result

    def is_surprising(self, result: dict):
        return result["variation"] != "unsurprising_variation"

    def generate_report(self, result: dict):
        results_path = os.path.join(self.output_dir, result["gene_pair"])
        os.makedirs(results_path, exist_ok=True)

        gene1, gene2 = result["gene1"], result["gene2"]
        operon_names1 = [tu["operonName"] for tu in result["gene1_known_transcription_units"]]
        operon_names2 = [tu["operonName"] for tu in result["gene2_known_transcription_units"]]
        operon_names_intersect = np.intersect1d(operon_names1, operon_names2)
        operon_name_description = "; ".join(operon_names_intersect)

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
                title=operon_name_description,
            )
        )
        fig.save(os.path.join(results_path, "phenotype_similarity.png"))

        report_file = os.path.join(results_path, "report.json")
        with open(report_file, "w") as f:
            json.dump(result, f, indent=4)


# %%
analyzer = OperonAnalyzer(
    adata, adata_case=adata_case, gene_to_tus=gene_to_tus, output_dir=SAVE_DIR
)
for gene_pair in tqdm(hypothesis_list["gene_pair"]):
    result = analyzer.test_operon(gene_pair)
# %%
n_discoveries_mmd = (analyzer.all_results["mmd_padj"] <= 0.05).sum()
# %%