from essential.gpu_utils import select_best_gpus

select_best_gpus()

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from essential.data import load_fitness_data


def compute_pairwise(df1, df2=None):
    if df2 is None:
        pairwise_d = cosine_similarity(df1)
        pairwise_d = pd.DataFrame(pairwise_d, index=df1.index, columns=df1.index)
    else:
        pairwise_d = cosine_similarity(df1, df2)
        pairwise_d = pd.DataFrame(pairwise_d, index=df1.index, columns=df2.index)
    return pairwise_d


fitness_df = load_fitness_data()
fitness_df_gene = fitness_df.groupby("gene")[["T1", "T2", "T3", "T4"]].mean()

flux_df = pd.read_csv("/workspace/experiments/01232026_fba/data/moma_fluxes.csv", index_col=0)
worker_df = pd.read_csv("/workspace/experiments/01232026_fba/data/worker_ids.csv", index_col=0)
wt_flux = pd.read_csv("/workspace/experiments/01232026_fba/data/wt_fluxes_0.csv", index_col=0)
growth_df = (
    pd.read_csv("/workspace/experiments/01232026_fba/data/fba_growth_ratios.csv", index_col=0)
    .merge(fitness_df_gene, left_index=True, right_index=True)
    .assign(
        fba_growth_type=lambda x: pd.Categorical(
            np.where(x["growth_ratio"] >= 0.5, "high", "low"), categories=["low", "high"]
        )
    )
    .merge(worker_df, left_index=True, right_index=True)
)

# preprocessing + low-dimensional embedding
reaction_std = flux_df.std(axis=0)


adata = sc.read_h5ad(
    "/workspace/data/251117_genomescale_CRISPRi/sample_mix_umi200_hvg500_pc25_neighbors10_mindist0.55.scvi.h5ad"
)
adata_ = adata.copy()
transcript_df = []
gene_names = []
for gene in tqdm(adata.obs["gene"].unique()):
    X_gene = adata[adata.obs["gene"] == gene].layers["cp10k"].toarray()
    if X_gene.shape[0] > 0:
        gene_names.append(gene)
        transcript_df.append(X_gene.mean(axis=0))
transcript_df = pd.DataFrame(transcript_df, index=gene_names, columns=adata.var_names)
transcript_df_ctrl = transcript_df.loc[lambda x: x.index.str.startswith("Control")]

corr_df = []
for n_top_genes in [100, 200, 500, 1000, 2000]:
    for reaction_std_thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        adata_ = adata.copy()
        sc.pp.highly_variable_genes(adata_, n_top_genes=n_top_genes)
        hvgs = adata_.var["highly_variable"]
        transcript_df_ = transcript_df.loc[:, hvgs]
        transcript_df_ctrl_ = transcript_df_ctrl.loc[:, hvgs]

        flux_df_ = flux_df.loc[:, reaction_std >= reaction_std_thresh]
        print(flux_df_.shape)
        wt_flux_df_ = wt_flux.loc[flux_df_.columns.values]

        flux_pairwise = compute_pairwise(flux_df_)
        flux_d_to_ctrl = compute_pairwise(flux_df_, wt_flux_df_.T)
        flux_d_to_ctrl.columns = ["flux_d_to_ctrl"]

        transcript_pairwise = compute_pairwise(transcript_df_)
        transcript_d_to_all_ctrl = compute_pairwise(transcript_df_, transcript_df_ctrl_)
        transcript_d_to_ctrl = transcript_d_to_all_ctrl.mean(1).to_frame("transcript_d_to_ctrl")

        dists_to_ctrl = pd.merge(
            transcript_d_to_ctrl,
            flux_d_to_ctrl,
            left_index=True,
            right_index=True,
            how="inner",
        )
        corr_ = stats.spearmanr(
            dists_to_ctrl["flux_d_to_ctrl"], dists_to_ctrl["transcript_d_to_ctrl"]
        )
        corr_df.append(
            {
                "n_top_genes": n_top_genes,
                "reaction_std_thresh": reaction_std_thresh,
                "corr": corr_.correlation,
            }
        )
corr_df = pd.DataFrame(corr_df)
corr_df.to_csv("dists_to_ctrl_corrs.csv")
