import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm


class MarginalEstimator:
    def __init__(self, adata, preprocess_mode="logmedian"):
        self.adata = adata
        self.preprocess_mode = preprocess_mode
    
    def fit(self, return_df=False):
        adata_ = self.adata.copy()
        if self.preprocess_mode == "logmedian":
            sc.pp.normalize_total(adata_)
            sc.pp.log1p(adata_)
        else:
            raise ValueError(f"Invalid preprocess mode: {self.preprocess_mode}")
        
        from scipy.stats import ks_2samp

        ctrl_adata = adata_[adata_.obs["consensus_target"] == "nontargeting"].copy()
        tf_knockdowns = adata_.obs["consensus_target"].unique()
        tf_knockdowns = [t for t in tf_knockdowns if t in adata_.var_names]

        scores = np.zeros((adata_.shape[1], len(tf_knockdowns)))
        for tf_knockdown_idx, tf_knockdown in tqdm(enumerate(tf_knockdowns)):
            target_adata = adata_[adata_.obs["consensus_target"] == tf_knockdown].copy()
            pvals = ks_2samp(ctrl_adata.X.toarray().squeeze(), target_adata.X.toarray().squeeze()).pvalue
            significance = - np.log10(pvals + 1e-10)
            
            scores[:, tf_knockdown_idx] = significance

        self.scores = pd.DataFrame(scores, index=adata_.var_names, columns=tf_knockdowns)
        if return_df:
            return self.scores
        return scores, tf_knockdowns

    def get_interaction_matrix(self, return_square=True, delta=None):
        if return_square:
            return self.scores
        else:
            df_ = (
                self.scores
                .unstack()
                .to_frame("score")
                .reset_index()
                # .rename(columns={"level_0": "target_gene", "level_1": "regulator_gene"})
                .rename(columns={"level_0": "regulator_gene", "level_1": "target_gene"})
                .assign(
                    target_gene_=lambda x: x["target_gene"],
                    regulator_gene_=lambda x: x["regulator_gene"],
                    target_gene=lambda x: x["target_gene"].str.lower(),
                    regulator_gene=lambda x: x["regulator_gene"].str.lower(),
                )
            )
            if delta is not None:
                df_.loc[:, "decision"] = df_["score"] > delta
            return df_