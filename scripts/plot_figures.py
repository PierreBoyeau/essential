# %%
import glob
import json

import pandas as pd
import plotnine as gg

from essential.utils import PLOTNINE_DEFAULT_THEME

# %%
PATH_TO_RESULTS = "/workspace/results/ecoli_rich_medium/*/*/distance_metrics_mmd.json"
all_results = []
for result_path in glob.glob(PATH_TO_RESULTS):
    with open(result_path, "r") as f:
        result = json.load(f)
    all_results.append(result)

all_results_df = pd.DataFrame(all_results).set_index("tag")

# %%
SELECTED_COLUMNS = [
    "target_dist_median_ratio_of_top_50_pred_pairs_to_global",
    "target_dist_median_ratio_of_top_100_pred_pairs_to_global",
    "target_dist_median_ratio_of_top_500_pred_pairs_to_global",
    "target_dist_median_ratio_of_top_1000_pred_pairs_to_global",
]

SELECTED_TAGS = [
    "fba_moma_default_mmd",
    "gene_graph_beta_1_mmd",
    "gene_graph_beta_10_mmd",
    "gene_graph_beta_50_mmd",
    "gene_graph_beta_100_mmd",
]
results_df_selected = all_results_df.loc[SELECTED_TAGS]
results_df_selected[SELECTED_COLUMNS]


# %%
results_df_selected.T
# %%
(
    gg.ggplot(
        results_df_selected.reset_index(),
        gg.aes(x="tag", y="target_dist_median_ratio_of_top_100_pred_pairs_to_global"),
    )
    + gg.geom_col()
    + gg.theme_minimal()
    + gg.coord_flip()
)
# %%
