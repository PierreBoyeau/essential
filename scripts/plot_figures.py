# %%
import plotnine as gg
import pandas as pd
import glob
import json
from essential.utils import PLOTNINE_DEFAULT_THEME

# %%
PATH_TO_RESULTS = "/workspace/results/ecoli_rich_medium/*/*/kernel_metrics.json"
all_results = []
for result_path in glob.glob(PATH_TO_RESULTS):
    with open(result_path, "r") as f:
        result = json.load(f)
    all_results.append(result)

all_results_df = pd.DataFrame(all_results).set_index("tag")

# %%
SELECTED_TAGS = [
    "gene_graph_beta_1",
    "gene_graph_beta_25",
    "fba_moma_default",
]

results_df_selected = all_results_df.loc[SELECTED_TAGS]

# %%
results_df_selected.T
# %%
(
    gg.ggplot(results_df_selected.reset_index(), gg.aes(x="tag", y="target_dist_median_of_top_100_pred_pairs"))
    + gg.geom_col()
)
# %%
