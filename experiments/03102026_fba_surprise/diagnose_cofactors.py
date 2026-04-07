# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from essential.gpu_utils import select_best_gpus

select_best_gpus()

import plotnine as gg
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from essential.fba import load_ecoli_rich_medium_model

# %%
pd.set_option("display.max_columns", 500)

SHARED_THEME = gg.theme(
    axis_text=gg.element_text(size=6),
    axis_title=gg.element_text(size=7),
    figure_size=(3, 2),
    title=gg.element_text(size=7),
    legend_text=gg.element_text(size=6),
)


def compute_pairwise(df1, df2=None, metric="cosine"):
    if metric == "cosine":
        metric_func = cosine_similarity
    elif metric == "euclidean":
        metric_func = euclidean_distances
    else:
        raise ValueError(f"Metric {metric} not supported")

    if df2 is None:
        pairwise_d = metric_func(df1)
        pairwise_d = pd.DataFrame(pairwise_d, index=df1.index, columns=df1.index)
    else:
        pairwise_d = metric_func(df1, df2)
        pairwise_d = pd.DataFrame(pairwise_d, index=df1.index, columns=df2.index)
    return pairwise_d


def plot_similarity_matrix(
    df, row_metadata, row_color_column, cmap="viridis", similarity_label="cosine similarity"
):
    """
    Plots a seaborn clustermap with a row colorbar.

    df: pd.DataFrame representing the similarity matrix.
    row_metadata: pd.DataFrame containing metadata for rows (must match df's rows).
    row_color_column: str, the column in row_metadata to map to colors.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import seaborn as sns

    cmap_obj = plt.get_cmap(cmap)
    norm = mcolors.Normalize(
        vmin=row_metadata[row_color_column].min(), vmax=row_metadata[row_color_column].max()
    )
    row_colors = row_metadata[row_color_column].map(lambda x: mcolors.to_hex(cmap_obj(norm(x))))

    g = sns.clustermap(
        df,
        row_colors=row_colors,
        xticklabels=False,
        yticklabels=False,
        cbar_pos=(0.02, 0.8, 0.05, 0.18),
        cbar_kws={"label": similarity_label},
    )

    cbar_ax = g.fig.add_axes([0.02, 0.55, 0.05, 0.18])
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), cax=cbar_ax)
    cb.set_label(row_color_column)

    return g


def hamming_distance(B: np.ndarray) -> np.ndarray:
    B_ = np.array(B)
    s = B_.sum(axis=1, keepdims=True)
    return s + s.T - 2 * (B_ @ B_.T)


def to_long_no_diagonal(df):
    return (
        df.stack()
        .reset_index()
        .rename(columns={"level_0": "gene1", "level_1": "gene2", 0: "distance"})
        .loc[lambda x: x["gene1"] != x["gene2"]]
        .assign(
            gene_pair=lambda x: x["gene1"] + "_" + x["gene2"],
        )
        .drop_duplicates(subset=["gene_pair"], keep="first")
    )


def get_gene_topology_stats(model, gene_names):
    """
    Extracts topological statistics for a list of genes from a cobrapy model.
    Returns a DataFrame indexed by gene name.
    """
    gene_stats = []

    # Create a mapping from gene names to gene IDs
    name_to_id = {g.name: g.id for g in model.genes}
    name_to_id.update({g.id: g.id for g in model.genes})

    for gene_name in tqdm(gene_names):
        try:
            gene_id = name_to_id[gene_name]
            gene = model.genes.get_by_id(gene_id)
        except KeyError:
            gene_stats.append(
                {"gene": gene_name, "num_reactions": np.nan, "num_downstream_reactions": np.nan}
            )
            continue

        gene_rxns = gene.reactions
        num_reactions = len(gene_rxns)
        products = set()
        for rxn in gene_rxns:
            products.update(rxn.products)

        downstream_reactions = set()
        for prod in products:
            for rxn in prod.reactions:
                if prod in rxn.reactants:
                    downstream_reactions.add(rxn.id)

        gene_stats.append(
            {
                "gene": gene_name,
                "num_reactions": num_reactions,
                "num_downstream_reactions": len(downstream_reactions),
            }
        )

    return pd.DataFrame(gene_stats).set_index("gene")


# %%
flux_df = pd.read_csv("/workspace/experiments/01232026_fba/data/moma_fluxes.csv", index_col=0)
worker_df = pd.read_csv("/workspace/experiments/01232026_fba/data/worker_ids.csv", index_col=0)
wt_flux = pd.read_csv("/workspace/experiments/01232026_fba/data/wt_fluxes_0.csv", index_col=0)
ecoli_model = load_ecoli_rich_medium_model()
growth_df_raw = pd.read_csv(
    "/workspace/experiments/01232026_fba/data/fba_growth_ratios.csv", index_col=0
)

topology_stats_df = get_gene_topology_stats(ecoli_model, growth_df_raw.index)

growth_df = (
    growth_df_raw.assign(
        fba_growth_type=lambda x: pd.Categorical(
            np.where(x["growth_ratio"] >= 0.5, "high", "low"), categories=["low", "high"]
        ),
        growth_score=lambda x: (x["growth"] - x["growth_wt"]) / x["growth_wt"],
        is_predicted_essential=lambda x: x["growth_ratio"] < 0.5,
    )
    .merge(worker_df, left_index=True, right_index=True)
    .join(topology_stats_df)
)

print(growth_df[["num_reactions", "num_downstream_reactions"]].head())

flux_df_bin = (flux_df.abs() >= 1e-6).astype(float)
flux_ham_dist = hamming_distance(flux_df_bin)
flux_ham_dist_df = pd.DataFrame(flux_ham_dist, index=flux_df_bin.index, columns=flux_df_bin.index)
# %%
pred_essential_genes = growth_df[growth_df["is_predicted_essential"] == True].index
sns.clustermap(
    flux_ham_dist_df.loc[pred_essential_genes, pred_essential_genes],
    xticklabels=False,
    yticklabels=False,
)
plt.show()
# %%
growth_df["num_downstream_reactions"].hist(bins=100)
# %%
flux_dist_selected = flux_ham_dist_df.loc[pred_essential_genes, pred_essential_genes]
growth_df_selected = growth_df.loc[pred_essential_genes].copy()
growth_df_selected["num_downstream_reactions_"] = np.clip(
    growth_df_selected["num_downstream_reactions"], 0, 50
)

plot_similarity_matrix(
    flux_dist_selected,
    growth_df_selected,
    "num_reactions",
)

# %%
