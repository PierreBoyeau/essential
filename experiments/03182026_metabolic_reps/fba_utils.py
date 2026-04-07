import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm


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


def hamming_distance(A: np.ndarray, B: np.ndarray = None) -> np.ndarray:
    A_ = np.array(A)
    s_A = A_.sum(axis=1, keepdims=True)
    if B is None:
        return s_A + s_A.T - 2 * (A_ @ A_.T)
    B_ = np.array(B)
    s_B = B_.sum(axis=1, keepdims=True)
    return s_A + s_B.T - 2 * (A_ @ B_.T)


def to_long_no_diagonal(df):
    return (
        df.stack()
        .reset_index()
        .rename(columns={"level_0": "gene1", "level_1": "gene2", 0: "distance"})
        .loc[lambda x: x["gene1"] != x["gene2"]]
        .assign(
            gene_pair=lambda x: np.where(
                x["gene1"] < x["gene2"],
                x["gene1"] + "_" + x["gene2"],
                x["gene2"] + "_" + x["gene1"],
            ),
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
