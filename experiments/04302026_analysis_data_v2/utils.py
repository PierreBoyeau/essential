import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import scanpy as sc
import seaborn as sns

from essential.data import load_fitness_data as _load_fitness
from essential.kegg_modules import kegg_module_to_graph, metabolic_to_operational_graph
from essential.pathway_discontinuity import PathwayDiscontinuity
from essential.plot_pathways import plot_pathway_results
from essential.utils import PLOTNINE_DEFAULT_THEME_2

ADATA_PATH = "/workspace/data/251117_genomescale_CRISPRi/sample_mix_umi200_hvg500_pc25_neighbors10_mindist0.55.h5ad"
FITNESS_DATA_PATH = "../../data/calvo2020_dcas9fitness/Supp_data2_log2FC.csv"


def load_adata(adata_path=None):
    adata = sc.read_h5ad(adata_path or ADATA_PATH)
    # sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_pca")
    # sc.tl.umap(adata, min_dist=0.5)
    adata.obs["UMAP1"] = adata.obsm["X_umap"][:, 0]
    adata.obs["UMAP2"] = adata.obsm["X_umap"][:, 1]
    return adata


def load_fitness_data():
    return (
        _load_fitness(FITNESS_DATA_PATH)
        .groupby("gene")[["T1", "T2", "T3", "T4"]]
        .mean()
        .reset_index()
    )


def rename_nodes(g, renamer):
    g = g.copy()
    for node, new_name in renamer.items():
        if node in g.nodes:
            g.nodes[node]["name"] = new_name
    return g


def run_module_analysis(
    adata,
    module_id,
    representation_obsm_key,
    perturbation_obs_key,
    renamer=None,
    global_sigma=5.0,
    mode="mmd_stat",
    threshold=0.2,
    control_perturbation_key=None,
):
    g = kegg_module_to_graph(module_id, "eco", print_info=False)
    if renamer:
        g = rename_nodes(g, renamer)
    op = metabolic_to_operational_graph(g)
    if control_perturbation_key is not None:
        op.add_node(control_perturbation_key)
        for node in list(op.nodes()):
            if node != control_perturbation_key:
                op.add_edge(control_perturbation_key, node)
    pda = PathwayDiscontinuity(
        adata,
        representation_obsm_key=representation_obsm_key,
        metabolic_graph=op,
        perturbation_obs_key=perturbation_obs_key,
        global_sigma=global_sigma,
        control_perturbation_key=control_perturbation_key,
    )
    results = pda.fit(threshold=threshold, mode=mode)
    return g, results


def plot_umap_equiv_classes(
    adata, gene_to_class, class_color_mapping, plot_legend=True, point_size=1.5
):
    obs_subset = adata.obs.loc[lambda x: x["target"].isin(gene_to_class.keys())].copy()
    obs_subset = obs_subset.sample(frac=1)
    obs_subset["equivalence_class"] = obs_subset["target"].map(gene_to_class)

    fig = (
        gg.ggplot(adata.obs, gg.aes(x="UMAP1", y="UMAP2"))
        + gg.geom_point(size=0.7, stroke=0)
        + gg.geom_point(obs_subset, gg.aes(color="equivalence_class"), size=point_size, stroke=0.0)
        + gg.scale_color_manual(values=class_color_mapping)
        + gg.theme_minimal()
    )
    if not plot_legend:
        fig = fig + gg.theme(legend_position="none")
    return fig


def plot_fitness_data(fitness_data, plot_info):
    gene_names = list(plot_info["gene_color_mapping"].keys())
    fitness_data_ = fitness_data.loc[lambda x: x["gene"].isin(gene_names)].copy()
    fitness_data_["gene"] = pd.Categorical(
        fitness_data_["gene"], categories=gene_names, ordered=True
    )

    return (
        gg.ggplot(fitness_data_, gg.aes(x="gene", y="T3", fill="gene"))
        + gg.scale_fill_manual(plot_info["gene_color_mapping"])
        + gg.geom_col()
        + PLOTNINE_DEFAULT_THEME_2
        + gg.theme(legend_position="none")
        + gg.coord_flip()
        + gg.labs(y="fitness (T+24h), (Calvo-Villamanan et al., 2020)", x="")
    )


def plot_lfc_heatmap(
    adata, gene_to_class, class_color_mapping=None, n_genes=10, figsize=(10, 4), lfc_range=(-3, 3)
):
    """Heatmap of one-vs-rest LFCs: rows = equivalence classes, cols = top marker genes.

    Returns (fig, de_dfs, lfc_df) where de_dfs is a dict of full DE results per class
    and lfc_df is the marker-gene LFC matrix. Returns (None, {}, None) if < 2 classes.
    """
    adata_sub = adata[adata.obs["target"].isin(gene_to_class.keys())].copy()
    adata_sub.obs["equiv_class"] = adata_sub.obs["target"].map(gene_to_class).astype(str)

    present = set(adata_sub.obs["equiv_class"].unique())
    classes = (
        [c for c in class_color_mapping if c in present] if class_color_mapping else sorted(present)
    )
    if len(classes) < 2:
        return None, {}, None

    sc.tl.rank_genes_groups(adata_sub, groupby="equiv_class", method="wilcoxon", reference="rest")

    de_dfs = {}
    marker_genes = []
    for cls in classes:
        df = sc.get.rank_genes_groups_df(adata_sub, group=cls).set_index("names")
        de_dfs[cls] = df
        marker_genes.extend(df.head(n_genes).index.tolist())
    marker_genes = list(dict.fromkeys(marker_genes))

    lfc_rows = {}
    for cls in classes:
        df = de_dfs[cls]
        lfc_rows[cls] = {
            g: df.loc[g, "logfoldchanges"] if g in df.index else np.nan for g in marker_genes
        }

    lfc_df = pd.DataFrame(lfc_rows).T.reindex(columns=marker_genes)

    row_colors = None
    if class_color_mapping is not None:
        row_colors = pd.Series(
            {cls: class_color_mapping.get(cls, "#888888") for cls in lfc_df.index},
            name="equiv. class",
        )

    cg = sns.clustermap(
        lfc_df,
        center=0,
        vmin=lfc_range[0],
        vmax=lfc_range[1],
        cmap="RdBu_r",
        row_colors=row_colors,
        row_cluster=False,
        col_cluster=False,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "log2 fold change"},
    )
    cg.ax_heatmap.set_xlabel("marker gene")
    cg.ax_heatmap.set_ylabel("equivalence class")
    return cg.fig, de_dfs, lfc_df


def plot_expr_clustermap(
    adata,
    gene_to_class,
    class_color_mapping,
    de_dfs,
    pval_threshold=0.1,
    figsize=(12, 8),
    layer="lognormalized",
    vmax_quantile=0.95,
):
    """Per-cell log-normalized expression clustermap for DE genes (pvals_adj <= pval_threshold in any class).

    Rows are cells grouped by equivalence class (no row clustering); columns are genes
    with hierarchical clustering. Row color bar encodes equivalence class.
    """
    import scipy.sparse as sp

    sig_genes = set()
    for df in de_dfs.values():
        sig_genes.update(df.index[df["pvals_adj"] <= pval_threshold].tolist())
    sig_genes = [g for g in adata.var_names if g in sig_genes]

    if not sig_genes:
        return None

    adata_sub = adata[adata.obs["target"].isin(gene_to_class.keys())].copy()
    adata_sub.obs["equiv_class"] = adata_sub.obs["target"].map(gene_to_class).astype(str)
    class_order = [c for c in class_color_mapping if c in adata_sub.obs["equiv_class"].unique()]
    adata_sub.obs["equiv_class"] = pd.Categorical(
        adata_sub.obs["equiv_class"], categories=class_order, ordered=True
    )
    adata_sub = adata_sub[adata_sub.obs.sort_values("equiv_class").index]

    X_total = adata[:, sig_genes].layers[layer]
    if sp.issparse(X_total):
        X_total = X_total.toarray()
    Xmax = np.quantile(X_total, vmax_quantile, axis=0)  # per-gene 95th percentile

    X = adata_sub[:, sig_genes].layers[layer]
    if sp.issparse(X):
        X = X.toarray()
    X = np.clip(X, 0, Xmax) / (Xmax + 1e-6)

    expr_df = pd.DataFrame(X, index=adata_sub.obs_names, columns=sig_genes)

    row_colors = adata_sub.obs["equiv_class"].map(class_color_mapping).rename("equiv. class")

    cg = sns.clustermap(
        expr_df,
        row_colors=row_colors,
        row_cluster=False,
        col_cluster=True,
        figsize=figsize,
        cmap="Greys",
        vmin=0,
        vmax=1.0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "log-normalized expression"},
    )
    return cg.figure


def plot_expr_heatmap_top_genes(
    adata,
    gene_to_class,
    class_color_mapping,
    de_dfs,
    top_n=100,
    pval_threshold=0.1,
    figsize=(12, 8),
    layer="lognormalized",
    vmax_quantile=0.95,
):
    """Per-cell log-normalized expression heatmap for top N genes per equivalence class.

    Rows are cells sorted by equivalence class (same ordering as plot_expr_clustermap).
    Columns are genes grouped by equivalence class, ranked by pvals_adj within each class.
    Column color bar shows which class each gene belongs to.
    """
    import scipy.sparse as sp

    seen = set()
    ordered_genes = []
    col_class = []
    for cls, df in de_dfs.items():
        sig = (
            df[df["pvals_adj"] <= pval_threshold]
            .loc[lambda x: x["logfoldchanges"] >= 0]
            .sort_values("logfoldchanges", ascending=False)
        )
        for g in sig.index[:top_n]:
            if g in adata.var_names and g not in seen:
                seen.add(g)
                ordered_genes.append(g)
                col_class.append(cls)

    if not ordered_genes:
        return None

    adata_sub = adata[adata.obs["target"].isin(gene_to_class.keys())].copy()
    adata_sub.obs["equiv_class"] = adata_sub.obs["target"].map(gene_to_class).astype(str)
    class_order = [c for c in class_color_mapping if c in adata_sub.obs["equiv_class"].unique()]
    adata_sub.obs["equiv_class"] = pd.Categorical(
        adata_sub.obs["equiv_class"], categories=class_order, ordered=True
    )
    adata_sub = adata_sub[adata_sub.obs.sort_values("equiv_class").index]

    X_total = adata[:, ordered_genes].layers[layer]
    if sp.issparse(X_total):
        X_total = X_total.toarray()
    Xmax = np.quantile(X_total, vmax_quantile, axis=0)  # per-gene 95th percentile
    X = adata_sub[:, ordered_genes].layers[layer]
    if sp.issparse(X):
        X = X.toarray()
    X = np.clip(X, 0, Xmax) / (Xmax + 1e-6)

    expr_df = pd.DataFrame(X, index=adata_sub.obs_names, columns=ordered_genes)

    row_colors = adata_sub.obs["equiv_class"].map(class_color_mapping).rename("equiv. class")
    col_colors = (
        pd.Series(col_class, index=ordered_genes).map(class_color_mapping).rename("gene class")
    )

    cg = sns.clustermap(
        expr_df,
        row_colors=row_colors,
        col_colors=col_colors,
        row_cluster=False,
        col_cluster=False,
        figsize=figsize,
        cmap="Greys",
        vmin=0,
        vmax=1.0,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "log-normalized expression (scaled)"},
    )
    return cg.figure


def save_module_outputs(
    adata,
    fitness_data,
    g,
    results,
    save_dir,
    point_size=2.0,
    umap_dpi=500,
    heatmap_n_genes=10,
    heatmap_figsize=(10, 4),
    heatmap_range=(-3, 3),
    adata_case=None,
    layer="lognormalized",
):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax, plot_info = plot_pathway_results(
        g,
        results,
        x_sep=0.4,
        y_sep=2,
        label_offset=(0, 0.4),
        edge_width=1.7,
        show_legend=False,
        node_mode="labels",
        isoenzyme_spacing=0.05,
    )
    plt.savefig(os.path.join(save_dir, "metabolic_graph.svg"))
    plt.close()

    fig_umap = (
        plot_umap_equiv_classes(
            adata,
            plot_info["gene_to_class"],
            plot_info["class_color_mapping"],
            plot_legend=False,
            point_size=point_size,
        )
        + PLOTNINE_DEFAULT_THEME_2
    )
    fig_umap.save(os.path.join(save_dir, "umap.png"), dpi=umap_dpi)

    if adata_case is not None:
        fig_umap_case = (
            plot_umap_equiv_classes(
                adata_case,
                plot_info["gene_to_class"],
                plot_info["class_color_mapping"],
                plot_legend=False,
                point_size=point_size,
            )
            + PLOTNINE_DEFAULT_THEME_2
        )
        fig_umap_case.save(os.path.join(save_dir, "umap_case.png"), dpi=umap_dpi)

    fig_heatmap, de_dfs, lfc_df = plot_lfc_heatmap(
        adata,
        plot_info["gene_to_class"],
        class_color_mapping=plot_info["class_color_mapping"],
        n_genes=heatmap_n_genes,
        figsize=heatmap_figsize,
        lfc_range=heatmap_range,
    )
    if fig_heatmap is not None:
        fig_heatmap.savefig(os.path.join(save_dir, "lfc_heatmap.svg"))
        plt.close()
    for cls, df in de_dfs.items():
        cls_slug = cls.replace(", ", "_").replace(" ", "_")
        df.to_csv(os.path.join(save_dir, f"de_{cls_slug}.csv"))
    if de_dfs:
        lines = ["# top DE genes\n"]
        for cls, df in de_dfs.items():
            genes = df.head(heatmap_n_genes).index.tolist()
            lines.append(f"## equivalence class: {cls}")
            lines.append(", ".join(genes) + "\n")
        with open(os.path.join(save_dir, "top_de_genes.md"), "w") as f:
            f.write("\n".join(lines))

    fig_expr_clustermap = None
    fig_expr_top_genes = None
    if de_dfs:
        fig_expr_clustermap = plot_expr_clustermap(
            adata,
            plot_info["gene_to_class"],
            plot_info["class_color_mapping"],
            de_dfs,
            layer=layer,
        )
        if fig_expr_clustermap is not None:
            fig_expr_clustermap.savefig(os.path.join(save_dir, "expr_clustermap.png"))
            plt.close()

        fig_expr_top_genes = plot_expr_heatmap_top_genes(
            adata,
            plot_info["gene_to_class"],
            plot_info["class_color_mapping"],
            de_dfs,
            layer=layer,
        )
        if fig_expr_top_genes is not None:
            fig_expr_top_genes.savefig(os.path.join(save_dir, "expr_top_genes.png"))
            plt.close()

    fig_fitness = None
    if fitness_data is not None:
        fig_fitness = plot_fitness_data(fitness_data, plot_info)
        fig_fitness.save(os.path.join(save_dir, "fitness.svg"))

    equiv_df = (
        pd.Series(plot_info["gene_to_class"], name="equivalence_class")
        .rename_axis("perturbation")
        .reset_index()
    )
    equiv_df.to_csv(os.path.join(save_dir, "equivalence_classes.tsv"), sep="\t", index=False)

    node_df = pd.DataFrame(
        [{"node_id": n, "name": g.nodes[n].get("name", n)} for n in g.nodes()],
    )
    node_df.to_csv(os.path.join(save_dir, "node_names.csv"), index=False)

    return plot_info, {
        "fig_metabolic_graph": fig,
        "fig_umap": fig_umap,
        "fig_umap_case": fig_umap_case if adata_case is not None else None,
        "fig_heatmap": fig_heatmap,
        "fig_expr_clustermap": fig_expr_clustermap,
        "fig_expr_top_genes": fig_expr_top_genes,
        "fig_fitness": fig_fitness,
        "equiv_df": equiv_df,
    }
