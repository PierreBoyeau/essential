# %%
import matplotlib.pyplot as plt
import scanpy as sc

plt.rcParams["svg.fonttype"] = "none"
# %%

ADATA_PATH = "/workspace/data/de122_lce75/adata_de122_lce75_merged.h5ad"
PERT_COL = "target"
CTRL_KEY = "nontargeting"
adata = sc.read_h5ad(ADATA_PATH)
adata.X = adata.layers["reads"].copy()
sc.pp.normalize_total(adata, target_sum=1e4)

# %%
adata_ctrl = adata[adata.obs[PERT_COL] == CTRL_KEY]
adata_pert = adata[adata.obs[PERT_COL] == "pgi"]

print(f"Control samples: {adata_ctrl.shape[0]}")
print(f"Perturbed samples: {adata_pert.shape[0]}")
# %%
mu_ctrl = adata_ctrl.X.mean(axis=0).A1
mu_pert = adata_pert.X.mean(axis=0).A1
# %%
CLIP_MIN = 3e-2
AXIS_LIM = (CLIP_MIN, 1e1)

mu_ctrl_clipped = mu_ctrl.clip(min=CLIP_MIN)
mu_pert_clipped = mu_pert.clip(min=CLIP_MIN)

HIGHLIGHT_GENES = {"sgrT": "#aa0000", "pgi": "#cc9900"}

FIGSIZE = (3, 3)
LABEL_FONTSIZE = 7
TICK_FONTSIZE = 6

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.scatter(mu_ctrl_clipped, mu_pert_clipped, color="#555555", s=4, alpha=0.5, linewidths=0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(AXIS_LIM)
ax.set_ylim(AXIS_LIM)
ax.plot(AXIS_LIM, AXIS_LIM, color="black", linewidth=0.5, zorder=0)

gene_names = list(adata.var_names)
for gene, color in HIGHLIGHT_GENES.items():
    idx = gene_names.index(gene)
    ax.scatter(
        mu_ctrl_clipped[idx],
        mu_pert_clipped[idx],
        color=color,
        s=40,
        zorder=5,
        label=gene,
        edgecolors="black",
        linewidths=0.5,
    )
ax.legend(fontsize=TICK_FONTSIZE)
ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
ax.set_xlabel("Reference expression (CP10K)", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Perturbed expression: pgi KD (CP10K)", fontsize=LABEL_FONTSIZE)

plt.tight_layout()
plt.savefig("pgi_kd_scatter.svg", dpi=300)
# %%
import pandas as pd
import plotly.express as px

df = pd.DataFrame(
    {
        "ctrl": mu_ctrl_clipped,
        "pert": mu_pert_clipped,
        "gene": adata.var_names,
    }
)
fig_px = px.scatter(
    df,
    x="ctrl",
    y="pert",
    hover_name="gene",
    log_x=True,
    log_y=True,
    range_x=[CLIP_MIN, 1e1],
    range_y=[CLIP_MIN, 1e1],
    color_discrete_sequence=["#555555"],
)
fig_px.update_traces(marker=dict(size=4, opacity=0.5))
fig_px.add_shape(
    type="line", x0=CLIP_MIN, y0=CLIP_MIN, x1=1e1, y1=1e1, line=dict(color="black", width=0.5)
)
fig_px.show()
# %%
