# %%
from metabolic_diffusion_kernel import MetabolicDiffusionKernel
from metabolic_shortest_distance import MetabolicShortestDistance

from essential.fba import load_ecoli_rich_medium_model

model = load_ecoli_rich_medium_model()
currency = [
    "atp_c",
    "atp_e",
    "adp_c",
    "adp_e",
    "h2o_c",
    "h2o_e",
    "h_c",
    "h_e",
    "nad_c",
    "nadh_c",
    "nadp_c",
    "nadph_c",
    "coa_c",
    "pi_c",
    "pi_e",
    "ppi_c",
]
msd = MetabolicShortestDistance(model, currency)

# %%
print("--- Shortest Distance ---")
print("Graph size:", msd.adj_matrix.shape)
print("Computing distance between first two genes:")
d = msd.compute_distance(msd.genes[0], msd.genes[1])
print(f"d({msd.genes[0]} -> {msd.genes[1]}) =", d)

print("Computing all distances...")
df = msd.compute_all_distances()
print(df.head())
print("DataFrame shape:", df.shape)

# %%
print("\n--- Diffusion Kernel ---")
mdk = MetabolicDiffusionKernel(model, currency, beta=1.0)
print("Computing similarity between first two genes:")
sim = mdk.compute_similarity(mdk.genes[0], mdk.genes[1])
print(f"sim({mdk.genes[0]}, {mdk.genes[1]}) =", sim)

print("Computing all similarities...")
df_sim = mdk.compute_all_similarities()
print(df_sim.head())
print("DataFrame shape:", df_sim.shape)

# %%
from metabolic_gene_graph_kernel import MetabolicGeneGraphKernel

print("\n--- Gene Graph Diffusion Kernel ---")
mggk = MetabolicGeneGraphKernel(model, currency, beta=1.0)
print("Computing similarity between first two genes:")
sim_gg = mggk.compute_similarity(mggk.genes[0], mggk.genes[1])
print(f"sim({mggk.genes[0]}, {mggk.genes[1]}) =", sim_gg)

print("Computing all similarities...")
df_sim_gg = mggk.compute_all_similarities()
print(df_sim_gg.head())
print("DataFrame shape:", df_sim_gg.shape)

# %%
