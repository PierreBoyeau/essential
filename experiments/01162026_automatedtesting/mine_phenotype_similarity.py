# %%
from kegg_utils import KEGGNavigator
from tqdm import tqdm
import pandas as pd
import plotnine as gg
import os

# %%
navigator = KEGGNavigator()

df = navigator.list_all_pathways_df("eco")
print(f"Found {len(df)} pathways.")
print(df.head())

# %%
gene_relationships = []

for pathway_id, pathway_description in tqdm(
    df[["pathway_id", "description"]].itertuples(index=False)
):
    kgml_met = navigator.get_pathway_kgml(pathway_id)
    parsed_met = navigator.parse_kgml_entries(kgml_met)
    G = navigator.build_gene_graph(parsed_met)
    for u, v, d in G.edges(data=True):
        gene_pair = f"{u}_{v}"
        interaction_type = d.get("interaction", "unknown")
        compound = d.get("compound", "")
        rel_types = list(d.get("types", []))
        rel_subtypes = list(d.get("subtypes", []))
        interaction_annotation = d.get("interaction_annotation", "")

        gene_relationships.append(
            {
                "pathway_id": pathway_id,
                "pathway_description": pathway_description,
                "interaction": interaction_type,
                "compound": compound,
                "relation_types": rel_types,
                "relation_subtypes": rel_subtypes,
                "interaction_annotation": interaction_annotation,
                "gene1": u,
                "gene2": v,
                "gene_pair": gene_pair,
            }
        )


# %%
gene_relationships_df = pd.DataFrame(gene_relationships)

output_dir = "outputs/phenotype_similarity/prior"
os.makedirs(output_dir, exist_ok=True)
gene_relationships_df.to_csv(os.path.join(output_dir, "kegg_edges.csv"), sep="\t")

# Consolidate gene pairs that appear in multiple pathways
consolidated_gene_pairs = (
    gene_relationships_df.assign(
        pathway_description_short=lambda x: x["pathway_description"].str.split(" - ").str[0]
    )
    .groupby("gene_pair")
    .agg(
        {
            "pathway_description_short": lambda x: "; ".join(x.unique()),
            "interaction": lambda x: "; ".join(x.unique()),
            "interaction_annotation": lambda x: "; ".join(x.unique()),
        }
    )
    .rename(
        columns={
            "pathway_description_short": "pathway_description_consolidated",
            "interaction": "interactions",
            "interaction_annotation": "interaction_annotations",
        }
    )
)

consolidated_gene_pairs.to_csv(os.path.join(output_dir, "kegg_edges_consolidated.csv"), sep="\t")


# %%
(
    gene_relationships_df.groupby(["pathway_id", "pathway_description"])["gene_pair"]
    .count()
    .sort_values(ascending=False)
)
# %%

gg.ggplot(gene_relationships_df, gg.aes("pathway_id")) + gg.geom_bar()

# %%
df_entries = parsed_met["entries"]
df_reactions = parsed_met["reactions"]


G = navigator.build_gene_graph(parsed_met)

for u, v, d in G.edges(data=True):
    compounds = d.get("compounds", set())
    # Note: parsed_met refers to the last loop iteration
    compound_info = parsed_met["entries"].loc[parsed_met["entries"]["name"].isin(compounds)]

# %%
