# %%
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

KEYS_OF_INTEREST = ["tuName", "operonName", "tuGenes", "confidenceLevel"]
SAVE_DIR = "outputs/operon"
operon_df = pd.read_csv("/workspace/data/RegulonDB/TUSet.tsv", sep="\t", comment="#")
operon_df.columns = operon_df.columns.str.replace(r"^\d+\)", "", regex=True)
operon_df_ = operon_df.query("confidenceLevel != 'W'")


# %%
# Build a dictionary mapping genes to their known TUs
gene_to_tus = defaultdict(list)
# Map all consecutive gene pairs in operons
gene_pairs_operon = []

for i, row in tqdm(operon_df_.iterrows(), total=len(operon_df_)):
    if pd.isna(row["tuGenes"]):
        continue
    genes_in_operon = [g for g in row["tuGenes"].strip(";").split(";") if g]
    tu_data = row.to_dict()
    for gene in genes_in_operon:
        gene_to_tus[gene].append(tu_data)

    if len(genes_in_operon) > 1:
        for pair_id in range(len(genes_in_operon) - 1):
            gene_1 = genes_in_operon[pair_id]
            gene_2 = genes_in_operon[pair_id + 1]
            gene_pairs_operon.append([gene_1, gene_2])

gene_to_tus = dict(gene_to_tus)
gene_pairs_operon = (
    pd.DataFrame(gene_pairs_operon, columns=["gene_1", "gene_2"])
    .drop_duplicates()
    .assign(gene_pair=lambda x: x.gene_1 + "_" + x.gene_2)
    .drop_duplicates(subset=["gene_pair"], keep="first")
)
print(f"Found {len(gene_pairs_operon)} gene pairs.")
# %%
gene_to_tus_file = os.path.join(SAVE_DIR, "gene_to_tus.json")
with open(gene_to_tus_file, "w") as f:
    json.dump(gene_to_tus, f, indent=4)

gene_pairs_operon_file = os.path.join(SAVE_DIR, "gene_pairs_operon.csv")
gene_pairs_operon.to_csv(gene_pairs_operon_file, index=False)

# %%
