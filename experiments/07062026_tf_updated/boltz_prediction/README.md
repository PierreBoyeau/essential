# Boltz2 binding prediction for TF–effector pairs

Predict ligand binding for transcription-factor / effector pairs using Boltz2,
for both curated (positive) and randomized (negative) pairs.

## Data source

`../../06222026_tf/tf_effector_reference/ledezmatejeida_2022_tf_annotations_resolved.csv`
— curated TF↔effector annotations (Ledezma-Tejeida 2022). Relevant columns:

- `Transcription factor`, `Uniprot ID` — the protein (sequence fetched from UniProt).
- `effector_name`, `kegg_id` — the ligand (SMILES fetched from KEGG + RDKit).
- `classification` — used to drop metals / iron-sulfur clusters.

## Pipeline

### 1. Build the pairs — `python build_pairs.py`

Produces `pairs.csv` with columns:
`pair_id, TF, uniprot_id, protein_sequence, effector_name, kegg_id, smiles, label`
(`label=1` positive, `label=0` negative).

What it does:

1. Keeps only rows with a KEGG C-number ligand and drops `Metals`
   (Zn, Cu, Fe … and `[2Fe-2S]`/`[4Fe-4S]` clusters) — Boltz affinity needs a
   real small-molecule SMILES.
2. Fetches each **unique** protein sequence from UniProt
   (`https://rest.uniprot.org/uniprotkb/{id}.fasta`).
3. Resolves each **unique** ligand SMILES from the KEGG MOL file
   (`https://rest.kegg.jp/get/cpd:{id}/mol`) via RDKit (canonical SMILES).
   Ligands with `< MIN_HEAVY_ATOMS` heavy atoms are dropped.
4. **Positives** = curated pairs where both sequence and SMILES resolved.
5. **Negatives** = random protein × ligand pairings (seed `RANDOM_SEED`),
   rejecting any pair that is a curated positive or already sampled.
   `NEG_PER_POS` controls the ratio (default 1:1).

All UniProt FASTAs and KEGG MOL/SMILES are cached under `cache/` so re-runs
don't re-hit the APIs. Configuration is a GLOBALS block at the top of the
script (no argparse).

### 2. Run predictions 


`python run_predictions.py`


### 3. Manual prediction

```bash
pair_id=ydci_L-Alanine
pair_id=ydci_3-sulfino-L-alanine
pair_id=yqhc_3-hydroxypropanal
pair_id=yqhc_acetaldehyde
pair_id=yqhc_propanal
pair_id=ygav_heme

pair_id=alsr_C02962_1

pair_id=ydci_L-Glutamate
pair_id=ydci_GABA
pair_id=yegw_adpg


time CUDA_VISIBLE_DEVICES=3 boltz predict configs_manual/${pair_id}.yaml \
  --use_msa_server \
  --use_potentials \
  --recycling_steps 10 \
  --diffusion_samples 25 \
  --out_dir results_manual/${pair_id}

```

long run:
real    4m38.144s
user    29m8.656s
sys     2m8.798s


```
pair_id=speed_test_new
time CUDA_VISIBLE_DEVICES=3 boltz predict configs_manual/${pair_id}.yaml \
  --use_msa_server \
  --out_dir results_manual/${pair_id}
```

short run:
real    1m41.558s
user    16m51.857s
sys     0m42.894s
