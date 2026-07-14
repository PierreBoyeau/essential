"""Build positive and negative ligand-protein pairs for Boltz2 binding prediction.

Positives are the curated TF <-> effector pairs from Ledezma-Tejeida 2022.
Negatives are random protein x ligand combinations that are NOT curated pairs.

Run with:  python build_pairs.py
Configuration lives in the GLOBALS block below (no argparse on purpose).
"""

import os
import random
import re
import time

import pandas as pd
import requests
from rdkit import Chem

# ----------------------------------------------------------------------------
# GLOBALS  (the only knobs)
# ----------------------------------------------------------------------------
ANNOTATIONS_CSV = "/workspace/experiments/06222026_tf/tf_effector_reference/ledezmatejeida_2022_tf_annotations_resolved_v2.csv"
OUTPUT_CSV = "/workspace/experiments/07062026_tf_updated/boltz_prediction/pairs.csv"
CACHE_DIR = "/workspace/experiments/07062026_tf_updated/boltz_prediction/cache"

RANDOM_SEED = 0  # reproducible negative sampling
NEG_PER_POS = 1  # 1:1 balanced negatives
MIN_HEAVY_ATOMS = 2  # drops bare single-atom ions that slip through the filter
DROP_METALS = True  # exclude classification == "Metals" (incl. Fe-S clusters)

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{id}.fasta"
KEGG_MOL_URL = "https://rest.kegg.jp/get/cpd:{id}/mol"
KEGG_MIN_INTERVAL = 0.4  # seconds between KEGG requests (~2.5 req/s)

_last_kegg_request = 0.0


# ----------------------------------------------------------------------------
# Fetching helpers
# ----------------------------------------------------------------------------
def fetch_protein_sequence(uniprot_id):
    """Return the amino-acid sequence for a UniProt ID, or None on failure.

    The raw FASTA is cached under cache/uniprot/{id}.fasta.
    """
    cache_path = os.path.join(CACHE_DIR, "uniprot", f"{uniprot_id}.fasta")
    if os.path.exists(cache_path):
        fasta = open(cache_path).read()
    else:
        resp = requests.get(UNIPROT_URL.format(id=uniprot_id))
        if not resp.ok or not resp.text.startswith(">"):
            print(f"  [WARN] UniProt fetch failed for {uniprot_id} ({resp.status_code})")
            return None
        fasta = resp.text
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        open(cache_path, "w").write(fasta)

    # Drop the header line and concatenate the sequence lines.
    lines = fasta.splitlines()
    sequence = "".join(l.strip() for l in lines if not l.startswith(">"))
    return sequence or None


def fetch_ligand_smiles(kegg_id):
    """Return a canonical SMILES for a KEGG compound ID, or None on failure.

    Fetches the KEGG MOL file and converts it with RDKit. The MOL and the
    resulting SMILES are cached under cache/kegg/{id}.mol and {id}.smi.
    Ligands with fewer than MIN_HEAVY_ATOMS heavy atoms are rejected.
    """
    global _last_kegg_request
    smi_path = os.path.join(CACHE_DIR, "kegg", f"{kegg_id}.smi")
    if os.path.exists(smi_path):
        return open(smi_path).read().strip() or None

    mol_path = os.path.join(CACHE_DIR, "kegg", f"{kegg_id}.mol")
    if os.path.exists(mol_path):
        mol_block = open(mol_path).read()
    else:
        # Rate-limit KEGG.
        wait = KEGG_MIN_INTERVAL - (time.time() - _last_kegg_request)
        if wait > 0:
            time.sleep(wait)
        _last_kegg_request = time.time()
        resp = requests.get(KEGG_MOL_URL.format(id=kegg_id))
        if not resp.ok or not resp.text.strip():
            print(f"  [WARN] KEGG mol fetch failed for {kegg_id} ({resp.status_code})")
            return None
        mol_block = resp.text
        os.makedirs(os.path.dirname(mol_path), exist_ok=True)
        open(mol_path, "w").write(mol_block)

    mol = Chem.MolFromMolBlock(mol_block)
    if mol is None:
        print(f"  [WARN] RDKit could not parse mol for {kegg_id}")
        return None
    if mol.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
        print(f"  [WARN] {kegg_id} has < {MIN_HEAVY_ATOMS} heavy atoms; dropped")
        return None

    smiles = Chem.MolToSmiles(mol)
    open(smi_path, "w").write(smiles)
    return smiles


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)

    df = pd.read_csv(ANNOTATIONS_CSV)
    print(f"Loaded {len(df)} annotation rows")

    # 1. Filter to usable small-molecule ligands.
    is_cnum = df["kegg_id"].astype(str).str.match(r"^C\d+$")
    keep = is_cnum
    if DROP_METALS:
        keep = keep & (df["classification"] != "Metals")
    df = df[keep].copy()
    print(
        f"{len(df)} rows after keeping C-number ligands"
        f"{' and dropping metals/clusters' if DROP_METALS else ''}"
    )

    # 2. Resolve each unique protein sequence and each unique ligand SMILES.
    print("Fetching protein sequences...")
    seq_by_uniprot = {u: fetch_protein_sequence(u) for u in sorted(df["Uniprot ID"].unique())}
    print("Fetching ligand SMILES...")
    smiles_by_kegg = {k: fetch_ligand_smiles(k) for k in sorted(df["kegg_id"].unique())}

    n_seq_fail = sum(v is None for v in seq_by_uniprot.values())
    n_lig_fail = sum(v is None for v in smiles_by_kegg.values())
    print(f"  proteins: {len(seq_by_uniprot) - n_seq_fail} resolved, {n_seq_fail} failed")
    print(f"  ligands:  {len(smiles_by_kegg) - n_lig_fail} resolved, {n_lig_fail} failed")

    # 3. Build the positive pairs (drop rows whose seq or SMILES failed, dedup).
    positives = []
    seen_keys = set()
    for _, row in df.iterrows():
        uid, kid = row["Uniprot ID"], row["kegg_id"]
        seq, smi = seq_by_uniprot.get(uid), smiles_by_kegg.get(kid)
        if seq is None or smi is None:
            continue
        key = (uid, kid)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        positives.append(
            {
                "TF": row["Transcription factor"],
                "uniprot_id": uid,
                "protein_sequence": seq,
                "effector_name": row["effector_name"],
                "kegg_id": kid,
                "smiles": smi,
                "label": 1,
            }
        )
    print(f"{len(positives)} positive pairs")

    # 4. Build negative pairs by random protein x ligand pairing.
    #    We keep readable metadata by drawing from the resolved protein/ligand pools.
    protein_pool = [p for p in positives]  # rows carry uniprot + TF + seq
    ligand_pool = [p for p in positives]  # rows carry kegg + effector + smiles
    positive_keys = set(seen_keys)

    n_negatives_target = NEG_PER_POS * len(positives)
    negatives = []
    neg_keys = set()
    attempts = 0
    max_attempts = 100 * n_negatives_target
    while len(negatives) < n_negatives_target and attempts < max_attempts:
        attempts += 1
        prot = random.choice(protein_pool)
        lig = random.choice(ligand_pool)
        key = (prot["uniprot_id"], lig["kegg_id"])
        if key in positive_keys or key in neg_keys:
            continue
        neg_keys.add(key)
        negatives.append(
            {
                "TF": prot["TF"],
                "uniprot_id": prot["uniprot_id"],
                "protein_sequence": prot["protein_sequence"],
                "effector_name": lig["effector_name"],
                "kegg_id": lig["kegg_id"],
                "smiles": lig["smiles"],
                "label": 0,
            }
        )
    if len(negatives) < n_negatives_target:
        print(f"  [WARN] only sampled {len(negatives)}/{n_negatives_target} negatives")
    print(f"{len(negatives)} negative pairs")

    # 5. Merge, assign a filesystem-safe unique pair_id, and write.
    pairs = pd.DataFrame(positives + negatives)
    pairs["pair_id"] = [
        f"{re.sub(r'[^A-Za-z0-9]', '', str(r.TF)).lower()}_{r.kegg_id}_{r.label}"
        for r in pairs.itertuples()
    ]
    pairs = pairs[
        [
            "pair_id",
            "TF",
            "uniprot_id",
            "protein_sequence",
            "effector_name",
            "kegg_id",
            "smiles",
            "label",
        ]
    ]
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pairs.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(pairs)} pairs to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
