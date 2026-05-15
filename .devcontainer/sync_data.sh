#!/usr/bin/env bash
# Mirror a whitelist of dataset folders from /ewsc (NFS, source of truth)
# to /data (local NVMe) for fast container-side I/O.
#
# Invoked by devcontainer.json `initializeCommand` on the host before the
# container starts. Safe to run manually at any time.

set -euo pipefail

SRC=/ewsc/pboyeau/data
DST=/data/pboyeau/data_essential

# Whitelist of dataset folders to mirror to local NVMe.
# Add a folder here the moment it becomes a runtime dependency.
DATASETS=(
    KEGG
    RegulonDB
    RegulonDB_PSSM
    # bigg
    calvo2020_dcas9fitness
    genomes
    ecoli_llm
    e_coli_llm_embeddings
    ecoli_evo2
    250516_TF_perturbseq
    251117_genomescale_CRISPRi
    Nov2025_DE122_genomescale_EZRDM_Glu_newpipeline_preprocessed
    260309_lce75_genomescale_ezrdm_glu_preprocessed
    de122_lce75
    Rapp_2026
    # Eaton_2025
)

mkdir -p "$DST"

args=()
for d in "${DATASETS[@]}"; do
    args+=(--include="/$d/***")
done
args+=(--exclude='/*')

rsync -a --size-only --delete --info=progress2 "${args[@]}" "$SRC/" "$DST/"
