#!/usr/bin/env bash
# Mirror a whitelist of dataset folders from /ewsc (NFS, source of truth)
# to /scratch (local NVMe) for fast container-side I/O.
#
# Invoked by devcontainer.json `initializeCommand` on the host before the
# container starts. Safe to run manually at any time.

set -euo pipefail

SRC=/ewsc/pboyeau/data
DST=/scratch/pboyeau/data_essential

# Whitelist of dataset folders to mirror to local NVMe.
# Add a folder here the moment it becomes a runtime dependency.
DATASETS=(
    KEGG
    RegulonDB
    RegulonDB_PSSM
    bigg
    calvo2020_dcas9fitness
    genomes
    ecoli_llm
    e_coli_llm_embeddings
    ecoli_evo2
    250516_TF_perturbseq
    251117_genomescale_CRISPRi
    Eaton_2025
)

mkdir -p "$DST"

args=()
for d in "${DATASETS[@]}"; do
    args+=(--include="/$d/***")
done
args+=(--exclude='/*')

rsync -aH --delete --info=progress2 -h "${args[@]}" "$SRC/" "$DST/"
