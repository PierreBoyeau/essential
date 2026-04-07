# Parallel Metabolic KO Simulation (FBA & MOMA)

This directory contains a parallelized pipeline for running Flux Balance Analysis (FBA) and Minimization of Metabolic Adjustment (MOMA) simulations for gene knockouts in an E. coli metabolic model.

## Overview

The simulation is computationally intensive and memory-heavy when processing all gene knockouts. This pipeline distributes the workload across multiple independent worker processes to avoid memory leaks and improve performance.

## Pipeline Components

1.  **`split_genes.py`**: Partitions the input list of genes (`iJO1366_genes.csv`) into multiple smaller chunks.
2.  **`predict_metabolic_ko_worker.py`**: The worker script that processes a single chunk of genes. It loads the model, computes FBA/MOMA for each knockout, and saves partial results.
3.  **`run_parallel_fba.sh`**: The orchestrator script. It calls the splitter, launches worker scripts in parallel (background processes), waits for completion, and triggers consolidation.
4.  **`consolidate_fba.py`**: Merges the partial results from all workers into final consolidated CSV files.

## Usage

To run the full simulation:

```bash
bash run_parallel_fba.sh
```

This will:
1.  Read `data/iJO1366_genes.csv`.
2.  Split it into chunks (default: 8 workers) in `data/chunks/`.
3.  Run 8 parallel workers.
4.  Consolidate outputs into:
    *   `data/fba_growth_ratios.csv`
    *   `data/fba_fluxes.csv`
    *   `data/moma_fluxes.csv`

## Configuration

You can modify parameters in `run_parallel_fba.sh`:
*   `NUM_WORKERS`: Number of parallel processes (default: 8).
*   `INPUT_GENES`: Path to the input gene list.
*   `OUTPUT_DIR`: Directory for output files.

## Requirements

*   `cobra` (COBRApy)
*   `pandas`
*   `numpy`
*   `tqdm`
*   `optlang`
*   `glpk` (Solver)



# Prompt for automated gene annotation 


```markdown

# task
You are Palsson, a microbiologist expert on E. coli.
Your task is to annotate the function of each of the following genes. 
Proceed in two steps:
1. for each gene, describe individually its function, the essential pathways and processes it might contribute to
2. provide an annotation for each gene

The annotations should be coarse. A starting point is the following list of annotations (that should be completed if you find a gene that does not fit in any of these annotations):
Cell Division
Cell Envelope & Membrane Biology
DNA Replication
Metabolism & Energetics
Ribosome Biogenesis
Ribosome Translation
Transcription & RNA Processing
 

# outputa formatting
Format your answer as follows:
1. Gene-by-gene description, in markdown format
2. Annotation, using the following format

{
   "gene1": "annotation1",
   "gene2", "annotation2",
   ...
}

# the gene list
accA, accB, accC, accD, acpP, aldA, argD, aroA, aroB, aroC, aroD, asd, bioA, bioB, bioC, bioD, bioF, bioH, cdsA, coaA, coaD, coaE, cyaY, cysG, dapA, dapB, dapD, dapE, dapF, dfp, dxr, dxs, fabB, fabD, fabG, fabH, fabI, fabZ, folB, folC, folE, folK, folP, gapA, glmM, glmS, glmU, gltX, gmk, gpsA, hemA, hemB, hemC, hemD, hemE, hemG, hemH, hemL, iscS, ispA, ispB, ispD, ispE, ispF, ispG, ispH, ispU, kdsA, kdsB, kdsC, lptA, lptB, lptC, lptF, lptG, lpxA, lpxB, lpxC, lpxD, lpxH, lpxK, luxS, metF, metK, moaA, moaC, moaD, moaE, mobA, moeA, moeB, mog, mraY, msbA, mtn, murA, murB, murC, murD, murE, murF, murG, murI, murJ, nadA, nadB, nadC, nadD, nadE, nadK, pabA, pabB, pabC, pdxA, pdxB, pdxJ, pgk, plsC, psd, pssA, pyrG, ribA, ribB, ribC, ribD, ribE, ribF, serC, thiL, thyA, tmk, tpiA, ubiA, ubiC, ubiD, ubiX, waaA, yrbG, zupT



```
