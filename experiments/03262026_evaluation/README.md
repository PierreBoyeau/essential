# Evaluation Pipeline

This directory contains the results and workflow for the evaluation pipeline.

## Reproducing the results
To reproduce this analysis, checkout the specific tag containing the legacy benchmark code:

```bash
git checkout eval_pipeline_v1
```

Then, run the pipeline:

```bash
snakemake --cores 4
```

## Analysis
The results of this pipeline are analyzed in the notebook: `essential/experiments/03262026_evaluation/analyze_results_v3.ipynb`