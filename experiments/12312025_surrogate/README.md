# Surrogate Model Experiment

This directory contains experiments for training surrogate models (MLP/Ridge) to predict transcriptomic outcomes from perturbations, using semi-supervised learning.

## Folder Structure

- **`run_experiment.py`**: The main script to run the experiment loop across different labeled data sizes and seeds. It loads data, trains models (`g_reg` and `f_reg`), and evaluates performance.
- **`tune_hyperparameters.py`**: A script using Optuna to find optimal hyperparameters (hidden dimensions) for the MLP models. It saves results to `hyperparameters.json`.
- **`regression_predictor.py`**: Contains the `RegressionPredictor` class (a JAX/Flax MLP) and the `ResNetBlock` definition.
- **`data_resources.py`**: Utilities for loading fitness data, CRISPRi data, and LLM embeddings.
- **`surrogate_training_*.ipynb`**: Jupyter notebooks for exploratory analysis and surrogate training on gene/spacer levels.

## Usage

### 1. Hyperparameter Tuning
First, run the tuning script to generate optimal hyperparameters for the MLP models. This will create a `hyperparameters.json` file.

```bash
for flag in "" "--use_pca"; do
    python experiments/12312025_surrogate/tune_hyperparameters.py --n_trials 20 $flag --fitness_cols T2
done

for flag in "" "--use_pca"; do
    python experiments/12312025_surrogate/tune_hyperparameters.py --n_trials 20 $flag --fitness_cols T4
done

for flag in "" "--use_pca"; do
    python experiments/12312025_surrogate/tune_hyperparameters.py --n_trials 20 $flag --fitness_cols all
done
```

### 2. Running the Experiment
Run the main experiment loop. It will automatically load the optimized hyperparameters if available.

```bash
python experiments/12312025_surrogate/run_experiment.py --model_type mlp --use_pca
```

Arguments:
- `--model_type`: Model architecture to use (`ridge` or `mlp`).
- `--use_pca`: If set, uses PCA components of transcriptomics as targets instead of raw gene expression.
- `--n_labeled`: List of labeled set sizes to test (default: 100 500 1000 2000).


```bash
for flag in "" "--use_pca"; do
    if [ -z "$flag" ]; then
        config_type="expression"
    else
        config_type="pca"
    fi
    for fitness_cols in T2 T4 all; do
        for seed in 0 1 2 3 4 5 6 7 8 9; do
            config_file="experiments/12312025_surrogate/hyperparameters_${config_type}_${fitness_cols}.json"
            python experiments/12312025_surrogate/run_experiment.py --model_type mlp $flag \
            --fitness_cols $fitness_cols --seed $seed
        done
    done
done


for flag in "" "--use_pca"; do
    if [ -z "$flag" ]; then
        config_type="expression"
    else
        config_type="pca"
    fi
    for fitness_cols in T2 T4 all; do
        for seed in 0 1 2 3 4 5 6 7 8 9; do
            config_file="experiments/12312025_surrogate/hyperparameters_${config_type}_${fitness_cols}.json"
            python experiments/12312025_surrogate/run_experiment.py --model_type ridge $flag \
            --fitness_cols $fitness_cols --seed $seed
        done
    done
done
```


New version at spacer level.
```bash
fitness_cols="all"
flag=""
for seed in 0 1 2 3 4; do
    python experiments/12312025_surrogate/run_experiment_spacer.py --model_type mlp $flag \
    --fitness_cols $fitness_cols --seed $seed
done
```