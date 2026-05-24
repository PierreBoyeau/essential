#!/bin/bash
set -euo pipefail

SPLITS="/workspace/experiments/05152026_cellbox_comeback/splits"
OUT="/workspace/experiments/05152026_cellbox_comeback"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_linear \
    --config.output_path="${OUT}/cellbox_linear.csv"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_causal \
    --config.models.cellbox.filter_regulators=true \
    --config.output_path="${OUT}/cellbox_causal.csv"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_linear_full \
    --config.split.train_extra_targets_path="${SPLITS}/non_tfs.txt" \
    --config.models.cellbox.training.max_val_perturbations=400 \
    --config.output_path="${OUT}/cellbox_linear_full.csv"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_causal_full \
    --config.models.cellbox.filter_regulators=true \
    --config.split.train_extra_targets_path="${SPLITS}/non_tfs.txt" \
    --config.models.cellbox.training.max_val_perturbations=400 \
    --config.output_path="${OUT}/cellbox_causal_full.csv"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=mean \
    --config.tag=mean \
    --config.output_path="${OUT}/mean.csv"
