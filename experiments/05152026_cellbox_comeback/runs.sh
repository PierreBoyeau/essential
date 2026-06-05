#!/bin/bash
set -euo pipefail

SPLITS="/workspace/experiments/05152026_cellbox_comeback/splits"
OUT="/workspace/experiments/05152026_cellbox_comeback/results"

# Generate splits (seed 0 — the fold used for all runs below, see config.py).
# python /workspace/experiments/05152026_cellbox_comeback/generate_splits.py \
#     --seed 0 \
#     --out_dir "${SPLITS}"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_linear \
    --config.output_path="${OUT}/cellbox_linear"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_causal \
    --config.models.cellbox.filter_regulators=true \
    --config.output_path="${OUT}/cellbox_causal"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_linear_full \
    --config.split.train_extra_targets_path="${SPLITS}/non_tfs.txt" \
    --config.models.cellbox.training.max_val_perturbations=400 \
    --config.output_path="${OUT}/cellbox_linear_full"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_causal_full \
    --config.models.cellbox.filter_regulators=true \
    --config.split.train_extra_targets_path="${SPLITS}/non_tfs.txt" \
    --config.models.cellbox.training.max_val_perturbations=400 \
    --config.output_path="${OUT}/cellbox_causal_full"

nsteps=10
python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_rollout_${nsteps}steps \
    --config.output_path="${OUT}/cellbox_rollout_${nsteps}steps" \
    --config.models.cellbox.train_mode=rollout \
    --config.models.cellbox.n_rollout_train=${nsteps} \
    --config.models.cellbox.n_rollout_val=${nsteps} \
    --config.models.cellbox.training.early_stopping_patience=10 \
    # --config.models.cellbox.training.early_stopping_metric="loss" \
    # --config.models.cellbox.training.early_stopping_mode="min"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_causal_rollout_${nsteps}steps \
    --config.models.cellbox.filter_regulators=true \
    --config.output_path="${OUT}/cellbox_causal_rollout_${nsteps}steps" \
    --config.models.cellbox.train_mode=rollout \
    --config.models.cellbox.n_rollout_train=${nsteps} \
    --config.models.cellbox.n_rollout_val=${nsteps} \
    --config.models.cellbox.training.validate_every_n_epochs=0 \
    --config.models.cellbox.training.early_stopping_patience=10 \
    # --config.models.cellbox.training.early_stopping_metric="loss" \
    # --config.models.cellbox.training.early_stopping_mode="min"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_rollout_full_${nsteps}steps \
    --config.split.train_extra_targets_path="${SPLITS}/non_tfs.txt" \
    --config.models.cellbox.training.max_val_perturbations=400 \
    --config.output_path="${OUT}/cellbox_rollout_full_${nsteps}steps" \
    --config.models.cellbox.train_mode=rollout \
    --config.models.cellbox.n_rollout_train=${nsteps} \
    --config.models.cellbox.n_rollout_val=${nsteps} \
    --config.models.cellbox.training.early_stopping_patience=10 \
    # --config.models.cellbox.training.early_stopping_metric="loss" \
    # --config.models.cellbox.training.early_stopping_mode="min"

python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_causal_rollout_full_${nsteps}steps \
    --config.models.cellbox.filter_regulators=true \
    --config.split.train_extra_targets_path="${SPLITS}/non_tfs.txt" \
    --config.models.cellbox.training.max_val_perturbations=400 \
    --config.output_path="${OUT}/cellbox_causal_rollout_full_${nsteps}steps" \
    --config.models.cellbox.train_mode=rollout \
    --config.models.cellbox.n_rollout_train=${nsteps} \
    --config.models.cellbox.n_rollout_val=${nsteps} \
    --config.models.cellbox.training.validate_every_n_epochs=0 \
    --config.models.cellbox.training.early_stopping_patience=10 \
    # --config.models.cellbox.training.early_stopping_metric="loss" \
    # --config.models.cellbox.training.early_stopping_mode="min"


python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=mean \
    --config.tag=mean \
    --config.output_path="${OUT}/mean"



nsteps=10
python /workspace/experiments/05152026_cellbox_comeback/run_prediction.py --config=config.py \
    --config.model_name=cellbox \
    --config.tag=cellbox_causal_rollout_${nsteps}steps \
    --config.models.cellbox.filter_regulators=true \
    --config.output_path="${OUT}/cellbox_causal_rollout_${nsteps}steps" \
    --config.models.cellbox.train_mode=rollout \
    --config.models.cellbox.n_rollout_train=${nsteps} \
    --config.models.cellbox.n_rollout_val=${nsteps} \
    --config.models.cellbox.training.validate_every_n_epochs=0 \
    --config.models.cellbox.training.early_stopping_patience=10 \
    --config.models.cellbox.training.n_epochs=5