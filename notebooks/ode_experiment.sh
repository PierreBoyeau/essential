#!/bin/bash

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/experiment_10312025_ablation_oneb"
python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_cellbox.py \
--config.processing.rt_bc="TACCAG" \
--config.training.batch_size=1000 \
--config.training.batch_size_eval=100 \
--output_path $OUTPUT_PATH

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/experiment_10312025_ablation_oneb"
python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_hardko.py \
--config.processing.rt_bc="TACCAG" \
--config.training.batch_size=1000 \
--config.training.batch_size_eval=100 \
--output_path $OUTPUT_PATH

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/experiment_10312025_ablation_oneb"
python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_sigmoidhardkozeroorder.py \
--config.processing.rt_bc="TACCAG" \
--config.training.batch_size=1000 \
--config.training.batch_size_eval=100 \
--output_path $OUTPUT_PATH

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/experiment_10312025_ablation_oneb"
python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_hardkozeroorder.py \
--config.processing.rt_bc="TACCAG" \
--config.training.batch_size=1000 \
--config.training.batch_size_eval=100 \
--output_path $OUTPUT_PATH

