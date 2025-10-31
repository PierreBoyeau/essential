#!/bin/bash

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/experiment_10302025_ablation_C"
python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_cellbox.py \
--output_path $OUTPUT_PATH

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/experiment_10302025_ablation_C"
python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_hardko.py \
--output_path $OUTPUT_PATH

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/experiment_10302025_ablation_C"
python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_sigmoidhardkozeroorder.py \
--output_path $OUTPUT_PATH

OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/experiment_10302025_ablation_C"
python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_hardkozeroorder.py \
--output_path $OUTPUT_PATH

