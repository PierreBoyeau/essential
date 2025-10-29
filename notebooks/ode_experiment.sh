OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/ode_experiment_10282025"


python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_cellbox.py \
--output_path $OUTPUT_PATH

python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_cellboxlowdim2.py \
--output_path $OUTPUT_PATH

python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_cellbox_onebatch.py \
--output_path $OUTPUT_PATH

python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/dynamic_cellbox_onebatchonec.py \
--output_path $OUTPUT_PATH

python /workspace/notebooks/ode_script.py \
--config=/workspace/src/essential/configs/models/steady_state_decay.py \
--output_path $OUTPUT_PATH