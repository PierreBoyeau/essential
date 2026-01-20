OUTPUT_PATH="/workspace/experiments/11072025_regulonDBknowledge/runs_litknowledge1"

for perc_targets_in_training in 0.1 0.2 0.5 1.0; do
    for config_name in linear_best; do
        for seed in 0; do
            tag="litknowledge_${config_name}_${perc_targets_in_training}_${seed}"
            python /workspace/experiments/11072025_regulonDBknowledge/ode_script_partiallitknowledge.py \
            --config=/workspace/experiments/11072025_regulonDBknowledge/$config_name.py \
            --config.tag=$tag \
            --output_path $OUTPUT_PATH \
            --heldout_targets /workspace/experiments/11072025_regulonDBknowledge/abc_transporter_genes.json \
            --perc_targets_in_training $perc_targets_in_training \
            --random_seed $seed
        done
    done
done
