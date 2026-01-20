OUTPUT_PATH="/workspace/experiments/11172025_presentation/runs_litknowledge1"
config_name="tanh"

for lambda_prior in 0.1 1.0 10.0; do
    for perc_targets_in_training in 0.1 0.2 0.5 1.0; do
        for seed in 0; do
            tag="litknowledge_${config_name}_${perc_targets_in_training}_${seed}_${lambda_prior}"
            python /workspace/experiments/11172025_presentation/ode_script_partiallitknowledge.py \
            --config=/workspace/experiments/11172025_presentation/configs/$config_name.py \
            --config.tag=$tag \
            --config.estimator.model_kwargs.lambda_prior=$lambda_prior \
            --config.training.n_epochs=2000 \
            --output_path $OUTPUT_PATH \
            --heldout_targets /workspace/experiments/11172025_presentation/abc_transporter_genes.json \
            --perc_targets_in_training $perc_targets_in_training \
            --random_seed $seed
        done
    done
done
