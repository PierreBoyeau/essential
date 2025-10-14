OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/ode_experiment_10142025C"

# for lambda_prior in 0 1e-7 1e-6 1e-5 1e-4 1e-3; do
for lambda_prior in 1e-4; do
    for rt_bc in all; do
        for consolidated_cluster in all; do
            for preprocess_mode in concentration; do
                for model_class in steady_state_forcing steady_state_decay multiplicative_knockdown multiplicative_knockdown_with_basal; do
                python ode_script.py \
                    --lambda_prior $lambda_prior \
                    --model_class $model_class \
                    --rt_bc $rt_bc \
                    --consolidated_cluster $consolidated_cluster \
                    --preprocess_mode $preprocess_mode \
                    --output_path $OUTPUT_PATH
                done
            done
        done
    done
done

# python marginal_script.py \
#     --preprocess_mode logmedian \
#     --output_path $OUTPUT_PATH

python validate_graph.py --folder $OUTPUT_PATH
