OUTPUT_PATH="/workspace/results/250516_TF_perturbseq/ode_experiment_10152025"

# for lambda_prior in 0 1e-7 1e-6 1e-5 1e-4 1e-3; do
for lambda_prior in 0 1e-2 1e-4 1e0; do
    for rt_bc in all; do
        for consolidated_cluster in all; do
            for preprocess_mode in concentration; do
                for model_class in dynamic_linear dynamic_linear_softplus dynamic_hardmultiplicative dynamic_multiplicative dynamic_cellbox; do
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

python validate_graph.py --folder $OUTPUT_PATH
