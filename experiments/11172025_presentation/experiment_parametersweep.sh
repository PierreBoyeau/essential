OUTPUT_PATH="/workspace/experiments/11172025_presentation/runs_parametersweep"


########################################################
python /workspace/scripts/marginal_script.py \
--preprocess_mode="logmedian" \
--rt_bc="all" \
--consolidated_cluster="all" \
--output_path $OUTPUT_PATH \
--tag="marginal_all"

# Model configs - all batches
for expression_type in logmedian concentration_fixed quantile; do
    for config_name in linear sigmoid2 tanh; do
        tag="${config_name}_${expression_type}"
        python /workspace/scripts/ode_script.py \
        --config=/workspace/experiments/11172025_presentation/configs/$config_name.py \
        --config.estimator.expression_type=$expression_type \
        --config.tag=$tag \
        --output_path $OUTPUT_PATH
    done
done


# Model configs - one batch for tanh only
for expression_type in logmedian concentration_fixed quantile; do
    for config_name in tanh; do
        tag="${config_name}_${expression_type}_onebatch"
        python /workspace/scripts/ode_script.py \
        --config=/workspace/experiments/11172025_presentation/configs/$config_name.py \
        --config.estimator.expression_type=$expression_type \
        --config.processing.rt_bc="TACCAG" \
        --config.tag=$tag \
        --output_path $OUTPUT_PATH
    done
done


# Model configs - all batches
BASE_CONFIG="tanh"
EXPRESSION_TYPE="concentration_fixed"
for lambda_prior in 0.001 0.01 0.1 1.0 10.0; do
    tag="${BASE_CONFIG}_${EXPRESSION_TYPE}_lambda_${lambda_prior}"
    python /workspace/scripts/ode_script.py \
    --config=/workspace/experiments/11172025_presentation/configs/$BASE_CONFIG.py \
    --config.estimator.expression_type=$EXPRESSION_TYPE \
    --config.estimator.model_kwargs.lambda_prior=$lambda_prior \
    --config.tag=$tag \
    --output_path $OUTPUT_PATH
done

# run last because it takes a while
BASE_CONFIG="tanhdynamic"
EXPRESSION_TYPE="concentration_fixed"
tag="${BASE_CONFIG}_${EXPRESSION_TYPE}"
python /workspace/scripts/ode_script.py \
    --config=/workspace/experiments/11172025_presentation/configs/$BASE_CONFIG.py \
    --config.estimator.expression_type=$EXPRESSION_TYPE \
    --config.tag=$tag \
    --output_path $OUTPUT_PATH \
    --config.training.log_topk_every_n_epochs=1 \
    --config.training.n_epochs=5




