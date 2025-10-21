SAVE_DIR="/workspace/results/simulation_experiment_1021_A"
DEFAULT_NOBS=100
DEFAULT_NPERTURBED=10
DEFAULT_T=0.01
DEFAULT_NGENES=100
DEFAULT_SPARSITY=0.003


for random_seed in $(seq 0 2); do
    for n_obs in 50 100 200 400 800; do
        python simulation_run.py --n_obs $n_obs --tag "n_obs" --save_dir $SAVE_DIR --random_seed $random_seed \
        --n_genes $DEFAULT_NGENES \
        --n_perturbed $DEFAULT_NPERTURBED \
        --t $DEFAULT_T \
        --sparsity $DEFAULT_SPARSITY
    done

    for n_perturbed in 1 2 5 10 20 40 80; do
        python simulation_run.py --n_perturbed $n_perturbed --tag "n_perturbed" --save_dir $SAVE_DIR --random_seed $random_seed \
        --n_obs $DEFAULT_NOBS \
        --n_genes $DEFAULT_NGENES \
        --t $DEFAULT_T \
        --sparsity $DEFAULT_SPARSITY
    done

    for t in 0.01 0.1 1.0 10.0; do
        python simulation_run.py --t $t --tag "t" --save_dir $SAVE_DIR --random_seed $random_seed \
        --n_obs $DEFAULT_NOBS \
        --n_genes $DEFAULT_NGENES \
        --n_perturbed $DEFAULT_NPERTURBED
    done
done
