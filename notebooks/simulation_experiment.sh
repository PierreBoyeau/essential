SAVE_DIR="/workspace/results/simulation_experiment_1020"

for random_seed in $(seq 0 100); do
    for n_obs in 200 400 800 1600 3200; do
        python simulation_run.py --n_obs $n_obs --tag "n_obs" --save_dir $SAVE_DIR --random_seed $random_seed
    done

    for n_perturbed in 10 20 40 80; do
        python simulation_run.py --n_perturbed $n_perturbed --tag "n_perturbed" --save_dir $SAVE_DIR --random_seed $random_seed
    done

    for t in 0.01 0.05 0.1 0.2 0.5 1.0 2.0; do
        python simulation_run.py --t $t --tag "t" --save_dir $SAVE_DIR --random_seed $random_seed
    done
done