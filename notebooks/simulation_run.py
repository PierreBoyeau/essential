from essential.simulator import ODESimulator
from essential.ode import ODEstimator

import click
import scipy.stats as stats
import os
import json
import hashlib


@click.command()
@click.option("--n_obs", default=200, help="Number of observations.")
@click.option("--n_perturbed", default=10, help="Number of perturbed genes.")
@click.option("--t", default=0.1, help="Time point.")
@click.option("--random_seed", default=0, help="Random seed.")
@click.option("--tag", default="default", help="Tag for the experiment.")
@click.option("--save_dir", default="results", help="Directory to save results.")
def main(n_obs, n_perturbed, t, random_seed, tag, save_dir):
    """Simulate and estimate ODEs."""
    config = {
        "n_obs": n_obs,
        "n_perturbed": n_perturbed,
        "t": t,
        "random_seed": random_seed,
        "tag": tag,
    }
    config_str = json.dumps(config, sort_keys=True)
    hash_id = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:8]

    output_dir = os.path.join(save_dir, hash_id)
    os.makedirs(output_dir, exist_ok=True)

    simulator = ODESimulator(
        n_genes=100,
        n_perturbed=n_perturbed,
        sparsity=0.1,
        model_class="dynamic_cellbox",
        random_seed=random_seed,
    )
    adata = simulator.simulate(n_obs, batch_size=100, t=t)
    Amat_gt = simulator.Amat

    estimator = ODEstimator(
        adata,
        model_class="dynamic_cellbox",
        pairing_strategy="exact",
        subset_treated=True,
    )
    estimator.fit(n_epochs=1000, batch_size=100, batch_size_eval=10)
    Amat_pred = estimator.get_interaction_matrix()

    corr_ = stats.pearsonr(Amat_gt.flatten(), Amat_pred.T.values.flatten())
    corr_reversed = stats.pearsonr(Amat_gt.flatten(), Amat_pred.values.T.flatten())
    print(f"Correlation: {corr_}")
    print(f"Correlation (reversed): {corr_reversed}")

    results = {
        "config": config,
        "corr": {"statistic": corr_[0], "pvalue": corr_[1]},
        "corr_reversed": {"statistic": corr_reversed[0], "pvalue": corr_reversed[1]},
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
