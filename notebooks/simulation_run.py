"""
Run a simulation experiment.

Default sparsity: 0.003
to align with the true sparsity of the interaction matrix (4175 genes * 0.003 = 12.525)
"""

from essential.simulator import ODESimulator
from essential.ode import ODEstimator

import click
import scipy.stats as stats
import os
import json
import hashlib


@click.command()
@click.option("--n_obs", default=100, help="Number of observations.")
@click.option("--n_genes", default=100, help="Number of genes.")
@click.option("--n_perturbed", default=10, help="Number of perturbed genes.")
@click.option("--t", default=0.1, help="Time point.")
@click.option("--sparsity", default=0.003, help="Sparsity of the interaction matrix.")
@click.option("--random_seed", default=0, help="Random seed.")
@click.option("--tag", default="default", help="Tag for the experiment.")
@click.option("--save_dir", default="results", help="Directory to save results.")
def main(n_obs, n_genes, n_perturbed, t, sparsity, random_seed, tag, save_dir):
    """Simulate and estimate ODEs."""
    config = {
        "n_obs": n_obs,
        "n_genes": n_genes,
        "n_perturbed": n_perturbed,
        "t": t,
        "random_seed": random_seed,
        "tag": tag,
    }
    config_str = json.dumps(config, sort_keys=True)
    hash_id = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:8]
    config["hash_id"] = hash_id

    output_dir = os.path.join(save_dir, hash_id)
    os.makedirs(output_dir, exist_ok=True)

    simulator = ODESimulator(
        n_genes=n_genes,
        n_perturbed=n_perturbed,
        sparsity=sparsity,
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
        expression_type="none",
    )
    estimator.fit(n_epochs=1000, batch_size=10, batch_size_eval=10)
    Amat_pred = estimator.get_interaction_matrix()

    corr_ = stats.pearsonr(Amat_gt.flatten(), Amat_pred.values.flatten())
    corr_reversed = stats.pearsonr(Amat_gt.flatten(), Amat_pred.values.T.flatten())
    print(f"Correlation: {corr_}")
    print(f"Correlation (reversed): {corr_reversed}")

    results = {
        "config": config,
        "corr": {"statistic": float(corr_[0]), "pvalue": float(corr_[1])},
        "corr_reversed": {"statistic": float(corr_reversed[0]), "pvalue": float(corr_reversed[1])},
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
