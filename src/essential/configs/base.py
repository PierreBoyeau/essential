"""Base configuration for ODE experiments.

This config contains all default parameters for running ODE estimation experiments.
Use this as a template or import and override specific sections for custom experiments.
"""

from ml_collections import config_dict
import subprocess
import os


def get_git_hash():
    """Get the current git commit hash."""
    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=file_dir)
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not_a_git_repo"


def get_config():
    """Returns the base configuration for ODE experiments."""
    config = config_dict.ConfigDict()
    config.tag = None
    config.git_hash = get_git_hash()

    # Data processing configuration
    config.processing = config_dict.ConfigDict()
    config.processing.adata_path = (
        "/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad"
    )
    config.processing.rt_bc = "all"
    config.processing.consolidated_cluster = "all"
    config.processing.latent_obsm_key = "X_scvi_rtbc_corrected"

    config.estimator = config_dict.ConfigDict()
    config.estimator.model_class = "dynamic_cellbox"
    config.estimator.expression_type = "concentration"
    config.estimator.pairing_strategy = "nn"
    config.estimator.subset_treated = True
    config.estimator.model_kwargs = config_dict.ConfigDict()
    config.estimator.model_kwargs.lambda_prior = 1.0
    config.estimator.model_kwargs.mode = "dynamic"

    config.training = config_dict.ConfigDict()
    config.training.learning_rate = 1e-3
    config.training.n_epochs = 100
    config.training.batch_size = 8000
    config.training.train_size = 0.9
    config.training.early_stopping_patience = 5
    config.training.early_stopping_metric = "reco_loss"
    config.training.log_every_n_steps = 1
    config.training.batch_size_eval = 128
    config.training.optimizer = None

    config.output_path = None
    return config
