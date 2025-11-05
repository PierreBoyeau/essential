"""Base configuration for ODE experiments.

This config contains all default parameters for running ODE estimation experiments.
Use this as a template or import and override specific sections for custom experiments.
"""

from ml_collections import config_dict
from essential.utils import get_git_hash
import subprocess


def get_config():
    """Returns the base configuration for ODE experiments."""
    config = config_dict.ConfigDict()
    config.tag = None
    config.git_hash = get_git_hash()

    config.do_lr_optimization = False
    config.lr_optimization_params = config_dict.ConfigDict()
    config.lr_optimization_params.n_trials = 50
    config.lr_optimization_params.n_steps_per_trial = 100
    config.lr_optimization_params.lr_min = 1e-5
    config.lr_optimization_params.lr_max = 1e2
    config.lr_optimization_params.batch_size = 100
    config.lr_optimization_params.dev_size = 0.2
    config.lr_optimization_params.random_seed = 42

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
    config.estimator.model_kwargs.Amask = None

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
