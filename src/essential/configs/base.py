"""Base configuration for ODE experiments.

This config contains all default parameters for running ODE estimation experiments.
Use this as a template or import and override specific sections for custom experiments.
"""

from ml_collections import config_dict


def get_config():
    """Returns the base configuration for ODE experiments."""
    config = config_dict.ConfigDict()

    # Data processing configuration
    config.processing = config_dict.ConfigDict()
    config.processing.adata_path = (
        "/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad"
    )
    config.processing.rt_bc = "all"
    config.processing.consolidated_cluster = "all"

    config.estimator = config_dict.ConfigDict()
    config.estimator.model_class = "dynamic_cellbox"
    config.estimator.expression_type = "concentration"
    config.estimator.pairing_strategy = None
    config.estimator.subset_treated = True
    config.estimator.model_kwargs = config_dict.ConfigDict()
    config.estimator.model_kwargs.lambda_prior = 1.0

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
