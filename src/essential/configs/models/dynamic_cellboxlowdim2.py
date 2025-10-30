"""Configuration for dynamic cellbox model experiments."""

from essential.configs.base import get_config as get_base_config


def get_config():
    """Returns config for dynamic cellbox model."""
    config = get_base_config()
    config.tag = "cellboxlowdim2"
    config.estimator.model_class = "dynamic_cellboxlowdim2"
    config.estimator.expression_type = "concentration"
    config.estimator.pairing_strategy = "nn"
    config.estimator.recompute_nns = False
    config.estimator.model_kwargs.lambda_prior = 0.0
    config.estimator.model_kwargs.n_latent = 128
    config.training.log_every_n_steps = 1000
    config.training.early_stopping_patience = 50
    config.training.n_epochs = 1200
    return config
