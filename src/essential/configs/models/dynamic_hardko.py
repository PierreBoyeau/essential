"""Configuration for dynamic cellbox model experiments."""

from essential.configs.base import get_config as get_base_config


def get_config():
    """Returns config for dynamic cellbox model."""
    config = get_base_config()
    config.tag = "dynamic_hardko"
    config.estimator.model_class = "dynamic_hardko"
    config.estimator.expression_type = "concentration"
    config.estimator.pairing_strategy = "nn"
    config.estimator.recompute_nns = False
    config.estimator.model_kwargs.lambda_prior = 3.81e-05
    config.training.n_epochs = 100
    return config
