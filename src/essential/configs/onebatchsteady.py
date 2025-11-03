"""Configuration for dynamic cellbox model experiments."""

from essential.configs.base import get_config as get_base_config


def get_config():
    """Returns config for dynamic cellbox model."""
    config = get_base_config()
    config.tag = "onebatchsteady"
    config.processing.rt_bc = "TACCAG"
    config.estimator.model_class = "linear"
    config.estimator.expression_type = "concentration"
    config.estimator.pairing_strategy = "nn"
    config.estimator.model_kwargs.lambda_prior = 0.0
    config.estimator.model_kwargs.mode = "steady"
    config.estimator.subset_treated = False

    config.training.n_epochs = 1000
    config.training.batch_size = 256
    config.training.learning_rate = 1e-3
    return config
