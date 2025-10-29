"""Configuration for dynamic cellbox model experiments."""

from essential.configs.base import get_config as get_base_config


def get_config():
    """Returns config for dynamic cellbox model."""
    config = get_base_config()
    config.tag = "cellbox_onebatch"
    config.processing.rt_bc = "TACCAG"
    config.estimator.model_class = "dynamic_cellbox"
    config.estimator.expression_type = "concentration"
    config.estimator.pairing_strategy = "nn"
    config.estimator.recompute_nns = True
    config.estimator.model_kwargs.lambda_prior = 3.81e-05

    config.training.n_epochs = 100
    config.training.batch_size = 256
    config.training.learning_rate = 1e-3
    return config
