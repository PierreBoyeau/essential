"""Configuration for dynamic cellbox model experiments."""

from essential.configs.base import get_config as get_base_config


def get_config():
    """Returns config for dynamic cellbox model."""
    config = get_base_config()
    config.tag = "steady_state_forcing"
    config.estimator.model_class = "steady_state_forcing"
    config.estimator.expression_type = "concentration"
    config.estimator.pairing_strategy = "none"  # does not use NNs, use this to avoid recomputation
    config.estimator.model_kwargs.lambda_prior = 3.81e-05

    config.training.learning_rate = 1e-2
    config.training.n_epochs = 5000
    config.training.early_stopping_patience = 100
    config.training.batch_size = 4000
    config.training.batch_size_eval = 128
    config.training.log_every_n_steps = 50
    return config
