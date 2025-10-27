"""Configuration for dynamic cellbox model experiments."""

from essential.configs.base import get_config as get_base_config


def get_config():
    """Returns config for dynamic cellbox model."""
    config = get_base_config()
    return config
