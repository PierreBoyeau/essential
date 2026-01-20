import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from base import get_config as get_base_config


def get_config():
    config = get_base_config()
    config.estimator.model_class = "tanh"
    config.estimator.model_kwargs.mode = "steady"
    config.estimator.expression_type = "concentration_fixed"
    config.training.n_epochs = 2000
    config.training.gradient_clip_norm = None
    return config
