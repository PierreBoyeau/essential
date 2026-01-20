import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from base import get_config as get_base_config


def get_config():
    config = get_base_config()

    config.processing.latent_obsm_key = "X_scvi_no_correction"

    config.estimator.model_class = "tanh"
    config.estimator.model_kwargs.mode = "dynamic"
    config.estimator.expression_type = "concentration_fixed"
    config.estimator.subset_treated = True
    config.training.n_epochs = 100
    config.training.batch_size = 64
    config.training.learning_rate = 1e-3
    config.training.gradient_clip_norm = None
    return config
