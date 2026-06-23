import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_config import get_config as _base_config

EXP = "/workspace/experiments/06052026_cellbox_noise"
TAG = "nb"


def get_config():
    config = _base_config()

    config.tag = TAG
    config.model.model_type = "nb"
    config.model.layer = "counts"
    config.model.layer_eval = "log1p"
    config.training.learning_rate = 1e-2
    config.training.n_epochs = 100
    config.training.validate_every_n_epochs = 100
    config.training.log_every_n_epochs = 1

    config.outputs.output_dir = f"{EXP}/results/{TAG}"

    return config
