import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_model import get_config as _base_config

EXP = "/workspace/experiments/06052026_cellbox_noise"
TAG = "nb_linear_residual"


def get_config():
    config = _base_config()

    config.tag = TAG
    config.filter_regulators = False
    config.model.model_type = "nb"
    config.model.mean_mode = "residual"
    config.model.layer = "counts"
    config.model.layer_eval = "log1p"
    config.model.train_mode = "rollout"
    config.model.n_rollout_train = 10
    config.model.n_rollout_val = 10

    config.training.n_epochs = 100
    config.training.learning_rate = 1e-3
    config.outputs.output_dir = f"{EXP}/results/{TAG}"

    return config
