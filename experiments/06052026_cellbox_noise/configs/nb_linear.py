import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_model import get_config as _base_config

EXP = "/workspace/experiments/06052026_cellbox_noise"
TAG = "nb_linear"


def get_config():
    config = _base_config()

    config.tag = TAG
    config.filter_regulators = False
    config.model.model_type = "nb"
    config.model.layer = "counts"
    config.model.layer_eval = "log1p"
    config.model.train_mode = "reconstruction"
    config.model.n_rollout_train = 1
    config.model.n_rollout_val = 10

    config.outputs.output_dir = f"{EXP}/results/{TAG}"

    return config
