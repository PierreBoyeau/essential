import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_model import get_config as _base_config

EXP = "/workspace/experiments/06052026_cellbox_noise"
TAG = "nb_causal_ds"


def get_config():
    config = _base_config()

    config.tag = TAG
    config.filter_regulators = True
    config.model.model_type = "nb_ds"
    config.model.layer = "counts"
    config.model.layer_eval = "log1p"
    config.model.train_mode = "rollout"
    config.model.n_rollout_train = 1
    config.model.n_rollout_val = 10

    config.training.tensorboard_log_dir = f"{EXP}/results/{TAG}/tensorboard"

    config.outputs.checkpoint_dir = f"{EXP}/results/{TAG}/checkpoint"
    config.outputs.adata_pred = f"{EXP}/results/{TAG}/adata_pred.h5ad"
    config.outputs.metrics_dir = f"{EXP}/results/{TAG}/metrics"

    return config
