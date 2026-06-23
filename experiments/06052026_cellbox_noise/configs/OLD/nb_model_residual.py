import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb_model import get_config as _base_config

EXP = "/workspace/experiments/06052026_cellbox_noise"
TAG = "nb_residual"


def get_config():
    config = _base_config()

    config.tag = TAG
    config.model.mean_mode = "residual"
    config.training.learning_rate = 1e-3
    config.training.n_epochs = 5

    config.training.tensorboard_log_dir = f"{EXP}/results/{TAG}/tensorboard"
    config.outputs.checkpoint_dir = f"{EXP}/results/{TAG}/checkpoint"
    config.outputs.adata_pred = f"{EXP}/results/{TAG}/adata_pred.h5ad"
    config.outputs.metrics_dir = f"{EXP}/results/{TAG}/metrics"

    return config
