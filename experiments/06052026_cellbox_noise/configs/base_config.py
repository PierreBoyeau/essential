import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_config import get_config as _data_config
from ml_collections import config_dict

EXP = "/workspace/experiments/06052026_cellbox_noise"
TAG = "baseline"


def get_config():
    config = _data_config()

    config.tag = TAG
    config.filter_regulators = True

    config.model.model_type = "gaussian"
    config.model.mean_mode = "absolute"
    config.model.standardize_inputs = False
    config.model.train_mode = "rollout"
    config.model.n_rollout_train = 10
    config.model.n_rollout_val = 10
    config.model.n_val_control = 64
    config.model.reg_embed_dim = 16
    config.model.reg_hidden_dim = 16

    config.training = config_dict.ConfigDict()
    config.training.learning_rate = 1e-3
    config.training.n_epochs = 10000
    config.training.batch_size = 256
    config.training.train_size = 0.8
    config.training.early_stopping_patience = 30
    config.training.early_stopping_metric = "loss"
    config.training.early_stopping_mode = "min"
    config.training.validate_every_n_epochs = 1
    config.training.log_every_n_epochs = 1
    config.training.split_by_perturbation = True
    config.training.gradient_clip_norm = None
    config.training.max_val_perturbations = None
    config.training.tensorboard_log_dir = f"{EXP}/results/{TAG}/tensorboard"

    config.outputs.checkpoint_dir = f"{EXP}/results/{TAG}/checkpoint"
    config.outputs.adata_pred = f"{EXP}/results/{TAG}/adata_pred.h5ad"
    config.outputs.metrics_dir = f"{EXP}/results/{TAG}/metrics"

    return config
