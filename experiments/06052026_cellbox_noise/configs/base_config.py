import sys

sys.path.insert(0, "/workspace/src")
from cellbox.config import get_config as _base_config

EXP = "/workspace/experiments/06052026_cellbox_noise"
TAG = "baseline"


def get_config():
    config = _base_config()

    config.tag = TAG

    config.perturbation_col = "target"
    config.control_key = "nontargeting"
    config.filter_regulators = True

    config.adata_train = f"{EXP}/data/adata_train_0.h5ad"
    config.adata_test = f"{EXP}/data/adata_tf_test_0.h5ad"
    config.adata_control = f"{EXP}/data/adata_control.h5ad"
    config.amask_path = f"{EXP}/data/amask.npy"

    config.model.model_type = "gaussian"
    config.model.layer = "log1p"
    config.model.train_mode = "rollout"
    config.model.n_rollout_train = 10
    config.model.layer_eval = "log1p"
    config.model.n_rollout_val = 10
    config.model.n_val_control = 64

    config.training.learning_rate = 1e-3
    config.training.n_epochs = 10000
    config.training.batch_size = 256
    config.training.train_size = 0.8
    config.training.early_stopping_patience = 50
    config.training.early_stopping_metric = "lfc_pearson_r"
    config.training.early_stopping_mode = "max"
    config.training.validate_every_n_epochs = 10
    config.training.log_every_n_epochs = 1
    config.training.split_by_perturbation = True
    config.training.gradient_clip_norm = None
    config.training.max_val_perturbations = None
    config.training.tensorboard_log_dir = f"{EXP}/results/{TAG}/tensorboard"

    config.outputs.checkpoint_dir = f"{EXP}/results/{TAG}/checkpoint"
    config.outputs.adata_pred = f"{EXP}/results/{TAG}/adata_pred.h5ad"
    config.outputs.metrics_dir = f"{EXP}/results/{TAG}/metrics"

    return config
