from ml_collections import config_dict
from essential.utils import get_git_hash


def get_config():
    """Returns the base configuration for ODE experiments."""
    config = config_dict.ConfigDict()
    config.tag = None
    config.git_hash = get_git_hash()

    # Data processing configuration
    config.processing = config_dict.ConfigDict()
    config.processing.adata_path = (
        "/workspace/data/250516_TF_perturbseq/250516_TF_perturbseq.annotated.h5ad"
    )
    config.processing.rt_bc = "all"
    config.processing.consolidated_cluster = "all"

    config.estimator = config_dict.ConfigDict()
    config.estimator.model_class = "sigmoid2_flow"
    config.estimator.expression_type = "logmedian"
    config.estimator.model_kwargs = config_dict.ConfigDict()
    config.estimator.model_kwargs.lambda_prior = 1.0
    config.estimator.model_kwargs.Amask = None

    config.training = config_dict.ConfigDict()
    config.training.learning_rate = 1e-3
    config.training.n_epochs = 1000
    config.training.batch_size = 128
    config.training.train_size = 0.7
    config.training.early_stopping_patience = 2000
    config.training.early_stopping_metric = "reco_loss"
    config.training.log_every_n_steps = 1
    config.training.batch_size_eval = 10000
    config.training.optimizer = None
    config.training.gradient_clip_norm = 1e5  # no gradient clipping

    config.output_path = None
    return config
