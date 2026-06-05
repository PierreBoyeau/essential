from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.tag = "cellbox"

    config.perturbation_col = "target"
    config.control_key = "nontargeting"
    config.filter_regulators = False

    config.adata_train = ""
    config.adata_test = ""
    config.adata_control = ""

    config.prepare_data = config_dict.ConfigDict()
    config.prepare_data.min_library_size = 1e3
    config.prepare_data.normalization = config_dict.ConfigDict()
    config.prepare_data.normalization.method = "log1p"
    config.prepare_data.normalization.target_sum = 1e4

    config.model = config_dict.ConfigDict()
    config.model.standardize_inputs = False
    config.model.train_mode = "reconstruction"
    config.model.n_rollout_train = 100
    config.model.n_val_control = 64
    config.model.n_rollout_val = 100

    config.training = config_dict.ConfigDict()
    config.training.learning_rate = 1e-3
    config.training.n_epochs = 5000
    config.training.batch_size = 256
    config.training.train_size = 0.9
    config.training.early_stopping_patience = 300
    config.training.early_stopping_metric = "reco_loss"
    config.training.early_stopping_mode = "min"
    config.training.log_every_n_epochs = 100
    config.training.gradient_clip_norm = None
    config.training.split_by_perturbation = False
    config.training.max_val_perturbations = None
    config.training.validate_every_n_epochs = 0

    config.outputs = config_dict.ConfigDict()
    config.outputs.checkpoint_dir = ""
    config.outputs.adata_pred = ""
    config.outputs.metrics_dir = ""

    return config
