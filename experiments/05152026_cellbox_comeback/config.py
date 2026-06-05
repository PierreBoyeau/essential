"""Configuration for the CellBox comeback prediction experiment.

Shared experiment settings (data, normalization, split, output) live at the
top level. Per-model settings live under ``config.models.<name>``.

Override any field from the command line::

    python run_prediction.py --config=config.py --config.normalization.method=library
    python run_prediction.py --config=config.py --config.models.cellbox.training.learning_rate=1e-2

Adding a new model: implement it in ``predictors.py`` and add its config block
to ``_base_config`` under ``config.models.<name>``.
"""

from ml_collections import config_dict


def _base_config():
    config = config_dict.ConfigDict()

    # --- which model to run ---
    config.model_name = "cellbox"
    config.tag = "cellbox"

    # --- data ---
    config.adata_path = "/workspace/data/de122_lce75/adata_de122_lce75_merged.h5ad"
    config.perturbation_col = "target"
    config.control_key = "nontargeting"
    config.min_library_size = 1e3
    config.experiment_subset = "lce75"
    config.seed = 0

    config.output_path = "/workspace/experiments/05152026_cellbox_comeback/metrics"

    # --- count normalization ---
    config.normalization = config_dict.ConfigDict()
    config.normalization.method = "log1p"
    config.normalization.target_sum = 1e4
    config.normalization.clip_percentile = None

    # --- train/test split ---
    config.split = config_dict.ConfigDict()
    _splits = "/workspace/experiments/05152026_cellbox_comeback/splits"
    config.split.train_targets_path = f"{_splits}/tf_train_0.txt"
    config.split.train_extra_targets_path = config_dict.placeholder(str)
    config.split.test_targets_path = f"{_splits}/tf_test_0.txt"

    # --- per-model configs ---
    config.models = config_dict.ConfigDict()

    config.models.cellbox = config_dict.ConfigDict()
    config.models.cellbox.filter_regulators = False
    config.models.cellbox.standardize_inputs = True
    config.models.cellbox.training = config_dict.ConfigDict()
    config.models.cellbox.training.learning_rate = 1e-3
    config.models.cellbox.training.n_epochs = 10000
    config.models.cellbox.training.batch_size = 256
    config.models.cellbox.training.train_size = 0.8
    config.models.cellbox.training.early_stopping_patience = 300
    config.models.cellbox.training.early_stopping_metric = "lfc_pearson"
    config.models.cellbox.training.early_stopping_mode = "max"
    config.models.cellbox.training.validate_every_n_epochs = 50
    config.models.cellbox.training.log_every_n_epochs = 1
    config.models.cellbox.training.gradient_clip_norm = None
    config.models.cellbox.training.split_by_perturbation = True
    config.models.cellbox.training.max_val_perturbations = config_dict.placeholder(int)
    config.models.cellbox.training.n_val_steps = 5
    config.models.cellbox.training.train_mode = "reconstruction"
    config.models.cellbox.training.n_train_steps = 5

    config.models.mean = config_dict.ConfigDict()

    return config


def get_config():
    return _base_config()
