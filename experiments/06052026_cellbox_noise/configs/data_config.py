from ml_collections import config_dict

EXP = "/workspace/experiments/06052026_cellbox_noise"


def get_config():
    config = config_dict.ConfigDict()

    config.perturbation_col = "target"
    config.control_key = "nontargeting"

    config.adata_train = f"{EXP}/data/adata_train_0.h5ad"
    config.adata_test = f"{EXP}/data/adata_tf_test_0.h5ad"
    config.adata_control = f"{EXP}/data/adata_control.h5ad"
    config.amask_path = f"{EXP}/data/amask.npy"

    config.model = config_dict.ConfigDict()
    config.model.layer = "log1p"
    config.model.layer_eval = "log1p"

    config.outputs = config_dict.ConfigDict()
    config.outputs.output_dir = ""
    config.outputs.adata_pred_filename = "adata_pred.h5ad"
    config.outputs.metrics_dir_name = "metrics"

    return config
