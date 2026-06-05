from cellbox import CellBoxEstimator


def run(config):
    adata_train, amask = CellBoxEstimator.prepare_data(
        adata_path=config.adata_train,
        min_library_size=config.prepare_data.min_library_size,
        perturbation_col=config.perturbation_col,
        normalization=config.prepare_data.normalization,
    )
    model = CellBoxEstimator(
        perturbation_col=config.perturbation_col,
        control_key=config.control_key,
        **config.model.to_dict(),
    )
    model.fit(
        adata_train,
        Amask=amask if config.filter_regulators else None,
        **config.training.to_dict(),
    )
    model.save(config.outputs.checkpoint_dir)


if __name__ == "__main__":
    from absl import app, flags
    from ml_collections import config_flags

    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file("config", None, "Path to config file")

    def main(_):
        run(FLAGS.config)

    app.run(main)
