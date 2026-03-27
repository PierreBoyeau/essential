import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import os
import subprocess
import hashlib
import json
import plotnine as gg


PLOTNINE_DEFAULT_THEME = gg.theme(
    axis_text=gg.element_text(size=6),
    axis_title=gg.element_text(size=7),
    figure_size=(3, 2),
    title=gg.element_text(size=7),
    legend_text=gg.element_text(size=6),
)


def get_hash(config):
    """Generate hash from config, excluding paths.

    Args:
        config: ConfigDict containing experiment configuration

    Returns:
        str: 8-character hash of the configuration
    """
    config_dict_copy = config.to_dict()
    # Exclude paths from hash to ensure reproducibility
    hash_dict = {
        k: v
        for k, v in config_dict_copy.items()
        if k not in ["output_path"] and not (k == "processing" and "adata_path" in str(v))
    }
    # Also exclude adata_path from processing if it exists
    if "processing" in hash_dict and isinstance(hash_dict["processing"], dict):
        hash_dict["processing"] = {
            k: v for k, v in hash_dict["processing"].items() if k != "adata_path"
        }

    str_config = json.dumps(hash_dict, sort_keys=True)
    return hashlib.sha256(str_config.encode()).hexdigest()[:8]


def get_git_hash():
    """Get the current git commit hash."""
    try:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=file_dir)
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not_a_git_repo"
