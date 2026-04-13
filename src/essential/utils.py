import hashlib
import json
import os
import subprocess

import numpy as np
import pandas as pd
import plotnine as gg
import sklearn.metrics as metrics

PLOTNINE_DEFAULT_THEME = gg.theme(
    axis_text=gg.element_text(size=6),
    axis_title=gg.element_text(size=7),
    figure_size=(3, 2),
    title=gg.element_text(size=7),
    legend_text=gg.element_text(size=6),
)

FONT_FAMILY = "Arial"

PLOTNINE_DEFAULT_THEME_2 = gg.theme(
    # Half-open frame: thin left + bottom axes only
    axis_line=gg.element_line(color="#333333", size=0.4),
    axis_line_x=gg.element_line(color="#333333", size=0.4),
    axis_line_y=gg.element_line(color="#333333", size=0.4),
    # Small outward ticks
    axis_ticks=gg.element_line(color="#333333", size=0.3),
    axis_ticks_length=3,
    axis_ticks_direction="out",
    axis_ticks_minor=gg.element_blank(),
    # Text
    axis_text=gg.element_text(size=6, family=FONT_FAMILY, color="#333333"),
    axis_title=gg.element_text(size=7, family=FONT_FAMILY, color="#333333", weight="normal"),
    title=gg.element_text(size=7, family=FONT_FAMILY, weight="bold"),
    # Clean background
    panel_background=gg.element_rect(fill="white", color="none"),
    panel_grid_major=gg.element_blank(),
    panel_grid_minor=gg.element_blank(),
    plot_background=gg.element_rect(fill="white", color="none"),
    panel_border=gg.element_blank(),
    # Sizing
    figure_size=(3, 2),
    # Legend
    legend_title=gg.element_text(size=6, family=FONT_FAMILY, color="#333333", weight="normal"),
    legend_text=gg.element_text(size=5, family=FONT_FAMILY, color="#333333"),
    legend_key_size=10,
    legend_key=gg.element_rect(fill="white", color="none"),
    legend_background=gg.element_rect(fill="white", color="#CCCCCC", size=0.3),
    legend_margin=0,
    legend_entry_spacing=2,
    # Facet strips
    strip_text=gg.element_text(size=6, family=FONT_FAMILY, color="#333333", weight="normal"),
    strip_background=gg.element_rect(fill="white", color="none"),
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
