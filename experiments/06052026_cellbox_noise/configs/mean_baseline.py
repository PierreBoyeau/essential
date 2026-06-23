import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_config import get_config as _data_config

EXP = "/workspace/experiments/06052026_cellbox_noise"
TAG = "mean_baseline"


def get_config():
    config = _data_config()

    config.tag = TAG

    config.outputs.output_dir = f"{EXP}/results/{TAG}"

    return config
