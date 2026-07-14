"""Run Boltz2 binding prediction for every ligand-protein pair in pairs.csv.

We render one Boltz YAML per pair into CONFIG_DIR, then invoke `boltz predict`
ONCE on that directory. Boltz loads the model a single time and iterates over
every config internally, instead of paying the checkpoint/trainer startup cost
per pair (the old per-pair-subprocess approach reloaded the model N times).

Resumability is free: Boltz skips any pair whose
`predictions/<id>/affinity_<id>.json` already exists (its own --override=False
behaviour, boltz/main.py:389), so an interrupted run can simply be re-launched.

Output layout (single batched call on a directory named `configs`):
    RESULTS_DIR/boltz_results_configs/predictions/<pair_id>/affinity_<pair_id>.json

Run with:  python run_predictions.py
Configuration lives in the GLOBALS block below (no argparse on purpose).
"""

import os
import subprocess
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# GLOBALS  (the only knobs)
# ----------------------------------------------------------------------------
PAIRS_CSV = "/workspace/experiments/07062026_tf_updated/boltz_prediction/pairs.csv"
TEMPLATE_YAML = "/workspace/experiments/07062026_tf_updated/boltz_configs/template.yaml"
CONFIG_DIR = "/workspace/experiments/07062026_tf_updated/boltz_prediction/configs"
RESULTS_DIR = "/workspace/experiments/07062026_tf_updated/boltz_prediction/results"

RECYCLING_STEPS = 10
DIFFUSION_SAMPLES = 25
USE_MSA_SERVER = True


def render_all_configs(template, pairs):
    """Write the filled Boltz YAML for every pair into CONFIG_DIR."""
    for pair in pairs.itertuples():
        config = template.format(
            protein_sequence=pair.protein_sequence,
            ligand_smiles=pair.smiles,
        )
        path = os.path.join(CONFIG_DIR, f"{pair.pair_id}.yaml")
        open(path, "w").write(config)
    print(f"Rendered {len(pairs)} configs into {CONFIG_DIR}")


def run_boltz_batch():
    """Invoke `boltz predict` once on the whole CONFIG_DIR."""
    cmd = [
        "boltz",
        "predict",
        CONFIG_DIR,
        "--recycling_steps",
        str(RECYCLING_STEPS),
        "--diffusion_samples",
        str(DIFFUSION_SAMPLES),
        "--out_dir",
        RESULTS_DIR,
    ]
    if USE_MSA_SERVER:
        cmd.append("--use_msa_server")
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def affinity_path(pair_id):
    """Where the batched run writes a pair's affinity score."""
    stem = Path(CONFIG_DIR).stem  # -> "configs"
    return os.path.join(
        RESULTS_DIR,
        f"boltz_results_{stem}",
        "predictions",
        pair_id,
        f"affinity_{pair_id}.json",
    )


def reconcile(pairs):
    """Report which pairs did / did not produce an affinity score.

    Batching trades the old per-pair try/except for a single call, so we check
    for missing outputs here rather than assuming success.
    """
    missing = [
        p.pair_id for p in pairs.itertuples() if not os.path.isfile(affinity_path(p.pair_id))
    ]
    done = len(pairs) - len(missing)
    print(f"\nReconcile: {done}/{len(pairs)} pairs have an affinity score.")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for pid in missing:
            print(f"  - {pid}")


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    template = open(TEMPLATE_YAML).read()
    pairs = pd.read_csv(PAIRS_CSV)
    print(f"Loaded {len(pairs)} pairs from {PAIRS_CSV}")

    render_all_configs(template, pairs)
    run_boltz_batch()
    reconcile(pairs)

    print("\nDone.")


if __name__ == "__main__":
    main()
