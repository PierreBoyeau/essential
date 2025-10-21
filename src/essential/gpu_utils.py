import os
import subprocess
import numpy as np


def select_best_gpus(n_gpus=1):
    """
    Selects the top N GPUs with the most available memory and sets CUDA_VISIBLE_DEVICES.

    This function queries nvidia-smi to determine the free memory for each GPU,
    identifies the `n_gpus` with the most free memory, and sets the
    CUDA_VISIBLE_DEVICES environment variable to a comma-separated list of their IDs.

    This should be called at the beginning of a script or notebook, before any
    GPU-initializing libraries (like JAX, PyTorch, or TensorFlow) are imported.

    Parameters
    ----------
    n_gpus : int, default=1
        The number of GPUs to select.
    """
    try:
        # Query nvidia-smi for free memory
        command = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
        free_memory = np.array([int(x) for x in result.stdout.strip().split("\n")])

        n_available = len(free_memory)
        print(f"Found {n_available} GPUs.")

        if n_gpus > n_available:
            print(
                f"Warning: Requested {n_gpus} GPUs, but only {n_available} are available. "
                f"Selecting all {n_available} GPUs."
            )
            n_gpus = n_available

        # Get the indices of the GPUs with the most free memory
        # argsort sorts in ascending order, so we take the last n_gpus elements
        best_gpu_ids = np.argsort(free_memory)[-n_gpus:]

        # Reverse the list to have the GPU with most memory first
        best_gpu_ids = best_gpu_ids[::-1]

        selected_gpus_str = ",".join(map(str, best_gpu_ids))
        selected_gpus_memory = free_memory[best_gpu_ids]

        print(
            f"Selecting GPU(s) {selected_gpus_str} with {selected_gpus_memory} MiB "
            "free memory respectively."
        )

        # Set the environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus_str

    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(
            "nvidia-smi command not found or failed to execute. "
            "Could not automatically select a GPU. "
            f"Error: {e}"
        )
