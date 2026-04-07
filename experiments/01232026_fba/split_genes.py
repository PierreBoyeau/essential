import argparse
import os

import numpy as np
import pandas as pd


def split_genes(input_file, num_chunks, output_dir):
    """
    Splits the genes CSV into multiple chunks.
    """
    df = pd.read_csv(input_file)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle for random distribution if needed, but sequential is fine here
    # df = df.sample(frac=1).reset_index(drop=True)

    chunks = np.array_split(df, num_chunks)

    created_files = []
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(output_dir, f"chunk_{i}.csv")
        chunk.to_csv(output_path, index=False)
        created_files.append(output_path)
        print(f"Created chunk {i} with {len(chunk)} genes: {output_path}")

    return created_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split genes CSV into chunks for parallel processing."
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input genes CSV file."
    )
    parser.add_argument("--num_chunks", type=int, default=8, help="Number of chunks to split into.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the chunks."
    )

    args = parser.parse_args()

    split_genes(args.input_file, args.num_chunks, args.output_dir)
