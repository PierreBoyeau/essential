"""
Explores reactions and bounds for the E. coli rich medium model.
"""

import os
import sys

import pandas as pd

# Add /workspace to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.essential.fba import load_ecoli_rich_medium_model


def main():
    # Load the model
    model = load_ecoli_rich_medium_model()

    # Extract bounds for each reaction
    data = []
    for reaction in model.reactions:
        data.append(
            {
                "reaction name": reaction.id,
                "lower bound": reaction.lower_bound,
                "upper bound": reaction.upper_bound,
            }
        )

    # Create a DataFrame and save it as a TSV
    df = pd.DataFrame(data)

    # Save to the same directory as this script
    output_path = os.path.join(os.path.dirname(__file__), "reaction_bounds.tsv")
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Successfully saved bounds to {output_path}")


if __name__ == "__main__":
    main()
