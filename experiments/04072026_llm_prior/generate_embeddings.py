import argparse
import json
import os

import numpy as np
import ollama
import pandas as pd
from tqdm import tqdm


def dict_to_description(card_dict):
    """
    Converts a dictionary of attributes into a structured text description.
    Assumes values are lists of strings or single strings.
    """
    lines = []
    for key, value in card_dict.items():
        if isinstance(value, list):
            val_str = ", ".join(str(v) for v in value)
        else:
            val_str = str(value)
        # Format key as Title Case (e.g. "substrates" -> "Substrates")
        key_formatted = key.replace("_", " ").title()
        lines.append(f"{key_formatted}: {val_str}")

    return "\n".join(lines)


def generate_embeddings(input_file, output_file, model_name="gemma4"):
    """
    Reads a JSON file mapping gene names to substrates and products,
    and uses Ollama to generate an embedding for each gene.
    """
    print(f"Loading gene cards from '{input_file}'...")
    with open(input_file, "r") as f:
        gene_cards = json.load(f)

    gene_names = list(gene_cards.keys())
    embeddings = []

    print(f"Generating embeddings using Ollama model '{model_name}' for {len(gene_names)} genes...")

    for gene in tqdm(gene_names, desc="Embedding genes"):
        card = gene_cards[gene]

        # We omit the gene name (if present) from the text to force the model to rely solely on the metabolic function
        card_without_name = {k: v for k, v in card.items() if k != "name"}
        description = dict_to_description(card_without_name)

        try:
            response = ollama.embeddings(model=model_name, prompt=description)
            embeddings.append(response["embedding"])
        except Exception as e:
            print(f"\nError generating embedding for {gene}: {e}")
            print(
                "Make sure Ollama is running and the model is pulled ('ollama serve' and 'ollama pull <model_name>')"
            )
            raise e

    print("Converting embeddings to DataFrame...")
    embeddings_matrix = np.array(embeddings)

    # Store as a Pandas DataFrame with gene names as the index.
    # This format is easy to load via pd.read_pickle() and compatible with pairwise_distances
    df_embeddings = pd.DataFrame(embeddings_matrix, index=gene_names)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_embeddings.to_pickle(output_file)
    print(f"Successfully saved embeddings to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="experiments/04072026_llm_prior/data/fba_filtered_metabolites_gene_cards.json",
        help="Input JSON file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/04072026_llm_prior/data/fba_filtered_metabolites_gene_embeddings.pkl",
        help="Output pickle file",
    )
    parser.add_argument("--model", type=str, default="embeddinggemma")

    args = parser.parse_args()

    generate_embeddings(input_file=args.input, output_file=args.out, model_name=args.model)
