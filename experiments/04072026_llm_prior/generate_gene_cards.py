import argparse
import json
import os
import sys

import cobra
from cobra.flux_analysis import pfba
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.metabolic_models import get_model


def generate_gene_cards(
    model_type="ecoli_rich_medium",
    flux_threshold=1e-6,
    output_file="experiments/04072026_llm_prior/data/fba_filtered_metabolites_gene_cards.json",
):
    """
    Generate gene cards using FBA to determine actual substrates and products based on flux direction.
    """
    print(f"Loading model '{model_type}'...")
    model = get_model(model_type)

    print("Running FBA to get fluxes...")
    try:
        model.solver = "glpk"
    except Exception:
        pass

    try:
        sol_wt = pfba(model)
        fluxes = sol_wt.fluxes
    except Exception:
        # Fallback to standard FBA if pFBA fails
        print("pFBA failed, falling back to standard FBA...")
        sol_wt = model.optimize()
        fluxes = sol_wt.fluxes

    print(f"Generating gene cards for {len(model.genes)} genes...")
    gene_cards = {}

    for gene in tqdm(model.genes, desc="Processing genes"):
        subs = set()
        prods = set()

        for rxn in gene.reactions:
            flux = fluxes.get(rxn.id, 0.0)

            if abs(flux) < flux_threshold:
                continue

            forward = flux > 0
            reverse = flux < 0

            rxn_reactants = [m.name for m in rxn.reactants]
            rxn_products = [m.name for m in rxn.products]

            if forward:
                subs.update(rxn_reactants)
                prods.update(rxn_products)
            elif reverse:
                subs.update(rxn_products)
                prods.update(rxn_reactants)

        gene_name = gene.name if gene.name else gene.id
        gene_cards[gene_name] = {"substrates": list(subs), "products": list(prods)}

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(gene_cards, f, indent=4)

    print(f"Successfully saved gene cards to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="ecoli_rich_medium", help="Metabolic model name"
    )
    parser.add_argument(
        "--threshold", type=float, default=1e-6, help="Flux threshold for active reactions"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/04072026_llm_prior/data/fba_filtered_metabolites_gene_cards.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    generate_gene_cards(model_type=args.model, flux_threshold=args.threshold, output_file=args.out)
