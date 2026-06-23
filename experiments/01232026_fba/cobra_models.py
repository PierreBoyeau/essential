import json

import cobra
import numpy as np
import pandas as pd


def get_model_components_df():
    """
    Load gene, reaction, and metabolite metadata into DataFrames.
    Optionally takes a model, but currently reloads from JSON source for full attributes.
    """
    with open("/workspace/data/bigg/iJO1366.json") as f:
        data = json.load(f)

    metabolite_info = data["metabolites"]
    reaction_info = data["reactions"]
    gene_info = data["genes"]

    metabolites_df = (
        pd.DataFrame(metabolite_info).set_index("id").assign(id_num=lambda x: np.arange(len(x)))
    )
    reactions_df = (
        pd.DataFrame(reaction_info).set_index("id").assign(id_num=lambda x: np.arange(len(x)))
    )
    genes_df = pd.DataFrame(gene_info).set_index("id").assign(id_num=lambda x: np.arange(len(x)))
    return metabolites_df, reactions_df, genes_df


def load_ecoli_rich_medium_model():
    """
    Load the E. coli iJO1366 model and configure it with EZ Rich Medium constraints.
    """
    model = cobra.io.load_json_model("/workspace/data/bigg/iJO1366.json")
    model.solver = "glpk"
    amino_acids_to_add = [
        "EX_ala__L_e",  # alanine
        "EX_arg__L_e",  # arginine
        "EX_asn__L_e",  # asparagine
        "EX_asp__L_e",  # aspartate
        "EX_cys__L_e",  # cysteine
        "EX_glu__L_e",  # glutamate
        "EX_gln__L_e",  # glutamine
        "EX_gly_e",  # glycine
        "EX_his__L_e",  # histidine
        "EX_ile__L_e",  # isoleucine
        "EX_leu__L_e",  # leucine
        "EX_lys__L_e",  # lysine
        "EX_met__L_e",  # methionine
        "EX_phe__L_e",  # phenylalanine
        "EX_pro__L_e",  # proline
        "EX_ser__L_e",  # serine
        "EX_thr__L_e",  # threonine
        "EX_trp__L_e",  # tryptophan
        "EX_tyr__L_e",  # tyrosine
        "EX_val__L_e",  # valine
    ]

    nucleobases_to_add = [
        "EX_ade_e",  # adenine (A in ACGU)
        "EX_csn_e",  # cytosine (C in ACGU)
        "EX_ura_e",  # uracil (U in ACGU)
        "EX_gua_e",  # guanine (G in ACGU)
    ]

    # Need to add (from VA solution):
    vitamins_to_add = [
        "EX_thm_e",  # thiamine (vitamin B1)
        "EX_pnto__R_e",  # pantothenate (vitamin B5)
        # The three below are not found
        # "EX_4abz_e",  # p-aminobenzoic acid (PABA)
        # "EX_4hbz_e",  # p-hydroxybenzoic acid
        # "EX_23dhb_e",  # 2,3-dihydroxybenzoic acid (enterobactin precursor)
    ]

    medium_import_reactions = amino_acids_to_add + nucleobases_to_add + vitamins_to_add

    # Get current medium configuration
    # In cobrapy, model.medium returns a dict of {reaction_id: uptake_limit}
    current_medium = model.medium

    # Update medium with new components
    for reaction_id in medium_import_reactions:
        if reaction_id in model.reactions:
            # Allow unlimited uptake (or specific value like 10.0)
            current_medium[reaction_id] = 1000.0
        else:
            print(f"Warning: Reaction {reaction_id} not found in model.")

    # Apply the updated medium to the model
    model.medium = current_medium
    return model
