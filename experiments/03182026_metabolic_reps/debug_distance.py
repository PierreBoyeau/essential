import sys

sys.path.append("/workspace/src/essential")
from fba import load_ecoli_rich_medium_model

model = load_ecoli_rich_medium_model()
ompN = next((g for g in model.genes if g.name == "ompN"), None)
phoE = next((g for g in model.genes if g.name == "phoE"), None)

if ompN and phoE:
    ompN_prods = set()
    phoE_subs = set()

    print("ompN reactions:")
    for rxn in ompN.reactions:
        print(f"  - {rxn.id}: {rxn.reaction} (bounds: {rxn.bounds})")
        if rxn.upper_bound > 0:
            ompN_prods.update([m.id for m in rxn.products])
        if rxn.lower_bound < 0:
            ompN_prods.update([m.id for m in rxn.reactants])

    print("\nphoE reactions:")
    for rxn in phoE.reactions:
        print(f"  - {rxn.id}: {rxn.reaction} (bounds: {rxn.bounds})")
        if rxn.upper_bound > 0:
            phoE_subs.update([m.id for m in rxn.reactants])
        if rxn.lower_bound < 0:
            phoE_subs.update([m.id for m in rxn.products])

    intersection = ompN_prods.intersection(phoE_subs)
    print("\nIntersection (ompN products that are phoE substrates):")
    print(intersection)
