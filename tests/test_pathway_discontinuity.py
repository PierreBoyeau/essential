import networkx as nx
import numpy as np
import scanpy as sc

from essential.pathway_discontinuity import PathwayDiscontinuity


def create_random_data():
    perturbation_classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    perturbation_assignments = []
    for perturbation_class in perturbation_classes:
        for i in range(100):
            perturbation_assignments.append(perturbation_class)
    perturbation_assignments = np.array(perturbation_assignments)
    n_obs = len(perturbation_assignments)
    n_vars = 5000
    X = np.random.randint(0, 100, size=(n_obs, n_vars))
    X[perturbation_assignments == "B"] += 50
    adata = sc.AnnData(X, obs={"perturbation_class": perturbation_assignments})

    # create graph
    G = nx.DiGraph()
    chain = [0, 1, 2, 3, 4]
    G.add_nodes_from(chain)
    for u, v, name in zip(chain[:-1], chain[1:], ["A", "B", "C", "D"]):
        G.add_edge(u, v, name=name)
    path = [5, 6, 7]
    G.add_nodes_from(path)
    for u, v, name in zip(path[:-1], path[1:], ["E", "F"]):
        G.add_edge(u, v, name=name)

    return adata, G


def test_pathway_discontinuity():
    adata, G = create_random_data()
    sc.pp.highly_variable_genes(adata, n_top_genes=500, flavor="seurat_v3")
    sc.pp.pca(adata, use_highly_variable=True, n_comps=50)

    pda = PathwayDiscontinuity(adata, representation_obsm_key="X_pca", metabolic_graph=G)

    pairs = pda._extract_consecutive_pairs(G)

    results = pda.fit(threshold=0.1, mode="mmd_stat")
    results.plot()
