from .base import MetabolicRepresentationMethod
from .fba_gene_graph import FBAGeneGraphMethod
from .fba_moma import FBAMOMAMethod
from .gene_graph import GeneGraphMethod

METHOD_REGISTRY = {
    "gene_graph": GeneGraphMethod,
    "fba_moma": FBAMOMAMethod,
    "fba_gene_graph": FBAGeneGraphMethod,
}

__all__ = [
    "MetabolicRepresentationMethod",
    "GeneGraphMethod",
    "FBAMOMAMethod",
    "FBAGeneGraphMethod",
    "METHOD_REGISTRY",
]
