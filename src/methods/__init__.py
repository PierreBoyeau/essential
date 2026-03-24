from .base import MetabolicRepresentationMethod
from .gene_graph import GeneGraphMethod
from .fba_moma import FBAMOMAMethod

METHOD_REGISTRY = {
    "gene_graph": GeneGraphMethod,
    "fba_moma": FBAMOMAMethod
}

__all__ = ["MetabolicRepresentationMethod", "GeneGraphMethod", "FBAMOMAMethod", "METHOD_REGISTRY"]
