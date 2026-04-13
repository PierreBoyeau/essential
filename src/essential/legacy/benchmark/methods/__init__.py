from .base import MetabolicRepresentationMethod
from .fba_gene_graph import FBAGeneGraphMethod
from .fba_moma import FBAMOMAMethod
from .gene_graph import GeneGraphMethod
from .llm_embedding import LLMEmbeddingMethod

METHOD_REGISTRY = {
    "gene_graph": GeneGraphMethod,
    "fba_moma": FBAMOMAMethod,
    "fba_gene_graph": FBAGeneGraphMethod,
    "llm_embedding": LLMEmbeddingMethod,
}

__all__ = [
    "MetabolicRepresentationMethod",
    "GeneGraphMethod",
    "FBAMOMAMethod",
    "FBAGeneGraphMethod",
    "LLMEmbeddingMethod",
    "METHOD_REGISTRY",
]
