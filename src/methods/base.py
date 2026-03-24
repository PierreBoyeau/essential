from abc import ABC, abstractmethod
import pandas as pd
import cobra
from typing import List, Any

class MetabolicRepresentationMethod(ABC):
    """
    Abstract base class for metabolic network representation methods.
    """
    
    @abstractmethod
    def fit(self, model: cobra.Model, genes: List[str], **kwargs) -> Any:
        """
        Fit the method using the metabolic model and a target list of genes.
        
        Args:
            model: COBRApy metabolic model.
            genes: List of gene IDs/names to generate representations for.
            **kwargs: Additional method-specific parameters.
        """
        pass
        
    @abstractmethod
    def get_kernel(self) -> pd.DataFrame:
        """
        Return a G x G symmetric kernel/similarity matrix.
        
        Returns:
            pd.DataFrame: A symmetric similarity matrix with gene names as index and columns.
        """
        pass
        
    @abstractmethod
    def get_expectations(self) -> pd.DataFrame:
        """
        Return a DataFrame of expected gene-gene interactions and their confidence.
        
        Returns:
            pd.DataFrame: DataFrame containing expected gene-gene pairs and similarity/confidence scores.
        """
        pass
