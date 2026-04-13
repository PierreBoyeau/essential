import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_centered_kernel(K: np.ndarray) -> np.ndarray:
    """Centers a kernel matrix.

    Args:
        K (np.ndarray): The input kernel matrix.

    Returns:
        np.ndarray: The centered kernel matrix.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def compute_cka(K: np.ndarray, L: np.ndarray) -> float:
    """Computes the Centered Kernel Alignment (CKA) between two kernel matrices.

    Args:
        K (np.ndarray): The first kernel matrix.
        L (np.ndarray): The second kernel matrix.

    Returns:
        float: The CKA value. Returns 0.0 if either centered kernel has zero Frobenius norm.
    """
    K_centered = compute_centered_kernel(K)
    L_centered = compute_centered_kernel(L)

    K_norm = np.linalg.norm(K_centered, "fro")
    L_norm = np.linalg.norm(L_centered, "fro")

    if K_norm == 0 or L_norm == 0:
        return 0.0

    return np.trace(K_centered @ L_centered) / (K_norm * L_norm)


def compute_cosine_similarity(K: np.ndarray, L: np.ndarray) -> float:
    """Computes the cosine similarity between two kernel matrices.

    Args:
        K (np.ndarray): The first kernel matrix.
        L (np.ndarray): The second kernel matrix.

    Returns:
        float: The cosine similarity. Returns 0.0 if either kernel has zero Frobenius norm.
    """
    K_norm = np.linalg.norm(K, "fro")
    L_norm = np.linalg.norm(L, "fro")

    if K_norm == 0 or L_norm == 0:
        return 0.0

    return np.trace(K @ L) / (K_norm * L_norm)


def compute_spearman_off_diagonal(K: np.ndarray, L: np.ndarray) -> float:
    """Computes the Spearman rank correlation between the upper off-diagonal elements of two matrices.

    Args:
        K (np.ndarray): The first matrix.
        L (np.ndarray): The second matrix.

    Returns:
        float: The Spearman correlation coefficient. Returns 0.0 if the off-diagonal elements of either matrix are constant.
    """
    # Extract upper triangle indices, excluding diagonal
    n = K.shape[0]
    iu = np.triu_indices(n, k=1)

    k_off_diag = K[iu]
    l_off_diag = L[iu]

    # Check for constant arrays to avoid NaNs
    if np.all(k_off_diag == k_off_diag[0]) or np.all(l_off_diag == l_off_diag[0]):
        return 0.0

    corr, _ = spearmanr(k_off_diag, l_off_diag)
    return corr


def compute_distance_matrix(K: pd.DataFrame) -> pd.DataFrame:
    """Computes a distance matrix from a kernel matrix.

    Args:
        K (pd.DataFrame): The input kernel matrix.

    Returns:
        pd.DataFrame: The computed distance matrix.
    """
    k_diag = np.diag(K)
    dist_squared = k_diag[:, None] + k_diag[None, :] - 2 * K
    dist_squared = np.clip(dist_squared, 0, None)
    D = np.sqrt(dist_squared)
    return D


def process_and_align(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aligns two DataFrames based on their common indices and extracts their values.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the aligned values of the first and second DataFrames.
    """
    df1.index = df1.index.astype(str)
    df1.columns = df1.columns.astype(str)
    df2.index = df2.index.astype(str)
    df2.columns = df2.columns.astype(str)
    common_genes = np.intersect1d(df1.index, df2.index)
    df1_aligned = df1.loc[common_genes, common_genes]
    df2_aligned = df2.loc[common_genes, common_genes]
    return df1_aligned, df2_aligned


def wide_to_long(
    df: pd.DataFrame,
    value_name: str,
    index_name: str,
    column_name: str,
    remove_diagonal: bool = False,
    remove_lower_triangle: bool = False,
) -> pd.DataFrame:
    """Converts a wide kernel matrix to a long DataFrame.

    Args:
        df (pd.DataFrame): The input kernel matrix.
        value_name (str): Name for the value column in the long format.
        index_name (str): Name for the index column.
        column_name (str): Name for the column name column.
        remove_diagonal (bool, optional): Whether to remove diagonal elements. Defaults to False.
        remove_lower_triangle (bool, optional): Whether to remove the lower triangle elements. Defaults to False.

    Returns:
        pd.DataFrame: The converted long DataFrame.
    """
    res_ = (
        df.stack().rename_axis(index=[index_name, column_name]).to_frame(value_name).reset_index()
    )
    if remove_diagonal:
        res_ = res_.loc[lambda x: x[index_name] != x[column_name]]
    if remove_lower_triangle:
        res_ = res_.loc[lambda x: x[index_name] < x[column_name]]
    return res_
