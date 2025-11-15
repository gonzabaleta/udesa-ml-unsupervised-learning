import numpy as np
from typing import Tuple, Optional


def fit_pca(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        X: Standardized data matrix of shape (n_samples, n_features)

    Returns:
        eigenvalues: Eigenvalues in descending order
        eigenvectors: Corresponding eigenvectors (principal components)
        explained_variance_ratio: Proportion of variance explained by each component
    """
    covariance_matrix = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues & vectors by eigenvalue magnitude
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute explained variance ratios
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    return eigenvalues, eigenvectors, explained_variance_ratio


def transform_pca(
    X: np.ndarray, W: np.ndarray, n_components: Optional[int] = None
) -> np.ndarray:
    """
    Transform data to lower-dimensional PCA space.

    Args:
        X: Data matrix to transform
        W: Principal components matrix
        n_components: Number of components to use (all if None)
    """
    if n_components is None:
        n_components = W.shape[1]

    # Project data to principal components
    return X @ W[:, :n_components]


def reconstruct_pca(
    Z: np.ndarray,
    W: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Reconstruct original data from PCA representation.

    Args:
        Z: Data in latent space
        W: Principal components matrix
        mean: Original data mean
        std: Original data standard deviation
        n_components: Number of components used in transformation
    """
    # Inverse PCA transformation
    X_std_reconstructed = Z @ W[:, :n_components].T

    # Denormalize using original statistics
    return X_std_reconstructed * std + mean
