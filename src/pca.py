import numpy as np


def fit_pca(X: np.ndarray):
    covariance_matrix = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # sort eigenvals & eigenvects by eigenval
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    return (
        eigenvalues,
        eigenvectors,
        explained_variance_ratio,
    )


def transform_pca(X: np.ndarray, W: np.ndarray, n_components: int = None):
    if n_components is None:
        n_components = W.shape[0]

    return X @ W[:, :n_components]


def reconstruct_pca(
    Z: np.ndarray, W: np.ndarray, mean: np.ndarray, std: np.ndarray, n_components
):
    X_std_reconstructed = Z @ W[:, :n_components].T
    return X_std_reconstructed * std + mean
