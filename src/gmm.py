import numpy as np
from typing import Tuple, List, Optional
from src.kmeans import kmeans
from src.utils import RANDOM_SEED


def gmm(
    X: np.ndarray,
    k: int,
    tolerance: float = 0.0001,
    max_iters: int = 300,
    seed: Optional[int] = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    """
    Gaussian Mixture Model clustering

    Args:
        X: Data matrix - (n_samples, n_features)
        k: Number of clusters
        tolerance: Convergence threshold for log-likelihood change
        max_iters: Maximum number of iterations
        seed: Random seed for reproducibility
    """
    n_samples, n_features = X.shape

    if seed:
        np.random.seed(seed)

    # initialize parameters
    # MEANS: initialize as centroids returned from a run of kmeans
    means, _, _ = kmeans(X, k, tolerance=0.01, max_iters=max_iters)

    # COVARIANCES: initialize as global dataset covariance
    covariances = np.array([np.cov(X.T) for _ in range(k)])

    # PRIORS: initialize uniformly for each class
    priors = np.ones(k) / k

    # use log likelihood as metric
    log_likelihoods = []
    responsibilities = np.ndarray((n_samples, k))

    for interation in range(max_iters):
        # E-step: compute responsibility of each cluster on each sample
        # compute numerator
        for i, sample in enumerate(X):
            for j in range(k):
                # p(x|cluster_j) * p(cluster_j)
                responsibilities[i, j] = priors[j] * multivariate_gaussian_pdf(
                    sample, means[j], covariances[j]
                )

        # compute denominator
        for i in range(n_samples):
            responsibilities[i, :] /= responsibilities[i, :].sum()

        # M-step: update parameters using responsibilities
        new_means = np.ndarray((k, n_features))
        new_covariances = np.ndarray((k, n_features, n_features))
        new_priors = np.ndarray(k)

        for j in range(k):
            # effective number of samples assigned to cluster j
            N_k = np.sum(responsibilities[:, j])

            # Update mean
            new_means[j] = np.sum(responsibilities[:, j : j + 1] * X, axis=0) / N_k

            # Update prior
            new_priors[j] = N_k / n_samples

            # Update covariance matrix
            diff = X - new_means[j]
            weighted_diff = responsibilities[:, j][:, np.newaxis] * diff
            new_covariances[j] = (weighted_diff.T @ diff) / N_k + 0.01 * np.eye(
                n_features
            )

        means = new_means
        covariances = new_covariances
        priors = new_priors

        # Check convergence based on log-likelihood change
        log_likelihood = compute_log_likelihood(X, means, covariances, priors)
        log_likelihoods.append(log_likelihood)

        if (
            len(log_likelihoods) > 1
            and abs(log_likelihood - log_likelihoods[-2]) < tolerance
        ):
            print(f"Converged at iteration {interation}")
            break

        print(f"Iteration {interation}: log likelihood = {log_likelihood}")

    # Final cluster assignments (most probable cluster)
    assignments = np.argmax(responsibilities, axis=1)

    return means, covariances, priors, assignments, log_likelihoods


def multivariate_gaussian_pdf(
    x: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> float:
    """
    Compute probability density function of multivariate Gaussian distribution.
    """
    k = len(mean)

    # Add regularization to avoid singular matrices
    cov_regularized = cov + 0.1 * np.eye(k)

    cov_det = np.linalg.det(cov_regularized)
    cov_inv = np.linalg.inv(cov_regularized)
    diff = x - mean
    exponent = -0.5 * (diff.T @ cov_inv @ diff)
    norm = np.sqrt((2 * np.pi) ** k * cov_det)

    return (1 / norm) * np.exp(exponent)


def compute_log_likelihood(
    X: np.ndarray, means: np.ndarray, covariances: np.ndarray, priors: np.ndarray
) -> float:
    """Compute total log-likelihood of data given current GMM parameters."""
    n_samples = X.shape[0]
    total_log_likelihood = 0

    for i in range(n_samples):
        sample_likelihood = 0
        for j in range(len(means)):
            sample_likelihood += priors[j] * multivariate_gaussian_pdf(
                X[i], means[j], covariances[j]
            )

        # Add small epsilon to avoid log(0)
        total_log_likelihood += np.log(sample_likelihood + 1e-10)

    return total_log_likelihood
