import numpy as np
from typing import Tuple, List, Optional
from src.utils import RANDOM_SEED


def distance_squared(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum((x1 - x2) ** 2)


def kmeans(
    X: np.ndarray,
    k: int,
    tolerance: float = 0.0001,
    max_iters: int = 300,
    seed: Optional[int] = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    K-means clustering algorithm.

    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of clusters
        tolerance: Convergence threshold for loss change
        max_iters: Maximum number of iterations
        seed: Random seed for reproducibility
    """
    n_samples, n_features = X.shape

    if seed:
        np.random.seed(seed)

    # initialize each centroid in one random sample
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()

    losses = []
    assignments = np.ndarray(n_samples, dtype=int)

    for interation in range(max_iters):
        # E-step: assign each sample to its closest centroid

        for i, sample in enumerate(X):
            # Calculate distances to all centroids
            distances = [distance_squared(sample, centroid) for centroid in centroids]
            best_centroid = np.argmin(distances)
            assignments[i] = int(best_centroid)

        # M-step: update centroids minimizing cost function (move to center of cluster)
        new_centroids = np.ndarray((k, n_features))
        for j in range(k):
            cluster_points = X[assignments == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                # empty cluster: keep current centroid
                new_centroids[j] = centroids[j]

        centroids = new_centroids

        # Check convergence based on loss change
        loss = kmeans_loss(X, centroids, assignments)
        losses.append(loss)

        if len(losses) > 1 and abs(loss - losses[-2]) < tolerance:
            print(f"Converged at iteration {interation}")
            break

        if interation % 10 == 0:
            print(f"Iteration {interation}: loss = {loss}")

    return centroids, assignments, losses


def kmeans_loss(X: np.ndarray, centroids: np.ndarray, assignments: np.ndarray) -> float:
    total_loss = 0.0
    n_samples = X.shape[0]

    for i in range(n_samples):
        cluster_id = assignments[i]
        distance = distance_squared(X[i], centroids[cluster_id])
        total_loss += distance

    return total_loss / n_samples
