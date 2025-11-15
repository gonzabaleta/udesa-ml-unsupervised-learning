import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def silhouette_score(X: np.ndarray, assignments: np.ndarray) -> float:
    """Compute average silhouette score for clustering"""
    S_total = 0
    n_samples = X.shape[0]

    for i in range(n_samples):
        cluster = assignments[i]
        cluster_samples = X[assignments == cluster]
        n_cluster_samples = cluster_samples.shape[0]

        if n_cluster_samples == 1:
            continue

        # a_i: average distance to points in same cluster
        a_i = 0
        for j in range(n_cluster_samples):
            a_i += np.linalg.norm(cluster_samples[j] - X[i])

        a_i /= n_cluster_samples - 1

        # b_i: minimum average distance to other clusters
        unique_clusters = np.unique(assignments)
        b_i = np.inf

        for j in unique_clusters:
            if j == cluster:
                continue

            cluster_j_samples = X[assignments == j]
            n_cluster_j_samples = cluster_j_samples.shape[0]

            avg_distance_to_cluster_j = 0
            for k in range(n_cluster_j_samples):
                avg_distance_to_cluster_j += np.linalg.norm(cluster_j_samples[k] - X[i])

            avg_distance_to_cluster_j /= n_cluster_j_samples

            if avg_distance_to_cluster_j < b_i:
                b_i = avg_distance_to_cluster_j

        S_total += (b_i - a_i) / max(a_i, b_i)

    return S_total / n_samples
