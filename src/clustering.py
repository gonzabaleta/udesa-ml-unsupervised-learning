import numpy as np


def kmeans(X, k, tolerance=0.0001, max_iters=300):
    n_samples, n_features = X.shape

    # initialize each centroid in one random sample
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()

    losses = []
    assignments = np.ndarray(n_samples, dtype=int)

    for interation in range(max_iters):
        # E-step: assign each sample to its closest centroid

        for i, sample in enumerate(X):
            distances = [distance_squared(sample, centroid) for centroid in centroids]
            best_centroid = np.argmin(distances)
            assignments[i] = int(best_centroid)

        # M-step: update centroids minimizing cost function (move to senter of cluster)
        new_centroids = np.ndarray((k, n_features))
        for j in range(k):
            cluster_points = X[assignments == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                # empty cluster, use same centroid
                new_centroids[j] = centroids[j]

        centroids = new_centroids

        # check for convergence
        loss = kmeans_loss(X, centroids, assignments)
        losses.append(loss)
        if len(losses) > 1 and abs(loss - losses[-2]) < tolerance:
            print(f"Converged at iteration {interation}")
            break

        if interation % 10 == 0:
            print(f"Iteration {interation}: loss = {loss}")

    return centroids, assignments, losses


def kmeans_loss(X, centroids, assignments):
    total_loss = 0.0
    n_samples = X.shape[0]

    for i in range(n_samples):
        cluster_id = assignments[i]
        distance = distance_squared(X[i], centroids[cluster_id])
        total_loss += distance

    return total_loss / n_samples


def distance_squared(x1, x2):
    return np.sum((x1 - x2) ** 2)
