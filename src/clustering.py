import numpy as np


def kmeans(X, k, tolerance=0.001, max_iters=300):
    centroids = np.zeros((k, X.shape[1]))
    assignments = np.zeros(X.shape[0])

    # initialize centroids in one random sample
    for i in range(k):
        centroids[i] = X[np.random.randint(X.shape[0])]

    improvement = np.inf

    while improvement > tolerance:
        # assign samples to nearest centroid
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            assignments[i] = np.argmin(distances)

        prev_centroids = centroids.copy()
        # update centroids
        for i in range(k):
            centroids[i] = np.mean(X[assignments == i], axis=0)

        improvement = np.linalg.norm(centroids - prev_centroids, axis=1).sum()

    return centroids, assignments
