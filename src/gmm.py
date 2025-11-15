import numpy as np
from src.kmeans import kmeans
from src.utils import RANDOM_SEED


def gmm(X, k, tolerance=0.0001, max_iters=300, seed=RANDOM_SEED):
    n_samples, n_features = X.shape

    if seed:
        np.random.seed(seed)

    # initialize parameters
    # MEANS: initialize as centroids returned from a run of kmeans
    means, _, _ = kmeans(X, k, tolerance=0.01, max_iters=max_iters)

    # COVARIANCES: initialize as global covariance
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
            N_k = 0
            mu_k = np.zeros(n_features)
            for i in range(n_samples):
                N_k += responsibilities[i, j]
                mu_k += responsibilities[i, j] * X[i]

            new_means[j] = mu_k / N_k
            new_priors[j] = N_k / n_samples

            diff = X - new_means[j]
            weighted_diff = responsibilities[:, j][:, np.newaxis] * diff
            new_covariances[j] = (weighted_diff.T @ diff) / N_k

        means = new_means
        covariances = new_covariances
        priors = new_priors

        # verify convergence
        log_likelihood = compute_log_likelihood(X, means, covariances, priors)
        log_likelihoods.append(log_likelihood)

        if (
            len(log_likelihoods) > 1
            and abs(log_likelihood - log_likelihoods[-2]) < tolerance
        ):
            print(f"Converged at iteration {interation}")
            break

        print(f"Iteration {interation}: log likelihood = {log_likelihood}")

    assignments = np.argmax(responsibilities, axis=1)

    return means, covariances, priors, assignments, log_likelihoods


def multivariate_gaussian_pdf(x, mean, cov):
    k = len(mean)

    cov_regularized = cov + 0.1 * np.eye(k)  # avoid singular matrices

    cov_det = np.linalg.det(cov_regularized)
    cov_inv = np.linalg.inv(cov_regularized)
    diff = x - mean

    exponent = -0.5 * (diff.T @ cov_inv @ diff)
    norm = np.sqrt((2 * np.pi) ** k * cov_det)

    return 1 / norm * np.exp(exponent)


def compute_log_likelihood(X, means, covariances, priors):
    n_samples = X.shape[0]
    total_log_likelihood = 0

    for i in range(n_samples):
        sample_likelihood = 0
        for j in range(len(means)):
            sample_likelihood += priors[j] * multivariate_gaussian_pdf(
                X[i], means[j], covariances[j]
            )
        total_log_likelihood += np.log(sample_likelihood + 1e-10)

    return total_log_likelihood
