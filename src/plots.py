import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Optional, Tuple, Dict, Any, List
from src.utils import IMAGE_SIZE, CLASS_LABEL_NAME, RANDOM_SEED

PLOTS_PATH = "plots/"


def finalize_plot(filename: Optional[str]) -> None:
    plt.tight_layout()
    if filename:
        plt.savefig(PLOTS_PATH + filename + ".png", dpi=300)
    plt.show()


def plot_images(
    df: pd.DataFrame,
    n: int = 15,
    ncols: int = 5,
    filename: str = "initial_faces",
    seed: int = RANDOM_SEED,
) -> None:
    """Display n random images from dataset in a grid layout."""
    np.random.seed(seed)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

    images_to_plot = df.sample(n=n)

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            pixels = images_to_plot.iloc[i, :-1].values
            image = pixels.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
            ax.imshow(image, cmap="gray")
            ax.axis("off")
            ax.set_title(f"Clase: {images_to_plot.iloc[i, -1]}")
        ax.axis("off")

    finalize_plot(filename)


def plot_images_by_class(
    df: pd.DataFrame,
    n_classes: int = 5,
    n_per_class: int = 3,
    filename: str = "initial_faces_by_class",
) -> None:
    """Plot sample images grouped by class in column layout."""
    classes = df[CLASS_LABEL_NAME].unique()[:n_classes]
    figsize = (n_classes * 5, n_per_class * 5)

    fig, axes = plt.subplots(n_per_class, n_classes, figsize=figsize)

    for j, class_ in enumerate(classes):
        class_samples = df[df[CLASS_LABEL_NAME] == class_].sample(n_per_class)

        for i in range(n_per_class):
            pixels = class_samples.iloc[i, :-1].values
            image = pixels.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
            axes[i, j].imshow(image, cmap="gray")
            axes[i, j].axis("off")

            if i == 0:
                axes[i, j].set_title(f"Clase {class_}", fontsize=30)

    finalize_plot(filename)


def plot_class_distribution(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 5),
    filename: str = "class_distribution",
) -> None:
    """Plot distribution of samples across different classes."""
    plt.figure(figsize=figsize)
    df[CLASS_LABEL_NAME].value_counts().plot(kind="bar", edgecolor=None)
    plt.xlabel("Clase")
    plt.ylabel("Número de muestras")
    finalize_plot(filename)


def plot_explained_variance(
    explained_variance_ratios: np.ndarray,
    figsize: Tuple[int, int] = (10, 5),
    filename: str = "explained_variance",
) -> None:
    """Plot cumulative explained variance ratio vs number of components."""
    plt.figure(figsize=figsize)
    plt.plot(
        range(1, len(explained_variance_ratios) + 1),
        np.cumsum(explained_variance_ratios),
    )
    plt.xlabel("Componente")
    plt.ylabel("Varianza Explicada Acumulada (%)")
    plt.axhline(y=0.90, color="r", linestyle="--", label="90% de varianza explicada")
    plt.legend()
    finalize_plot(filename)


def plot_reconstruction_comparison(
    original_images: np.ndarray,
    reconstructed_images_pca: np.ndarray,
    reconstructed_images_ae: Optional[np.ndarray] = None,
    n_images: int = 10,
    filename: str = "reconstruction_comparison",
    seed: int = RANDOM_SEED,
) -> None:
    """Compare original images with PCA and autoencoder reconstructions."""
    np.random.seed(seed)
    indices = np.random.choice(len(original_images), n_images, replace=False)

    nrows = 3 if reconstructed_images_ae is not None else 2

    fig, axes = plt.subplots(nrows, n_images, figsize=(2 * n_images, 2 * nrows))

    if nrows == 2:
        axes = axes.reshape(2, n_images)

    for i, idx in enumerate(indices):
        original = original_images[idx].reshape(IMAGE_SIZE)
        axes[0, i].imshow(original, cmap="gray")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        for spine in axes[0, i].spines.values():
            spine.set_visible(False)
        if i == 0:
            axes[0, i].set_ylabel("Original", rotation=90, va="center", labelpad=15)

        # PCA
        reconstructed_pca = reconstructed_images_pca[idx].reshape(IMAGE_SIZE)
        axes[1, i].imshow(reconstructed_pca, cmap="gray")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        for spine in axes[1, i].spines.values():
            spine.set_visible(False)
        if i == 0:
            axes[1, i].set_ylabel("PCA", rotation=90, va="center", labelpad=15)

        # Autoencoder
        if reconstructed_images_ae is not None:
            reconstructed_ae = reconstructed_images_ae[idx].reshape(IMAGE_SIZE)
            axes[2, i].imshow(reconstructed_ae, cmap="gray")
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
            for spine in axes[2, i].spines.values():
                spine.set_visible(False)
            if i == 0:
                axes[2, i].set_ylabel(
                    "Autoencoder", rotation=90, va="center", labelpad=15
                )

    finalize_plot(filename)


def plot_clusteres_2d(
    data: np.ndarray,
    assignments: np.ndarray,
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    figsize: Tuple[int, int] = (10, 8),
    filename: str = "2d_reduction",
) -> None:
    """Plot 2D scatter of clustered data points."""
    plt.figure(figsize=figsize)

    unique_clusters = np.unique(assignments)
    colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster in enumerate(unique_clusters):
        mask = assignments == cluster
        plt.scatter(
            data[mask, 0],
            data[mask, 1],
            color=colors[i],
            label=f"Cluster {cluster}",
            alpha=0.7,
            s=50,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    # plt.legend()
    finalize_plot(filename)


def plot_silhouette_comparison(
    kmeans_results: Dict[int, Dict[str, Any]],
    gmm_results: Dict[int, Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 8),
    filename: str = "silhouette_comparison",
) -> None:
    """Plot silhouette score comparison between K-means and GMM algorithms."""
    k_values_kmeans = list(kmeans_results.keys())
    silhouette_kmeans = [kmeans_results[k]["silhouette_score"] for k in k_values_kmeans]

    k_values_gmm = list(gmm_results.keys())
    silhouette_gmm = [gmm_results[k]["silhouette_score"] for k in k_values_gmm]

    best_k_kmeans = k_values_kmeans[np.argmax(silhouette_kmeans)]
    best_k_gmm = k_values_gmm[np.argmax(silhouette_gmm)]

    plt.figure(figsize=figsize)

    plt.plot(
        k_values_kmeans,
        silhouette_kmeans,
        "bo-",
        label="K-means",
        linewidth=2,
        markersize=8,
    )

    plt.plot(
        k_values_gmm,
        silhouette_gmm,
        "ro-",
        label="GMM",
        linewidth=2,
        markersize=8,
    )

    plt.xlabel("Cantidad de Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True, alpha=0.3)

    plt.axvline(
        x=best_k_kmeans,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Mejor K K-means = {best_k_kmeans}",
    )
    plt.axvline(
        x=best_k_gmm,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mejor K GMM = {best_k_gmm}",
    )

    plt.legend()
    finalize_plot(filename=filename)


def plot_elbow_method(
    kmeans_results: Dict[int, Dict[str, Any]],
    gmm_results: Dict[int, Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 8),
    filename: str = "marginal_gains",
) -> None:
    """
    Plot marginal gains for K-means and GMM algorithms."""
    k_values_all = sorted(list(kmeans_results.keys()))
    k_values_gains = k_values_all[1:]

    kmeans_gains = []
    gmm_gains = []

    for i, k in enumerate(k_values_gains):
        prev_k = k_values_all[i]

        loss_prev = kmeans_results[prev_k]["losses"][-1]
        loss_curr = kmeans_results[k]["losses"][-1]
        kmeans_gains.append(loss_prev - loss_curr)

        loglik_prev = gmm_results[prev_k]["log_likelihoods"][-1]
        loglik_curr = gmm_results[k]["log_likelihoods"][-1]
        gmm_gains.append(loglik_curr - loglik_prev)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(k_values_gains, kmeans_gains, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Cantidad de Clusters (K)")
    ax1.set_ylabel("Ganancia Marginal (Reducción de Loss)")
    ax1.set_title("Ganancia Marginal: K-means")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    ax2.plot(k_values_gains, gmm_gains, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Cantidad de Clusters (K)")
    ax2.set_ylabel("Ganancia Marginal (Aumento de Log-Likelihood)")
    ax2.set_title("Ganancia Marginal: GMM")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()
    finalize_plot(filename=filename)


def get_40_colors() -> np.ndarray:
    """Generate 40 distinct colors by combining different colormaps."""
    colors1 = plt.cm.Set1(np.linspace(0, 1, 9))
    colors2 = plt.cm.Set2(np.linspace(0, 1, 8))
    colors3 = plt.cm.Set3(np.linspace(0, 1, 12))
    colors4 = plt.cm.tab20(np.linspace(0, 1, 20))

    all_colors = np.vstack([colors1, colors2, colors3, colors4[:11]])  # 9+8+12+11=40
    return all_colors


def plot_cluster_composition(
    assignments: np.ndarray,
    y_true: np.ndarray,
    figsize: Tuple[int, int] = (12, 8),
    filename: Optional[str] = None,
) -> None:
    """Plot stacked bar chart showing class composition of each cluster."""
    unique_clusters = np.unique(assignments)
    unique_classes = np.unique(y_true)

    data = []
    for cluster in unique_clusters:
        cluster_data = []
        for class_id in unique_classes:
            count = np.sum((assignments == cluster) & (y_true == class_id))
            cluster_data.append(count)
        data.append(cluster_data)

    data = np.array(data)

    plt.figure(figsize=figsize)

    colors = get_40_colors()

    bottom = np.zeros(len(unique_clusters))
    for i, class_id in enumerate(unique_classes):
        plt.bar(
            unique_clusters,
            data[:, i],
            bottom=bottom,
            label=f"Clase {class_id}",
            color=colors[i],
            alpha=0.8,
        )
        bottom += data[:, i]

    plt.xlabel("Cluster")
    plt.ylabel("Número de muestras")
    plt.xticks(unique_clusters)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

    finalize_plot(filename=filename)


def plot_cluster_entropy(
    assignments: np.ndarray,
    y_true: np.ndarray,
    figsize: Tuple[int, int] = (12, 6),
    filename: str = "cluster_entropy",
) -> Tuple[List[float], List[int]]:
    """Plot entropy of each cluster as a measure of homogeneity."""
    unique_clusters = np.unique(assignments)
    entropies = []
    cluster_sizes = []

    for cluster_id in unique_clusters:
        mask = assignments == cluster_id
        cluster_labels = y_true[mask]
        cluster_sizes.append(len(cluster_labels))

        unique, counts = np.unique(cluster_labels, return_counts=True)
        probabilities = counts / len(cluster_labels)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        entropies.append(entropy)

    plt.figure(figsize=figsize)

    colors = plt.cm.RdYlGn_r(np.array(entropies) / max(entropies))

    plt.bar(unique_clusters, entropies, color=colors, alpha=0.8)

    for i, (cluster_id, entropy, size) in enumerate(
        zip(unique_clusters, entropies, cluster_sizes)
    ):
        plt.text(
            cluster_id,
            entropy + 0.05,
            f"n={size}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xlabel("Cluster")
    plt.ylabel("Entropía")
    plt.grid(axis="y", alpha=0.3)

    avg_entropy = np.mean(entropies)
    plt.axhline(
        y=avg_entropy,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Entropía promedio: {avg_entropy:.2f}",
    )
    plt.legend()
    plt.xticks(unique_clusters)
    finalize_plot(filename=filename)

    return entropies, cluster_sizes


def plot_eigenvectors(
    W: np.ndarray,
    n_components: int = 5,
    figsize: Optional[Tuple[int, int]] = None,
    filename: Optional[str] = None,
) -> None:
    """Plot first n principal components (eigenvectors) as eigenfaces."""
    if figsize is None:
        cols = min(n_components, 5)
        rows = (n_components + cols - 1) // cols
        figsize = (cols * 3, rows * 3)

    rows = (n_components + 4) // 5
    cols = min(n_components, 5)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if n_components == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i in range(n_components):
        eigenvector = W[:, i]
        eigenface = eigenvector.reshape(IMAGE_SIZE)

        ax = axes[i] if n_components > 1 else axes[0]

        ax.imshow(eigenface, cmap="gray")
        ax.set_title(f"Autovector {i+1}")
        ax.axis("off")

    for i in range(n_components, len(axes)):
        axes[i].axis("off")

    finalize_plot(filename=filename)
