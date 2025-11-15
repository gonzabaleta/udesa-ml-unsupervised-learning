import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.utils import IMAGE_SIZE, CLASS_LABEL_NAME, RANDOM_SEED

PLOTS_PATH = "plots/"


def finalize_plot(filename):
    plt.tight_layout()
    plt.savefig(PLOTS_PATH + filename + ".png", dpi=300)
    plt.show()


def plot_images(
    df: pd.DataFrame,
    n: int = 15,
    ncols: int = 5,
    filename: str = "initial_faces",
    seed: int = RANDOM_SEED,
):
    """
    Display n random images from dataset
    """
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
    df: pd.DataFrame, n_classes=5, n_per_class=3, filename="initial_faces_by_class"
):
    classes = df[CLASS_LABEL_NAME].unique()[:n_classes]
    figsize = (n_per_class * 5, n_classes * 5)

    fig, axes = plt.subplots(n_classes, n_per_class, figsize=figsize)
    for i, class_ in enumerate(classes):
        class_samples = df[df[CLASS_LABEL_NAME] == class_].sample(n_per_class)

        for j in range(n_per_class):
            pixels = class_samples.iloc[j, :-1].values
            image = pixels.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
            axes[i, j].imshow(image, cmap="gray")
            axes[i, j].axis("off")
            axes[i, j].set_title(f"Clase: {class_samples.iloc[j, -1]}")

        # add title for first image in class
        axes[i, 0].set_ylabel(f"Clase {class_}")

    finalize_plot(filename)


def plot_class_distribution(
    df: pd.DataFrame, figsize=(10, 5), filename="class_distribution"
):
    plt.figure(figsize=figsize)
    df[CLASS_LABEL_NAME].value_counts().plot(kind="bar")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad")
    finalize_plot(filename)


def plot_explained_variance(
    explained_variance_ratios: np.ndarray,
    figsize=(10, 5),
    filename="explained_variance",
):
    plt.figure(figsize=figsize)
    plt.plot(
        range(1, len(explained_variance_ratios) + 1),
        np.cumsum(explained_variance_ratios),
    )
    plt.xlabel("Componente")
    plt.ylabel("Proporción de varianza explicada")
    plt.axhline(y=0.90, color="r", linestyle="--")
    finalize_plot(filename)


def plot_reconstruction_comparison(
    original_images: np.ndarray,
    reconstructed_images_pca: np.ndarray,
    reconstructed_images_ae: np.ndarray = None,
    n_images: int = 10,
    filename: str = "reconstruction_comparison",
    seed: int = RANDOM_SEED,
):
    # Select random images from the set
    np.random.seed(seed)
    indices = np.random.choice(len(original_images), n_images, replace=False)

    nrows = 3 if reconstructed_images_ae is not None else 2

    fig, axes = plt.subplots(nrows, n_images, figsize=(2 * n_images, 2 * nrows))

    if nrows == 2:
        axes = axes.reshape(2, n_images)

    for i, idx in enumerate(indices):
        original = original_images[idx].reshape(IMAGE_SIZE)
        axes[0, i].imshow(original, cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        reconstructed_pca = reconstructed_images_pca[idx].reshape(IMAGE_SIZE)
        axes[1, i].imshow(reconstructed_pca, cmap="gray")
        axes[1, i].set_title("Reconstrucción PCA")
        axes[1, i].axis("off")

        if reconstructed_images_ae is not None:
            reconstructed_ae = reconstructed_images_ae[idx].reshape(IMAGE_SIZE)
            axes[2, i].imshow(reconstructed_ae, cmap="gray")
            axes[2, i].set_title("Reconstrucción AE")
            axes[2, i].axis("off")

    finalize_plot(filename)


def plot_scatter_2d(
    data: np.ndarray,
    labels: np.ndarray = None,
    xlabel="Componente 1",
    ylabel="Componente 2",
    figsize=(10, 8),
    filename="2d_reduction",
):
    plt.figure(figsize=figsize)

    if labels is not None:
        unique_labels = np.unique(labels)
        print(f"Unique labels: {unique_labels}")
        markers = ["o", "s", "^", "v", "<", ">", "p", "*", "h", "H", "D", "d"]
        cmap = plt.cm.get_cmap("tab20")

        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = cmap(i / len(unique_labels))
            marker = markers[i % len(markers)]
            plt.scatter(
                data[mask, 0],
                data[mask, 1],
                color=color,
                # marker=marker,
                label=f"Clase {label}",
                alpha=0.7,
                s=50,
            )

            plt.legend()
    else:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.7, s=50)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    finalize_plot(filename)


def plot_silhouette_comparison(
    kmeans_results, gmm_results, figsize=(12, 8), filename="silhouette_comparison"
):
    """
        Grafica silhouette score vs K para K-means y GMM en el mismo
    gráfico

        Args:
            kmeans_results: Dict con keys=K y values=dict con
    'silhouette_score'
            gmm_results: Dict con keys=K y values=dict con
    'silhouette_score'
            figsize: Tuple con tamaño de la figura
            filename: String con nombre del archivo para guardar
    (opcional)
    """

    # Extraer datos de K-means
    k_values_kmeans = list(kmeans_results.keys())
    silhouette_kmeans = [kmeans_results[k]["silhouette_score"] for k in k_values_kmeans]

    # Extraer datos de GMM
    k_values_gmm = list(gmm_results.keys())
    silhouette_gmm = [gmm_results[k]["silhouette_score"] for k in k_values_gmm]

    # Encontrar mejores K
    best_k_kmeans = k_values_kmeans[np.argmax(silhouette_kmeans)]
    best_k_gmm = k_values_gmm[np.argmax(silhouette_gmm)]

    plt.figure(figsize=figsize)

    # Plot con labels que incluyen mejor K
    plt.plot(
        k_values_kmeans,
        silhouette_kmeans,
        "bo-",
        label=f"K-means",
        linewidth=2,
        markersize=8,
    )

    plt.plot(
        k_values_gmm,
        silhouette_gmm,
        "ro-",
        label=f"GMM",
        linewidth=2,
        markersize=8,
    )

    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Comparación Silhouette Score: K-means vs GMM")
    plt.grid(True, alpha=0.3)

    # Líneas verticales para mejores K
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
    kmeans_results, gmm_results, figsize=(12, 8), filename="8_marginal_gains"
):
    """
        Grafica la ganancia marginal para K-means y GMM

        Args:
            kmeans_results: Dict con keys=K y values=dict con 'losses'
            gmm_results: Dict con keys=K y values=dict con
    'log_likelihoods'
            figsize: Tuple con tamaño de la figura
            filename: String con nombre del archivo para guardar
    (opcional)
    """
    import matplotlib.pyplot as plt

    # Obtener K values ordenados
    k_values_all = sorted(list(kmeans_results.keys()))
    k_values_gains = k_values_all[1:]  # Empezar desde el segundo K

    # Calcular ganancias marginales
    kmeans_gains = []
    gmm_gains = []

    for i, k in enumerate(k_values_gains):
        prev_k = k_values_all[i]  # K anterior

        # K-means gain: cuánto BAJA el loss
        loss_prev = kmeans_results[prev_k]["losses"][-1]
        loss_curr = kmeans_results[k]["losses"][-1]
        kmeans_gains.append(loss_prev - loss_curr)

        # GMM gain: cuánto SUBE el log-likelihood
        loglik_prev = gmm_results[prev_k]["log_likelihoods"][-1]
        loglik_curr = gmm_results[k]["log_likelihoods"][-1]
        gmm_gains.append(loglik_curr - loglik_prev)

    # Crear subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot K-means gains
    ax1.plot(k_values_gains, kmeans_gains, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Número de Clusters (K)")
    ax1.set_ylabel("Ganancia Marginal (Reducción de Loss)")
    ax1.set_title("Ganancia Marginal: K-means")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Plot GMM gains
    ax2.plot(k_values_gains, gmm_gains, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Número de Clusters (K)")
    ax2.set_ylabel("Ganancia Marginal (Aumento de Log-Likelihood)")
    ax2.set_title("Ganancia Marginal: GMM")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    plt.tight_layout()
    finalize_plot(filename=filename)
