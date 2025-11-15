import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
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
    figsize = (n_classes * 5, n_per_class * 5)

    fig, axes = plt.subplots(n_per_class, n_classes, figsize=figsize)

    for j, class_ in enumerate(classes):
        class_samples = df[df[CLASS_LABEL_NAME] == class_].sample(n_per_class)

        for i in range(n_per_class):
            pixels = class_samples.iloc[i, :-1].values
            image = pixels.reshape(IMAGE_SIZE[0], IMAGE_SIZE[1])
            axes[i, j].imshow(image, cmap="gray")
            axes[i, j].axis("off")

            # Solo title en la primera fila
            if i == 0:
                axes[i, j].set_title(f"Clase {class_}")

    finalize_plot(filename)


def plot_class_distribution(
    df: pd.DataFrame, figsize=(10, 5), filename="class_distribution"
):
    plt.figure(figsize=figsize)
    df[CLASS_LABEL_NAME].value_counts().plot(kind="bar", edgecolor=None)
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
    plt.ylabel("Proporción de varianza explicada acumulada")
    plt.axhline(y=0.90, color="r", linestyle="--", label="90% de varianza explicada")
    plt.legend()
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
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        # Quitar bordes
        for spine in axes[0, i].spines.values():
            spine.set_visible(False)
        # Ylabel solo en la primera columna con más espacio
        if i == 0:
            axes[0, i].set_ylabel("Original", rotation=90, va="center", labelpad=15)

        reconstructed_pca = reconstructed_images_pca[idx].reshape(IMAGE_SIZE)
        axes[1, i].imshow(reconstructed_pca, cmap="gray")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        # Quitar bordes
        for spine in axes[1, i].spines.values():
            spine.set_visible(False)
        # Ylabel solo en la primera columna con más espacio
        if i == 0:
            axes[1, i].set_ylabel(
                "Reconstrucción PCA", rotation=90, va="center", labelpad=15
            )

        if reconstructed_images_ae is not None:
            reconstructed_ae = reconstructed_images_ae[idx].reshape(IMAGE_SIZE)
            axes[2, i].imshow(reconstructed_ae, cmap="gray")
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
            # Quitar bordes
            for spine in axes[2, i].spines.values():
                spine.set_visible(False)
            # Ylabel solo en la primera columna con más espacio
            if i == 0:
                axes[2, i].set_ylabel(
                    "Reconstrucción AE", rotation=90, va="center", labelpad=15
                )

    finalize_plot(filename)


def plot_clusteres_2d(
    data,
    assignments,
    xlabel="Componente 1",
    ylabel="Componente 2",
    figsize=(10, 8),
    filename="2d_reduction",
):
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
    plt.legend()
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


def get_40_colors():
    """
    Genera 40 colores combinando diferentes colormaps
    """
    colors1 = plt.cm.Set1(np.linspace(0, 1, 9))
    colors2 = plt.cm.Set2(np.linspace(0, 1, 8))
    colors3 = plt.cm.Set3(np.linspace(0, 1, 12))
    colors4 = plt.cm.tab20(np.linspace(0, 1, 20))

    # Combinar y tomar las primeras 40
    all_colors = np.vstack([colors1, colors2, colors3, colors4[:11]])  # 9+8+12+11=40
    return all_colors


def plot_cluster_composition(assignments, y_true, figsize=(12, 8), filename=None):
    """
    Heatmap elegante de composición de clusters
    """
    unique_clusters = np.unique(assignments)
    unique_classes = np.unique(y_true)

    # Preparar datos
    data = []
    for cluster in unique_clusters:
        cluster_data = []
        total = np.sum(assignments == cluster)
        for class_id in unique_classes:
            count = np.sum((assignments == cluster) & (y_true == class_id))
            cluster_data.append(count)
        data.append(cluster_data)

    data = np.array(data)

    plt.figure(figsize=figsize)

    # Colores distintivos
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
    plt.ylabel("Número de Muestras")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)

    finalize_plot(filename=filename)


def plot_cluster_entropy(
    assignments, y_true, figsize=(12, 6), filename="cluster_entropy"
):
    """
    Grafica la entropía de cada cluster como medida de homogeneidad
    """
    unique_clusters = np.unique(assignments)
    entropies = []
    cluster_sizes = []

    for cluster_id in unique_clusters:
        mask = assignments == cluster_id
        cluster_labels = y_true[mask]
        cluster_sizes.append(len(cluster_labels))

        # Calcular entropía
        unique, counts = np.unique(cluster_labels, return_counts=True)
        probabilities = counts / len(cluster_labels)
        entropy = -np.sum(
            probabilities * np.log2(probabilities + 1e-10)
        )  # +epsilon para evitar log(0)
        entropies.append(entropy)

    plt.figure(figsize=figsize)

    # Crear bar chart con colores basados en entropía (más rojo = más entropía)
    colors = plt.cm.RdYlGn_r(np.array(entropies) / max(entropies))

    plt.bar(unique_clusters, entropies, color=colors, alpha=0.8)

    # Agregar texto con tamaño del cluster
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

    # Línea de referencia para entropía promedio
    avg_entropy = np.mean(entropies)
    plt.axhline(
        y=avg_entropy,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Entropía promedio: {avg_entropy:.2f}",
    )
    plt.legend()

    finalize_plot(filename=filename)

    return entropies, cluster_sizes


def plot_eigenvectors(W, n_components=5, figsize=None, filename=None):
    if figsize is None:
        # Calcular tamaño automático basado en número de componentes
        cols = min(n_components, 5)
        rows = (n_components + cols - 1) // cols
        figsize = (cols * 3, rows * 3)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Si solo hay una fila o columna, convertir a array 2D
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

        im = ax.imshow(eigenface, cmap="gray")
        ax.set_title(f"Autovector {i+1}")
        ax.axis("off")

    # Ocultar subplots extras
    for i in range(n_components, len(axes)):
        axes[i].axis("off")

    finalize_plot(filename=filename)
