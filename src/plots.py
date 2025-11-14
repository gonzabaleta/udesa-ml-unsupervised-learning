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
