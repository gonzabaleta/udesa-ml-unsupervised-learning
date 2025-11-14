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
    reconstructed_images: np.ndarray,
    n_images: int = 10,
    filename: str = "reconstruction_comparison",
    seed: int = RANDOM_SEED,
):
    # Select random images from the set
    np.random.seed(seed)
    indices = np.random.choice(len(original_images), n_images, replace=False)

    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 6))

    for i, idx in enumerate(indices):
        original = original_images[idx].reshape(IMAGE_SIZE)
        axes[0, i].imshow(original, cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        reconstructed = reconstructed_images[idx].reshape(IMAGE_SIZE)
        axes[1, i].imshow(reconstructed, cmap="gray")
        axes[1, i].set_title("Reconstrucción")
        axes[1, i].axis("off")

    finalize_plot(filename)
