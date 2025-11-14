import numpy as np
import pandas as pd

# constants
DATA_PATH = "data/caras.csv"
IMAGE_SIZE = (64, 64)
INPUT_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]
CLASS_LABEL_NAME = "person_id"
RANDOM_SEED = 42


def train_val_split(
    df: pd.DataFrame, val_size: float = 0.2, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)

    # we will split the samples of each class separately and then join them back
    train_dfs = []
    test_dfs = []

    # Split each class separately to maintain proportions
    for class_ in df[CLASS_LABEL_NAME].unique():
        class_data = df[df[CLASS_LABEL_NAME] == class_]
        n_samples_val = int(len(class_data) * val_size)

        # Split class data
        class_data_shuffled = class_data.sample(frac=1, random_state=seed)
        train_dfs.append(class_data_shuffled.iloc[n_samples_val:])
        test_dfs.append(class_data_shuffled.iloc[:n_samples_val])

    train_df = pd.concat(train_dfs).sample(frac=1, random_state=seed)
    test_df = pd.concat(test_dfs).sample(frac=1, random_state=seed)

    return train_df, test_df


def df_to_np(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.iloc[:, :-1].values  # pixels
    y = df.iloc[:, -1].values  # classes

    return X, y


def standardize(
    X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if (mean is None) or (std is None):
        mean = X.mean(axis=0)
        std = X.std(axis=0)

    X_std = (X - mean) / std
    return X_std, mean, std


def select_subset_classes(data, labels, n_classes):
    unique_labels = np.unique(labels)
    selected_labels = unique_labels[:n_classes]
    mask = np.isin(labels, selected_labels)
    return data[mask], labels[mask]
