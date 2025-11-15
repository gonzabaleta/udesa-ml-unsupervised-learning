import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, TypedDict
from src.metrics import mse

Tensor = torch.Tensor

ModelConfig = TypedDict(
    "ModelConfig",
    {
        "encoder_layers": List[int],
        "latent_size": int,
        "decoder_layers": List[int],
        "dropout": float,
        "lr": float,
        "name": str,
    },
)


class Autoencoder(nn.Module):
    def __init__(
        self,
        encoder_layers: List[int],
        latent_size: int,
        decoder_layers: List[int],
        dropout: float = 0.0,
    ) -> None:
        super(Autoencoder, self).__init__()

        encoder_modules = []

        # create encoder layers
        for i in range(len(encoder_layers) - 1):
            encoder_modules.append(nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
            encoder_modules.append(nn.SiLU())

            # Add dropout in all layers except last
            if dropout > 0 and i < len(encoder_layers) - 2:
                encoder_modules.append(nn.Dropout(dropout))

        # encoder output (no activation)
        encoder_modules.append(nn.Linear(encoder_layers[-1], latent_size))
        self.encoder = nn.Sequential(*encoder_modules)

        # decoder
        decoder_modules = [nn.Linear(latent_size, decoder_layers[0]), nn.ReLU()]

        for i in range(len(decoder_layers) - 1):
            decoder_modules.append(nn.Linear(decoder_layers[i], decoder_layers[i + 1]))

            if i < len(decoder_layers) - 2:
                decoder_modules.append(nn.SiLU())
                if dropout > 0:
                    decoder_modules.append(nn.Dropout(dropout))

        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: Tensor | np.ndarray, numpy=False) -> Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        if numpy:
            return self.encoder(x).detach().numpy()

        return self.encoder(x)

    def decode(self, z: Tensor | np.ndarray, numpy=False) -> Tensor:
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32)

        if numpy:
            return self.decoder(z).detach().numpy()

        return self.decoder(z)

    def reconstruct(self, z: Tensor | np.ndarray, mu, sigma, numpy=False):
        return self.decode(z, numpy) * sigma + mu


def dataset_to_tensor(X_train, X_val):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    return X_train_tensor, X_val_tensor


def prepare_autoencoder_data(
    X_tarin: np.ndarray, X_val: np.ndarray, batch_size: int = None
) -> Tuple[DataLoader, DataLoader]:
    if batch_size is None:
        batch_size = X_tarin.shape[0]

    X_train_tensor, X_val_tensor = dataset_to_tensor(X_tarin, X_val)

    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0

    for (batch,) in dataloader:
        # forward
        X_hat = model(batch)
        loss = criterion(X_hat, batch)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module
) -> float:
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for (batch,) in dataloader:
            X_hat = model(batch)
            loss = criterion(X_hat, batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    min_delta: float = 0.001,
) -> Tuple[List[float], List[float]]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


def compare_models(
    model_configs: List[ModelConfig], X_train, X_val, epochs, input_size
):
    results = {}
    best_val_mse = float("inf")
    best_model = None

    train_loader, val_loader = prepare_autoencoder_data(X_train, X_val)
    X_train_tensor, X_val_tensor = dataset_to_tensor(X_train, X_val)

    for config in model_configs:
        print(f"Training model: {config['name']}")

        model = Autoencoder(
            encoder_layers=[input_size] + config["encoder_layers"],
            latent_size=config["latent_size"],
            decoder_layers=config["decoder_layers"] + [input_size],
            dropout=config["dropout"],
        )

        train_losses, val_losses = train_autoencoder(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=config["lr"],
            patience=30,
            min_delta=0.0001,
        )

        model.eval()
        with torch.no_grad():
            X_train_hat = model.decode(model.encode(X_train_tensor))
            X_val_hat = model.decode(model.encode(X_val_tensor))

        train_mse = mse(X_train_tensor.numpy(), X_train_hat.numpy())
        val_mse = mse(X_val_tensor.numpy(), X_val_hat.numpy())

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = config["name"]

        results[config["name"]] = {
            "model": model,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_val_loss": val_losses[-1],
            "train_mse": train_mse,
            "val_mse": val_mse,
            "config": config,
        }

    print(f"Best model: {best_model}")
    print(f"Best val mse: {best_val_mse}")
    return results, best_model
