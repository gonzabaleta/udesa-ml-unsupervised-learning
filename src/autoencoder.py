import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder_layers, latent_size, decoder_layers):
        super(Autoencoder, self).__init__()

        encoder_modules = []

        # create encoder layers
        for i in range(len(encoder_layers) - 1):
            encoder_modules.append(nn.Linear(encoder_layers[i], encoder_layers[i + 1]))
            encoder_modules.append(nn.ReLU())

        # encoder output ( no activation
        encoder_modules.append(nn.Linear(encoder_layers[-1], latent_size))
        self.encoder = nn.Sequential(*encoder_modules)
