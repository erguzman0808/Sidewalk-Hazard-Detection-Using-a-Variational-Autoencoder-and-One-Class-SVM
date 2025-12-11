import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for the progress bar

class CNN_VAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super(CNN_VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),  # 320 x 240 -> 160 x 120
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # 160 x 120 -> 80 x 60
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # 80 x 60 -> 40 x 30
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),  # 40 x 30 -> 20 x 15
            #nn.ReLU(),
            #nn.Conv2d(128, 256, 5, stride=2, padding=2)  # 20 x 15 -> 10 x 8
        )

        # Latent space
        self.fc_mu = nn.Linear(256 * 20 * 15, latent_dim)
        self.fc_logvar = nn.Linear(256 * 20 * 15, latent_dim)
        self.fc_latent = nn.Linear(latent_dim, 256 * 20 * 15)

        # Decoder
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 10 x 8 -> 20 x 15
            #nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 20 x 15 -> 40 x 30
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 40 x 30 -> 80 x 60
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 80 x 60 -> 160 x 120
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)  # 160 x 120 -> 320 x 240
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_ = self.encoder[0](x)  # Conv2d(3→32)
        x_ = self.encoder[1](x_)  # ReLU
        x_ = self.encoder[2](x_)  # Conv2d(32→64)
        x_ = self.encoder[3](x_)  # ReLU
        x_ = self.encoder[4](x_)  # Conv2d(64→128) ← this is what you want
        third_conv_output = x_   # Shape: (B, 128, 40, 30)

        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        z = self.fc_latent(z)
        z = z.view(z.size(0), 256, 20, 15)
        decoded = self.decoder(z)
        return decoded, mu, logvar, self.reparameterize(mu, logvar), encoded
