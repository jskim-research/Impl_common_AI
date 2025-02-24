import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision import transforms, datasets


class G(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0, bias=False),  # (512, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2, bias=False),  # (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 2, 2, bias=False),  # (1, 28, 28)
            nn.Tanh()
        )

    def forward(self, z):
        """
        Args:
            z: latent vector (batch_size, latent_dim)
        Returns:
            generated image (batch_size, 1, 28, 28)
        """
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.generator(z)
        out = (out + 1) / 2
        return out

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),  # (64, 14, 14)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # (512, 1, 1)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 1, 1, 0, bias=False),  # (1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: input image (batch_size, 784)
        Returns:
            probability of real image (batch_size, 1)
        """
        x = x.view(-1, 1, 28, 28)
        out = self.discriminator(x).view(-1, 1)
        return out
