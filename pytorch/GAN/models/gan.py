"""GAN 구현

논문 제목: Generative Adversarial Nets
"""


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torchvision import transforms, datasets


class G(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            # nn.Tanh()
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent vector (batch_size, latent_dim)
        Returns:
            generated image (batch_size, 784)
        """
        out = self.generator(z)
        return out


class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()  # range (-1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input image (batch_size, 784)
        Returns:
            probability of real image (batch_size, 1) with range [0, 1]
        """
        out = self.discriminator(x)
        out = (out + 1) / 2
        return out