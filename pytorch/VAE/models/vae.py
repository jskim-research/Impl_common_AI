import torch
import torch.nn as nn
from typing import Tuple


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),  # (B, 32, 14, 14)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),  # (B, 64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 16, 1, 1)
            nn.Flatten(),
            nn.Linear(64, 2 * latent_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor (B, 1, 28, 28)

        Returns:
            mean and log_var (B, 2 * latent_dim) for re-parameterization trick
        """
        out = self.encoder(x)
        # 모델 결과를 log_var 로 해석하고, 이후 exp 를 붙여줌으로써 양수 보장 + 너무 작은 값 또는 큰 값을 조정하여 기울기 소실 또는 폭발 방지
        return out[:, :self.latent_dim], out[:, self.latent_dim:]


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # ConvTranspose2d shape (I-1) * S + K - 2*P
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1),  # (B, 128, 2, 2)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (B, 64, 4, 4)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B, 32, 8, 8)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # (B, 16, 16, 16)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 1, 4, 2, 3),  # (B, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: latent tensor (B, latent_dim)

        Returns:
            output image tensor decoded from z (B, 1, 28, 28)
        """
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.decoder(z)
        out = (out + 1) / 2
        return out