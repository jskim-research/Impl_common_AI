import torch
import torch.nn as nn
import models
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torch.utils.data.dataset import random_split
from typing import Tuple


def step(x: torch.Tensor, encoder: nn.Module, decoder: nn.Module, latent_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: input image tensor (B, 1, 28, 28)
        encoder: model that outputs mean and std (B, 2 * latent_dim) for re-parameterization trick
        decoder: model that outputs generated image (B, 1, 28, 28)

    Returns:
        Regularization loss (scalar), Reconstruction loss (scalar)
    """
    device = x.device
    mean, log_var = encoder(x)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn((x.size(0), latent_dim)).to(device)
    z = mean + std * eps
    gen_x = decoder(z)

    # Regularization loss
    # Minimiaze D_KL (q(z|x) || p(z)) => let q(z|x) be form of prior p(z)
    regularization_loss = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())

    # sum (log p(x|z)) -> batch sum 하면서 monte carlo sampling 한 효과 (sampling # = 1)
    # p(x|z) = N(x | gen_x, sigma^2) => gen_x 가 x 를 평균으로 한 gaussian distribution 내에 나오도록 함
    sigma = 0.5
    D = x.size(1) * x.size(2) * x.size(3)
    # batch 에 대한 summation 고려하여 x.size(0) 곱해줌
    # normalization factor 의 경우 상수값이라 실제 학습에 영향이 없어 loss 에 쓰진 않음
    log_norm_factor = x.size(0) * D * torch.log(sigma * torch.tensor(2 * torch.pi).sqrt())
    reconstruction_loss = 0.5 * (((x - gen_x) / sigma) ** 2).sum()
    return regularization_loss, reconstruction_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError("Config file not found")

    batch_size = config["training"]["batch_size"]
    latent_dim = config["training"]["latent_dim"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    save_freq = config["training"]["save_freq"]
    save_folder = config["paths"]["save_folder"]
    plot_folder = config["paths"]["plot_folder"]
    model_name = config["model"]["name"]

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    dataset = datasets.MNIST(root="data",
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())

    train_len = int(len(dataset) * 0.8)
    valid_len = len(dataset) - train_len

    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

    test_dataset = datasets.MNIST(root="data",
                                  train=False,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=4,
                                               pin_memory=True,
                                               persistent_workers=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=4,
                                               pin_memory=True,
                                               persistent_workers=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    q = models.vae.ConvEncoder(latent_dim)
    p = models.vae.ConvDecoder(latent_dim)

    q.to(device)
    p.to(device)

    optimizer = torch.optim.Adam(list(q.parameters()) + list(p.parameters()), lr=lr)

    for epoch in range(epochs):
        q.train()
        p.train()
        for x, y in train_loader:
            x = x.to(device)
            reg_loss, rec_loss = step(x, q, p, latent_dim)
            loss = reg_loss + rec_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % save_freq == 0:
            q.eval()
            p.eval()
            reg_total_loss = 0
            rec_total_loss = 0
            with torch.no_grad():
                for x, y in valid_loader:
                    reg_loss, rec_loss = step(x.to(device), q, p, latent_dim)
                    reg_total_loss += reg_loss
                    rec_total_loss += rec_loss

                z = torch.randn((8, latent_dim)).to(device)
                generated_image = p(z)
            reg_total_loss /= valid_len
            rec_total_loss /= valid_len
            torch.save(q.state_dict(), os.path.join(save_folder, "encoder.pth"))
            torch.save(p.state_dict(), os.path.join(save_folder, "decoder.pth"))
            save_image(generated_image, os.path.join(plot_folder, f"image_epoch_{epoch + 1}.png"), nrow=4)
            print(f"Epoch [{epoch + 1}/{epochs}], Validation Regularization Loss: {reg_total_loss:.4f}, Validation Reconstruction Loss: {rec_total_loss:.4f}")


