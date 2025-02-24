"""
구현 사항
1. Parzen window log likelihood estimation
- 생성된 데이터가 실제 데이터 상에서 얼마나 likelihood 가 높은지 측정하는 방식
- Cross validation 기반 kernel bandwidth 탐색 구현
- 계산 안정성을 위해 log sum exp trick 사용 (log sum exp(x) = max + log sum exp(x - max))

"""


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils
import yaml
import os
import argparse
import torch.profiler
import models
from typing import Tuple
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import KFold


def calculate_lls(data: torch.Tensor, target: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    """
    Calculate log likelihoods of target data

    Args:
        data: data tensor (N1, 1, 28, 28)
        target: evaluation target tensor (N2, 1, 28, 28)
        bandwidth: bandwidth tensor (1)
    Returns:
        log likelihoods (N2)
    """
    data = data.view(1, data.size(0), -1)  # (1, N1, 784)
    target = target.view(target.size(0), 1, -1)  # (N2, 1, 784)
    dimension = data.size(2)

    split_size = 10
    log_likelihoods = []

    for split_start in range(0, target.size(0), split_size):
        split_target = target[split_start:split_start + split_size]
        # log likelihood 계산
        a = (split_target - data) / bandwidth  # (split_size, N1, 784)
        t = -0.5 * (a ** 2).sum(dim=2)  # (split_size, N1)

        max_t = t.max(1, keepdims=True).values  # (split_size, 1)
        # log sum exp trick
        E = max_t.squeeze() + torch.log(torch.mean(torch.exp(t - max_t), dim=1))  # (split_size)
        Z = dimension * torch.log((bandwidth * torch.tensor(2 * torch.pi)).sqrt())  # (split_size)
        log_likelihoods.extend(E - Z)

    if len(log_likelihoods) != target.size(0):
        raise ValueError("Log likelihoods size mismatch")

    log_likelihoods = torch.stack(log_likelihoods, dim=0)
    return log_likelihoods


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open("configs/base_train_gan_config.yaml") as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError("Config file not found")

    print(config)

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if model_name == "gan":
        g = models.gan.G(latent_dim)
        d = models.gan.D()
    elif model_name == "dcgan":
        g = models.dcgan.G(latent_dim)
        d = models.dcgan.D()
    else:
        raise ValueError(f"Model name ({model_name}) not found")

    g.to(device)
    d.to(device)

    generator_path = os.path.join(save_folder, "generator.pth")
    discriminator_path = os.path.join(save_folder, "discriminator.pth")

    g.load_state_dict(torch.load(generator_path, weights_only=True))
    d.load_state_dict(torch.load(discriminator_path, weights_only=True))

    dataset = datasets.MNIST(root="data",
                             train=True,
                             download=True,
                             transform=transforms.ToTensor())

    # train, valid data 80%, 20% 로 나누기
    train_len = int(len(dataset) * 0.8)
    valid_len = len(dataset) - train_len

    train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len])

    test_dataset = datasets.MNIST(root="data",
                                  train=False,
                                  transform=transforms.ToTensor())

    # Evaluation
    # parzen window log likelihood estimation

    # dataset -> tensor 변환
    all_train_data = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    all_train_len = torch.tensor(all_train_data.size(0)).to(device)
    all_test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)

    dimension = all_train_data.size(1) * all_train_data.size(2) * all_train_data.size(3)

    # Cross validation 기반 best bandwidth 탐색
    # bandwidth 후보군
    bandwidths = torch.logspace(-1, 0, 10).to(device)

    limit_size = 1000
    best_mll = -float("inf")
    best_bandwidth = None

    with torch.no_grad():
        z = torch.randn((limit_size, latent_dim)).to(device)
        fake_images = g(z).view(-1, 1, 28, 28)

        # Find best bandwidth
        for bandwidth in bandwidths:
            valid_data = all_train_data[:limit_size]

            lls = calculate_lls(valid_data, fake_images, bandwidth)

            # mean of log likelihood estimates
            mll = lls.mean()

            if mll > best_mll:
                # best bandwidth 갱신
                best_mll = mll
                best_bandwidth = bandwidth

            print(f"bandwidth: {bandwidth:.4f}, mll: {mll:.4f}")

    eval_batch_size = 64
    eval_total_num = 10000  # test data 개수와 맞춤
    eval_epochs = eval_total_num // eval_batch_size

    mean_lls = []
    std_lls = []

    with torch.no_grad():
        all_lls = []
        for _ in range(eval_epochs):
            z = torch.randn((eval_batch_size, latent_dim)).to(device)
            fake_images = g(z).view(-1, 1, 28, 28)
            lls = calculate_lls(all_test_data, fake_images, best_bandwidth)
            all_lls.extend(lls)

    all_lls = torch.stack(all_lls)
    print(f"Mean ll: {all_lls.mean():.4f}, Std ll: {all_lls.std():.4f}")
