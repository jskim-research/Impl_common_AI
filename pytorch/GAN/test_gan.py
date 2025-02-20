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
from typing import Tuple
from torch.optim import Adam
from torchvision import transforms, datasets
from models.gan import G, D
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import KFold


def gaussian_kernel(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Gaussian kernel function

    Args:
        x: input tensor (1, 28, 28)
        mu: gaussian mean tensor (N, 1, 28, 28)
        sigma: standard deviation (1)

    Returns:
        Gaussian kernel tensor
    """
    D = x.size(0) * x.size(1) * x.size(2)
    a = (x - mu) / sigma

    t = -0.5 * (a ** 2).sum(dim=(1, 2, 3))
    E = torch.exp(t)
    Z = (1 / (torch.tensor(2 * torch.pi)).sqrt())  # normalization constant
    return Z * E


def calculate_mean_std_ll(data: torch.Tensor, target: torch.Tensor, bandwidth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean and standard deviation of log likelihood estimation

    Args:
        data: data tensor (N1, 1, 28, 28)
        target: evaluation target tensor (N2, 1, 28, 28)
        bandwidth: bandwidth tensor (1)
    Returns:
        mean and standard deviation of log likelihood estimation
    """
    data = data.view(1, data.size(0), -1)  # (1, N1, 784)
    target = target.view(target.size(0), 1, -1)  # (N2, 1, 784)

    device = data.device
    data_len = torch.tensor(data.size(0))
    dimension = data.size(2)

    split_size = 10
    log_likelihoods = []

    for split_start in range(0, target.size(0), split_size):
        split_target = target[split_start:split_start + split_size]
        # log likelihood 계산
        a = (split_target - data) / bandwidth  # (split_size, N1, 784)
        t = -0.5 * (a ** 2).sum(dim=2)  # (split_size, N1)

        # log mean exp (pixel dimension 에 대해선 mean 값 취함)
        max_t = t.max(1, keepdims=True).values  # (split_size, 1)
        E = max_t.squeeze() + torch.log(torch.mean(torch.exp(t - max_t), dim=1))  # (split_size)
        Z = dimension * torch.log((bandwidth * torch.tensor(2 * torch.pi)).sqrt())  # (split_size)
        log_likelihoods += E - Z

    log_likelihoods = torch.stack(log_likelihoods, dim=0)
    mean_ll = torch.mean(log_likelihoods)
    std_ll = torch.std(log_likelihoods)

    return mean_ll, std_ll


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

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(plot_folder, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    g = G(latent_dim)
    d = D()
    g.to(device)
    d.to(device)

    generator_path = os.path.join(save_folder, "generator.pth")
    discriminator_path = os.path.join(save_folder, "discriminator.pth")

    # g.load_state_dict(torch.load(generator_path, weights_only=True))
    # d.load_state_dict(torch.load(discriminator_path, weights_only=True))

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
    # parzen window log likelihood estimation 구현 필요

    # dataset -> tensor 변환
    all_train_data = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    all_train_len = torch.tensor(all_train_data.size(0)).to(device)
    all_test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)

    dimension = all_train_data.size(1) * all_train_data.size(2) * all_train_data.size(3)

    # Cross validation 기반 best bandwidth 탐색
    # bandwidth 후보군
    bandwidths = torch.logspace(-1, 0, 10).to(device)
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)
    best_mll = -float("inf")
    best_bandwidth = None

    with torch.no_grad():
        for bandwidth in bandwidths:
            mll = 0

            for fold, (train_idx, valid_idx) in enumerate(kfold.split(all_train_data)):
                train_fold = all_train_data[train_idx]
                valid_fold = all_train_data[valid_idx]

                fold_mll, _ = calculate_mean_std_ll(train_fold, valid_fold, bandwidth)
                mll += fold_mll
                print(f"bandwidth: {bandwidth:.4f}, fold-{fold}-mll: {fold_mll:.4f}")

            # mean of log likelihood estimates of all folds
            mll /= k_folds

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
        for _ in range(eval_epochs):
            z = torch.randn((eval_batch_size, latent_dim)).to(device)
            fake_images = g(z).view(-1, 1, 28, 28)
            mean_ll, std_ll = calculate_mean_std_ll(all_test_data, fake_images, best_bandwidth)
            mean_lls.append(mean_ll)
            std_lls.append(std_ll)

    mean_lls = torch.stack(mean_lls)
    std_lls = torch.stack(std_lls)

    print(f"Mean ll: {torch.mean(mean_lls):.4f}, Std ll: {torch.mean(std_lls):.4f}")
