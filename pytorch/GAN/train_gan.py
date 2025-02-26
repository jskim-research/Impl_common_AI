import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils
import yaml
import os
import argparse
import models
from torch.optim import Adam
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader


def step_d(generator: nn.Module, discriminator: nn.Module, x: torch.Tensor, loss_func: nn.Module, latent_dim: int) -> torch.Tensor:
    """
    One step of training or validation for discriminator

    Args:
        generator: generator model
        discriminator: discriminator model
        x: input image tensor (B, 1, 28, 28)
        y: label tensor (B, 1) which represents real or fake as 1 or 0
        loss_func: loss function
        latent_dim: latent dimension

    Returns:
         Loss tensor (scalar)
    """
    device = x.device
    batch_size = x.size(0)
    # sample minibatch of noise samples from noise prior p_g(z)
    z = torch.randn((batch_size, latent_dim)).to(device)
    real_x = x.reshape(batch_size, -1).to(device)

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # maximize V(D,G) = E_x~p_data(x)[log D(x)] + E_z~p_z(z)[log(1-D(G(z)))]
    # 진짜는 진짜로, 가짜는 가짜로 구분하도록 discriminator loss 계산
    real_loss = loss_func(discriminator(real_x), real_labels)
    fake_images = generator(z)
    fake_loss = loss_func(discriminator(fake_images), fake_labels)
    loss_d = real_loss + fake_loss

    return loss_d


def step_g(generator: nn.Module, discriminator: nn.Module, x: torch.Tensor, loss_func: nn.Module, latent_dim: int) -> torch.Tensor:
    """
    One step of training or validation for generator

    Args:
        generator: generator model
        discriminator: discriminator model
        x: input image tensor (B, 1, 28, 28)
        loss_func: loss function
        latent_dim: latent dimension

    Returns:
        Loss tensor (scalar)

    """
    batch_size = x.size(0)
    device = x.device
    # sample minibatch of noise samples from noise prior p_g(z)
    z = torch.randn((batch_size, latent_dim)).to(device)
    fake_images = generator(z)
    real_labels = torch.ones((batch_size, 1)).to(device)

    # maximize V(D,G) = E_z~p_z(z)[log D(G(z))]
    # 가짜 이미지를 진짜로 구분하도록 generator 학습
    # minimize E_z~p_z(z)[log (1 - D(G(z)))] 가 아닌 maximize E_z~p_z(z)[log D(G(z))] 를 사용함으로써 초반 학습 속도 향상
    loss_g = loss_func(discriminator(fake_images), real_labels)

    return loss_g


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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4,  # 데이터 로딩을 위한 서브 프로세스 개수
                              pin_memory=True,  # OS 가 데이터 페이지를 swap out (to disk) 하지 않도록 보장
                              persistent_workers=True)  # worker가 종료되지 않고 계속해서 데이터를 로드할 수 있도록 함 (프로세스 생성과 초기화에 소요되는 시간 감소)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              drop_last=True,
                              num_workers=4,
                              pin_memory=True,
                              persistent_workers=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             drop_last=True,
                             num_workers=4,
                             pin_memory=True,
                             persistent_workers=True)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    if model_name == "gan":
        g = models.gan.G(latent_dim)
        g.apply(weights_init)
        d = models.gan.D()
    elif model_name == "dcgan":
        g = models.dcgan.G(latent_dim)
        g.apply(weights_init)
        d = models.dcgan.D()
    else:
        raise ValueError(f"Model name ({model_name}) not found")

    criterion = nn.BCELoss()  # p(y) = p^y * (1-p)^(1-y) 를 최대화하는 loss (negative log 걸겠지?)

    g.to(device)
    d.to(device)

    optim_g = Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_d = Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        g.train()
        d.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            loss_d = step_d(g, d, x, criterion, latent_dim)
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()

            loss_g = step_g(g, d, x, criterion, latent_dim)
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

        if (epoch + 1) % save_freq == 0:
            # Model validation and save
            g.eval()
            d.eval()
            valid_loss_g = 0
            valid_loss_d = 0
            valid_len = 0
            with torch.no_grad():
                for x, y in valid_loader:
                    x = x.to(device)
                    y = y.to(device)

                    valid_loss_d += step_d(g, d, x, criterion, latent_dim).item()
                    valid_loss_g += step_g(g, d, x, criterion, latent_dim).item()
                    valid_len += x.size(0)

                valid_loss_d /= valid_len
                valid_loss_g /= valid_len
                print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss D: {valid_loss_d:.4f}, Validation Loss G: {valid_loss_g:.4f}")

                # Generate image for visualization
                z = torch.randn((4, latent_dim)).to(device)
                fake_images = g(z)

            torch.save(g.state_dict(), os.path.join(save_folder, "generator.pth"))
            torch.save(d.state_dict(), os.path.join(save_folder, "discriminator.pth"))
            torchvision.utils.save_image(fake_images.view(fake_images.size(0), 1, 28, 28), os.path.join(plot_folder, f'fake_image_epoch_{epoch + 1}.png'), nrow=4)
