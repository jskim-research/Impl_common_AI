import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
import os
import argparse
from torch.optim import Adam
from torchvision import transforms, datasets
from models.gan import G, D


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

    train_dataset = datasets.MNIST(root="data",
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())

    test_dataset = datasets.MNIST(root="data",
                                  train=False,
                                  transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=True)

    # pixel value range [0, 1]

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    g = G(latent_dim)
    g.apply(weights_init)
    d = D()

    criterion = nn.BCELoss()  # p(y) = p^y * (1-p)^(1-y) 를 최대화하는 loss (negative log 걸겠지?)

    g.to(device)
    d.to(device)

    # Optimizer 에 따른 영향도 당연히 크고
    optim_g = Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_d = Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for x, y in train_loader:
            x.to(device)
            y.to(device)

            # sample minibatch of noise samples from noise prior p_g(z)
            z = torch.randn((batch_size, latent_dim)).to(device)
            real_x = x.reshape(batch_size, -1).to(device)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # maximize V(D,G) = E_x~p_data(x)[log D(x)] + E_z~p_z(z)[log(1-D(G(z)))]
            # 진짜는 진짜로, 가짜는 가짜로 구분하도록 discriminator 학습
            real_loss = criterion(d(real_x), real_labels)
            fake_images = g(z)
            fake_loss = criterion(d(fake_images), fake_labels)
            loss_d = real_loss + fake_loss

            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()

            # maximize V(D,G) = E_z~p_z(z)[log D(G(z))]
            # 가짜 이미지를 진짜로 구분하도록 generator 학습
            fake_images = g(z)
            loss_g = criterion(d(fake_images), real_labels)
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

        if (epoch + 1) % save_freq == 0:
            torch.save(g.state_dict(), os.path.join(save_folder, "generator.pth"))
            torch.save(d.state_dict(), os.path.join(save_folder, "discriminator.pth"))
            plt.imshow(fake_images.view(fake_images.size(0), 1, 28, 28)[0].squeeze().cpu().detach().numpy(), cmap='gray')
            plt.savefig(os.path.join(plot_folder, f'fake_image_epoch_{epoch + 1}.png'))
            plt.close()

        print(loss_d, loss_g)


    # Evaluation
    # parzen window log likelihood estimation 구현 필요
