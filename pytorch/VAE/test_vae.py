import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import models
import argparse
import os
import yaml
from torchvision import transforms, datasets
from torchvision.utils import save_image


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

    test_dataset = datasets.MNIST(root="data",
                                  train=False,
                                  transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=True)
    q = models.vae.ConvEncoder(latent_dim)
    p = models.vae.ConvDecoder(latent_dim)

    q.to(device)
    p.to(device)

    q.load_state_dict(torch.load(os.path.join(save_folder, "encoder.pth"), weights_only=True))
    p.load_state_dict(torch.load(os.path.join(save_folder, "decoder.pth"), weights_only=True))

    q.eval()
    p.eval()

    test_batch_size = 12
    eps = torch.randn((test_batch_size, latent_dim))
    z = torch.zeros((test_batch_size, latent_dim)) + torch.ones((test_batch_size, latent_dim)) * eps
    z = z.to(device)
    gen_x = p(z)
    save_image(gen_x, os.path.join(plot_folder, f"randomly_generated_image.png"), nrow=4)

    x, y = next(iter(test_loader))
    x = x[:test_batch_size].to(device)
    mean, log_var = q(x)
    std = torch.exp(0.5 * log_var)
    eps = torch.randn((x.size(0), latent_dim)).to(device)
    z = mean + std * eps
    gen_x = p(z)
    save_image(torch.cat([x, gen_x], dim=0), os.path.join(plot_folder, f"reconstructed_image.png"), nrow=4)
