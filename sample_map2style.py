import argparse
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision import transforms, utils
from PIL import Image
from distributed import reduce_loss_dict
from tqdm import tqdm
import math
import lpips

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from model import Generator, Discriminator, MappingNetwork, Encoder
# from finegan_config import finegan_config


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise


def child_to_parent(c_code, c_dim, p_dim):
    ratio = c_dim / p_dim
    cid = torch.argmax(c_code,  dim=1)
    pid = (cid / ratio).long()
    p_code = torch.zeros([c_code.size(0), p_dim], device=c_code.device)
    for i in range(c_code.size(0)):
        p_code[i][pid[i]] = 1
    return p_code

def loss_geocross(latent, n_latent):
    if len(latent.size()) == 2:
        return torch.zeros(1, device=latent.device)
    else:
        X = latent.view(-1, 1, n_latent, 512)
        Y = latent.view(-1, n_latent, 1, 512)
        A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
        B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
        D = 2*torch.atan2(A, B)
        D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
        return D

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def manual_sample_codes(code, code_id):
    _code = torch.zeros(code.size(0), code.size(1), device=code.device)
    for i in range(code.size(0)):
        if i >= len(code_id):
            _id = code_id[-1]
        else:
            _id = code_id[i]
        if _id >= code.size(1):
            _id = 0
        _code[i, _id] = 1

    return _code


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="sample fine img")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument(
        "paths", metavar="PATHS", nargs="+", help="path to image files or directory of files to be projected")
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus")

    args = parser.parse_args()

    resize = 128

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if os.path.isdir(args.paths[0]):
        img_files = [os.path.join(args.paths[0], f) for f in os.listdir(
            args.paths[0]) if '.png' in f]
    else:
        img_files = args.paths

    imgs = []
    for imgfile in sorted(img_files):
        img = transform(Image.open(imgfile).convert("RGB"))
        if img.size()[1:3] == (128, 128):
            imgs.append(img)
            if len(imgs) >= args.batch:
                break

    imgs = torch.stack(imgs, 0).to(device)
    batch = imgs.size(0)

    assert args.ckpt is not None
    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    train_args = ckpt['args']

    style_generator = Generator(
        size=train_args.size,
        style_dim=train_args.latent,
        n_mlp=train_args.n_mlp,
        channel_multiplier=train_args.channel_multiplier
    ).to(device)

    # style_discriminator = Discriminator(
    #     size=train_args.size
    # ).to(device)

    if train_args.mp_arch == 'vanilla':
        mpnet = MappingNetwork(
            num_ws=style_generator.n_latent,
            w_dim=train_args.latent
        ).to(device)
    # https://github.com/bryandlee/stylegan2-encoder-pytorch
    elif train_args.mp_arch == 'encoder':
        mpnet = Encoder(
            size=128,
            num_ws=style_generator.n_latent,
            w_dim=train_args.latent
        ).to(device)

    style_generator.load_state_dict(ckpt["style_g"])
    # style_discriminator.load_state_dict(ckpt["style_d"])
    mpnet.load_state_dict(ckpt["mp"])

    style_generator.eval()
    # style_discriminator.eval()
    mpnet.eval()

    with torch.no_grad():
        wp_code = mpnet(imgs)
        style_img, _ = style_generator([wp_code],
            input_is_latent=True,
            randomize_noise=False)

        # print(style_discriminator(style_img))

    utils.save_image(
        style_img,
        f"map2style/map.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )