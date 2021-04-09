import argparse
import math
import random
import os
import copy

from numpy.core.fromnumeric import resize
import dnnlib

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from torch_utils import image_transforms

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from model import Generator
from finegan_config import finegan_config


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def lerp_list(li1, li2, weight):
    li = []
    for i in range(len(li1)):
        li.append(torch.lerp(li1[i], li2[i], weight))
    return li

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="mpnet trainer")

    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--style_model",
        type=str,
        default=None,
        help="path to stylegan",
    )
    parser.add_argument(
        "--fine_model",
        type=str,
        default=None,
        help="path to finegan",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--latent_ckpt", type=str, default=None, help="latent code stored"
    )
    parser.add_argument(
        "--n_interpo", type=int, default=1, help="number of interpolation"
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    args.z_dim = finegan_config['Z_DIM']
    args.b_dim = finegan_config['FINE_GRAINED_CATEGORIES']
    args.p_dim = finegan_config['SUPER_CATEGORIES']
    args.c_dim = finegan_config['FINE_GRAINED_CATEGORIES']

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    style_generator = Generator(
        size=args.size,
        style_dim=args.latent,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).to(device)

    assert args.style_model is not None
    print("load style model:", args.style_model)

    style_dict = torch.load(args.style_model, map_location=lambda storage, loc: storage)

    try:
        ckpt_name = os.path.basename(args.style_model)
        args.start_iter = int(os.path.splitext(ckpt_name)[0])

    except ValueError:
        pass

    style_generator.load_state_dict(style_dict["g_ema"])

    if args.latent_ckpt is None:
        noise1 = mixing_noise(args.batch, args.latent, args.mixing, device)
        noise2 = mixing_noise(args.batch, args.latent, args.mixing, device)
        _, latent1 = style_generator(noise1, return_latents=True)
        _, latent2 = style_generator(noise2, return_latents=True)
    else:
        projection = torch.load(args.latent_ckpt)
        latent1 = projection['fine_sample/fine3.png']['latent'].unsqueeze(0)
        latent2 = projection['fine_sample/fine7.png']['latent'].unsqueeze(0)

    interval = 1 / (args.n_interpo + 1)

    result_img = None
    for i in range(args.n_interpo + 2):
        weight = i * interval
        latent = torch.lerp(latent1, latent2, weight)
        style_img, _ = style_generator([latent], input_is_latent=True)

        if result_img is None:
            result_img = style_img
        else:
            result_img = torch.cat([result_img, style_img])

    utils.save_image(
        result_img,
        f"style_project/interpolate.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )