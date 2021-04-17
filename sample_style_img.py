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

os.environ["CUDA_VISIBLE_DEVICES"]="1"

from model import Generator, MappingNetwork, G_NET
from finegan_config import finegan_config

from dataset import MultiResolutionDataset

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="mpnet trainer")

    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=8,
        help="number of the samples generated during training",
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
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

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

    discriminator = Discriminator(
        args.size
    ).to(device)

    assert args.style_model is not None
    print("load style model:", args.style_model)

    style_dict = torch.load(args.style_model, map_location=lambda storage, loc: storage)

    try:
        ckpt_name = os.path.basename(args.style_model)
        args.start_iter = int(os.path.splitext(ckpt_name)[0])

    except ValueError:
        pass

    style_generator.load_state_dict(style_dict["g_ema"], strict=False)
    style_generator.eval()
    discriminator.load_state_dict(style_dict["d"], strict=False)
    discriminator.eval()

    with torch.no_grad():
        sample_z = torch.randn(args.batch, args.latent, device=device)
        sample_img, _ = style_generator([sample_z])
        print(discriminator(sample_img))

    utils.save_image(
        sample_img,
        f"style_sample.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )

