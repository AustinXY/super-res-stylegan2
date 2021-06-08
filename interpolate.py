import argparse
import math
import random
import os
import copy

from numpy.core.fromnumeric import resize
import dnnlib
from PIL import Image

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from torch_utils import image_transforms

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

from model import Generator, Encoder, _Encoder
from mixnmatch_model import G_NET

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

def child_to_parent(c_code, c_dim, p_dim):
    ratio = c_dim / p_dim
    cid = torch.argmax(c_code,  dim=1)
    pid = (cid / ratio).long()
    p_code = torch.zeros([c_code.size(0), p_dim], device=c_code.device)
    for i in range(c_code.size(0)):
        p_code[i][pid[i].item()] = 1
    return p_code

def sample_codes(batch, z_dim, b_dim, p_dim, c_dim, device):
    z = torch.randn(batch, z_dim, device=device)
    c = torch.zeros(batch, c_dim, device=device)
    cid = np.random.randint(c_dim, size=batch)
    for i in range(batch):
        c[i, cid[i]] = 1

    p = child_to_parent(c, c_dim, p_dim)
    b = c.clone()
    return z, b, p, c

def rand_sample_codes(prev_z=None, prev_b=None, prev_p=None, prev_c=None, rand_code=['b', 'p']):
    '''
    rand code default: keeping z and c
    '''

    if prev_z is not None:
        device = prev_z.device
        batch = prev_z.size(0)
    elif prev_b is not None:
        device = prev_b.device
        batch = prev_b.size(0)
    elif prev_p is not None:
        device = prev_p.device
        batch = prev_p.size(0)
    elif prev_c is not None:
        device = prev_c.device
        batch = prev_c.size(0)
    else:
        sys.exit(0)

    if 'z' in rand_code:
        z = torch.randn(batch, prev_z.size(1), device=device)
    else:
        z = prev_z

    if 'b' in rand_code:
        b = torch.zeros(batch, prev_b.size(1), device=device)
        bid = np.random.randint(prev_b.size(1), size=batch)
        for i in range(batch):
            b[i, bid[i]] = 1
    else:
        b = prev_b

    if 'p' in rand_code:
        p = torch.zeros(batch, prev_p.size(1), device=device)
        pid = np.random.randint(prev_p.size(1), size=batch)
        for i in range(batch):
            p[i, pid[i]] = 1
    else:
        p = prev_p

    if 'c' in rand_code:
        c = torch.zeros(batch, prev_c.size(1), device=device)
        cid = np.random.randint(prev_c.size(1), size=batch)
        for i in range(batch):
            c[i, cid[i]] = 1
    else:
        c = prev_c

    return z, b, p, c

def lerp_list(li1, li2, weight):
    li = []
    for i in range(len(li1)):
        li.append(torch.lerp(li1[i], li2[i], weight))
    return li

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="mpnet trainer")

    parser.add_argument(
        "--batch", type=int, default=1, help="batch sizes for each gpus"
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to stylegan",
    )

    parser.add_argument(
        "--n_interpo", type=int, default=8, help="number of interpolation"
    )

    parser.add_argument(
        "--end_pt0", type=str, default='fine_sample/fine1.png', help="end point 0"
    )
    parser.add_argument(
        "--end_pt1", type=str, default='fine_sample/fine3.png', help="end point 1"
    )

    args = parser.parse_args()

    assert args.ckpt is not None
    print("load style model:", args.ckpt)

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

    ckpt_args = ckpt['args']

    args.z_dim = finegan_config[ckpt_args.ds_name]['Z_DIM']
    args.b_dim = finegan_config[ckpt_args.ds_name]['FINE_GRAINED_CATEGORIES']
    args.p_dim = finegan_config[ckpt_args.ds_name]['SUPER_CATEGORIES']
    args.c_dim = finegan_config[ckpt_args.ds_name]['FINE_GRAINED_CATEGORIES']

    try:
        args.injidx = ckpt_args.injidx
    except:
        args.injidx = None

    g_ema = Generator(
        size=ckpt_args.size,
        style_dim=ckpt_args.latent,
        n_mlp=ckpt_args.n_mlp,
        channel_multiplier=ckpt_args.channel_multiplier
    ).to(device)

    try:
        mp_nws = ckpt_args.mp_nws
    except:
        mp_nws = g_ema.n_latent

    mpnet = Encoder(
        size=128,
        num_ws=mp_nws,
        w_dim=ckpt_args.latent
    ).to(device)
    mpnet.eval()

    fine_generator = G_NET(ds_name=ckpt_args.ds_name).to(device)
    fine_generator.eval()

    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )
    try:
        g_ema.load_state_dict(ckpt["g_ema"])
    except:
        g_ema.load_state_dict(ckpt["style_g"])

    fine_generator.load_state_dict(ckpt['fine'])

    try:
        mpnet.load_state_dict(ckpt["mp"])
    except:
        mpnet = _Encoder(
            size=128,
            num_ws=mp_nws,
            w_dim=ckpt_args.latent
        ).to(device)
        mpnet.eval()
        mpnet.load_state_dict(ckpt["mp"])

    # endpt0_img = transform(Image.open(args.end_pt0).convert("RGB")).unsqueeze(0).to(device)
    # endpt1_img = transform(Image.open(args.end_pt1).convert("RGB")).unsqueeze(0).to(device)
    z, b, p, c = sample_codes(1, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
    z, b, p, c = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c, rand_code=['b', 'p'])
    z1, b1, p1, c1 = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c, rand_code=['z', 'b', 'p', 'c'])
    z2, b2, p2, c2 = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c, rand_code=['z', 'b', 'p', 'c'])


    with torch.no_grad():
        endpt0_img = fine_generator(z, b, p, c)
        endpt1_img = fine_generator(z1, b1, p, c, z_fg=z)
        endpt0_w = mpnet(endpt0_img)
        endpt1_w = mpnet(endpt1_img)

        interval = 1 / (args.n_interpo + 1)
        result_img = None
        for i in range(args.n_interpo + 2):
            weight = i * interval
            latent = torch.lerp(endpt0_w, endpt1_w, weight)
            style_img, _ = g_ema(latent, inject_index=args.injidx, input_is_latent=True, randomize_noise=False)

            if result_img is None:
                result_img = style_img
            else:
                result_img = torch.cat([result_img, style_img], dim=0)

        endpt2_img = fine_generator(z, b, p1, c1, z_fg=z1)
        endpt3_img = fine_generator(z1, b1, p1, c1, z_fg=z1)
        endpt0_w = mpnet(endpt2_img)
        endpt1_w = mpnet(endpt3_img)
        for i in range(args.n_interpo + 2):
            weight = i * interval
            latent = torch.lerp(endpt0_w, endpt1_w, weight)
            style_img, _ = g_ema(latent, inject_index=args.injidx, input_is_latent=True, randomize_noise=False)
            result_img = torch.cat([result_img, style_img], dim=0)

    utils.save_image(
        result_img,
        f"style_project/interpolate.png",
        nrow=args.n_interpo+2,
        normalize=True,
        range=(-1, 1),
    )

    utils.save_image(
        torch.cat([endpt0_img, endpt1_img, endpt2_img, endpt3_img],dim=0),
        f"style_project/endpoints.png",
        nrow=2,
        normalize=True,
        range=(-1, 1),
    )