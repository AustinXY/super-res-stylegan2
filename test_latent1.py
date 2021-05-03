import argparse
import os
import random

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
from finegan_config import finegan_config

# import seaborn as sns
# import matplotlib
import sys
# matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

from model import Generator, Discriminator, MappingNetwork, Encoder, G_NET
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


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def sample_codes(batch, z_dim, b_dim, p_dim, c_dim, device):
    z = torch.randn(batch, z_dim, device=device)
    c = torch.zeros(batch, c_dim, device=device)
    cid = np.random.randint(c_dim, size=batch)
    for i in range(batch):
        c[i, cid[i]] = 1

    p = child_to_parent(c, c_dim, p_dim)
    b = c.clone()
    return z, b, p, c


def rand_sample_codes(prev_z, prev_b, prev_p, prev_c, rand_code=['b', 'p']):
    '''
    rand code default: keeping z and c
    '''
    device = prev_z.device
    batch = prev_z.size(0)
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

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)
    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="sample fine img")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument(
        "--params", type=str, default='map2style/params.pt', help="path to the model checkpoint")
    parser.add_argument(
        "--batch", type=int, default=1, help="batch sizes for each gpus")
    parser.add_argument("--thrld", type=float, default=0.1)

    # parser.add_argument('--load_param', type=str, default='')
    args = parser.parse_args()

    resize = 128

    assert args.ckpt is not None
    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    train_args = ckpt['args']

    args.z_dim = finegan_config[train_args.ds_name]['Z_DIM']
    args.b_dim = finegan_config[train_args.ds_name]['FINE_GRAINED_CATEGORIES']
    args.p_dim = finegan_config[train_args.ds_name]['SUPER_CATEGORIES']
    args.c_dim = finegan_config[train_args.ds_name]['FINE_GRAINED_CATEGORIES']

    style_generator = Generator(
        size=train_args.size,
        style_dim=train_args.latent,
        n_mlp=train_args.n_mlp,
        channel_multiplier=train_args.channel_multiplier
    ).to(device)

    style_discriminator = Discriminator(
        size=train_args.size
    ).to(device)

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

    fine_generator = G_NET(ds_name=train_args.ds_name).to(device)

    style_generator.load_state_dict(ckpt["style_g"])
    style_discriminator.load_state_dict(ckpt["style_d"])
    mpnet.load_state_dict(ckpt["mp"])
    fine_generator.load_state_dict(ckpt["fine"])

    style_generator.eval()
    style_discriminator.eval()
    mpnet.eval()
    fine_generator.eval()

    z = None
    b = None
    p = None
    c0 = None
    c1 = None
    latent = None

    params = torch.load(args.params, map_location=lambda storage, loc: storage)
    z = params['z'].to(device)
    b = params['b'].to(device)
    p = params['p'].to(device)
    c0 = params['c0'].to(device)
    c1 = params['c1'].to(device)
    # latent = params['rand_latent'].to(device)

    with torch.no_grad():
        _z, _b, _p, _c0 = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
        if z is None:
            z = _z
        if b is None:
            b = _b
        if p is None:
            p = _p
        if c0 is None:
            c0 = _c0

        _, _, _, _c1 = rand_sample_codes(z, b, p, c0, rand_code=['c'])
        if c1 is None:
            c1 = _c1

        img0 = fine_generator(z, b, p, c0)
        wp0 = mpnet(img0)
        s_img0, _ = style_generator([wp0],
            input_is_latent=True,
            randomize_noise=False)
        imgs = img0
        s_imgs = s_img0

        img1 = fine_generator(z, b, p, c1)
        wp1 = mpnet(img1)
        s_img1, _ = style_generator([wp1],
            input_is_latent=True,
            randomize_noise=False)
        imgs = torch.cat((imgs, img1), 0)
        s_imgs = torch.cat((s_imgs, s_img1), 0)

        if latent is None:
            noise = mixing_noise(args.batch, train_args.latent, train_args.mixing, device)
            rand_img, latent = style_generator(noise,
                return_latents=True,
                randomize_noise=False)
        else:
            rand_img, _ = style_generator([latent],
                input_is_latent=True,
                randomize_noise=False)

        arti_wp = latent - wp0 + wp1

        arti_img, _ = style_generator([arti_wp],
            input_is_latent=True,
            randomize_noise=False)

    utils.save_image(
        imgs,
        f"map2style/fine1.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )

    utils.save_image(
        rand_img,
        f"map2style/rand1.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )

    utils.save_image(
        arti_img,
        f"map2style/arti1.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )

    utils.save_image(
        s_imgs,
        f"map2style/style1.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )

    torch.save(
        {
            "rand_latent": latent,
            "z": z,
            "b": b,
            "p": p,
            "c0": c0,
            "c1": c1,
        },
        f"map2style/params.pt",
    )