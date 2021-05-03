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

import sys

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from model import Generator, MappingNetwork, Encoder, G_NET, ImgMixer
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

def tuple2device(noise, device):
    for item in noise:
        item.to(device)
    return noise

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="sample fine img")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument(
        "--batch", type=int, default=1, help="batch sizes for each gpus")
    parser.add_argument(
        "--mix_recon", action="store_true", help="whether mix mapping net reconstructed image")
    parser.add_argument(
        "--rand_nosie", action="store_true", help="whether mix mapping net reconstructed image")

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

    # mixer = ImgMixer(
    #     size=train_args.mxr_size,
    #     num_ws=style_generator.n_latent,
    #     w_dim=train_args.latent
    # ).to(device)

    fine_generator = G_NET(ds_name=train_args.ds_name).to(device)

    style_generator.load_state_dict(ckpt["style_g"])
    fine_generator.load_state_dict(ckpt["fine"])
    mpnet.load_state_dict(ckpt["mp"])
    # mixer.load_state_dict(ckpt["mxr"])

    style_generator.eval()
    mpnet.eval()
    fine_generator.eval()
    # mixer.eval()

    train_args.mixing = 1

    ii1 = None
    ii2 = None
    noise1 = None
    noise2 = None
    noise3 = None
    noise4 = None
    noise5 = None
    noise6 = None

    # try:
    params = torch.load('mixer_testdir/params.pt', map_location=lambda storage, loc: storage)
    ii1 = 8
    # ii2 = 10
    ii2 = 8
    # noise1 = [e.to(device) for e in params['noise1']]
    # ii1 = params['ii1']
    # noise2 = [e.to(device) for e in params['noise2']]
    # ii2 = params['ii2']
    # except:
    #     pass

    with torch.no_grad():
        if noise1 is None:
            noise1 = mixing_noise(args.batch, train_args.latent, train_args.mixing, device)

        sample_img1, _ = style_generator(noise1, inject_index=ii1, return_latents=True, randomize_noise=args.rand_nosie)
        # _sample_img1 = F.interpolate(sample_img1, size=(128, 128), mode='area')

        # noise2 = noise1
        if noise2 is None:
            # delta = torch.randn_like(noise1[0])
            noise2 = mixing_noise(args.batch, train_args.latent, train_args.mixing, device)
            noise2 = [noise1[0], noise2[1]]

        sample_img2, _ = style_generator(noise2, inject_index=ii2, return_latents=True, randomize_noise=args.rand_nosie)

        if noise3 is None:
            l2_code = noise2[1][0:1].repeat(args.batch, 1)
            noise3 = mixing_noise(args.batch, train_args.latent, train_args.mixing, device)
            noise3 = [noise1[0], l2_code]

        sample_img3, _ = style_generator(noise3, inject_index=ii2, return_latents=True, randomize_noise=args.rand_nosie)

        if noise4 is None:
            l2_code = noise2[1][1:2].repeat(args.batch, 1)
            noise4 = mixing_noise(args.batch, train_args.latent, train_args.mixing, device)
            noise4 = [noise1[0], l2_code]

        sample_img4, _ = style_generator(noise4, inject_index=ii2, return_latents=True, randomize_noise=args.rand_nosie)

        if noise5 is None:
            l2_code = noise2[1][2:3].repeat(args.batch, 1)
            noise5 = mixing_noise(args.batch, train_args.latent, train_args.mixing, device)
            noise5 = [noise1[0], l2_code]

        sample_img5, _ = style_generator(noise5, inject_index=ii2, return_latents=True, randomize_noise=args.rand_nosie)

        if noise6 is None:
            l2_code = noise2[1][3:4].repeat(args.batch, 1)
            noise6 = mixing_noise(args.batch, train_args.latent, train_args.mixing, device)
            noise6 = [noise1[0], l2_code]

        sample_img6, _ = style_generator(noise6, inject_index=ii2, return_latents=True, randomize_noise=args.rand_nosie)


        # _sample_img2 = F.interpolate(sample_img2, size=(128, 128), mode='area')

        # w0 = mpnet(_sample_img1)
        # w1 = mpnet(_sample_img2)

        # if args.mix_recon:
        #     sample_img1, _ = style_generator([w0], input_is_latent=True, randomize_noise=args.rand_nosie)
        #     sample_img2, _ = style_generator([w1], input_is_latent=True, randomize_noise=args.rand_nosie)

        #     _sample_img1 = F.interpolate(sample_img1, size=(train_args.mxr_size, train_args.mxr_size), mode='area')
        #     _sample_img2 = F.interpolate(sample_img2, size=(train_args.mxr_size, train_args.mxr_size), mode='area')

        # mix_w = mixer(_sample_img1, _sample_img1)

        # mix_sample, *_ = style_generator([mix_w], input_is_latent=True, randomize_noise=args.rand_nosie)
        # arti_sample, *_ = style_generator([w1 - mix_w + w0], input_is_latent=True, randomize_noise=args.rand_nosie)

        samples = torch.cat((sample_img2, sample_img3, sample_img4, sample_img5, sample_img6), dim=0)
        # samples = torch.cat((sample_img1, sample_img2, mix_sample, arti_sample), dim=0)

        # z0, b0, p0, c0 = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
        # z0, b0, p0, c0 = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=c0, rand_code=['b', 'p'])
        # z1, b1, p1, c1 = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=c0, rand_code=['z', 'b', 'p', 'c'])

        # shape_img = fine_generator(z0, b0, p0, c0)
        # color_img = fine_generator(z1, b1, p1, c1)
        # # target_img = fine_generator(z0, b0, p0, c1)

        # shape_w = mpnet(shape_img)
        # color_w = mpnet(color_img)
        # # target_w = mpnet(target_img)


        # shape_img, _ = style_generator([shape_w], input_is_latent=True)
        # color_img, _ = style_generator([color_w], input_is_latent=True)
        # # target_img, _ = style_generator([target_w], input_is_latent=True)

        # _shape_img = F.interpolate(shape_img, size=(train_args.mxr_size, train_args.mxr_size), mode='area')
        # _color_img = F.interpolate(color_img, size=(train_args.mxr_size, train_args.mxr_size), mode='area')
        # mix_w = mixer(_shape_img, _color_img)

        # mix_img, _ = style_generator([mix_w], input_is_latent=True)
        # arti_img, _ = style_generator([color_w - mix_w + shape_w], input_is_latent=True)

        # imgs = torch.cat((shape_img, color_img, mix_img, arti_img), dim=0)

    utils.save_image(
        samples,
        f"mixer_testdir/style4.png",
        nrow=args.batch,
        normalize=True,
        range=(-1, 1),
    )

    # utils.save_image(
    #     imgs,
    #     f"mixer_testdir/fine4.png",
    #     nrow=args.batch,
    #     normalize=True,
    #     range=(-1, 1),
    # )

    torch.save(
        {
            "noise1": noise1,
            "noise2": noise2,
            "ii1": ii1,
            "ii2": ii2
        },
        f"mixer_testdir/params.pt",
    )