import argparse
import math
import random
import os
import copy
import sys

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
from PIL import Image

from model import Generator, MappingNetwork, G_NET, Encoder, ImplicitMixer2, Discriminator

from finegan_config import finegan_config

# try:
import wandb

# except ImportError:
#     wandb = None

from op import conv2d_gradfix
from dataset import MultiResolutionDataset

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
# from non_leaking import augment, AdaptiveAugment

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


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


def train(args, fine_generator, style_generator, mpnet, mixer, context_discriminator, mxr_optim, ctx_optim, device):

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # cur_nimg = 0
    loss_dict = {}

    if args.distributed:
        mp_module = mpnet.module
        fine_g_module = fine_generator.module
        style_g_module = style_generator.module
        mxr_module = mixer.module
    else:
        mp_module = mpnet
        fine_g_module = fine_generator
        style_g_module = style_generator
        mxr_module = mixer

    fine_generator.eval()
    fine_generator.requires_grad_(False)
    style_generator.eval()
    style_generator.requires_grad_(False)
    mpnet.eval()
    mpnet.requires_grad_(False)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        mixer.train()
        mixer.requires_grad_(True)

        z0, b0, p0, c0 = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
        if not args.tie_code:
            z0, b0, p0, c0 = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=c0, rand_code=['b', 'p'])

        z1, b1, p1, c1 = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=c0, rand_code=['z', 'b', 'p', 'c'])

        shape_img = fine_generator(z0, b0, p0, c0)
        color_img = fine_generator(z1, b1, p1, c1)
        target_img = fine_generator(z0, b0, p0, c1)

        shape_w = mpnet(shape_img)
        color_w = mpnet(color_img)
        target_w = mpnet(target_img)

        mix_w = mixer(shape_w, color_w)
        w_loss = F.mse_loss(mix_w, target_w) * args.w_mse

        if args.constrain_img:
            target_img, _ = style_generator([target_w], input_is_latent=True, randomize_noise=False)
            mix_img, _ = style_generator([mix_w], input_is_latent=True, randomize_noise=False)
            img_loss = F.mse_loss(mix_img, target_img) * args.img_mse
        else:
            img_loss = torch.zeros(1, device=device)

        mxr_loss = w_loss + img_loss
        loss_dict["w"] = w_loss
        loss_dict["img"] = img_loss

        mixer.zero_grad()
        mxr_loss.backward()
        mxr_optim.step()

        ############# ############# #############
        loss_reduced = reduce_loss_dict(loss_dict)
        w_loss_val = loss_reduced["w"].mean().item()
        img_loss_val = loss_reduced["img"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"w: {w_loss_val:.4f}; img: {img_loss_val:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "W MSE": w_loss_val,
                        "Img MSE": img_loss_val,
                    }
                )

            if i % 1000 == 0:
                with torch.no_grad():
                    mixer.eval()

                    z0, b0, p0, c0 = sample_codes(args.n_sample, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
                    if not args.tie_code:
                        z0, b0, p0, c0 = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=c0, rand_code=['b', 'p'])

                    z1, b1, p1, c1 = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=c0, rand_code=['z', 'b', 'p', 'c'])

                    shape_img = fine_generator(z0, b0, p0, c0)
                    color_img = fine_generator(z1, b1, p1, c1)
                    target_img = fine_generator(z0, b0, p0, c1)

                    shape_w = mpnet(shape_img)
                    color_w = mpnet(color_img)
                    target_w = mpnet(target_img)
                    mix_w = mixer(shape_w, color_w)

                    _shape_img, _ = style_generator([shape_w], input_is_latent=True)
                    _color_img, _ = style_generator([color_w], input_is_latent=True)
                    _target_img, _ = style_generator([target_w], input_is_latent=True)
                    _mix_img, _ = style_generator([mix_w], input_is_latent=True)

                    noise1 = mixing_noise(args.n_sample, args.latent, args.mixing, device)
                    sample_img1, _ = style_generator(noise1, return_latents=True)
                    _sample_img1 = F.interpolate(sample_img1, size=(128, 128), mode='area')

                    noise2 = mixing_noise(args.n_sample, args.latent, args.mixing, device)
                    sample_img2, _ = style_generator(noise2, return_latents=True)
                    _sample_img2 = F.interpolate(sample_img2, size=(128, 128), mode='area')

                    w1 = mpnet(_sample_img1)
                    w2 = mpnet(_sample_img2)

                    mix_w = mixer(w1, w2)
                    mix_sample, _ = style_generator([mix_w], input_is_latent=True)

                    utils.save_image(
                        _shape_img,
                        f"cdn_training_dir/sample/{str(i).zfill(6)}_0.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        _color_img,
                        f"cdn_training_dir/sample/{str(i).zfill(6)}_1.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        _target_img,
                        f"cdn_training_dir/sample/{str(i).zfill(6)}_2.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        _mix_img,
                        f"cdn_training_dir/sample/{str(i).zfill(6)}_3.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        sample_img1,
                        f"cdn_training_dir/sample/{str(i).zfill(6)}_4.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        sample_img2,
                        f"cdn_training_dir/sample/{str(i).zfill(6)}_5.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        mix_sample,
                        f"cdn_training_dir/sample/{str(i).zfill(6)}_6.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    if wandb and args.wandb:
                        wandb.log(
                            {
                                "fine mix": [wandb.Image(Image.open(f"cdn_training_dir/sample/{str(i).zfill(6)}_3.png").convert("RGB"))],
                                "style mix": [wandb.Image(Image.open(f"cdn_training_dir/sample/{str(i).zfill(6)}_6.png").convert("RGB"))]
                            }
                        )

            if i % 40000 == 0 and i != args.start_iter:
                torch.save(
                    {
                        "style_g": style_g_module.state_dict(),
                        "fine": fine_g_module.state_dict(),
                        "mp": mp_module.state_dict(),
                        "mxr": mxr_module.state_dict(),
                        "mxr_optim": mxr_optim.state_dict(),
                        "args": args,
                        "cur_iter": i,
                    },
                    f"cdn_training_dir/checkpoint/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="cdn trainer")
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
        "--ckpt",
        type=str,
        default=None,
        help="path to previous trained checkpoint",
    )
    parser.add_argument(
        "--mp_ckpt",
        type=str,
        default=None,
        help="path to mapping ckpt",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    parser.add_argument(
        "--tie_code", action="store_true", help="use tied codes"
    )
    parser.add_argument(
        "--constrain_img", action="store_true", help="use img mse loss as well"
    )
    parser.add_argument("--w_mse", type=float, default=1, help="mse weight for w")
    parser.add_argument("--img_mse", type=float, default=1e-1, help="mse weight for img")

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

    if args.mp_ckpt is not None:
        mp_ckpt = torch.load(args.mp_ckpt, map_location=lambda storage, loc: storage)
        mp_args = mp_ckpt['args']
    elif args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        mp_args = ckpt['args']
        args.start_iter = ckpt['cur_iter']
    else:
        print('need to at least load one of mp_ckpt or ckpt')
        sys.exit(0)

    args.ds_name = mp_args.ds_name
    args.size = mp_args.size
    args.latent = mp_args.latent
    args.n_mlp = mp_args.n_mlp
    args.channel_multiplier = mp_args.channel_multiplier
    args.mp_arch = mp_args.mp_arch
    args.mixing = mp_args.mixing

    args.z_dim = finegan_config[args.ds_name]['Z_DIM']
    args.b_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']
    args.p_dim = finegan_config[args.ds_name]['SUPER_CATEGORIES']
    args.c_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']


    style_generator = Generator(
        size=args.size,
        style_dim=args.latent,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).to(device)

    context_discriminator = Discriminator(
        size=args.size,
        channel_multiplier=args.channel_multiplier
    ).to(device)

    fine_generator = G_NET(ds_name=args.ds_name).to(device)

    if args.mp_arch == 'vanilla':
        mpnet = MappingNetwork(
            num_ws=style_generator.n_latent,
            w_dim=args.latent
        ).to(device)
    # https://github.com/bryandlee/stylegan2-encoder-pytorch
    elif args.mp_arch == 'encoder':
        mpnet = Encoder(
            size=128,
            num_ws=style_generator.n_latent,
            w_dim=args.latent
        ).to(device)

    mixer = ImplicitMixer2(
        num_ws=style_generator.n_latent,
        w_dim=args.latent,
    ).to(device)

    ctx_optim = optim.Adam(
        context_discriminator.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    mxr_optim = optim.Adam(
        mixer.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    if args.ckpt is not None:
        print("loading checkpoint:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        mixer.load_state_dict(ckpt["mxr"])
        context_discriminator.load_state_dict(ckpt["ctx"])
        mxr_optim.load_state_dict(ckpt["mxr_optim"])
        ctx_optim.load_state_dict(ckpt["ctx_optim"])

        if args.mp_ckpt is None:
            mpnet.load_state_dict(ckpt["mp"])
            style_generator.load_state_dict(ckpt["style_g"])
            fine_generator.load_state_dict(ckpt["fine"])

    if args.mp_ckpt is not None:
        print("load mapping model:", args.mp_ckpt)
        mp_ckpt = torch.load(args.mp_ckpt, map_location=lambda storage, loc: storage)
        mpnet.load_state_dict(mp_ckpt["mp"])
        style_generator.load_state_dict(mp_ckpt["style_g"])
        fine_generator.load_state_dict(mp_ckpt["fine"])

    if args.distributed:
        sys.exit()
        # style_generator = nn.parallel.DistributedDataParallel(
        #     style_generator,
        #     device_ids=[args.local_rank],
        #     output_device=args.local_rank,
        #     broadcast_buffers=False,
        # )

        # fine_generator = nn.parallel.DistributedDataParallel(
        #     fine_generator,
        #     device_ids=[args.local_rank],
        #     output_device=args.local_rank,
        #     broadcast_buffers=False,
        # )

        # mpnet = nn.parallel.DistributedDataParallel(
        #     mpnet,
        #     device_ids=[args.local_rank],
        #     output_device=args.local_rank,
        #     broadcast_buffers=False,
        # )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="mixer net")

    train(args, fine_generator, style_generator, mpnet, mixer, context_discriminator, mxr_optim, ctx_optim, device)
