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

from model import Generator, MappingNetwork, G_NET, Encoder, Decomposer, Composer

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


def train(args, fine_generator, style_generator, mpnet, decomposer, composer, dpr_optim, cpr_optim, device):

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # cur_nimg = 0
    loss_dict = {}

    if args.distributed:
        mp_module = mpnet.module
        fine_g_module = fine_generator.module
        style_g_module = style_generator.module
        dpr_module = decomposer.module
        cpr_module = composer.module
    else:
        mp_module = mpnet
        fine_g_module = fine_generator
        style_g_module = style_generator
        dpr_module = decomposer
        cpr_module = composer

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

        decomposer.train()
        composer.train()

        # ############# train decomposer #############
        decomposer.requires_grad_(True)
        composer.requires_grad_(False)

        z0, b0, p0, c0 = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
        if not args.tie_code:
            z0, b0, p0, c0 = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=c0, rand_code=['b', 'p'])

        z1, b1, p1, _ = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=None, rand_code=['z', 'b', 'p'])
        _,  _,  _, c1 = rand_sample_codes(prev_z=None, prev_b=None, prev_p=None, prev_c=c0, rand_code=['c'])

        ref_img = fine_generator(z0, b0, p0, c0)
        vc_img = fine_generator(z1, b1, p1, c0)
        ivc_img = fine_generator(z0, b0, p0, c1)

        ref_w = mpnet(ref_img)
        vc_w = mpnet(vc_img)
        ivc_w = mpnet(ivc_img)

        # variance, invariant code reconstruction
        vc0, ivc0 = decomposer(ref_w)
        vc1, _ = decomposer(vc_w)
        _, ivc2 = decomposer(ivc_w)

        # vc_loss = torch.mean(cos(vc1, vc0))
        # ivc_loss = torch.mean(cos(ivc2, ivc0))
        vc_loss = F.mse_loss(vc1, vc0)
        ivc_loss = F.mse_loss(ivc2, ivc0)
        dpr_loss = vc_loss + ivc_loss

        loss_dict["vc"] = vc_loss
        loss_dict["ivc"] = ivc_loss
        loss_dict["dpr"] = dpr_loss

        decomposer.zero_grad()
        dpr_loss.backward()
        dpr_optim.step()

        ############ train composer #############
        decomposer.requires_grad_(True)
        composer.requires_grad_(True)

        # recompose finegan mapped w
        z, b, p, c = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
        if not args.tie_code:
            z, b, p, c = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c, rand_code=['b', 'p'])

        fine_img = fine_generator(z, b, p, c)
        wp_code = mpnet(fine_img)
        vc, ivc = decomposer(wp_code)
        rec_wp = composer(vc, ivc)

        fine_cpr_loss = F.mse_loss(rec_wp, wp_code) * 1e2
        loss_dict["f_cpr"] = fine_cpr_loss

        # recompose random sampled stylegan w
        # noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        # wp_img, wp_code = style_generator(noise, return_latents=True)
        # vc, ivc = decomposer(wp_code)
        # rec_wp = composer(vc, ivc)
        # rec_img, _ = style_generator([rec_wp], input_is_latent=True, randomize_noise=False)
        # style_cpr_loss = F.mse_loss(rec_img, wp_img) * 1e2
        style_cpr_loss = torch.zeros(1, device=device)
        loss_dict["s_cpr"] = style_cpr_loss

        cpr_loss = fine_cpr_loss + style_cpr_loss
        loss_dict["cpr"] = cpr_loss

        decomposer.zero_grad()
        composer.zero_grad()
        cpr_loss.backward()
        dpr_optim.step()
        cpr_optim.step()

        ############# ############# #############
        loss_reduced = reduce_loss_dict(loss_dict)
        dpr_loss_val = loss_reduced["dpr"].mean().item()
        vc_loss_val = loss_reduced["vc"].mean().item()
        ivc_loss_val = loss_reduced["ivc"].mean().item()
        cpr_score_val = loss_reduced["cpr"].mean().item()
        fcpr_score_val = loss_reduced["f_cpr"].mean().item()
        scpr_score_val = loss_reduced["s_cpr"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"dpr: {dpr_loss_val:.4f}; cpr: {cpr_score_val:.4f}; s_cpr: {scpr_score_val:.4f}; f_cpr: {fcpr_score_val:.4f}; vc: {vc_loss_val:.4f}; ivc: {ivc_loss_val:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Decomposer": dpr_loss_val,
                        "Composer": cpr_score_val,
                        "Style_recompose": scpr_score_val,
                        "Fine_recompose": fcpr_score_val,
                        "Variance_code": vc_loss_val,
                        "Invariance_code": ivc_loss_val,
                    }
                )

            if i % 1000 == 0:
                with torch.no_grad():
                    decomposer.eval()
                    composer.eval()

                    z0, b0, p0, c0 = sample_codes(args.n_sample, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
                    z0, b0, p0, c0 = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=c0, rand_code=['b', 'p'])

                    z1, b1, p1, _ = rand_sample_codes(prev_z=z0, prev_b=b0, prev_p=p0, prev_c=None, rand_code=['z', 'b', 'p'])
                    _,  _,  _, c1 = rand_sample_codes(prev_z=None, prev_b=None, prev_p=None, prev_c=c0, rand_code=['c'])

                    ref_img = fine_generator(z0, b0, p0, c0)
                    vc_img = fine_generator(z1, b1, p1, c0)
                    ivc_img = fine_generator(z0, b0, p0, c1)

                    ref_w = mpnet(ref_img)
                    vc_w = mpnet(vc_img)
                    ivc_w = mpnet(ivc_img)

                    vc0, ivc0 = decomposer(ref_w)
                    vc1, ivc1 = decomposer(vc_w)
                    vc2, ivc2 = decomposer(ivc_w)

                    rec_ref_w = composer(vc0, ivc0)
                    rec_vc_w = composer(vc1, ivc1)
                    rec_ivc_w = composer(vc2, ivc2)

                    noise = mixing_noise(args.n_sample, args.latent, args.mixing, device)
                    _, wp_code = style_generator(noise, return_latents=True)
                    vc, ivc = decomposer(wp_code)
                    rec_w = composer(vc, ivc)

                    style_li = []
                    for w in [ref_w, vc_w, ivc_w, wp_code]:
                        style_img, _ = style_generator([w], input_is_latent=True)
                        style_li.append(style_img)

                    rec_li = []
                    for w in [rec_ref_w, rec_vc_w, rec_ivc_w, rec_w]:
                        rec_img, _ = style_generator([w], input_is_latent=True)
                        rec_li.append(rec_img)

                    for j in range(4):
                        utils.save_image(
                            style_li[j],
                            f"cdn_training_dir/sample/{str(i).zfill(6)}_style_{j}.png",
                            nrow=8,
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            rec_li[j],
                            f"cdn_training_dir/sample/{str(i).zfill(6)}_recon_{j}.png",
                            nrow=8,
                            normalize=True,
                            range=(-1, 1),
                        )

            if i % 200000 == 0 and i != args.start_iter:
                torch.save(
                    {
                        "style_g": style_g_module.state_dict(),
                        "fine": fine_g_module.state_dict(),
                        "mp": mp_module.state_dict(),
                        "dpr": dpr_module.state_dict(),
                        "cpr": cpr_module.state_dict(),
                        "dpr_optim": dpr_optim.state_dict(),
                        "cpr_optim": cpr_optim.state_dict(),
                        "args": args,
                        "cur_iter": i,
                    },
                    f"cdn_training_dir/checkpoint/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="cdn trainer")
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
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

    parser.add_argument("--vc_dim", type=int, default=512, help="variance code dim")
    parser.add_argument("--ivc_dim", type=int, default=4096, help="invariance code dim")

    ## weights
    # parser.add_argument("--adv", type=float, default=1, help="weight of the adv loss")
    # parser.add_argument("--mse", type=float, default=1e2, help="weight of the mse loss")
    # parser.add_argument("--lrl", type=float, default=1e1, help="weight of the latent recon loss")

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
        ckpt = torch.load(args.mp_ckpt, map_location=lambda storage, loc: storage)
        mp_args = ckpt['args']
    elif args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        mp_args = ckpt['args']
        args.start_iter = ckpt['cur_iter']
    else:
        print('need to at least load one of mp_ckpt or ckpt')
        sys.exit(0)

    args.z_dim = finegan_config[mp_args.ds_name]['Z_DIM']
    args.b_dim = finegan_config[mp_args.ds_name]['FINE_GRAINED_CATEGORIES']
    args.p_dim = finegan_config[mp_args.ds_name]['SUPER_CATEGORIES']
    args.c_dim = finegan_config[mp_args.ds_name]['FINE_GRAINED_CATEGORIES']


    style_generator = Generator(
        size=mp_args.size,
        style_dim=mp_args.latent,
        n_mlp=mp_args.n_mlp,
        channel_multiplier=mp_args.channel_multiplier
    ).to(device)

    fine_generator = G_NET(ds_name=mp_args.ds_name).to(device)

    if mp_args.mp_arch == 'vanilla':
        mpnet = MappingNetwork(
            num_ws=style_generator.n_latent,
            w_dim=mp_args.latent
        ).to(device)
    # https://github.com/bryandlee/stylegan2-encoder-pytorch
    elif mp_args.mp_arch == 'encoder':
        mpnet = Encoder(
            size=128,
            num_ws=style_generator.n_latent,
            w_dim=mp_args.latent
        ).to(device)

    decomposer = Decomposer(
        num_ws=style_generator.n_latent,
        w_dim=mp_args.latent,
        vc_dim=args.vc_dim,
        ivc_dim=args.ivc_dim
    ).to(device)

    composer = Composer(
        num_ws=style_generator.n_latent,
        w_dim=mp_args.latent,
        vc_dim=args.vc_dim,
        ivc_dim=args.ivc_dim
    ).to(device)

    dpr_optim = optim.Adam(
        decomposer.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    cpr_optim = optim.Adam(
        composer.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    if args.ckpt is not None:
        print("loading checkpoint:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        decomposer.load_state_dict(ckpt["dpr"])
        composer.load_state_dict(ckpt["cpr"])
        dpr_optim.load_state_dict(ckpt["dpr_optim"])
        cpr_optim.load_state_dict(ckpt["cpr_optim"])

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
        wandb.init(project="code disentangle")

    train(args, fine_generator, style_generator, mpnet, decomposer, composer, dpr_optim, cpr_optim, device)
