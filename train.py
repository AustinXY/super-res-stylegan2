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

from model import Generator, MappingNetwork, G_NET
from finegan_config import finegan_config

# try:
import wandb

# except ImportError:
#     wandb = None

from dataset import MultiResolutionDataset

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment, AdaptiveAugment


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname == 'MnetConv':
        nn.init.constant_(m.mask_conv.weight.data, 1)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True, only_inputs=True
    )
    # grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    grad_penalty = grad_real.square().sum([1,2,3])
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths

def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def child_to_parent(c_code, c_dim, p_dim):
    ratio = c_dim / p_dim
    cid = torch.argmax(c_code,  dim=1)
    pid = (cid / ratio).long()
    p_code = torch.zeros([c_code.size(0), p_dim], device=c_code.device)
    for i in range(c_code.size(0)):
        p_code[i][pid[i]] = 1
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


def rand_sample_codes(prev_z, prev_b, prev_p, prev_c, device, rand_code=['z', 'b', 'p', 'c']):
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


def train(args, fine_generator, style_generator, mpnet, mp_optim, device):

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    cur_nimg = 0
    loss_dict = {}

    if args.distributed:
        mp_module = mpnet.module
    else:
        mp_module = mpnet


    # criterion_construct = nn.MSELoss()
    criterion_reconstruct = nn.MSELoss().to(device)

    style_generator.eval()
    style_generator.requires_grad_(False)
    fine_generator.eval()
    fine_generator.requires_grad_(False)
    mpnet.train()
    mpnet.requires_grad_(True)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        ############# train child discriminator #############
        z, b, p, c = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
        z, b, p, c = rand_sample_codes(z, b, p, c, device)

        fine_img = fine_generator(z, b, p, c)

        utils.save_image(
            fine_img,
            f"tt.png",
            nrow=8,
            normalize=True,
            range=(-1, 1),
        )

        sys

        wp_code = mpnet(fine_img)
        style_img, _ = style_generator([wp_code], input_is_latent=True)

        _style_img = F.interpolate(style_img, size=(128, 128), mode='area')
        recon_loss = criterion_reconstruct(_style_img, fine_img)

        mpnet.zero_grad()
        recon_loss.backward()
        mp_optim.step()

        cur_nimg += args.batch

        loss_reduced = reduce_loss_dict(loss_dict)

        recon_loss_val = loss_reduced["recon"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"recon: {recon_loss_val:.4f}; "
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Reconstruction": recon_loss_val,
                    }
                )

            if i % 1000 == 0:
                with torch.no_grad():
                    # mpnet.eval()

                    z, b, p, c = sample_codes(args.n_sample, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
                    z, b, p, c = rand_sample_codes(z, b, p, c, device)

                    fine_img = fine_generator(z, b, p, c)
                    wp_code = mpnet(fine_img)
                    style_img, _ = style_generator([wp_code], input_is_latent=True)

                    utils.save_image(
                        fine_img,
                        f"sample/{str(i).zfill(6)}_0.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        style_img,
                        f"sample/{str(i).zfill(6)}_1.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 100000 == 0 and i != 0:
                torch.save(mp_module.state_dict(),
                    f"checkpoint/{str(i).zfill(6)}.pt",
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
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
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

    mpnet = MappingNetwork(
        num_ws=style_generator.n_latent,
        w_dim=args.latent
    ).to(device)

    fine_generator = G_NET().to(device)

    mp_optim = optim.Adam(
        mpnet.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )


    assert args.style_model is not None
    print("load style model:", args.style_model)

    style_dict = torch.load(args.style_model, map_location=lambda storage, loc: storage)

    try:
        ckpt_name = os.path.basename(args.style_model)
        args.start_iter = int(os.path.splitext(ckpt_name)[0])

    except ValueError:
        pass

    style_generator.load_state_dict(style_dict["g"])

    assert args.fine_model is not None
    print("load fine model:", args.fine_model)

    fine_dict = torch.load(args.fine_model, map_location=lambda storage, loc: storage)
    fine_generator.load_state_dict(fine_dict)

    # torch.save(fine_generator.module.state_dict(),
    #     f"../data/fine_model/lsuncar1600k.pt",
    # )

    if args.distributed:
        style_generator = nn.parallel.DistributedDataParallel(
            style_generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        fine_generator = nn.parallel.DistributedDataParallel(
            fine_generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )


    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="mapping net")

    train(args, fine_generator, style_generator, mpnet, mp_optim, device)
