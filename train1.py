import argparse
import math
import random
import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

from model import Generator, G_NET, Encoder, UNet
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

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

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
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

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

def binarization_loss(mask):
    return torch.min(1-mask, mask).mean()

def train(args, fine_generator, style_generator, mpnet, mknet, mp_optim, mk_optim, device):

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # cur_nimg = 0
    loss_dict = {}

    if args.distributed:
        mp_module = mpnet.module
        mk_module = mknet.module
        fine_g_module = fine_generator.module
        style_g_module = style_generator.module
    else:
        mp_module = mpnet
        mk_module = mknet
        fine_g_module = fine_generator
        style_g_module = style_generator


    # criterion_construct = nn.MSELoss()
    # criterion_reconstruct = nn.MSELoss().to(device)

    if args.trunc:
        truncation = 0.7
        trunc = style_generator.mean_latent(4096).detach()
        trunc.requires_grad_(False)
    else:
        truncation = 1
        trunc = None

    style_generator.eval()
    style_generator.requires_grad_(False)
    fine_generator.eval()
    fine_generator.requires_grad_(False)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        mpnet.train()
        mknet.train()

        ############# train mk network #############
        mknet.requires_grad_(True)
        # mpnet.requires_grad_(False)

        z, b, p, c = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
        if not args.tie_code:
            z, b, p, c = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c, rand_code=['b', 'p'])

        fine_img, mask = fine_generator(z, b, p, c, rtn_mk=True)
        pred_mask = mknet(fine_img)

        bin_loss = binarization_loss(pred_mask) * args.bin
        mk_loss = F.mse_loss(pred_mask, mask) * args.mk

        mknet_loss = mk_loss + bin_loss
        loss_dict["mk"] = mk_loss / args.mk
        loss_dict["bin"] = bin_loss / args.bin

        mknet.zero_grad()
        mknet_loss.backward()
        mk_optim.step()

        ############# train mapping network #############
        mpnet.requires_grad_(True)
        mknet.requires_grad_(False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        style_img, latent = style_generator(noise, return_latents=True, randomize_noise=False)
        _style_img = F.interpolate(style_img, size=(128, 128), mode='area')
        wp_code = mpnet(_style_img)

        mp_loss = F.mse_loss(wp_code, latent) * args.mp
        loss_dict["mp"] = mp_loss / args.mp

        z, b, p, c0 = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
        if not args.tie_code:
            z, b, p, c0 = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c0, rand_code=['b', 'p'])

        _, _, _, c1 = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c0, rand_code=['c'])

        fine_img0, _ = fine_generator(z, b, p, c0)
        fine_img1, _ = fine_generator(z, b, p, c1)
        w0 = mpnet(fine_img0)
        w1 = mpnet(fine_img1)
        fine_img0, _ = style_generator([w0], input_is_latent=True, randomize_noise=False)
        fine_img1, _ = style_generator([w1], input_is_latent=True, randomize_noise=False)
        fine_img0 = F.interpolate(fine_img0, size=(128, 128), mode='area')
        fine_img1 = F.interpolate(fine_img1, size=(128, 128), mode='area')
        mask0 = mknet(fine_img0)
        mask1 = mknet(fine_img1)
        bg_mask0 = torch.ones_like(mask0) - mask0
        bg_mask1 = torch.ones_like(mask1) - mask1

        fg_sim = F.mse_loss(mask0, mask1)
        bg_sim = F.cosine_similarity((bg_mask0 * fine_img0).view(args.batch, -1), (
            bg_mask1 * fine_img1).view(args.batch, -1)).mean()

        sim_loss = fg_sim + bg_sim
        loss_dict["sim"] = sim_loss

        mp_loss += sim_loss

        mpnet.zero_grad()
        mp_loss.backward()
        mp_optim.step()

        ############# ############# #############
        loss_reduced = reduce_loss_dict(loss_dict)
        mp_loss_val = loss_reduced["mp"].mean().item()
        mk_loss_val = loss_reduced["mk"].mean().item()
        bin_loss_val = loss_reduced["bin"].mean().item()
        sim_loss_val = loss_reduced["sim"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"mp: {mp_loss_val:.4f}; mk: {mk_loss_val:.4f}; bin: {bin_loss_val:.4f}, sim: {sim_loss_val:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "MP": mp_loss_val,
                        "MK": mk_loss_val,
                        "Bin": bin_loss_val,
                        "Sim": sim_loss_val,
                    }
                )

            if i % 500 == 0:
                with torch.no_grad():
                    mpnet.eval()
                    mknet.eval()

                    z, b, p, c = sample_codes(args.n_sample, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
                    if not args.tie_code:
                        z, b, p, c = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c, rand_code=['b', 'p'])

                    fine_img, _ = fine_generator(z, b, p, c)
                    wp_code = mpnet(fine_img)
                    rec_fine, _ = style_generator([wp_code], input_is_latent=True)

                    _rec_fine = F.interpolate(rec_fine, size=(128, 128), mode='area')

                    rec_mk = mknet(_rec_fine)

                    noise = mixing_noise(args.n_sample, args.latent, args.mixing, device)
                    style_img, _ = style_generator(noise, return_latents=True)
                    _style_img = F.interpolate(style_img, size=(128, 128), mode='area')
                    wp_code = mpnet(_style_img)
                    rec_style, _ = style_generator([wp_code], input_is_latent=True)

                    utils.save_image(
                        fine_img,
                        f"sample/{str(i).zfill(6)}_0.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        rec_fine,
                        f"sample/{str(i).zfill(6)}_1.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        rec_mk,
                        f"sample/{str(i).zfill(6)}_2.png",
                        nrow=8,
                        normalize=True,
                        range=(0, 1),
                    )

                    utils.save_image(
                        style_img,
                        f"sample/{str(i).zfill(6)}_3.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        rec_style,
                        f"sample/{str(i).zfill(6)}_4.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )
                    if wandb and args.wandb:
                        wandb.log(
                            {
                                "fine image": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_0.png").convert("RGB"))],
                                "recon fine": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_1.png").convert("RGB"))],
                                "recon mask": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_2.png").convert("RGB"))],
                                "style image": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_3.png").convert("RGB"))],
                                "recon style": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_4.png").convert("RGB"))],
                            }
                        )

            if i % 40000 == 0 and i != args.start_iter:
                torch.save(
                    {
                        "style_g": style_g_module.state_dict(),
                        "fine": fine_g_module.state_dict(),
                        "mp": mp_module.state_dict(),
                        "mk": mk_module.state_dict(),
                        "mp_optim": mp_optim.state_dict(),
                        "mk_optim": mk_optim.state_dict(),
                        "args": args,
                        "cur_iter": i,
                    },
                    f"checkpoint/{str(i).zfill(6)}_1_.pt",
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
        "--ckpt",
        type=str,
        default=None,
        help="path to previous trained checkpoint",
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
    parser.add_argument("--lr_mp", type=float, default=2e-3, help="mapping network learning rate")
    # parser.add_argument("--lr_d", type=float, default=2e-5, help="discriminator learning rate")
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
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument(
        "--trunc", action="store_true", help="use truncation"
    )
    parser.add_argument('--mp_arch', type=str, default='encoder',
                        help='model architectures (vanilla | encoder)')
    parser.add_argument(
        "--tie_code", action="store_true", help="use tied codes"
    )
    parser.add_argument('--ds_name', type=str, default='STANFORDCAR',
                        help='dataset used for training finegan (LSUNCAR | CUB | STANFORDCAR)')

    ## weights
    # parser.add_argument("--adv", type=float, default=1, help="weight of the adv loss")
    # parser.add_argument("--mse", type=float, default=1e2, help="weight of the mse loss")
    parser.add_argument("--mp", type=float, default=1, help="weight of the latent recon loss")
    parser.add_argument("--mk", type=float, default=1e1, help="weight of mask recon")
    parser.add_argument("--bin", type=float, default=1, help="weight of mask recon")

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

    args.z_dim = finegan_config[args.ds_name]['Z_DIM']
    args.b_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']
    args.p_dim = finegan_config[args.ds_name]['SUPER_CATEGORIES']
    args.c_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']

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

    mknet = UNet(
        n_channels = 3,
        n_classes = 1,
        bilinear = True,
    ).to(device)

    mp_optim = optim.Adam(
        mpnet.parameters(),
        lr=args.lr_mp,
        betas=(0, 0.99),
    )

    mk_optim = optim.Adam(
        mknet.parameters(),
        lr=args.lr_mp,
        betas=(0, 0.99),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        train_args = ckpt['args']
        args.start_iter = ckpt['cur_iter']

        mpnet.load_state_dict(ckpt["mp"])
        mknet.load_state_dict(ckpt["mk"])


        if args.style_model is None:
            style_generator.load_state_dict(ckpt["style_g"])
            mp_optim.load_state_dict(ckpt["mp_optim"])

        if args.fine_model is None:
            fine_generator.load_state_dict(ckpt["fine"])
            mk_optim.load_state_dict(ckpt["mk_optim"])

    # if specify stylegan checkpoint, overwrite stylegan from ckpt
    if args.style_model is not None:
        print("load style model:", args.style_model)
        style_dict = torch.load(args.style_model, map_location=lambda storage, loc: storage)
        style_generator.load_state_dict(style_dict["g_ema"])
        # d_optim.load_state_dict(style_dict["d_optim"])

    # if specify finegan checkpoint, overwrite finegan from ckpt
    if args.fine_model is not None:
        print("load fine model:", args.fine_model)
        fine_dict = torch.load(args.fine_model, map_location=lambda storage, loc: storage)
        fine_generator.load_state_dict(fine_dict)

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

        mpnet = nn.parallel.DistributedDataParallel(
            mpnet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        mknet = nn.parallel.DistributedDataParallel(
            mknet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="map net style distribute")

    train(args, fine_generator, style_generator, mpnet, mknet, mp_optim, mk_optim, device)
