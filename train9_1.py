import argparse
import math
import random
import os
import gc

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
from torch import mul, nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from PIL import Image
from finegan_config import finegan_config
import wandb

from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from model import _MGenerator, Discriminator, UNet, _MuVarEncoder
from mixnmatch_model import G_NET
from scene_model import Encoder

from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment


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
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

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

    path_mean = mean_path_length + decay * \
        (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    # if n_noise == 1:
    #     noises = torch.randn(1, batch, latent_dim, device=device)
    #     noises = torch.cat((noises, noises), dim=0)
    #     return noises

    noises = torch.randn(n_noise, batch, latent_dim, device=device)

    return noises

# def make_noise(batch, latent_dim, n_noise, device):
#     if n_noise == 1:
#         noises = torch.randn(batch, 1, latent_dim, device=device)
#         noises = torch.cat((noises, noises), dim=1)
#         return noises

#     noises = torch.randn(batch, n_noise, latent_dim, device=device)
#     return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return make_noise(batch, latent_dim, 1, device)


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


def fill_mask(mask, pad_px=3):
    kernel = torch.ones(1, 1, 2*pad_px+1, 2*pad_px+1, device=mask.device)
    filled_mask = F.conv2d(mask, kernel, padding=pad_px)
    return filled_mask


def get_bin_mask(mask, thrsh=0.8):
    bin_mask = torch.where(mask >= thrsh, torch.ones_like(
        mask), torch.zeros_like(mask))
    return bin_mask


def process_mask(mask, thrsh0=0.8, thrsh1=0.3, pad_px=3):
    bin_mask = get_bin_mask(mask, thrsh=thrsh0)
    if pad_px == 0:
        return bin_mask

    filled_mask = fill_mask(bin_mask, pad_px)
    bin_mask = get_bin_mask(filled_mask, thrsh=thrsh1)
    return bin_mask


def approx_bin_mask(img, mknet, thrsh0=0.8, thrsh1=0.3, pad_px=3):
    _img = F.interpolate(img, size=(128, 128), mode='area')
    mask = F.interpolate(mknet(_img), size=(512, 512), mode='area')
    mask = process_mask(mask, thrsh0, thrsh1, pad_px)
    return mask

def train(args, generator, fine_generator, fine2style, g_optim, g_ema, device):

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        f2s_module = fine2style.module
        fine_module = fine_generator.module
    else:
        g_module = generator
        f2s_module = fine2style
        fine_module = fine_generator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(
            args.ada_target, args.ada_length, 8, device)

    fine_generator.eval()
    requires_grad(fine_generator, False)

    # args.injidx = None

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        fine2style.train()

        ############# train generator network #############
        requires_grad(generator, True)
        requires_grad(fine2style, True)

        z, b, p, c = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
        if not args.tie_code:
            z, b, p, c = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c, rand_code=['b', 'p'])

        fine_img = fine_generator(z, b, p, c)
        style_z, kl_loss = fine2style(fine_img)

        # assert style_z.size() == (1, args.batch, args.latent)

        fake_img, _ = generator(style_z)

        # fake_img = F.interpolate(fake_img, size=(128, 128), mode='bicubic')
        # rec_loss = vgg(fake_img, fine_img)

        rec_loss = F.mse_loss(fake_img, fine_img)

        loss_dict["rec"] = rec_loss
        loss_dict["kl"] = kl_loss

        # loss = rec_loss * 1 + kl_loss * args.kl

        fine2style.zero_grad()
        generator.zero_grad()
        rec_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        rec_loss_val = loss_reduced["rec"].item()
        kl_loss_val = loss_reduced["kl"].item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {rec_loss_val:.4f}; g: {kl_loss_val:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "KL": kl_loss_val,
                        "Reconstruction": rec_loss_val,
                    }
                )

            if i % 500 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    fine2style.eval()

                    z, b, p, c = sample_codes(8, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
                    if not args.tie_code:
                        z, b, p, c = rand_sample_codes(prev_z=z, prev_b=b, prev_p=p, prev_c=c, rand_code=['b', 'p'])

                    fine_img = fine_generator(z, b, p, c)
                    style_z, _ = fine2style(fine_img)
                    style_img, _ = g_ema(style_z)

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

                    if wandb and args.wandb:
                        wandb.log(
                            {
                                "fine image": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_0.png").convert("RGB"))],
                                "fine2style image": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_1.png").convert("RGB"))],
                            }
                        )

            if i % 100000 == 0 and i != args.start_iter:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "f2s": f2s_module.state_dict(),
                        "fine": fine_module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "cur_itr": i
                    },
                    f"checkpoint/{str(i).zfill(6)}_9_1.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--arch', type=str, default='stylegan2',
                        help='model architectures (stylegan2 | swagan)')
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

    parser.add_argument("--n_noise", type=int, default=1)
    parser.add_argument("--injidx", type=int, default=10)
    parser.add_argument("--style_dim", type=int, default=512)
    parser.add_argument("--fg_dim", type=int, default=256)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--scene_size", type=tuple, default=(512, 512),
                        help='size of image (H*W), used in defining dataset and model')
    parser.add_argument("--prior_size", type=tuple,
                        default=(128, 128), help='input size to encoder')
    parser.add_argument("--starting_height_size", type=int, default=4,
                        help='encoder feature passed to generator, support 4,8,16,32.')

    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--guide_reg_every",
        type=int,
        default=1,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument(
        "--fine_ckpt",
        type=str,
        default=None,
        help="path to the fine ckpt",
    )
    parser.add_argument(
        "--style_ckpt",
        type=str,
        default=None,
        help="path to the style ckpt",
    )
    parser.add_argument("--lr", type=float, default=0.002,
                        help="learning rate")
    parser.add_argument("--lr_cog", type=float, default=0.004,
                        help="learning rate")
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
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--tie_code", action="store_true", help="use tied codes"
    )
    parser.add_argument('--ds_name', type=str, default='STANFORDCAR',
                        help='dataset used for training finegan (LSUNCAR | CUB | STANFORDCAR)')

    parser.add_argument("--mse", type=float, default=4, help="mse weight")
    parser.add_argument("--guide_mse_fg", type=float, default=5, help="mse weight")
    parser.add_argument("--guide_mse_bg", type=float, default=5, help="mse weight")
    parser.add_argument("--kl", type=float, default=0.01, help="mse weight")

    parser.add_argument("--mk_thrsh0", type=float, default=0.5, help="Threshold for mask")
    parser.add_argument("--mk_thrsh1", type=float, default=0.3, help="Threshold for mask")
    parser.add_argument("--mk_pdpx", type=int, default=2, help="Threshold for mask")

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8
    args.mp_nws = 2

    args.start_iter = 0

    args.z_dim = finegan_config[args.ds_name]['Z_DIM']
    args.b_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']
    args.p_dim = finegan_config[args.ds_name]['SUPER_CATEGORIES']
    args.c_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']

    generator = _MGenerator(
        size=args.size,
        style_dim=args.latent,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema = _MGenerator(
        size=args.size,
        style_dim=args.latent,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    fine_generator = G_NET(ds_name=args.ds_name).to(device)

    args.prior_size = (128, 128)
    args.starting_height_size = 4

    fine2style = Encoder(
        args
    ).to(device)

    # cog_optim = optim.Adam(
    #     list(fine2style.parameters()) + list(generator.parameters()),
    #     lr=args.lr_cog,
    #     betas=(0, 0.99),
    # )

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        list(fine2style.parameters()) + list(generator.parameters()),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )


    assert args.ckpt is not None or args.fine_ckpt is not None
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        args.start_iter = ckpt['cur_itr']

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        fine2style.load_state_dict(ckpt['f2s'])
        style2fine.load_state_dict(ckpt['s2f'])
        fine_generator.load_state_dict(ckpt["fine"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        cog_optim.load_state_dict(ckpt["cog_optim"])

    if args.fine_ckpt is not None:
        print("load fine model:", args.fine_ckpt)
        fine_dict = torch.load(args.fine_ckpt, map_location=lambda storage, loc: storage)
        fine_generator.load_state_dict(fine_dict)

    if args.style_ckpt is not None:
        print("load fine model:", args.style_ckpt)
        style_dict = torch.load(args.style_ckpt, map_location=lambda storage, loc: storage)
        generator.load_state_dict(style_dict["g"])
        discriminator.load_state_dict(style_dict["d"])
        g_ema.load_state_dict(style_dict["g_ema"])
        g_optim.load_state_dict(style_dict["g_optim"])
        d_optim.load_state_dict(style_dict["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        fine2style = nn.parallel.DistributedDataParallel(
            fine2style,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        fine_generator = nn.parallel.DistributedDataParallel(
            fine_generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="guide 9_1")

    train(args, generator, fine_generator, fine2style,
          g_optim, g_ema, device)