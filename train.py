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
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"]="0"

from model import Generator, MappingNetwork, G_NET, Encoder
from sbg_model import StyledGenerator

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

@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style


def sample_codes(batch, z_dim, device):
    z = torch.randn(batch, z_dim, device=device)
    return z


def train(args, loader, sbg_generator, style_generator, style_discriminator, mpnet, mp_optim, d_optim, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # cur_nimg = 0
    loss_dict = {}

    if args.distributed:
        mp_module = mpnet.module
        sbg_g_module = sbg_generator.module
        style_g_module = style_generator.module
        style_d_module = style_discriminator.module
    else:
        mp_module = mpnet
        sbg_g_module = sbg_generator
        style_g_module = style_generator
        style_d_module = style_discriminator


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
    sbg_generator.eval()
    sbg_generator.requires_grad_(False)
    style_discriminator.train()
    # mpnet.requires_grad_(True)

    sbg_mean_style = get_mean_style(sbg_generator, device)
    sbg_step = int(math.log(args.sbg_size, 2)) - 2

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        mpnet.train()

        ############# train discriminator #############
        mpnet.requires_grad_(False)
        style_discriminator.requires_grad_(True)

        z = sample_codes(args.batch, args.z_dim, device)

        sbg_img = sbg_generator(
            z,
            step=sbg_step,
            alpha=1,
            mean_style=sbg_mean_style,
            style_weight=0.7,
        )

        wp_code = mpnet(sbg_img)
        fake_img, _ = style_generator([wp_code],
                                      input_is_latent=True,
                                      truncation=truncation,
                                      truncation_latent=trunc,
                                      randomize_noise=False)

        fake_pred = style_discriminator(fake_img)
        real_pred = style_discriminator(real_img)

        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        style_discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = style_discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)

            style_discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss


        ############# train mapping network #############
        mpnet.requires_grad_(True)
        style_discriminator.requires_grad_(False)

        z = sample_codes(args.batch, args.z_dim, device)

        sbg_img = sbg_generator(
            z,
            step=sbg_step,
            alpha=1,
            mean_style=sbg_mean_style,
            style_weight=0.7,
        )

        wp_code = mpnet(sbg_img)
        style_img, _ = style_generator([wp_code],
                                       input_is_latent=True,
                                       truncation=truncation,
                                       truncation_latent=trunc,
                                       randomize_noise=False)

        # adv loss
        fake_pred = style_discriminator(style_img)
        adv_loss = g_nonsaturating_loss(fake_pred)
        loss_dict["adv"] = adv_loss

        # mse loss
        _style_img = F.interpolate(style_img, size=(args.sbg_size, args.sbg_size), mode='area')
        mse_loss = F.mse_loss(_style_img, sbg_img)
        loss_dict["mse"] = mse_loss

        g_loss = adv_loss * args.adv + mse_loss * args.mse

        mpnet.zero_grad()
        g_loss.backward()
        mp_optim.step()

        ############# down sample image latent reconstruct #############
        if args.ltnt_recon_every != 0 and i % args.ltnt_recon_every == 0:
            mpnet.requires_grad_(True)
            style_discriminator.requires_grad_(False)

            noise = mixing_noise(args.batch, args.latent, args.mixing, device)

            style_img, latent = style_generator(noise, return_latents=True)
            _style_img = F.interpolate(style_img, size=(args.sbg_size, args.sbg_size), mode='area')
            wp_code = mpnet(_style_img)

            latent_recon_loss = F.mse_loss(wp_code, latent)

            loss_dict["lrl"] = latent_recon_loss

            mpnet.zero_grad()
            (latent_recon_loss * args.lrl).backward()
            mp_optim.step()

        ############# ############# #############
        loss_reduced = reduce_loss_dict(loss_dict)
        d_loss_val = loss_reduced["d"].mean().item()
        adv_loss_val = loss_reduced["adv"].mean().item()
        mse_loss_val = loss_reduced["mse"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        if args.ltnt_recon_every != 0 and i % args.ltnt_recon_every == 0:
            latent_recon_loss_val = loss_reduced["lrl"].mean().item()

        if get_rank() == 0:
            if args.ltnt_recon_every == 0:
                pbar.set_description(
                    (
                        f"mse: {mse_loss_val:.4f}; adv: {adv_loss_val:.4f};"
                    )
                )
            else:
                pbar.set_description(
                    (
                        f"mse: {mse_loss_val:.4f}; adv: {adv_loss_val:.4f}; lrl: {latent_recon_loss_val:.4f}"
                    )
                )

            if wandb and args.wandb:
                if args.ltnt_recon_every != 0 and i % args.ltnt_recon_every == 0:
                    wandb.log(
                        {
                            "Discriminator": d_loss_val,
                            "Adversarial": adv_loss_val,
                            "MSE": mse_loss_val,
                            "Real Score": real_score_val,
                            "Fake Score": fake_score_val,
                            "latent reconstruction": latent_recon_loss_val,
                        }
                    )
                else:
                    wandb.log(
                        {
                            "Discriminator": d_loss_val,
                            "Adversarial": adv_loss_val,
                            "MSE": mse_loss_val,
                            "Real Score": real_score_val,
                            "Fake Score": fake_score_val,
                        }
                    )

            if i % 500 == 0:
                with torch.no_grad():
                    mpnet.eval()

                    z = sample_codes(8, args.z_dim, device)

                    sbg_img = sbg_generator(
                        z,
                        step=sbg_step,
                        alpha=1,
                        mean_style=sbg_mean_style,
                        style_weight=0.7,
                    )

                    wp_code = mpnet(sbg_img)
                    style_img, _ = style_generator([wp_code], input_is_latent=True)

                    utils.save_image(
                        sbg_img,
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
                                "lr image": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_0.png").convert("RGB"))],
                                "hr image": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_1.png").convert("RGB"))],
                            }
                        )

            if i % 30000 == 0 and i != 0:
                torch.save(
                    {
                        "style_g": style_g_module.state_dict(),
                        "style_d": style_d_module.state_dict(),
                        "sbg_g": sbg_g_module.state_dict(),
                        "mp": mp_module.state_dict(),
                        "mp_optim": mp_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="mpnet trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
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
        "--sbg_size", type=int, default=128, help="image sizes for the sbg model"
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
        "--sbg_model",
        type=str,
        default=None,
        help="path to finegan",
    )
    parser.add_argument("--lr_mp", type=float, default=2e-3,
                        help="mapping network learning rate")
    parser.add_argument("--lr_d", type=float, default=2e-5,
                        help="discriminator learning rate")
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
    parser.add_argument(
        "--ltnt_recon_every", type=int, default=0, help="down sample images reconstruct"
    )
    parser.add_argument('--mp_arch', type=str, default='vanilla',
                        help='model architectures (vanilla | encoder)')

    ## weights
    parser.add_argument("--adv", type=float, default=1, help="weight of the adv loss")
    parser.add_argument("--mse", type=float, default=1e2, help="weight of the mse loss")
    parser.add_argument("--lrl", type=float, default=1e1, help="weight of the latent recon loss")


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

    args.z_dim = 512

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

    style_discriminator = Discriminator(
        size=args.size,
        channel_multiplier=args.channel_multiplier
    ).to(device)

    sbg_generator = StyledGenerator(
        code_dim=args.z_dim
    ).to(device)

    if args.mp_arch == 'vanilla':
        mpnet = MappingNetwork(
            num_ws=style_generator.n_latent,
            w_dim=args.latent
        ).to(device)
    # https://github.com/bryandlee/stylegan2-encoder-pytorch
    elif args.mp_arch == 'encoder':
        mpnet = Encoder(
            size=args.sbg_size,
            num_ws=style_generator.n_latent,
            w_dim=args.latent
        ).to(device)

    d_optim = optim.Adam(
        style_discriminator.parameters(),
        lr=args.lr_d,
        betas=(0, 0.99),
    )

    mp_optim = optim.Adam(
        mpnet.parameters(),
        lr=args.lr_mp,
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

    style_generator.load_state_dict(style_dict["g_ema"], strict=False)
    style_discriminator.load_state_dict(style_dict["d"])
    # d_optim.load_state_dict(style_dict["d_optim"])

    assert args.sbg_model is not None
    print("load sbg model:", args.sbg_model)

    sbg_dict = torch.load(args.sbg_model, map_location=lambda storage, loc: storage)

    # sbg_generator.load_state_dict(sbg_dict['generator'])
    sbg_generator.load_state_dict(sbg_dict)

    if args.distributed:
        style_generator = nn.parallel.DistributedDataParallel(
            style_generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        style_discriminator = nn.parallel.DistributedDataParallel(
            style_discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        sbg_generator = nn.parallel.DistributedDataParallel(
            sbg_generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="ffhq test")

    train(args, loader, sbg_generator, style_generator, style_discriminator, mpnet, mp_optim, d_optim, device)
