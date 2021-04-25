import argparse
import os

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

os.environ["CUDA_VISIBLE_DEVICES"]="1"

from model import Generator, Discriminator, MappingNetwork, Encoder
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

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


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


def train(args, loader, fine_img, style_generator, style_discriminator, mpnet, device):
    # loader = sample_data(loader)

    pbar = range(args.training_steps)
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

    n_mean_latent = 10000
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = style_generator.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    noises_single = style_generator.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    latent_in = mpnet(fine_img).detach().clone()
    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.training_steps))
    latent_path = []

    for i in pbar:
        t = i / args.training_steps
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        img_gen, _ = style_generator([latent_n], input_is_latent=True, noise=noises)
        fake_pred = style_discriminator(img_gen)

        batch, channel, height, width = img_gen.shape

        if height > 128:
            factor = height // 128

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)
        l1_loss = F.l1_loss(img_gen, imgs)
        adv_loss = g_nonsaturating_loss(fake_pred)
        geo_loss = loss_geocross(latent_in, style_generator.n_latent)

        loss = args.vgg * p_loss + args.noise_regularize * n_loss + args.mse * \
            mse_loss + args.adv * adv_loss + args.l1 * l1_loss + args.geo * geo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {args.vgg * p_loss.item():.4f}; adversarial: {args.adv * adv_loss.item():.4f}; geo: {args.geo * geo_loss.item():.4f}; "
                f"noise regularize: {args.noise_regularize * n_loss.item():.4f}; mse: {args.mse * mse_loss.item():.4f}; l1: {args.l1 * l1_loss.item():.4f}; lr: {lr:.4f}"
            )
        )

        if i % int(args.training_steps//10) == 0:
                utils.save_image(
                    img_gen,
                    f"map2style/{str(i).zfill(6)}.png",
                    nrow=8,
                    normalize=True,
                    range=(-1, 1),
                )

                utils.save_image(
                    img_gen,
                    f"map2style/curr.png",
                    nrow=8,
                    normalize=True,
                    range=(-1, 1),
                )

    return img_gen


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="sample fine img")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument(
        "paths", metavar="PATHS", nargs="+", help="path to image files or directory of files to be projected")
    parser.add_argument(
        "--training_steps", type=int, default=0, help="number of training steps for the input batch of images")
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus")

    # training arguments if use training mode
    parser.add_argument("--lr", type=float, default=1e-1, help="mapping network learning rate")
    parser.add_argument("--lr_d", type=float, default=2e-4, help="discriminator learning rate")
    parser.add_argument("--lmdb", type=str, help="path to the lmdb dataset")
    # parser.add_argument("--load_optim", action="store_true", help="load optimiser or create new ones")
    parser.add_argument("--vgg", type=float, default=0, help="weight of the vgg loss")
    parser.add_argument("--mse", type=float, default=10, help="weight of the mse loss")
    parser.add_argument("--geo", type=float, default=0, help="weight of the geo loss")
    parser.add_argument("--l1", type=float, default=0, help="weight of the l1 loss")
    parser.add_argument("--adv", type=float, default=1, help="weight of the adv loss")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    args = parser.parse_args()

    resize = 128

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    if os.path.isdir(args.paths[0]):
        img_files = [os.path.join(args.paths[0], f) for f in os.listdir(
            args.paths[0]) if '.png' in f]
    else:
        img_files = args.paths

    imgs = []
    for imgfile in sorted(img_files):
        img = transform(Image.open(imgfile).convert("RGB"))
        if img.size()[1:3] == (128, 128):
            imgs.append(img)
            if len(imgs) >= args.batch:
                break

    imgs = torch.stack(imgs, 0).to(device)
    batch = imgs.size(0)

    assert args.ckpt is not None
    print("load model:", args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    train_args = ckpt['args']

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

    style_generator.load_state_dict(ckpt["style_g"])
    style_discriminator.load_state_dict(ckpt["style_d"])
    mpnet.load_state_dict(ckpt["mp"])

    style_generator.eval()
    style_discriminator.eval()
    mpnet.eval()

    with torch.no_grad():
        wp_code = mpnet(imgs)
        style_img, _ = style_generator([wp_code],
            input_is_latent=True,
            randomize_noise=False)

        print(style_discriminator(style_img))

    if args.training_steps > 0:
        from dataset import MultiResolutionDataset
        from torch.utils import data

        # d_optim = optim.Adam(
        #     style_discriminator.parameters(),
        #     lr=args.lr_d,
        #     betas=(0, 0.99),
        # )

        # if args.load_optim:
        #     mp_optim.load_state_dict(ckpt["mp_optim"])
        #     d_optim.load_state_dict(ckpt["d_optim"])

        # transform = transforms.Compose(
        #     [
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        #     ]
        # )

        # dataset = MultiResolutionDataset(args.lmdb, transform, train_args.size)
        # loader = data.DataLoader(
        #     dataset,
        #     batch_size=batch,
        #     sampler=data_sampler(dataset, shuffle=True, distributed=False),
        #     drop_last=True,
        # )
        loader = None

        style_img = train(args, loader, imgs, style_generator, style_discriminator, mpnet, device)

    utils.save_image(
        style_img,
        f"map2style/map.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )