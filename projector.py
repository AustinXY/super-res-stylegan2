import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, utils
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator, Discriminator


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


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


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

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--vgg", type=float, default=0, help="weight of the vgg loss")
    parser.add_argument("--mse", type=float, default=1, help="weight of the mse loss")
    parser.add_argument("--geo", type=float, default=0, help="weight of the geo loss")
    parser.add_argument("--l1", type=float, default=0, help="weight of the l1 loss")
    parser.add_argument("--adv", type=float, default=0, help="weight of the adv loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, 128)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    discriminator = Discriminator(args.size).to(device)
    discriminator.load_state_dict(torch.load(args.ckpt)["d"])
    discriminator.eval()
    discriminator.requires_grad_(False)

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(args.step))
    latent_path = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())
        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)
        fake_pred = discriminator(img_gen)

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
        geo_loss = loss_geocross(latent_in, g_ema.n_latent)

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

    img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)


    with torch.no_grad():
        print(discriminator(img_gen))

    filename = 'style_project/' + os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"

    img_ar = make_image(img_gen)

    result_file = {}
    result_img = None
    for i, input_name in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i : i + 1])

        result_file[input_name] = {
            "img": img_gen[i],
            "latent": latent_in[i],
            "noise": noise_single,
        }

        img_name = 'style_project/' + os.path.splitext(os.path.basename(input_name))[0] + "-project.png"
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)

    utils.save_image(
        img_gen,
        f"style_project/project.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )
    torch.save(result_file, filename)
