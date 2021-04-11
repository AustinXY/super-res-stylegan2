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
from pulse import PULSE

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
    parser.add_argument("--mse", type=float, default=1, help="weight of the mse loss")
    parser.add_argument("--adv", type=float, default=1, help="weight of the adv loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    ## parse argument
    parser.add_argument('--seed', type=int, help='manual seed to use')
    parser.add_argument('--loss_str', type=str,
                        default="100*L2+0.05*GEOCROSS", help='Loss function to use')
    parser.add_argument('--eps', type=float, default=1e-3,
                        help='Target for downscaling loss (L2)')
    parser.add_argument('--noise_type', type=str,
                        default='trainable', help='zero, fixed, or trainable')
    parser.add_argument('--num_trainable_noise_layers', type=int,
                        default=5, help='Number of noise layers to optimize')
    parser.add_argument('--tile_latent', action='store_true',
                        help='Whether to forcibly tile the same latent 18 times')
    parser.add_argument('--bad_noise_layers', type=str, default="17",
                        help='List of noise layers to zero out to improve image quality')
    parser.add_argument('--opt_name', type=str, default='adam',
                        help='Optimizer to use in projected gradient descent')
    parser.add_argument('--learning_rate', type=float, default=0.4,
                        help='Learning rate to use during optimization')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of optimization steps')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Whether to store and save intermediate HR and LR images during optimization')
    parser.add_argument('--verbose', action='store_false',
                        help='using verbose mode')
    args = parser.parse_args()

    # resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            # transforms.Resize(resize),
            # transforms.CenterCrop(resize),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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

    model = PULSE(g_ema, device, verbose=args.verbose)

    discriminator = Discriminator(args.size).to(device)
    discriminator.load_state_dict(torch.load(args.ckpt)["d"])
    discriminator.eval()
    discriminator.requires_grad_(False)

    img_gen = model(
        ref_im = imgs,
        seed = args.seed,
        loss_str = args.loss_str,
        eps = args.eps,
        noise_type = args.noise_type,
        num_trainable_noise_layers = args.num_trainable_noise_layers,
        tile_latent = args.tile_latent,
        bad_noise_layers = args.bad_noise_layers,
        opt_name = args.opt_name,
        learning_rate = args.learning_rate,
        steps = args.steps,
        lr_schedule = args.lr_schedule,
        save_intermediate = args.save_intermediate)

    with torch.no_grad():
        print(discriminator(img_gen))

    utils.save_image(
        img_gen,
        f"pulse_project/project.png",
        nrow=8,
        normalize=True,
        range=(0, 1),
    )
