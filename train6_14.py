import argparse
import math
import random
import os
import gc
import copy
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.transforms.functional import convert_image_dtype

from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

from model import Generator, Discriminator, UNet

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
    #     return torch.randn(1, batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device)

    return noises


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

def get_mask(model, img, unnorm=False):
    if unnorm:
        img = un_normalize(img)
    normalized_batch = transforms.functional.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    output = model(normalized_batch)['out']
    # sem_classes = [
    #     '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    # ]
    # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    # cid = sem_class_to_idx['car']

    normalized_masks = torch.nn.functional.softmax(output, dim=1)
    bool_mk = (normalized_masks.argmax(1) == 7)
    return bool_mk.float().unsqueeze(1)


def norm_ip(img, low, high):
    img_ = img.clamp(min=low, max=high)
    img_.sub_(low).div_(max(high - low, 1e-5))
    return img_

def norm_range(t, value_range=(-1,1)):
    if value_range is not None:
        return norm_ip(t, value_range[0], value_range[1])
    else:
        return norm_ip(t, float(t.min()), float(t.max()))


def un_normalize(img):
    img_ = norm_range(img).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
    return convert_image_dtype(img_, dtype=torch.float)


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, segnet, mknet, mk_optim):
    loader = sample_data(loader)

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
        d_module = discriminator.module
        mk_module = mknet.module
    else:
        g_module = generator
        d_module = discriminator
        mk_module = mknet

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(
            args.ada_target, args.ada_length, 8, device)

    args.injidx = None
    segnet.requires_grad_(False)
    mknet.train()

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        ############# train mask network #############
        requires_grad(mknet, True)

        real_mask = get_mask(segnet, real_img, unnorm=True)
        fake_mask = mknet(real_img)

        mk_loss = F.mse_loss(fake_mask, real_mask)

        loss_dict["mk"] = mk_loss

        mknet.zero_grad()
        mk_loss.backward()
        mk_optim.step()

        ############# train discriminator network #############
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise, inject_index=args.injidx)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        ############# train generator network #############
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, weights = generator(noise, inject_index=args.injidx, return_weights=True)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        neg_loss = torch.tensor(0.0, device=device)
        for w in weights:
            oc = w.size(1)
            ic = w.size(2)

            neg_loss += torch.sum(F.relu(torch.sum(w[:, 0:oc//2, ic//2:], dim=(-1,-2))))
            neg_loss += torch.sum(F.relu(torch.sum(w[:, oc//2:, 0:ic//2], dim=(-1,-2))))

        loss_dict["g"] = g_loss
        loss_dict['neg'] = neg_loss

        loss = g_loss + neg_loss * args.neg

        generator.zero_grad()
        loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True, inject_index=args.injidx)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        # ############# guide disentangle #############
        requires_grad(generator, True)
        requires_grad(mknet, False)

        guide_regularize = i % args.guide_reg_every == 0
        # guide_regularize = False
        if guide_regularize:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
            style_img, outs = generator(noise, inject_index=args.injidx, return_outs=True)

            mask = mknet(style_img)
            sep_loss = torch.tensor(0.0, device=device)
            for o in outs[7:]:
                n_channel = o.size(1)
                _mask = F.interpolate(mask, size=(o.size(-2), o.size(-1)), mode='area')
                _rmask = torch.ones_like(_mask) - _mask

                g1 = o[:, 0:n_channel//2]
                g2 = o[:, n_channel//2:n_channel]

                sep_loss += F.mse_loss(g1*_mask, g1)
                sep_loss += F.mse_loss(g2*_rmask, g2)

            loss_dict["sep"] = sep_loss

            loss = sep_loss * args.sep

            generator.zero_grad()
            loss.backward()
            g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()
        neg_loss_val = loss_reduced["neg"].mean().item()
        sep_loss_val = loss_reduced["sep"].item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                        "weight neg loss": neg_loss_val,
                        "separation loss": sep_loss_val,
                    }
                )

            if i % 500 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    img_noise = g_module.make_noise()

                    noise = mixing_noise(8, args.latent, args.mixing, device)
                    style_img1, ssc1 = g_ema(noise, inject_index=args.injidx, noise=img_noise, return_ssc=True)

                    noise = mixing_noise(8, args.latent, args.mixing, device)
                    style_img2, ssc2 = g_ema(noise, inject_index=args.injidx, noise=img_noise, return_ssc=True)

                    ssc3 = copy.deepcopy(ssc1)
                    for l in range(len(ssc3)):
                        channel = ssc3[l].size(1)
                        ssc3[l][:, channel//2:] = ssc2[l][:, channel//2:]

                    style_img3, _ = g_ema(ssc3, input_is_ssc=True, inject_index=args.injidx, noise=img_noise)

                    utils.save_image(
                        style_img1,
                        f"sample/{str(i).zfill(6)}_0.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        style_img2,
                        f"sample/{str(i).zfill(6)}_1.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    utils.save_image(
                        style_img3,
                        f"sample/{str(i).zfill(6)}_2.png",
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )

                    if wandb and args.wandb:
                        wandb.log(
                            {
                                "style image1": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_0.png").convert("RGB"))],
                                "style image2": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_1.png").convert("RGB"))],
                                "mix style image": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_2.png").convert("RGB"))],
                            }
                        )

            if i % 10000 == 0 and i != args.start_iter:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                        "cur_itr": i
                    },
                    f"checkpoint/{str(i).zfill(6)}_6_14.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
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
        default=4,
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
        default=8,
        help="interval of the applying path length regularization",
    )
    # parser.add_argument(
    #     "--mse_reg_every",
    #     type=int,
    #     default=10,
    #     help="interval of the applying path length regularization",
    # )
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
    parser.add_argument("--lr", type=float, default=0.002,
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

    parser.add_argument("--kl_lambda", type=float, default=0.01)
    parser.add_argument("--mse", type=float, default=4, help="mse weight")
    parser.add_argument("--dis1", type=float, default=0.2, help="mse weight")
    parser.add_argument("--dis2", type=float, default=0.5, help="mse weight")

    parser.add_argument("--neg", type=float, default=10, help="mse weight")
    parser.add_argument("--mk", type=float, default=10, help="mse weight")
    parser.add_argument("--sep", type=float, default=1000, help="mse weight")


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

    args.start_iter = 0

    segnet = deeplabv3_resnet101(pretrained=True, progress=False).to(device)
    segnet.eval()

    mknet = UNet(
        n_channels=3,
        n_classes=1,
        bilinear=True,
    ).to(device)

    generator = Generator(
        size=args.size,
        style_dim=args.latent,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        sep_mode=True,
        negative_slope=0.0001
    ).to(device)

    discriminator = Discriminator(
        size=args.size,
        channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema = Generator(
        size=args.size,
        style_dim=args.latent,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        sep_mode=True,
        negative_slope=0.0001
    ).to(device)

    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    mk_optim = optim.Adam(
        mknet.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        args.start_iter = ckpt['cur_itr']
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])
        mknet.load_state_dict(ckpt['mk'])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        mk_optim.load_state_dict(ckpt['mk_optim'])

    if args.distributed:

        segnet = nn.parallel.DistributedDataParallel(
            segnet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        mknet = nn.parallel.DistributedDataParallel(
            mknet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
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
        sampler=data_sampler(dataset, shuffle=True,
                             distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="guide 6_14")

    train(args, loader, generator, discriminator,
          g_optim, d_optim, g_ema, device, segnet, mknet, mk_optim)
