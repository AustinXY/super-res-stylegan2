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
from finegan_config import finegan_config

# import seaborn as sns
# import matplotlib
import sys
# matplotlib.use('Agg')

os.environ["CUDA_VISIBLE_DEVICES"]="1"

from model import Generator, Discriminator, MappingNetwork, Encoder, G_NET
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


def sample_codes(batch, z_dim, b_dim, p_dim, c_dim, device):
    z = torch.randn(batch, z_dim, device=device)
    c = torch.zeros(batch, c_dim, device=device)
    cid = np.random.randint(c_dim, size=batch)
    for i in range(batch):
        c[i, cid[i]] = 1

    p = child_to_parent(c, c_dim, p_dim)
    b = c.clone()
    return z, b, p, c


def rand_sample_codes(prev_z, prev_b, prev_p, prev_c, rand_code=['b', 'p']):
    '''
    rand code default: keeping z and c
    '''
    device = prev_z.device
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


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="sample fine img")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument(
        "paths", metavar="PATHS", nargs="+", help="path to image files or directory of files to be projected")
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument("--thrld", type=float, default=0.1)
    parser.add_argument('--ds_name', type=str, default='STANFORDCAR',
                        help='dataset used for training finegan (LSUNCAR | CUB | STANFORDCAR)')

    args = parser.parse_args()

    args.z_dim = finegan_config[args.ds_name]['Z_DIM']
    args.b_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']
    args.p_dim = finegan_config[args.ds_name]['SUPER_CATEGORIES']
    args.c_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']

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

    fine_generator = G_NET(ds_name=args.ds_name).to(device)

    style_generator.load_state_dict(ckpt["style_g"])
    style_discriminator.load_state_dict(ckpt["style_d"])
    mpnet.load_state_dict(ckpt["mp"])
    fine_generator.load_state_dict(ckpt["fine"])

    style_generator.eval()
    style_discriminator.eval()
    mpnet.eval()
    fine_generator.eval()

    # with torch.no_grad():
    #     z0, b0, p0, c0 = sample_codes(args.batch, args.z_dim, args.b_dim, args.p_dim, args.c_dim, device)
    #     _, _, _, c1 = rand_sample_codes(z0, b0, p0, c0, rand_code=['c'])
    #     z1, _, _, _ = rand_sample_codes(z0, b0, p0, c1, rand_code=['z'])

    #     img0 = fine_generator(z0, b0, p0, c0)
    #     wp0 = mpnet(img0)
    #     s_img0, _ = style_generator([wp0],
    #         input_is_latent=True,
    #         randomize_noise=False)
    #     imgs = img0
    #     s_imgs = s_img0

    #     # z, b0, p1, c1 = rand_sample_codes(z, b0, p0, c0, rand_code=['p', 'c', 'z'])
    #     img1 = fine_generator(z1, b0, p0, c0)
    #     wp1 = mpnet(img1)
    #     s_img1, _ = style_generator([wp1],
    #         input_is_latent=True,
    #         randomize_noise=False)
    #     imgs = torch.cat((imgs, img1), 0)
    #     s_imgs = torch.cat((s_imgs, s_img1), 0)

    #     # z, b1, p1, c1 = rand_sample_codes(z, b0, p1, c1, rand_code=['b'])
    #     img2 = fine_generator(z1, b0, p0, c1)
    #     wp2 = mpnet(img2)
    #     s_img2, _ = style_generator([wp2],
    #         input_is_latent=True,
    #         randomize_noise=False)
    #     imgs = torch.cat((imgs, img2), 0)
    #     s_imgs = torch.cat((s_imgs, s_img2), 0)

    #     img3 = fine_generator(z0, b0, p0, c1)
    #     wp3 = mpnet(img3)
    #     s_img3, _ = style_generator([wp3],
    #         input_is_latent=True,
    #         randomize_noise=False)
    #     imgs = torch.cat((imgs, img3), 0)
    #     s_imgs = torch.cat((s_imgs, s_img3), 0)

    #     arti_wp = wp0 - wp1 + wp2
    #     arti_img, _ = style_generator([arti_wp],
    #         input_is_latent=True,
    #         randomize_noise=False)

    # utils.save_image(
    #     imgs,
    #     f"map2style/fine.png",
    #     nrow=8,
    #     normalize=True,
    #     range=(-1, 1),
    # )

    # utils.save_image(
    #     s_imgs,
    #     f"map2style/style.png",
    #     nrow=8,
    #     normalize=True,
    #     range=(-1, 1),
    # )

    # utils.save_image(
    #     arti_img,
    #     f"map2style/arti.png",
    #     nrow=8,
    #     normalize=True,
    #     range=(-1, 1),
    # )

    # sys.exit(0)



    with torch.no_grad():
        wp_code = mpnet(imgs)
        style_img, _ = style_generator([wp_code],
            input_is_latent=True,
            randomize_noise=False)

        print(style_discriminator(style_img))

    mean_wp = wp_code.mean(0)
    # sum_diff = torch.zeros_like(wp_code[0])
    # diff = []
    # for i in range(0, wp_code.size(0)):
    #     temp_diff = torch.abs(wp_code[i] - mean_wp)
    #     # temp_diff = torch.where(temp_diff < args.thrld, torch.zeros_like(temp_diff), temp_diff)
    #     sum_diff += temp_diff
    #     diff.append(temp_diff)

    # inv_coords = (sum_diff/batch) <= args.thrld
    # # inv_coords = _sum == 0
    # mask = torch.zeros_like(wp_code[0]).masked_fill(inv_coords, 1.)
    # rev_mask = torch.ones_like(mask) - mask

    # z1 = torch.randn(1, 512, device=device)
    # latent1 = style_generator.style(z1)

    # z2 = torch.randn(1, 512, device=device)
    # latent2 = style_generator.style(z2)

    # rand_img1, _ = style_generator([latent1],
    #     input_is_latent=True,
    #     randomize_noise=False)

    # rand_img2, _ = style_generator([latent2],
    #     input_is_latent=True,
    #     randomize_noise=False)

    # # arti_wp = torch.rand_like(wp_code[0])
    # # arti_wp = latent1 * rev_mask.unsqueeze(0) + (mask * wp_code[0]).unsqueeze(0)

    # arti_wp = latent1 * rev_mask.unsqueeze(0) + latent2 * mask.unsqueeze(0)

    # arti_img, _ = style_generator([arti_wp],
    #     input_is_latent=True,
    #     randomize_noise=False)

    # chanb_wp = latent1 * rev_mask.unsqueeze(0) + (mask * wp_code[0]).unsqueeze(0)
    # chanb_img, _ = style_generator([chanb_wp],
    #     input_is_latent=True,
    #     randomize_noise=False)

    # # print(mask)
    # # for i in range(len(diff)):
    # svm = sns.heatmap(mask.detach().cpu().numpy(), cmap='coolwarm')
    # figure = svm.get_figure()
    # figure.savefig(f'map2style/mask.png', dpi=400)
    # figure.clf()


    mean_img, _ = style_generator([mean_wp.unsqueeze(0)],
        input_is_latent=True,
        randomize_noise=False)

    utils.save_image(
        style_img,
        f"map2style/map.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )

    # utils.save_image(
    #     rand_img1,
    #     f"map2style/rand1.png",
    #     nrow=8,
    #     normalize=True,
    #     range=(-1, 1),
    # )

    # utils.save_image(
    #     rand_img2,
    #     f"map2style/rand2.png",
    #     nrow=8,
    #     normalize=True,
    #     range=(-1, 1),
    # )

    utils.save_image(
        mean_img,
        f"map2style/mean.png",
        nrow=8,
        normalize=True,
        range=(-1, 1),
    )

    # utils.save_image(
    #     chanb_img,
    #     f"map2style/chanb.png",
    #     nrow=8,
    #     normalize=True,
    #     range=(-1, 1),
    # )