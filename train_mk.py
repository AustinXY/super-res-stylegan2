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


def train(args, loader, device, segnet, mknet, mk_optim):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    loss_dict = {}

    if args.distributed:
        mk_module = mknet.module
    else:
        mk_module = mknet

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

        loss_reduced = reduce_loss_dict(loss_dict)

        mk_loss_val = loss_reduced["mk"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"mk: {mk_loss_val:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "MK": mk_loss_val,
                    }
                )

            if i % 500 == 0:
                    utils.save_image(
                        fake_mask,
                        f"sample/{str(i).zfill(6)}_mk.png",
                        nrow=8,
                        normalize=True,
                        range=(0, 1),
                    )

                    if wandb and args.wandb:
                        wandb.log(
                            {
                                "mask": [wandb.Image(Image.open(f"sample/{str(i).zfill(6)}_mk.png").convert("RGB"))],
                            }
                        )

            if i % 100000 == 0 and i != args.start_iter:
                torch.save(
                    {
                        "mk": mk_module.state_dict(),
                        "mk_optim": mk_optim.state_dict(),
                        "args": args,
                        "cur_itr": i
                    },
                    f"checkpoint/{str(i).zfill(6)}_mk.pt",
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

    parser.add_argument("--mk", type=float, default=10, help="mse weight")

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

    mk_optim = optim.Adam(
        mknet.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        args.start_iter = ckpt['cur_itr']
        mknet.load_state_dict(ckpt['mk'])
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
        wandb.init(project="mk")

    train(args, loader, device, segnet, mknet, mk_optim)
