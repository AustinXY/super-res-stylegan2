import argparse
import os


import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision import transforms, utils

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# from model import G_NET
from mixnmatch_model import G_NET
from finegan_config import finegan_config


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
    z_fg = torch.randn(batch, z_dim, device=device)
    c = torch.zeros(batch, c_dim, device=device)
    cid = np.random.randint(c_dim, size=batch)
    for i in range(batch):
        c[i, cid[i]] = 1

    p = child_to_parent(c, c_dim, p_dim)
    b = c.clone()
    return z, b, p, c, z_fg


def rand_sample_codes(prev_z, prev_b, prev_p, prev_c, rand_code=['z', 'b', 'p', 'c']):
    batch = prev_z.size(0)
    device = prev_z.device

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


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="sample fine img")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to finegan",
    )
    parser.add_argument(
        "--rand_code",
        type=str,
        default='z',
        nargs='+',
        help="path to finegan",
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample", type=int, default=8, help="number of sample for each batch item"
    )
    parser.add_argument(
        "--ds_name", type=str, default='STANFORDCAR', help="name of dataset"
    )
    parser.add_argument(
        "--rtn_img", type=str, default='fnl', help="return image type"
    )
    args = parser.parse_args()

    args.z_dim = finegan_config[args.ds_name]['Z_DIM']
    args.b_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']
    args.p_dim = finegan_config[args.ds_name]['SUPER_CATEGORIES']
    args.c_dim = finegan_config[args.ds_name]['FINE_GRAINED_CATEGORIES']

    fine_generator = G_NET(args.ds_name).to(device)
    # fine_generator = torch.nn.DataParallel(fine_generator)
    # print(fine_generator)

    assert args.ckpt is not None
    print("load fine model:", args.ckpt)

    fine_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    fine_generator.load_state_dict(fine_dict)

    print(args.rand_code)

    #########
    fine_generator.eval()
    with torch.no_grad():
        img_li = []
        z, b, p, c, z_fg = sample_codes(args.batch, args.z_dim,
                                args.b_dim, args.p_dim, args.c_dim, device)
        for i in range(args.n_sample):
            # p = manual_sample_codes(p, [i+10])
            fine_img = fine_generator(z_fg, b, p, c, z_fg=z)
            # fine_img = fine_generator(z, b, p, c, rtn_img='pmk')
            img_li.append(fine_img)
            z_fg, b, p, c = rand_sample_codes(z_fg, b, p, c, rand_code=args.rand_code)

        fnl_img = None

        _range = (-1, 1)
        if 'mk' in args.rtn_img:
            _range = (0, 1)

        for i in range(len(img_li)):

            if fnl_img is None:
                fnl_img = img_li[i]
            else:
                fnl_img = torch.cat([fnl_img, img_li[i]])

            utils.save_image(
                img_li[i],
                f"fine_sample/fine{str(i)}.png",
                nrow=8,
                normalize=True,
                range=_range,
            )

        utils.save_image(
            fnl_img,
            f"fine_sample/fine.png",
            nrow=8,
            normalize=True,
            range=_range,
        )
