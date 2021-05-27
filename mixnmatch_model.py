import math
import random
import functools
import operator
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from finegan_config import finegan_config

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


def sameBlock(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

## mxnmch g ###################################################################


class GET_IMAGE(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class GET_MASK(nn.Module):
    def __init__(self, ngf):
        super(GET_MASK, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 1),
            # nn.Conv2d(ngf, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class BACKGROUND_STAGE(nn.Module):
    def __init__(self, ngf, ds_name='CUB'):
        super().__init__()

        self.ngf = ngf
        in_dim = finegan_config[ds_name]['Z_DIM'] + finegan_config[ds_name]['FINE_GRAINED_CATEGORIES']

        self.fc = nn.Sequential( nn.Linear(in_dim, ngf*4*4 * 2, bias=False), nn.BatchNorm1d(ngf*4*4 * 2), GLU())
        # 1024*4*4
        self.upsample1 = upBlock(ngf, ngf // 2)
        # 512*8*8
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # 256*16*16
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # 128*32*32
        self.upsample4 = upBlock(ngf // 8, ngf // 8)
        # 128*64*64
        self.upsample5 = upBlock(ngf // 8, ngf // 16)
        # 64*128*128

    def forward(self, z_input, input):
        in_code = torch.cat([z_input, input], dim=1)
        out_code = self.fc(in_code).view(-1, self.ngf, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)
        return out_code


class PARENT_STAGE(nn.Module):
    def __init__(self, ngf, ds_name='CUB'):
        super().__init__()

        self.ngf = ngf
        in_dim = finegan_config[ds_name]['Z_DIM'] + finegan_config[ds_name]['SUPER_CATEGORIES']
        self.code_len = finegan_config[ds_name]['SUPER_CATEGORIES']

        self.fc = nn.Sequential(nn.Linear(in_dim, ngf*4*4 * 2, bias=False), nn.BatchNorm1d(ngf*4*4 * 2), GLU())
        # 512*4*4
        self.upsample1 = upBlock(ngf, ngf//2)
        # 256*8*8
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # 128*16*16
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # 64*32*32
        self.upsample4 = upBlock(ngf // 8, ngf // 32)
        # 16*64*64
        self.upsample5 = upBlock(ngf // 32, ngf // 32)
        # 16*128*128
        self.jointConv = sameBlock(finegan_config[ds_name]['SUPER_CATEGORIES']+ngf//32, ngf//32)
        # (16+20)*128*128 --> 16*128*128
        self.residual = self._make_layer(3, ngf//32)
        # 16*128*128

    def _make_layer(self,num_residual,ngf):
        layers = []
        for _ in range(num_residual):
            layers.append( ResBlock(ngf) )
        return nn.Sequential(*layers)

    def forward(self, z_input, input):
        in_code = torch.cat([z_input, input], dim=1)
        out_code = self.fc(in_code).view(-1, self.ngf, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)

        h, w  = out_code.size(2),out_code.size(3)
        input = input.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
        out_code = torch.cat( (out_code, input), dim=1)
        out_code = self.jointConv(out_code)
        out_code = self.residual(out_code)
        return out_code


class CHILD_STAGE(nn.Module):
    def __init__(self, ngf, num_residual=2, ds_name='CUB'):
        super().__init__()

        self.ngf = ngf
        self.code_len = finegan_config[ds_name]['FINE_GRAINED_CATEGORIES']
        self.num_residual = num_residual

        self.jointConv = sameBlock( self.code_len+self.ngf, ngf*2 )
        self.residual = self._make_layer()
        self.samesample = sameBlock(ngf*2, ngf)

    def _make_layer(self):
        layers = []
        for _ in range(self.num_residual):
            layers.append( ResBlock(self.ngf*2) )
        return nn.Sequential(*layers)

    def forward(self, h_code, code):
        h, w  = h_code.size(2),h_code.size(3)
        code = code.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
        h_c_code = torch.cat((code, h_code), 1)

        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code


class G_NET(nn.Module):
    def __init__(self, ds_name='CUB'):
        super(G_NET, self).__init__()

        ngf = finegan_config[ds_name]['GF_DIM']
        self.scale_fimg = nn.UpsamplingBilinear2d(size=[126, 126])

        # Background stage
        self.background_stage = BACKGROUND_STAGE( ngf*8 )
        self.background_image = GET_IMAGE(ngf//2)

        # Parent stage networks
        self.parent_stage = PARENT_STAGE( ngf*8 )
        self.parent_image = GET_IMAGE( ngf//4 )
        self.parent_mask = GET_MASK( ngf//4 )

        # Child stage networks
        self.child_stage = CHILD_STAGE( ngf//4 )
        self.child_image = GET_IMAGE( ngf//4 )
        self.child_mask = GET_MASK( ngf//4 )

    def forward(self, z_code, b_code, p_code, c_code, z_fg=None, rtn_type='fnl', rtn_mk=False):
        fake_imgs = []  # Will contain [background image, parent image, child image]
        fg_imgs = []  # Will contain [parent foreground, child foreground]
        mk_imgs = []  # Will contain [parent mask, child mask]
        fg_mk = []  # Will contain [masked parent foreground, masked child foreground]

        if z_fg is None:
            z_fg = z_code.clone()

        # Background stage
        temp = self.background_stage( z_code, b_code )
        fake_img1 = self.background_image( temp )  # Background image
        fake_img1_126 = self.scale_fimg(fake_img1)
        fake_imgs.append(fake_img1_126)

        # Parent stage
        p_temp = self.parent_stage(z_fg, p_code)
        fake_img2_foreground = self.parent_image(p_temp)  # Parent foreground
        fake_img2_mask = self.parent_mask(p_temp)  # Parent mask
        fg_masked2 = fake_img2_foreground*fake_img2_mask # masked_parent
        fake_img2 = fg_masked2 + fake_img1*(1-fake_img2_mask)  # Parent image
        fg_mk.append(fg_masked2)
        fake_imgs.append(fake_img2)
        fg_imgs.append(fake_img2_foreground)
        mk_imgs.append(fake_img2_mask)

        # Child stage
        temp = self.child_stage(p_temp, c_code)
        fake_img3_foreground = self.child_image(temp)  # Child foreground
        fake_img3_mask = self.child_mask(temp)  # Child mask
        fg_masked3 = torch.mul(fake_img3_foreground, fake_img3_mask) # masked child
        fake_img3 = fg_masked3 + fake_img2*(1-fake_img3_mask)  # Child image
        fg_mk.append(fg_masked3)
        fake_imgs.append(fake_img3)
        fg_imgs.append(fake_img3_foreground)
        mk_imgs.append(fake_img3_mask)

        if rtn_mk:
            return fake_img3, fake_img2_mask

        rtn = fake_img3
        if rtn_type == 'pmk':
            rtn = fake_img2_mask
        elif rtn_type == 'cmk':
            rtn = fake_img3_mask
        elif rtn_type == 'bg':
            rtn = fake_img1
        elif rtn_type == 'cmsk':
            rtn = fg_masked3

        return rtn
