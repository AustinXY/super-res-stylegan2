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


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor,
                        down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1,
                        down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(
                pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(
                    0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(
                    input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(
            in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(
                f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation *
                    (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(
                1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class _ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(_ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4],
                        activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


# ############## fine G networks ################################################
# Upsale the spatial size by a factor of 2

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


def convlxl(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=13, stride=1,
                     padding=1, bias=False)


def child_to_parent(child_c_code, classes_child, classes_parent):
    ratio = classes_child / classes_parent
    arg_parent = torch.argmax(child_c_code, dim=1) / ratio
    parent_c_code = torch.zeros([child_c_code.size(0), classes_parent]).cuda()
    for i in range(child_c_code.size(0)):
        parent_c_code[i][arg_parent[i].type(torch.LongTensor)] = 1
    return parent_c_code


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

# Keep the spatial size


def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


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


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, c_flag, ds_name='CUB'):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.c_flag = c_flag
        if self.c_flag == 1:
            self.in_dim = finegan_config[ds_name]['Z_DIM'] + finegan_config[ds_name]['SUPER_CATEGORIES']
        elif self.c_flag == 2:
            self.in_dim = finegan_config[ds_name]['Z_DIM'] + finegan_config[ds_name]['FINE_GRAINED_CATEGORIES']

        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.upsample5 = upBlock(ngf // 16, ngf // 16)

    def forward(self, z_code, code):
        in_code = torch.cat((code, z_code), 1)
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)
        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, use_hrc=1, num_residual=2, ds_name='CUB'):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if use_hrc == 1:  # For parent stage
            self.ef_dim = finegan_config[ds_name]['SUPER_CATEGORIES']

        else:            # For child stage
            self.ef_dim = finegan_config[ds_name]['FINE_GRAINED_CATEGORIES']

        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim
        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.samesample = sameBlock(ngf, ngf // 2)

    def forward(self, h_code, code):
        s_size = h_code.size(2)
        code = code.view(-1, self.ef_dim, 1, 1)
        code = code.repeat(1, 1, s_size, s_size)
        h_c_code = torch.cat((code, h_code), 1)
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code


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

class ____G_NET(nn.Module):
    def __init__(self, ds_name='CUB'):
        super(G_NET, self).__init__()
        self.gf_dim = finegan_config[ds_name]['GF_DIM']
        self.ds_name = ds_name
        self.define_module()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')
        self.scale_fimg = nn.UpsamplingBilinear2d(size=[126, 126])

    def define_module(self):
        #Background stage
        self.h_net1_bg = INIT_STAGE_G(self.gf_dim * 16, 2, ds_name=self.ds_name)
        # Background generation network
        self.img_net1_bg = GET_IMAGE(self.gf_dim)

        # Parent stage networks
        self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, 1, ds_name=self.ds_name)
        self.h_net2 = NEXT_STAGE_G(self.gf_dim, use_hrc=1, ds_name=self.ds_name)
        # Parent foreground generation network
        self.img_net2 = GET_IMAGE(self.gf_dim // 2)
        # Parent mask generation network
        self.img_net2_mask = GET_MASK(self.gf_dim // 2)

        # Child stage networks
        self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2, use_hrc=0, ds_name=self.ds_name)
        # Child foreground generation network
        self.img_net3 = GET_IMAGE(self.gf_dim // 4)
        # Child mask generation network
        self.img_net3_mask = GET_MASK(self.gf_dim // 4)

    def forward(self, z_code, bg_code, p_code, c_code, z_fg=None, rtn_type='fnl', rtn_mk=False):

        fake_imgs = []  # Will contain [background image, parent image, child image]
        fg_imgs = []  # Will contain [parent foreground, child foreground]
        mk_imgs = []  # Will contain [parent mask, child mask]
        fg_mk = []  # Will contain [masked parent foreground, masked child foreground]

        #Background stage
        h_code1_bg = self.h_net1_bg(z_code, bg_code)
        fake_img1 = self.img_net1_bg(h_code1_bg)  # Background image
        # Resizing fake background image from 128x128 to the resolution which background discriminator expects: 126 x 126.
        fake_img1_126 = self.scale_fimg(fake_img1)
        fake_imgs.append(fake_img1_126)

        #Parent stage
        h_code1 = self.h_net1(z_fg, p_code)
        h_code2 = self.h_net2(h_code1, p_code)
        fake_img2_foreground = self.img_net2(h_code2)  # Parent foreground
        fake_img2_mask = self.img_net2_mask(h_code2)  # Parent mask
        ones_mask_p = torch.ones_like(fake_img2_mask)
        opp_mask_p = ones_mask_p - fake_img2_mask
        fg_masked2 = torch.mul(fake_img2_foreground, fake_img2_mask)
        fg_mk.append(fg_masked2)
        bg_masked2 = torch.mul(fake_img1, opp_mask_p)
        fake_img2_final = fg_masked2 + bg_masked2  # Parent image
        fake_imgs.append(fake_img2_final)
        fg_imgs.append(fake_img2_foreground)
        mk_imgs.append(fake_img2_mask)

        #Child stage
        h_code3 = self.h_net3(h_code2, c_code)
        fake_img3_foreground = self.img_net3(h_code3)  # Child foreground
        fake_img3_mask = self.img_net3_mask(h_code3)  # Child mask
        ones_mask_c = torch.ones_like(fake_img3_mask)
        opp_mask_c = ones_mask_c - fake_img3_mask
        fg_masked3 = torch.mul(fake_img3_foreground, fake_img3_mask)
        fg_mk.append(fg_masked3)
        bg_masked3 = torch.mul(fake_img2_final, opp_mask_c)
        fake_img3_final = fg_masked3 + bg_masked3  # Child image
        fake_imgs.append(fake_img3_final)
        fg_imgs.append(fake_img3_foreground)
        mk_imgs.append(fake_img3_mask)

        if rtn_mk:
            return fake_img3_final, fake_img2_mask

        # return fake_imgs, fg_imgs, mk_imgs, fg_mk
        rtn = fake_img3_final
        if rtn_type == 'pmk':
            rtn = fake_img2_mask
        elif rtn_type == 'cmk':
            rtn = fake_img3_mask
        elif rtn_type == 'bg':
            rtn = fake_img1
        return rtn, None

#----------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = nn.Sequential(
            ConvLayer(in_channel, out_channel, 3, activate=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            ConvLayer(out_channel, out_channel, 3,
                      activate=False, downsample=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = torch.sigmoid(self.outc(x))
        return logits


class Encoder(nn.Module):
    def __init__(self, size, num_ws, img_channels=3, w_dim=512):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }

        self.w_dim = w_dim
        self.n_latents = num_ws

        log_size = int(math.log(size, 2))
        convs = [ConvLayer(img_channels, channels[size], 1)]

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(_ResBlock(in_channel, out_channel))
            in_channel = out_channel

        convs.append(EqualConv2d(in_channel, self.n_latents*self.w_dim, 4, padding=0, bias=False))

        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        out = self.convs(input)
        return out.view(len(input), self.n_latents, self.w_dim)


class LinearModule(nn.Module):
    def __init__(self, in_channel, out_channel, activation=True, normalize=False):
        super().__init__()
        if normalize:
            self.fc = nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif activation:
            self.fc = nn.Sequential(
                nn.Linear(in_channel, out_channel),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.fc = nn.Linear(in_channel, out_channel)
        # self.fc = EqualLinear(in_channel, out_channel, activation="fused_lrelu")
    def forward(self, x):
        x = self.fc(x)
        return x


# class LinearDown(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#         self.fc = LinearModule(in_channel, out_channel)
#     def forward(self, x):
#         x = self.fc(x)
#         return x


# class LinearUp(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#         self.up = LinearModule(in_channel, out_channel)
#         # self.down = LinearModule(out_channel*2, out_channel)
#     def forward(self, x, _x):
#         x = self.up(x)
#         # x = self.down(torch.cat((x, _x), dim=1))
#         return x


# decompose w_plus into variance code and invariance code
class Decomposer(nn.Module):
    def __init__(self, num_ws, w_dim, vc_dim, ivc_dim):
        super().__init__()
        self.vc_dim = vc_dim
        self.ivc_dim = ivc_dim
        full_dim = num_ws * w_dim
        self.fc0 = LinearModule(full_dim, full_dim)
        self.fc1 = LinearModule(full_dim, full_dim//4)
        self.fc2 = LinearModule(full_dim//4, full_dim//16)
        self.fc3 = LinearModule(full_dim//16, vc_dim+ivc_dim)

    def forward(self, w):
        x = self.fc0(w.view(w.size(0), -1))  # // 1
        x = self.fc1(x)  # // 4
        x = self.fc2(x)  # // 16
        x = self.fc3(x)  # fnl

        vc = x[:, 0: self.vc_dim]
        ivc = x[:, self.vc_dim: self.vc_dim+self.ivc_dim]
        return vc, ivc

# compose variance code and invariance code into w_plus
class Composer(nn.Module):
    def __init__(self, num_ws, w_dim, vc_dim, ivc_dim):
        super().__init__()
        self.num_ws = num_ws
        self.w_dim = w_dim
        full_dim = num_ws * w_dim
        self.fc0 = LinearModule(vc_dim+ivc_dim, full_dim//16)
        self.fc1 = LinearModule(full_dim//16, full_dim//4)
        self.fc2 = LinearModule(full_dim//4, full_dim)
        self.fc3 = LinearModule(full_dim, full_dim)

    def forward(self, vc, ivc):
        x = self.fc0(torch.cat((vc, ivc), dim=1))  # // 16
        x = self.fc1(x)  # // 4
        x = self.fc2(x)  # // 1
        x = self.fc3(x)  # // 1
        return x.view(vc.size(0), self.num_ws, self.w_dim)




############# only distil color coding ##################
# distill variance code from w_plus
class Distiller(nn.Module):
    def __init__(self, num_ws, w_dim, vc_dim):
        super().__init__()
        self.vc_dim = vc_dim
        full_dim = num_ws * w_dim
        self.fc0 = LinearModule(full_dim, full_dim)
        self.fc1 = LinearModule(full_dim, full_dim//4)
        self.fc2 = LinearModule(full_dim//4, full_dim//16)
        self.fc3 = LinearModule(full_dim//16, vc_dim, normalize=False)

    def forward(self, w):
        x = self.fc0(w.view(w.size(0), -1))
        x = self.fc1(x)
        x = self.fc2(x)
        vc = self.fc3(x)
        return vc

# mix variance code into w_plus
class Mixer(nn.Module):
    def __init__(self, num_ws, w_dim, vc_dim):
        super().__init__()
        self.num_ws = num_ws
        self.w_dim = w_dim
        full_dim = num_ws * w_dim
        self.fc0 = LinearModule(vc_dim+full_dim, full_dim)
        self.fc1 = LinearModule(full_dim, full_dim)
        self.fc2 = LinearModule(full_dim, full_dim, normalize=False)

    def forward(self, w, vc):
        x = self.fc0(torch.cat((w.view(w.size(0), -1), vc), dim=1))
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x.view(-1, self.num_ws, self.w_dim)


############# implicitly mix code ##################
# mix 2 w codes into one
class ImplicitMixer(nn.Module):
    def __init__(self, num_ws, w_dim):
        super().__init__()
        self.num_ws = num_ws
        self.w_dim = w_dim
        full_dim = num_ws * w_dim
        self.fc = nn.Sequential(
            LinearModule(full_dim*2, full_dim),
            LinearModule(full_dim, full_dim),
            # LinearModule(full_dim, full_dim),
            LinearModule(full_dim, full_dim, activation=False, normalize=False)
        )

    def forward(self, w0, w1):
        batch = w0.size(0)
        in_w = torch.cat((w0.view(batch, -1), w1.view(batch, -1)), dim=1)
        w = self.fc(in_w)
        return w.view(batch, self.num_ws, self.w_dim)


class ImplicitMixer1(nn.Module):
    def __init__(self, num_ws, w_dim):
        super().__init__()
        self.num_ws = num_ws
        self.w_dim = w_dim
        full_dim = num_ws * w_dim
        self.full_dim = full_dim

        self.fc0_0 = LinearModule(full_dim, full_dim)
        self.fc1_0 = LinearModule(full_dim, full_dim)
        self.fc2_0 = LinearModule(full_dim, full_dim, activation=True, normalize=False)

        self.fc0_1 = LinearModule(full_dim, full_dim)
        self.fc1_1 = LinearModule(full_dim, full_dim)
        self.fc2_1 = LinearModule(full_dim, full_dim, activation=True, normalize=False)

        self.fc2 = nn.Sequential(
            # LinearModule(full_dim, full_dim, activation=True, normalize=False),
            LinearModule(full_dim, full_dim, activation=False, normalize=False),
        )

    def forward(self, w0, w1):
        batch = w0.size(0)
        w0 = self.fc0_0(w0.view(batch, self.full_dim))
        w0 = self.fc1_0(w0)
        w0 = self.fc2_0(w0)

        w1 = self.fc0_1(w1.view(batch, self.full_dim))
        w1 = self.fc1_1(w1)
        w1 = self.fc2_1(w1)

        w = w0 + w1
        w = self.fc2(w)
        return w.view(batch, self.num_ws, self.w_dim)


class ImplicitMixer2(nn.Module):
    def __init__(self, num_ws, w_dim):
        super().__init__()
        self.num_ws = num_ws
        self.w_dim = w_dim
        full_dim = num_ws * w_dim
        self.full_dim = full_dim

        self.layers = nn.ModuleList()
        for _ in range(num_ws):
            self.layers.append(
                nn.Sequential(
                    LinearModule(w_dim*2, w_dim*2, normalize=True),
                    # LinearModule(w_dim*2, w_dim*2),
                    LinearModule(w_dim*2, w_dim),
                    # LinearModule(w_dim, w_dim),
                )
            )

        self.final_fc = nn.Sequential(
            # LinearModule(full_dim, full_dim),
            LinearModule(full_dim, full_dim, normalize=True),
            LinearModule(full_dim, full_dim),
            LinearModule(full_dim, full_dim, activation=False, normalize=False),
        )

    def forward(self, w0, w1):
        batch = w0.size(0)
        wp = torch.cat((w0, w1), dim=2)
        ws = []
        for i in range(self.num_ws):
            layer = self.layers[i]
            w = wp[:, i]
            w = layer(w)
            ws.append(w)

        w = torch.cat(ws, dim=1)
        w = self.final_fc(w)
        return w.view(batch, self.num_ws, self.w_dim)


class ImgMixer(nn.Module):
    def __init__(self, size, num_ws, img_channels=3, w_dim=512):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }

        self.w_dim = w_dim
        self.n_latents = num_ws

        log_size = int(math.log(size, 2))
        convs0 = [ConvLayer(img_channels, channels[size], 1)]
        convs1 = [ConvLayer(img_channels, channels[size], 1)]

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs0.append(_ResBlock(in_channel, out_channel))
            convs1.append(_ResBlock(in_channel, out_channel))
            in_channel = out_channel
        self.convs0 = nn.Sequential(*convs0)
        self.convs1 = nn.Sequential(*convs1)

        # convs.append(EqualConv2d(in_channel, self.n_latents*self.w_dim, 4, padding=0, bias=False))

        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4 * 2, channels[4] * 4 * 4, activation="fused_lrelu"),
            EqualLinear(channels[4] * 4 * 4, w_dim * num_ws),
            # EqualLinear(w_dim * num_ws, w_dim * num_ws),
        )

    def forward(self, img0, img1):
        batch = img0.size(0)
        x0 = self.convs0(img0).view(batch, -1)
        x1 = self.convs1(img1).view(batch, -1)

        x = self.final_linear(torch.cat((x0, x1), dim=1))

        return x.view(batch, self.n_latents, self.w_dim)




## mxnmch g ###################################################################
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

        return rtn, None