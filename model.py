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


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1,1,False)
    def forward(self):
        return 0

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
        sep_mode=False,
        is_skip=False,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.sep_mode = sep_mode
        self.style_dim = style_dim
        self.is_skip = is_skip

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

        if not sep_mode:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        else:
            self.modulation = EqualLinear(style_dim//2, in_channel//2, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style, input_is_ssc=False):
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

        if not input_is_ssc:
            if not self.sep_mode:
                style = self.modulation(style)
            else:
                style = torch.cat([self.modulation(style[:, 0:self.style_dim//2]),
                                   self.modulation(style[:, self.style_dim//2:])], dim=1)

        style = style.view(batch, 1, in_channel, 1, 1)
        if not self.sep_mode:
            _style = style
        else:
            # if not self.is_skip:
            #     fgc = style[:,:,0:in_channel//2]
            #     fgc = torch.cat([fgc, torch.zeros_like(fgc)], dim=2)
            #     fgc = fgc.repeat(1,self.out_channel//2,1,1,1)
            #     bgc = style[:,:,in_channel//2:]
            #     bgc = bgc.repeat(1,self.out_channel//2,2,1,1)
            #     _style = torch.cat([fgc, bgc], dim=1)
            # else:
            if self.out_channel % 2 == 0:
                oc1 = self.out_channel // 2
            else:
                oc1 = (self.out_channel + 1) // 2
            oc2 = self.out_channel - oc1

            fgc = style[:,:,0:in_channel//2]
            fgc = torch.cat([fgc, torch.zeros_like(fgc)], dim=2)
            fgc = fgc.repeat(1,oc1,1,1,1)
            bgc = style[:,:,in_channel//2:]
            bgc = torch.cat([torch.zeros_like(bgc), bgc], dim=2)
            bgc = bgc.repeat(1,oc2,1,1,1)
            _style = torch.cat([fgc, bgc], dim=1)

        weight = self.scale * self.weight * _style

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

        return out, style.view(batch, in_channel)


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

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class WpMask(nn.Module):
    def __init__(self, n_latent, style_dim):
        super().__init__()

        self.wpmk = nn.Parameter(torch.randn(1, n_latent, style_dim))

    def forward(self):
        out = torch.sigmoid(self.wpmk)
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
        sep_mode=False,
        negative_slope=0.2,
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
            sep_mode=sep_mode,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel, negative_slope=negative_slope)

    def forward(self, input, style, noise=None, input_is_ssc=False):
        out, s = self.conv(input, style, input_is_ssc=input_is_ssc)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out, s


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, im_channel=3, upsample=True, blur_kernel=[1, 3, 3, 1], sep_mode=False):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(
            in_channel, im_channel, 1, style_dim, demodulate=False, sep_mode=sep_mode, is_skip=True)
        self.bias = nn.Parameter(torch.zeros(1, im_channel, 1, 1))

    def forward(self, input, style, skip=None, input_is_ssc=False):
        out, s = self.conv(input, style, input_is_ssc=input_is_ssc)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out, s


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        im_channel=3,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        sep_mode=False,
        negative_slope=0.2,
        starting_feature_size=4,
        no_skip=False,
    ):
        super().__init__()

        self.size = size
        self.sep_mode = sep_mode
        self.no_skip = no_skip

        self.style_dim = style_dim

        layers = [PixelNorm()]

        if not sep_mode:
            for _ in range(n_mlp):
                layers.append(
                    EqualLinear(
                        style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                    )
                )
        else:
            for _ in range(n_mlp):
                layers.append(
                    EqualLinear(
                        style_dim//2, style_dim//2, lr_mul=lr_mlp, activation="fused_lrelu"
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

        #################################
        # self.channels = {
        #     4: 128,
        #     8: 128,
        #     16: 128,
        #     32: 128,
        #     64: 64 * channel_multiplier,
        #     128: 32 * channel_multiplier,
        #     256: 64 * channel_multiplier,
        #     512: 32 * channel_multiplier,
        #     1024: 16 * channel_multiplier,
        # }
        #################################

        starting_n_channel = self.channels[starting_feature_size]
        self.input = ConstantInput(starting_n_channel)
        self.conv1 = StyledConv(
            starting_n_channel, starting_n_channel, 3, style_dim, blur_kernel=blur_kernel, sep_mode=sep_mode, negative_slope=negative_slope
        )
        self.to_rgb1 = ToRGB(starting_n_channel, style_dim, im_channel=im_channel, upsample=False, sep_mode=sep_mode)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - int(math.log(starting_feature_size, 2))) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = starting_n_channel

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(
                f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(int(math.log(starting_feature_size, 2))+1, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    sep_mode=sep_mode,
                    negative_slope=negative_slope,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, sep_mode=sep_mode, negative_slope=negative_slope
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim, im_channel=im_channel, sep_mode=sep_mode))

            in_channel = out_channel

        self.n_latent = (self.log_size-int(math.log(starting_feature_size, 2))+1) * 2

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
        return torch.cat([self.style(s).unsqueeze(1) for s in input], dim=1)

    def forward(
        self,
        styles,
        input_feats=[],
        starting_features=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        input_is_ssc=False,
        return_ssc=False,
        return_img_only=False,
        return_feats=False,
        return_feats_only=False,
        return_skips_n_feats=False,
        return_skips=False,
        return_ssc_only=False,
    ):
        if not self.sep_mode:
            if (not input_is_latent) and (not input_is_ssc):
                styles = [self.style(s) for s in styles]
        else:
            if (not input_is_latent) and (not input_is_ssc):
                styles = [torch.cat([self.style(s[:, 0:self.style_dim//2]),
                                     self.style(s[:, self.style_dim//2:])] , dim=1) for s in styles]

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

        if (not input_is_latent) and (not input_is_ssc):
            if len(styles) < 2:
                inject_index = self.n_latent
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)

                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(
                    1, self.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)

        # input is latent
        elif input_is_latent:
            if styles.size(1) == 1:
                inject_index = self.n_latent
                latent = styles.repeat(1, inject_index, 1)

            elif styles.size(1) == 2:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)

                latent = styles[:, 0:1].repeat(1, inject_index, 1)
                latent2 = styles[:, 1:2].repeat(
                    1, self.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)

            else:
                latent = styles

        # input is stylespace code
        elif input_is_ssc:
            latent = styles

        feats = []
        skips = []
        oix = 0
        if not input_is_ssc:
            ssc = []

            batch = latent.size(0)

            out = starting_features
            if starting_features is None:
                out = self.input(batch)

            feats.append(out)
            if oix < len(input_feats):
                out = input_feats[oix]
                oix += 1

            out, s = self.conv1(out, latent[:, 0], noise=noise[0], input_is_ssc=input_is_ssc)
            ssc.append(s)
            feats.append(out)
            if oix < len(input_feats):
                out = input_feats[oix]
                oix += 1

            skip, s = self.to_rgb1(out, latent[:, 1], input_is_ssc=input_is_ssc)
            ssc.append(s)
            skips.append(skip)

            i = 1
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out, s = conv1(out, latent[:, i], noise=noise1, input_is_ssc=input_is_ssc)
                ssc.append(s)
                feats.append(out)
                if oix < len(input_feats):
                    out = input_feats[oix]
                    oix += 1

                out, s = conv2(out, latent[:, i + 1], noise=noise2, input_is_ssc=input_is_ssc)
                ssc.append(s)
                feats.append(out)
                if oix < len(input_feats):
                    out = input_feats[oix]
                    oix += 1

                if self.no_skip:
                    skip = None
                skip, s = to_rgb(out, latent[:, i + 2], skip, input_is_ssc=input_is_ssc)
                ssc.append(s)
                skips.append(skip)

                i += 2

        else:
            batch = latent[0].size(0)

            out = starting_features
            if starting_features is None:
                out = self.input(batch)

            feats.append(out)
            if oix < len(input_feats):
                out = input_feats[oix]
                oix += 1

            out, _ = self.conv1(out, latent[0], noise=noise[0], input_is_ssc=input_is_ssc)
            feats.append(out)
            if oix < len(input_feats):
                out = input_feats[oix]
                oix += 1

            skip, _ = self.to_rgb1(out, latent[1], input_is_ssc=input_is_ssc)
            skips.append(skip)

            i = 2
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out, _ = conv1(out, latent[i], noise=noise1, input_is_ssc=input_is_ssc)
                feats.append(out)
                if oix < len(input_feats):
                    out = input_feats[oix]
                    oix += 1

                out, _ = conv2(out, latent[i + 1], noise=noise2, input_is_ssc=input_is_ssc)
                feats.append(out)
                if oix < len(input_feats):
                    out = input_feats[oix]
                    oix += 1

                if self.no_skip:
                    skip = None
                skip, _ = to_rgb(out, latent[i + 2], skip, input_is_ssc=input_is_ssc)
                skips.append(skip)

                i += 3

            ssc = latent

        image = skip

        if return_skips:
            return skips

        if return_skips_n_feats:
            return skips, feats

        if return_feats:
            return image, feats

        if return_ssc:
            return image, ssc

        if return_ssc_only:
            return ssc

        if return_img_only:
            return image

        if return_feats_only:
            return feats

        if return_latents:
            return image, latent

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
    def __init__(self, size, im_channel=3, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
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

        convs = [ConvLayer(im_channel, channels[size], 1)]

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

# class G_NET(nn.Module):
#     def __init__(self, ds_name='CUB'):
#         super(G_NET, self).__init__()
#         self.gf_dim = finegan_config[ds_name]['GF_DIM']
#         self.ds_name = ds_name
#         self.define_module()
#         self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.scale_fimg = nn.UpsamplingBilinear2d(size=[126, 126])

#     def define_module(self):
#         #Background stage
#         self.h_net1_bg = INIT_STAGE_G(self.gf_dim * 16, 2, ds_name=self.ds_name)
#         # Background generation network
#         self.img_net1_bg = GET_IMAGE(self.gf_dim)

#         # Parent stage networks
#         self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, 1, ds_name=self.ds_name)
#         self.h_net2 = NEXT_STAGE_G(self.gf_dim, use_hrc=1, ds_name=self.ds_name)
#         # Parent foreground generation network
#         self.img_net2 = GET_IMAGE(self.gf_dim // 2)
#         # Parent mask generation network
#         self.img_net2_mask = GET_MASK(self.gf_dim // 2)

#         # Child stage networks
#         self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2, use_hrc=0, ds_name=self.ds_name)
#         # Child foreground generation network
#         self.img_net3 = GET_IMAGE(self.gf_dim // 4)
#         # Child mask generation network
#         self.img_net3_mask = GET_MASK(self.gf_dim // 4)

#     def forward(self, z_code, bg_code, p_code, c_code, z_fg=None, rtn_type='fnl', rtn_mk=False):

#         fake_imgs = []  # Will contain [background image, parent image, child image]
#         fg_imgs = []  # Will contain [parent foreground, child foreground]
#         mk_imgs = []  # Will contain [parent mask, child mask]
#         fg_mk = []  # Will contain [masked parent foreground, masked child foreground]

#         #Background stage
#         h_code1_bg = self.h_net1_bg(z_code, bg_code)
#         fake_img1 = self.img_net1_bg(h_code1_bg)  # Background image
#         # Resizing fake background image from 128x128 to the resolution which background discriminator expects: 126 x 126.
#         fake_img1_126 = self.scale_fimg(fake_img1)
#         fake_imgs.append(fake_img1_126)

#         #Parent stage
#         h_code1 = self.h_net1(z_fg, p_code)
#         h_code2 = self.h_net2(h_code1, p_code)
#         fake_img2_foreground = self.img_net2(h_code2)  # Parent foreground
#         fake_img2_mask = self.img_net2_mask(h_code2)  # Parent mask
#         ones_mask_p = torch.ones_like(fake_img2_mask)
#         opp_mask_p = ones_mask_p - fake_img2_mask
#         fg_masked2 = torch.mul(fake_img2_foreground, fake_img2_mask)
#         fg_mk.append(fg_masked2)
#         bg_masked2 = torch.mul(fake_img1, opp_mask_p)
#         fake_img2_final = fg_masked2 + bg_masked2  # Parent image
#         fake_imgs.append(fake_img2_final)
#         fg_imgs.append(fake_img2_foreground)
#         mk_imgs.append(fake_img2_mask)

#         #Child stage
#         h_code3 = self.h_net3(h_code2, c_code)
#         fake_img3_foreground = self.img_net3(h_code3)  # Child foreground
#         fake_img3_mask = self.img_net3_mask(h_code3)  # Child mask
#         ones_mask_c = torch.ones_like(fake_img3_mask)
#         opp_mask_c = ones_mask_c - fake_img3_mask
#         fg_masked3 = torch.mul(fake_img3_foreground, fake_img3_mask)
#         fg_mk.append(fg_masked3)
#         bg_masked3 = torch.mul(fake_img2_final, opp_mask_c)
#         fake_img3_final = fg_masked3 + bg_masked3  # Child image
#         fake_imgs.append(fake_img3_final)
#         fg_imgs.append(fake_img3_foreground)
#         mk_imgs.append(fake_img3_mask)

#         if rtn_mk:
#             return fake_img3_final, fake_img2_mask

#         # return fake_imgs, fg_imgs, mk_imgs, fg_mk
#         rtn = fake_img3_final
#         if rtn_type == 'pmk':
#             rtn = fake_img2_mask
#         elif rtn_type == 'cmk':
#             rtn = fake_img3_mask
#         elif rtn_type == 'bg':
#             rtn = fake_img1
#         return rtn, None

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

        # convs.append(EqualConv2d(in_channel, self.n_latents*self.w_dim, 4, padding=0, bias=False))
        self.convs = nn.Sequential(*convs)

        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4] * self.n_latents, activation="fused_lrelu"),
            EqualLinear(channels[4] * self.n_latents, self.w_dim * self.n_latents))

    def forward(self, input, return_li=True, n_first=False):
        batch = input.size(0)
        out = self.convs(input)
        out = self.final_linear(out.view(batch, -1))
        # if use_sigmoid:
        # out = torch.tanh(out) * 5
        out = out.view(batch, self.n_latents, self.w_dim)
        if n_first:
            return out.transpose(0, 1)
        return out

        if self.n_latents > 1:
            if return_li:
                return list(out.view(self.n_latents, batch, self.w_dim).unbind(0))
            return out.view(batch, self.n_latents, self.w_dim)
        else:
            if return_li:
                return [out.view(batch, self.w_dim)]
            return out.view(batch, self.w_dim)


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


class Classifier(nn.Module):
    def __init__(self, size, img_channels=3, b_dim=200, p_dim=20, c_dim=200):
        super().__init__()

        channels = {
            1: 512,
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

        self.b_dim = b_dim
        self.p_dim = p_dim
        self.c_dim = c_dim

        log_size = int(math.log(size, 2))
        convs = [ConvLayer(img_channels, channels[size], 1)]

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(_ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        # convs.append(EqualConv2d(in_channel, self.n_latents*self.w_dim, 4, padding=0, bias=False))

        self.b_conv = nn.Sequential(
            EqualConv2d(channels[4], channels[1], 4, padding=0, bias=False))

        self.p_conv = nn.Sequential(
            EqualConv2d(channels[4], channels[1], 4, padding=0, bias=False))

        self.c_conv = nn.Sequential(
            EqualConv2d(channels[4], channels[1], 4, padding=0, bias=False))

        self.b_final = nn.Sequential(
            EqualLinear(channels[1], channels[1], activation="fused_lrelu"),
            EqualLinear(channels[1], self.b_dim))

        self.p_final = nn.Sequential(
            EqualLinear(channels[1], channels[1], activation="fused_lrelu"),
            EqualLinear(channels[1], self.p_dim))

        self.c_final = nn.Sequential(
            EqualLinear(channels[1], channels[1], activation="fused_lrelu"),
            EqualLinear(channels[1], self.c_dim))

    def forward(self, input):
        batch = input.size(0)
        out = self.convs(input)
        b_out = self.b_conv(out)
        pred_b = self.b_final(b_out.view(batch, -1))
        pred_b = torch.sigmoid(pred_b).view(batch, self.b_dim)

        p_out = self.p_conv(out)
        pred_p = self.p_final(p_out.view(batch, -1))
        pred_p = torch.sigmoid(pred_p).view(batch, self.p_dim)

        c_out = self.c_conv(out)
        pred_c = self.c_final(c_out.view(batch, -1))
        pred_c = torch.sigmoid(pred_c).view(batch, self.c_dim)
        # if use_sigmoid:
        # out = torch.tanh(out) * 5
        return (pred_b, pred_p, pred_c)


class _Encoder(nn.Module):
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
        batch = input.size(0)
        out = self.convs(input)
        return out.view(batch, self.n_latents, self.w_dim)


class W_Discriminator(nn.Module):
    def __init__(self, num_ws, w_dim):
        super().__init__()
        self.num_ws = num_ws
        self.w_dim = w_dim
        full_dim = num_ws * w_dim
        self.full_dim = full_dim

        self.fc = nn.Sequential(
            EqualLinear(full_dim, full_dim, activation="fused_lrelu"),
            EqualLinear(full_dim, full_dim//4, activation="fused_lrelu"),
            EqualLinear(full_dim//4, full_dim//16, activation="fused_lrelu"),
            EqualLinear(full_dim//16, full_dim//32, activation="fused_lrelu"),
            EqualLinear(full_dim//32, 1))

    def forward(self, w):
        out = self.fc(w.view(-1, self.full_dim))
        return out


class MuVarEncoder(nn.Module):
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

        # convs.append(EqualConv2d(in_channel, self.n_latents*self.w_dim, 4, padding=0, bias=False))
        self.convs = nn.Sequential(*convs)

        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4] * self.n_latents, activation="fused_lrelu"),)
            # EqualLinear(channels[4] * self.n_latents, self.w_dim * self.n_latents))
        self.mu_linear = EqualLinear(channels[4] * self.n_latents, self.w_dim * self.n_latents)
        self.var_linear = EqualLinear(channels[4] * self.n_latents, self.w_dim * self.n_latents)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def get_kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, input, return_li=True, n_first=False):
        batch = input.size(0)
        out = self.convs(input)
        out = self.final_linear(out.view(batch, -1))
        mu = self.mu_linear(out)
        logvar = self.var_linear(out)
        z = self.reparameterize(mu, logvar)
        # if use_sigmoid:
        # out = torch.tanh(out) * 5
        z = z.view(batch, self.n_latents, self.w_dim)
        loss = self.get_kl_loss(mu, logvar)

        if n_first:
            return z.transpose(0, 1), loss

        return z, loss


class MKGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        use_w_mix=False,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for _ in range(n_mlp):
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
            self.channels[4], self.channels[4]+1, 3, style_dim, blur_kernel=blur_kernel
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
                    out_channel+1,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel+1, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        if use_w_mix:
            self.wpmk = WpMask(self.n_latent, style_dim)

    def mix_wp(self, fg_wp, bg_wp):
        batch = fg_wp.size(0)
        wpmk = self.wpmk()
        fg_mk = wpmk.repeat(batch, 1, 1)
        bg_mk = torch.ones_like(fg_mk) - fg_mk
        wp = fg_wp * fg_mk + bg_wp * bg_mk
        return wp, wpmk

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
        return torch.cat([self.style(s).unsqueeze(1) for s in input], dim=1)

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
        input_is_ssc=False,
        return_ssc=False,
        return_img_only=False,
        return_outs=False,
    ):
        if (not input_is_latent) and (not input_is_ssc):
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

        if (not input_is_latent) and (not input_is_ssc):
            if len(styles) < 2:
                inject_index = self.n_latent
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)

                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(
                    1, self.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)

        # input is latent
        elif input_is_latent:
            if styles.size(1) == 1:
                inject_index = self.n_latent
                latent = styles.repeat(1, inject_index, 1)

            elif styles.size(1) == 2:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)

                latent = styles[:, 0:1].repeat(1, inject_index, 1)
                latent2 = styles[:, 1:2].repeat(
                    1, self.n_latent - inject_index, 1)

                latent = torch.cat([latent, latent2], 1)

            else:
                latent = styles

        # input is stylespace code
        elif input_is_ssc:
            latent = styles

        outs = []
        if not input_is_ssc:
            ssc = []

            batch = latent.size(0)

            out = self.input(batch)
            # outs.append(out)

            out, s = self.conv1(out, latent[:, 0], noise=noise[0], input_is_ssc=input_is_ssc)
            ssc.append(s)
            outs.append(out)

            _out = out[:, 0:-1].clone()
            skip, s = self.to_rgb1(_out, latent[:, 1], input_is_ssc=input_is_ssc)
            ssc.append(s)

            i = 1
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out, s = conv1(_out, latent[:, i], noise=noise1, input_is_ssc=input_is_ssc)
                ssc.append(s)
                outs.append(out)

                _out = out[:, 0:-1].clone()
                out, s = conv2(_out, latent[:, i + 1], noise=noise2, input_is_ssc=input_is_ssc)
                ssc.append(s)
                outs.append(out)

                _out = out[:, 0:-1].clone()
                skip, s = to_rgb(_out, latent[:, i + 2], skip, input_is_ssc=input_is_ssc)
                ssc.append(s)

                i += 2

        else:
            batch = latent[0].size(0)

            out = self.input(batch)
            # outs.append(out)

            out, _ = self.conv1(out, latent[0], noise=noise[0], input_is_ssc=input_is_ssc)
            outs.append(out)

            skip, _ = self.to_rgb1(out[:, 0:-1], latent[1], input_is_ssc=input_is_ssc)

            i = 2
            for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
            ):
                out, _ = conv1(out[:, 0:-1], latent[i], noise=noise1, input_is_ssc=input_is_ssc)
                outs.append(out)

                out, _ = conv2(out[:, 0:-1], latent[i + 1], noise=noise2, input_is_ssc=input_is_ssc)
                outs.append(out)

                skip, _ = to_rgb(out[:, 0:-1], latent[i + 2], skip, input_is_ssc=input_is_ssc)

                i += 3

            ssc = latent

        image = skip

        if return_outs:
            return outs

        if return_ssc:
            return image, ssc

        if return_img_only:
            return image

        if return_latents:
            return image, latent

        return image, None


class NoSkipGenerator(nn.Module):
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
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

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

            self.to_rgbs.append(ToRGB(out_channel, style_dim, upsample=False))

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
                    truncation_latent + truncation * (style - truncation_latent)
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
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent.size(0))
        out, _ = self.conv1(out, latent[:, 0], noise=noise[0])

        # skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out, _ = conv1(out, latent[:, i], noise=noise1)
            out, _ = conv2(out, latent[:, i + 1], noise=noise2)
            skip, _ = to_rgb(out, latent[:, i + 2])

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class SEncoder(nn.Module):
    def __init__(self, size, dim_li, img_channels=3):
        super().__init__()

        channels = {
            4: 256,
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16
        }

        log_size = int(math.log(size, 2))
        convs = [ConvLayer(img_channels, channels[size], 1)]

        in_channel = channels[size]
        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(_ResBlock(in_channel, out_channel))
            in_channel = out_channel

        # convs.append(EqualConv2d(in_channel, self.n_latents*self.w_dim, 4, padding=0, bias=False))
        self.convs = nn.Sequential(*convs)

        self.fc = EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu")

        self.fnl_fc_li = nn.ModuleList()
        for s_dim in dim_li:
            self.fnl_fc_li.append(EqualLinear(channels[4], s_dim))

    def forward(self, input):
        batch = input.size(0)
        out = self.convs(input)

        out = self.fc(out.view(batch, -1))
        ssc = []
        for fc in self.fnl_fc_li:
            ssc.append(fc(out))

        return ssc


# class SmallUnet(nn.Module):
#     "Here we aggregate feature from different resolution. It is actually an up branch of Unet"
#     def __init__(self, size_to_channel):
#         super().__init__()

#         self.convs = nn.ModuleList()

#         size = 4
#         extra_channel = 0

#         for i in range(len(size_to_channel)):
#             in_channel = size_to_channel[size] + extra_channel
#             upsample = i != (len(size_to_channel)-1)
#             self.convs.append(ConvLayer(in_channel, 512, 3, upsample=upsample))
#             size *= 2
#             extra_channel = 512

#     def forward(self, feature_list):
#         "feature_list should be ordered from small to big: BS*C1*4*?, BS*C2*8*?, BS*C3*16*?,..."

#         for conv, feature in zip(self.convs, feature_list):
#             if feature.shape[2] != 4:
#                 feature = torch.cat([feature,previos], dim=1)
#             previos = conv(feature)

#         return previos


class StarttingFeatureEncoder(nn.Module):
    def __init__(self, size, out_feature_size, channel_multiplier=2):
        super().__init__()
        self.out_feature_size = out_feature_size
        channels = {4: 512,
                    8: 512,
                    16: 512,
                    32: 512,
                    64: 512,
                    128: 128 * channel_multiplier,
                    256: 64 * channel_multiplier,
                    512: 32 * channel_multiplier,
                    1024: 16 * channel_multiplier}

        self.convs1 = ConvLayer(3, channels[size], 1)

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        self.convs2 = nn.ModuleList()
        for i in range(log_size, int(math.log(out_feature_size, 2)), -1):
            out_size = 2 ** (i-1)
            out_channel = channels[out_size]
            self.convs2.append(_ResBlock(in_channel, out_channel))
            in_channel = out_channel

        # self.unet = SmallUnet(size_to_channel)

    def forward(self, input):
        out = self.convs1(input)
        for conv in self.convs2:
            out = conv(out)

        return out


class TwoStageGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        im_channel=3,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        sep_mode=False,
        negative_slope=0.2,
        starting_feature_size=16,
        ):
        super().__init__()

        self.fg_generator = Generator(
            size=size,
            style_dim=style_dim,
            n_mlp=n_mlp,
            im_channel=im_channel,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            lr_mlp=lr_mlp,
            sep_mode=sep_mode,
            negative_slope=negative_slope
        )

        self.bg_generator = Generator(
            size=size,
            style_dim=style_dim,
            n_mlp=n_mlp,
            im_channel=im_channel,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            lr_mlp=lr_mlp,
            sep_mode=sep_mode,
            negative_slope=negative_slope,
            starting_feature_size=starting_feature_size
        )

        self.fg_encoder = StarttingFeatureEncoder(
            size=size,
            out_feature_size=starting_feature_size,
            channel_multiplier=channel_multiplier
        )

    def forward(
        self,
        fg_styles,
        bg_styles,
        input_feats=[],
        starting_features=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        input_is_ssc=False,
        return_ssc=False,
        return_img_only=False,
        return_feats=False,
        return_feats_only=False,
        return_skips_n_feats=False,
        return_skips=False,
        return_ssc_only=False,
        return_weights=False,
        ):

        fg_img, fg_out = self.fg_generator(
            fg_styles,
            )

        bg_starting_features = self.fg_encoder(fg_img)

        bg_img, bg_out = self.bg_generator(
            bg_styles,
            starting_features=bg_starting_features
        )

        fnl_img = fg_img + bg_img
        return (fg_img, bg_img, fnl_img), (fg_out, bg_out)


class SepGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        negative_slope=0.2,
        no_skip=False,
        ):
        super().__init__()

        self.generator = Generator(
            size=size,
            style_dim=style_dim,
            n_mlp=n_mlp,
            im_channel=6,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            lr_mlp=lr_mlp,
            sep_mode=True,
            negative_slope=negative_slope,
            no_skip=no_skip
        )

    def make_noise(self):
        return self.generator.make_noise()

    def forward(
        self,
        styles,
        input_feats=[],
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        input_is_ssc=False,
        return_ssc=False,
        return_img_only=False,
        return_feats=False,
        return_feats_only=False,
        return_skips_n_feats=False,
        return_skips=False,
        return_ssc_only=False,
        return_separately=False
        ):

        img, out = self.generator(
            styles,
            input_is_ssc=input_is_ssc,
            return_latents=return_latents,
            return_feats=return_feats,
            return_ssc=return_ssc,
            noise=noise,
            )

        fg_img = img[:, 0:3]
        bg_img = img[:, 3:6]

        fnl_img = fg_img + bg_img
        if not return_separately:
            return fnl_img, out

        return [fg_img, bg_img, fnl_img], out


class SepWithMkGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        negative_slope=0.2,
        no_skip=False,
        ):
        super().__init__()

        self.generator = Generator(
            size=size,
            style_dim=style_dim,
            n_mlp=n_mlp,
            im_channel=7,
            channel_multiplier=channel_multiplier,
            blur_kernel=blur_kernel,
            lr_mlp=lr_mlp,
            sep_mode=True,
            negative_slope=negative_slope,
            no_skip=no_skip
        )

    def make_noise(self):
        return self.generator.make_noise()

    def forward(
        self,
        styles,
        input_feats=[],
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        input_is_ssc=False,
        return_ssc=False,
        return_img_only=False,
        return_feats=False,
        return_feats_only=False,
        return_skips_n_feats=False,
        return_skips=False,
        return_ssc_only=False,
        return_separately=False
        ):

        imgs, out = self.generator(
            styles,
            input_is_ssc=input_is_ssc,
            return_latents=return_latents,
            return_feats=return_feats,
            return_ssc=return_ssc,
            noise=noise,
            )

        fg_img = imgs[:, 0:3]
        mask = imgs[:, 3:4]
        bg_img = imgs[:, 4:7]

        mask = torch.sigmoid(mask)
        rmask = torch.ones_like(mask) - mask

        fnl_img = fg_img * mask + bg_img * rmask
        if not return_separately:
            return fnl_img, out

        return [fg_img, bg_img, fnl_img, mask], out