import math
import random
import torch
from torch import nn
from torch.nn import functional as F



class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)
        return out * math.sqrt(2)




class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)



class Upsample(nn.Module):
    def __init__(self, factor=2):
        super().__init__()
        """
        It is used to be more complicated one. They implemented their own upsampling (cpp and cuda)
        I did not check the details, but it looks like they use a fixed blur kernel to achieve upsample.
        I do not know what are benefits (would be appreciate if someone could point out the reasons),
        thus I replace their upsample with the simpler one: bilinear upsampling.

        """
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode='bilinear')





class Blur(nn.Module):
    def __init__(self):
        super().__init__()

        kernel=[1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]  # 1*1*3*3
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        return F.conv2d( x, kernel, stride=1, padding=1, groups=x.size(1) )



class EqualLinear(nn.Module):
    def __init__( self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=False ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None

        self.activation = LeakyReLU() if activation else None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):

        x = F.linear( x, self.weight * self.scale )

        if self.bias is not None:
            x += self.bias.unsqueeze(0) * self.lr_mul

        if self.activation is not None:
            x = self.activation(x)

        return x





class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter( torch.randn(out_channel, in_channel, kernel_size, kernel_size) )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight*self.scale, bias=self.bias, stride=self.stride, padding=self.padding)





class ModulatedConv2d(nn.Module):
    def __init__( self, in_channel, out_channel, kernel_size, style_dim, demodulate=True):
        super().__init__()

        assert kernel_size == 1 or kernel_size == 3
        self.kernel_size = kernel_size
        self.padding = 1 if kernel_size == 3 else 0
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.demodulate = demodulate

        self.scale = 1 / math.sqrt( in_channel * kernel_size**2 )

        self.weight = nn.Parameter( torch.randn(1, out_channel, in_channel, kernel_size, kernel_size) )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)


    def forward(self, input, style):

        # get original size
        batch, in_channel, height, width = input.shape

        # apply afine transformation on style
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        # modulation
        weight = self.scale * self.weight * style # BS * OUT * IN * H * W

        # demodulation (if not apply, then output statistic not have uint variance)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        # apply conv ( grouped convolutions tirck mentioned in the supp )
        weight = weight.view( batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size )
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
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
        self.input = nn.Parameter( torch.randn(1, channel, size, size) )

    def forward(self, input):
        # input is only used for get batch size
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out




class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True ):
        super().__init__()

        if upsample:
            self.upsample = Upsample()

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out




class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, upsample=False ):
        super().__init__()
        """
        Unlike before, in unsample case, they used TransConv2d to do upsampling and at the end a Blur is
        applied. Here we first apply bilinear upsample and then we do modulated conv

        """

        self.upsample = Upsample() if upsample else None

        self.conv = ModulatedConv2d( in_channel, out_channel, 3, style_dim )
        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(out_channel))
        self.activate = LeakyReLU()



    def forward(self, x, style, noise=None):

        # apply upsample if necessary
        if self.upsample is not None:
            x = self.upsample(x)

        # modulated conv ( output should be close to unit variance )
        out = self.conv(x, style)


        # add noise B and bias b
        out = self.noise(out, noise=noise)
        out = out + self.bias.view(1,-1,1,1)

        # act
        out = self.activate(out)

        return out




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ConvLayer(nn.Module):
    def __init__( self, in_channel, out_channel, kernel_size, downsample=False, upsample=False, bias=True, activate=True ):
        super().__init__()

        assert not (downsample and upsample), 'what you want?'
        assert kernel_size == 1 or kernel_size == 3
        padding = 1 if kernel_size == 3 else 0
        layers = []

        if downsample:
            layers.append( Blur() )
            layers.append( EqualConv2d( in_channel, out_channel, kernel_size, 2, padding, bias=bias ) )
        elif upsample:
            layers.append( Upsample() )
            layers.append( EqualConv2d( in_channel, out_channel, kernel_size, padding=padding, bias=bias ) )
        else:
            layers.append( EqualConv2d( in_channel, out_channel, kernel_size, padding=padding, bias=bias ) )

        if activate:
            layers.append( LeakyReLU() )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)





class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer( in_channel, out_channel, 1, downsample=True, activate=False, bias=False )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


