import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from model import ModulatedConv2d, StyledConv, ToRGB, _MGenerator


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

        kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]  # 1*1*3*3
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        return F.conv2d(x, kernel, stride=1, padding=1, groups=x.size(1))


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(
            out_dim).fill_(bias_init)) if bias else None

        self.activation = LeakyReLU() if activation else None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):

        x = F.linear(x, self.weight * self.scale)

        if self.bias is not None:
            x += self.bias.unsqueeze(0) * self.lr_mul

        if self.activation is not None:
            x = self.activation(x)

        return x


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(
            out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weight*self.scale, bias=self.bias, stride=self.stride, padding=self.padding)


class _ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True):
        super().__init__()

        assert kernel_size == 1 or kernel_size == 3
        self.kernel_size = kernel_size
        self.padding = 1 if kernel_size == 3 else 0
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.demodulate = demodulate

        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(
            1, out_channel, in_channel, kernel_size, kernel_size))

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

    def forward(self, input, style):

        # get original size
        batch, in_channel, height, width = input.shape

        # apply afine transformation on style
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        # modulation
        weight = self.scale * self.weight * style  # BS * OUT * IN * H * W

        # demodulation (if not apply, then output statistic not have uint variance)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        # apply conv ( grouped convolutions tirck mentioned in the supp )
        weight = weight.view(batch * self.out_channel,
                             in_channel, self.kernel_size, self.kernel_size)
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
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        # input is only used for get batch size
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class _ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True):
        super().__init__()

        if upsample:
            self.upsample = Upsample()

        self.conv = _ModulatedConv2d(
            in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class _StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, upsample=False):
        super().__init__()
        """
        Unlike before, in unsample case, they used TransConv2d to do upsampling and at the end a Blur is
        applied. Here we first apply bilinear upsample and then we do modulated conv

        """

        self.upsample = Upsample() if upsample else None

        self.conv = _ModulatedConv2d(in_channel, out_channel, 3, style_dim)
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
        out = out + self.bias.view(1, -1, 1, 1)

        # act
        out = self.activate(out)

        return out


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, upsample=False, bias=True, activate=True):
        super().__init__()

        assert not (downsample and upsample), 'what you want?'
        assert kernel_size == 1 or kernel_size == 3
        padding = 1 if kernel_size == 3 else 0
        layers = []

        if downsample:
            layers.append(Blur())
            layers.append(EqualConv2d(in_channel, out_channel,
                                      kernel_size, 2, padding, bias=bias))
        elif upsample:
            layers.append(Upsample())
            layers.append(EqualConv2d(in_channel, out_channel,
                                      kernel_size, padding=padding, bias=bias))
        else:
            layers.append(EqualConv2d(in_channel, out_channel,
                                      kernel_size, padding=padding, bias=bias))

        if activate:
            layers.append(LeakyReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(in_channel, out_channel, 1,
                              downsample=True, activate=False, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Generator(nn.Module):
    def __init__(self, args, device, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, ):
        super().__init__()

        self.args = args
        self.device = device

        layers = [PixelNorm()]
        for i in range(args.n_mlp):
            layers.append(EqualLinear(args.style_dim, args.style_dim,
                                      lr_mul=lr_mlp, activation='fused_lrelu'))
        self.style = nn.Sequential(*layers)

        self.channels = {4: 512,
                         8: 512,
                         16: 512,
                         32: 512,
                         64: 256 * args.channel_multiplier,
                         128: 128 * args.channel_multiplier,
                         256: 64 * args.channel_multiplier,
                         512: 32 * args.channel_multiplier,
                         1024: 16 * args.channel_multiplier}

        self.encoder = Encoder(args)

        self.w_over_h = args.scene_size[1] / args.scene_size[0]
        assert self.w_over_h.is_integer(), 'non supported scene_size'
        self.w_over_h = int(self.w_over_h)

        self.log_size = int(math.log(
            args.scene_size[0], 2)) - int(math.log(self.args.starting_height_size, 2))
        self.num_layers = self.log_size * 2
        self.n_latent = self.log_size * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        self.input = ConstantInput(self.channels[4], self.args.starting_height_size)

        expected_out_size = self.args.starting_height_size
        layer_idx = 0
        for _ in range(self.log_size):
            expected_out_size *= 2
            shape = [1, 1, expected_out_size, expected_out_size*self.w_over_h]
            self.noises.register_buffer(
                f'noise_{layer_idx}', torch.zeros(*shape))
            self.noises.register_buffer(
                f'noise_{layer_idx+1}', torch.zeros(*shape))
            layer_idx += 2

        in_channel = self.channels[self.args.starting_height_size]
        expected_out_size = self.args.starting_height_size
        for _ in range(self.log_size):
            expected_out_size *= 2
            out_channel = self.channels[expected_out_size]
            self.convs.append(_StyledConv(
                in_channel, out_channel, args.style_dim, upsample=True))
            self.convs.append(_StyledConv(
                out_channel, out_channel, args.style_dim, ))
            self.to_rgbs.append(_ToRGB(out_channel, args.style_dim))
            in_channel = out_channel

    def make_noise(self):

        expected_out_size = self.args.starting_height_size
        noises = []
        for _ in range(self.log_size):
            expected_out_size *= 2
            noises.append(torch.randn(1, 1, expected_out_size,
                                      expected_out_size*self.w_over_h, device=self.device))
            noises.append(torch.randn(1, 1, expected_out_size,
                                      expected_out_size*self.w_over_h, device=self.device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.args.style_dim, device=self.device)
        latent = self.style(latent_in).mean(0, keepdim=True).unsqueeze(1)
        return latent

    def get_latent(self, input):
        return self.style(input)

    def __prepare_starting_feature(self, global_pri, styles, input_type):
        feature, z, loss = self.encoder(global_pri)
        if input_type == None:
            styles = [z]
            input_type = 'z'
        return feature, styles, input_type, loss

    def __prepare_letent(self, styles, inject_index, truncation, truncation_latent,  input_type):
        "This is a private function to prepare w+ space code needed during forward"

        if input_type == 'z':
            styles = [self.style(s).unsqueeze(1)
                      for s in styles]  # each one is bs*1*512
        elif input_type == 'w':
            styles = [s.unsqueeze(1) for s in styles]  # each one is bs*1*512
        else:
            return styles  # w+ case

        # truncate each w
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation *
                               (style-truncation_latent))
            styles = style_t

        # duplicate and concat into BS * n_latent * code_len
        if len(styles) == 1:
            latent = styles[0].repeat(1, self.n_latent, 1)
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent1 = styles[0].repeat(1, inject_index, 1)
            latent2 = styles[1].repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        else:
            latent = torch.cat(styles, 1)

        return latent

    def __prepare_noise(self, noise, randomize_noise):
        "This is a private function to prepare noise needed during forward"

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise_{i}')
                         for i in range(self.num_layers)]

        return noise

    def forward(self, global_pri, styles=None, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_type=None, noise=None, randomize_noise=True, return_loss=True):
        """
        global_pri: a tensor with the shape BS*C*self.prior_size*self.prior_size. Here, in background training,
                    it should be semantic map + edge map, so it should have channel 151+1

        styles: it should be either list or a tensor or None.
                List Case:
                    a list containing either z code or w code.
                    and each code (whether z or w, specify by input_type) in this list should be bs*len_of_code.
                    And number of codes should be 1 or 2 or self.n_latent.
                    When len==1, later this code will be broadcast into bs*self.n_latent*512
                    if it is 2 then it will perform style mixing. If it is self.n_latent, then each of them will
                    provide style for each layer.
                Tensor Case:
                    then it has to be bs*self.n_latent*code_len, which means it is a w+ code.
                    In this case input_type should be 'w+', and for now we do not support truncate,
                    we assume the input is a ready-to-go latent code from w+ space
                None Case:
                    Then z code will be derived from global_pri also. In this case input_type shuold be None

        return_latents: if true w+ code: bs*self.n_latent*512 tensor, will be returned

        inject_index: int value, it will be specify for style mixing, only will be used when len(styles)==2

        truncation: whether each w will be truncated

        truncation_latent: if given then it should be calculated from mean_latent function. It has size 1*1*512
                           if truncation, then this latent must be given

        input_type: input type of styles, None, 'z', 'w' 'w+'

        noise: if given then recommand to run make_noise first to get noise and then use that as input. if given
               randomize_noise will be ignored

        randomize_noise: if true then each forward will use different noise, if not a pre-registered fixed noise
                         will be used for each forward.

        return_loss: if return kl loss.

        """
        if input_type == 'z' or input_type == 'w':
            assert len(styles) in [
                1, 2, self.n_latent], 'number of styles must be 1, 2 or self.n_latent'
        elif input_type == 'w+':
            assert styles.ndim == 3 and styles.shape[1] == self.n_latent
        elif input_type == None:
            assert styles == None
        else:
            assert False, 'not supported input_type'

        start_feature, styles, input_type, loss = self.__prepare_starting_feature(
            global_pri, styles, input_type)

        latent = self.__prepare_letent(
            styles, inject_index, truncation, truncation_latent, input_type)
        noise = self.__prepare_noise(noise, randomize_noise)

        # # # start generating # # #
        start_feature = self.input(global_pri)
        out = start_feature
        # print(start_feature.size())
        # sys.exit()
        skip = None

        i = 0

        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[::2], noise[1::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = torch.tanh(skip)

        output = {'image': image}
        if return_latents:
            output['latent'] = latent
        if return_loss:
            output['klloss'] = loss

        return output




# class Padder():
#     def __init__(self, scene_size):

#         if scene_size[0]<scene_size[1]:
#             # height is smaller, so pad up and bottem
#             p = int( (scene_size[1]-scene_size[0])/2 )
#             self.pad = nn.ZeroPad2d((0, 0, p, p))
#         else:
#             # width is smaller, so pad left and right
#             p = int( (scene_size[0]-scene_size[1])/2 )
#             self.pad = nn.ZeroPad2d((p, p, 0, 0))

#     def __call__(self, x):

#         return self.pad(x)


class Padder():
    def __init__(self, scene_size):

        print('scene size will be splitted in D, thus input and output batch dimension is not the same')

        if scene_size[0]<scene_size[1]:
            # height is smaller, so split width: 3rd dimension
            self.dim = 3
            self.size = scene_size[0]
        else:
            # width is smaller, so split height: 2nd dimension
            self.dim = 2
            self.size = scene_size[1]

    def __call__(self, x):
        xx = torch.split(x, self.size, self.dim)
        return torch.cat( [xx[0], xx[1]], dim=0 )



class Discriminator(nn.Module):
    def __init__(self, args, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * args.channel_multiplier,
            128: 128 * args.channel_multiplier,
            256: 64 * args.channel_multiplier,
            512: 32 * args.channel_multiplier,
            1024: 16 * args.channel_multiplier,
        }

        if args.scene_size[0] == args.scene_size[1]:
            input_size = args.scene_size[0]
            self.need_handle_size = False
        else:
            input_size = min(args.scene_size)  ########## if padding, then this should be max
            self.need_handle_size = True
            self.padder = Padder(args.scene_size)


        log_size = int( math.log(input_size,2) )
        convs = [ ConvLayer(3, channels[input_size], 1) ]

        in_channel = channels[input_size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential( EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                                           EqualLinear(channels[4], 1) )

    def forward(self, input):

        if self.need_handle_size:
            input = self.padder(input)

        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view( group, -1, self.stddev_feat, channel // self.stddev_feat, height, width )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out







class SmallUnet(nn.Module):
    "Here we aggregate feature from different resolution. It is actually an up branch of Unet"
    def __init__(self, size_to_channel):
        super().__init__()

        self.convs = nn.ModuleList()

        size = 4
        extra_channel = 0

        for i in range(  len(size_to_channel)  ):
            in_channel = size_to_channel[size] + extra_channel
            upsample = i != (len(size_to_channel)-1)
            self.convs.append(     ConvLayer(in_channel, 512, 3, upsample=upsample)      )
            size *= 2
            extra_channel = 512

    def forward(self, feature_list):
        "feature_list should be ordered from small to big: BS*C1*4*?, BS*C2*8*?, BS*C3*16*?,..."

        for conv, feature in zip(self.convs, feature_list):
            if feature.shape[2] != 4:
                feature = torch.cat( [feature,previos], dim=1 )
            previos = conv(feature)

        return previos


class Encoder(nn.Module):
    def __init__(self, args, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.args = args

        channels = {4: 512,
                    8: 512,
                    16: 512,
                    32: 512,
                    64: 512,
                    128: 128 * args.channel_multiplier,
                    256: 64 * args.channel_multiplier,
                    512: 32 * args.channel_multiplier,
                    1024: 16 * args.channel_multiplier}

        self.convs1 = ConvLayer(3, channels[args.prior_size[0]], 1)

        log_size = int(math.log(args.prior_size[0], 2))

        in_channel = channels[args.prior_size[0]]
        size_to_channel = {} # indicate from which resolution we provide spatial feature
        self.convs2 = nn.ModuleList()
        for i in range(log_size, 2, -1):
            out_size = 2 ** (i-1)
            out_channel = channels[ out_size ]
            if 4 <= out_size <= args.starting_height_size:
                size_to_channel[out_size] = out_channel
            self.convs2.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        # self.unet = SmallUnet( size_to_channel )

        w_over_h = args.prior_size[1] / args.prior_size[0]
        assert w_over_h.is_integer(), 'non supported scene_size'

        self.final_linear = EqualLinear(channels[4] * 4 * 4 * int(w_over_h), channels[4], activation='fused_lrelu')
        self.mu_linear = EqualLinear(channels[4], args.style_dim)
        self.var_linear = EqualLinear(channels[4], args.style_dim)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu


    def get_kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    def forward(self, input):
        batch = input.shape[0]
        intermediate_feature = []

        out = self.convs1(input)
        for conv in self.convs2:
            out = conv(out)
            if 4 <= out.shape[2] <= self.args.starting_height_size:
                intermediate_feature.append(out)
        # starting_feature = self.unet( intermediate_feature[::-1] )

        out = self.final_linear(out.view(batch, -1))
        mu = self.mu_linear(out)
        logvar = self.var_linear(out)
        z = self.reparameterize(mu, logvar)

        loss = self.get_kl_loss(mu, logvar)
        # return starting_feature, z, loss
        return None, z, loss


class _Generator(nn.Module):
    def __init__(self, args, device, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01, ):
        super().__init__()

        self.args = args
        self.device = device

        layers = [PixelNorm()]
        for i in range(args.n_mlp):
            layers.append( EqualLinear( args.style_dim, args.style_dim, lr_mul=lr_mlp, activation='fused_lrelu' )  )
        self.style = nn.Sequential(*layers)

        self.channels = {   4: 512,
                            8: 512,
                            16: 512,
                            32: 512,
                            64: 256 * args.channel_multiplier,
                            128: 128 * args.channel_multiplier,
                            256: 64 * args.channel_multiplier,
                            512: 32 * args.channel_multiplier,
                            1024: 16 * args.channel_multiplier }

        self.encoder = _Encoder(args)

        self.w_over_h = args.scene_size[1] / args.scene_size[0]
        assert self.w_over_h.is_integer(), 'non supported scene_size'
        self.w_over_h = int(self.w_over_h)

        self.log_size = int(math.log(args.scene_size[0], 2)) - int(math.log(self.args.starting_height_size, 2))
        self.num_layers = self.log_size * 2
        self.n_latent = self.log_size * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        expected_out_size = self.args.starting_height_size
        layer_idx = 0
        for _ in range(self.log_size):
            expected_out_size *= 2
            shape = [1, 1, expected_out_size, expected_out_size*self.w_over_h]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.zeros(*shape))
            self.noises.register_buffer(f'noise_{layer_idx+1}', torch.zeros(*shape))
            layer_idx += 2

        in_channel = self.channels[self.args.starting_height_size]
        expected_out_size = self.args.starting_height_size
        for _ in range(self.log_size):
            expected_out_size *= 2
            out_channel = self.channels[expected_out_size]
            self.convs.append( _StyledConv( in_channel, out_channel, args.style_dim, upsample=True ) )
            self.convs.append( _StyledConv(out_channel, out_channel, args.style_dim, ) )
            self.to_rgbs.append(_ToRGB(out_channel, args.style_dim))
            in_channel = out_channel



    def make_noise(self):

        expected_out_size = self.args.starting_height_size
        noises = []
        for _ in range(self.log_size):
            expected_out_size *= 2
            noises.append( torch.randn(1, 1, expected_out_size, expected_out_size*self.w_over_h, device=self.device) )
            noises.append( torch.randn(1, 1, expected_out_size, expected_out_size*self.w_over_h, device=self.device) )

        return noises


    def mean_latent(self, n_latent):
        latent_in = torch.randn( n_latent, self.args.style_dim, device=self.device)
        latent = self.style(latent_in).mean(0, keepdim=True).unsqueeze(1)
        return latent


    def get_latent(self, input):
        return self.style(input)


    def __prepare_starting_feature(self, global_pri, styles, input_type):
        feature, z, loss = self.encoder(global_pri)
        if input_type == None:
            styles = [z]
            input_type = 'z'
        return  feature, styles, input_type, loss



    def __prepare_letent(self, styles, inject_index, truncation, truncation_latent,  input_type):
        "This is a private function to prepare w+ space code needed during forward"

        if input_type == 'z':
            styles = [self.style(s).unsqueeze(1) for s in styles]  # each one is bs*1*512
        elif input_type == 'w':
            styles = [s.unsqueeze(1) for s in styles]  # each one is bs*1*512
        else:
            return styles # w+ case

        # truncate each w
        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append( truncation_latent + truncation * (style-truncation_latent)  )
            styles = style_t

        # duplicate and concat into BS * n_latent * code_len
        if len(styles) == 1:
            latent = styles[0].repeat(1, self.n_latent, 1)
        elif len(styles) == 2:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            latent1 = styles[0].repeat(1, inject_index, 1)
            latent2 = styles[1].repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)
        else:
            latent = torch.cat( styles, 1 )

        return latent



    def __prepare_noise(self, noise, randomize_noise):
        "This is a private function to prepare noise needed during forward"

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [ getattr(self.noises, f'noise_{i}') for i in range(self.num_layers) ]

        return noise



    def forward(self, global_pri, styles=None, return_latents=False, inject_index=None, truncation=1, truncation_latent=None, input_type=None, noise=None, randomize_noise=True, return_loss=True):

        """
        global_pri: a tensor with the shape BS*C*self.prior_size*self.prior_size. Here, in background training,
                    it should be semantic map + edge map, so it should have channel 151+1

        styles: it should be either list or a tensor or None.
                List Case:
                    a list containing either z code or w code.
                    and each code (whether z or w, specify by input_type) in this list should be bs*len_of_code.
                    And number of codes should be 1 or 2 or self.n_latent.
                    When len==1, later this code will be broadcast into bs*self.n_latent*512
                    if it is 2 then it will perform style mixing. If it is self.n_latent, then each of them will
                    provide style for each layer.
                Tensor Case:
                    then it has to be bs*self.n_latent*code_len, which means it is a w+ code.
                    In this case input_type should be 'w+', and for now we do not support truncate,
                    we assume the input is a ready-to-go latent code from w+ space
                None Case:
                    Then z code will be derived from global_pri also. In this case input_type shuold be None

        return_latents: if true w+ code: bs*self.n_latent*512 tensor, will be returned

        inject_index: int value, it will be specify for style mixing, only will be used when len(styles)==2

        truncation: whether each w will be truncated

        truncation_latent: if given then it should be calculated from mean_latent function. It has size 1*1*512
                           if truncation, then this latent must be given

        input_type: input type of styles, None, 'z', 'w' 'w+'

        noise: if given then recommand to run make_noise first to get noise and then use that as input. if given
               randomize_noise will be ignored

        randomize_noise: if true then each forward will use different noise, if not a pre-registered fixed noise
                         will be used for each forward.

        return_loss: if return kl loss.

        """
        if input_type == 'z' or input_type == 'w':
            assert len(styles) in [1,2,self.n_latent], 'number of styles must be 1, 2 or self.n_latent'
        elif input_type == 'w+':
            assert styles.ndim == 3 and styles.shape[1] == self.n_latent
        elif input_type == None:
            assert styles == None
        else:
            assert False, 'not supported input_type'


        start_feature, styles, input_type, loss = self.__prepare_starting_feature(global_pri, styles, input_type)
        latent = self.__prepare_letent(styles, inject_index, truncation, truncation_latent, input_type)
        noise = self.__prepare_noise(noise, randomize_noise)

        # # # start generating # # #

        out = start_feature
        skip = None

        i = 0

        for conv1, conv2, noise1, noise2, to_rgb in zip( self.convs[::2], self.convs[1::2], noise[::2], noise[1::2], self.to_rgbs ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = torch.tanh(skip)

        output = { 'image': image }
        if return_latents:
            output['latent'] =  latent
        if return_loss:
            output['klloss'] =  loss

        return output


class _Encoder(nn.Module):
    def __init__(self, args, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.args = args

        channels = {4: 512,
                    8: 512,
                    16: 512,
                    32: 512,
                    64: 512,
                    128: 128 * args.channel_multiplier,
                    256: 64 * args.channel_multiplier,
                    512: 32 * args.channel_multiplier,
                    1024: 16 * args.channel_multiplier}

        self.convs1 = ConvLayer(3, channels[args.prior_size[0]], 1)

        log_size = int(math.log(args.prior_size[0], 2))

        in_channel = channels[args.prior_size[0]]
        size_to_channel = {}  # indicate from which resolution we provide spatial feature
        self.convs2 = nn.ModuleList()
        for i in range(log_size, 2, -1):
            out_size = 2 ** (i-1)
            out_channel = channels[out_size]
            if 4 <= out_size <= args.starting_height_size:
                size_to_channel[out_size] = out_channel
            self.convs2.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        self.unet = SmallUnet( size_to_channel )

        w_over_h = args.prior_size[1] / args.prior_size[0]
        assert w_over_h.is_integer(), 'non supported scene_size'

        self.final_linear = EqualLinear(
            channels[4] * 4 * 4 * int(w_over_h), channels[4], activation='fused_lrelu')
        self.mu_linear = EqualLinear(channels[4], args.style_dim)
        self.var_linear = EqualLinear(channels[4], args.style_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def get_kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, input):
        batch = input.shape[0]
        intermediate_feature = []

        out = self.convs1(input)
        for conv in self.convs2:
            out = conv(out)
            if 4 <= out.shape[2] <= self.args.starting_height_size:
                intermediate_feature.append(out)
        starting_feature = self.unet( intermediate_feature[::-1] )

        out = self.final_linear(out.view(batch, -1))
        mu = self.mu_linear(out)
        logvar = self.var_linear(out)
        z = self.reparameterize(mu, logvar)

        # print(z)
        # sys.exit()
        loss = self.get_kl_loss(mu, logvar)
        return starting_feature, z, loss
