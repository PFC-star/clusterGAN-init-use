from __future__ import print_function
# 引入transformer
try:
    import numpy as np
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    
    from itertools import chain as ichain

    from clusgan.utils import tlog, softmax, initialize_weights, calc_gradient_penalty

    import math

    from op import FusedLeakyReLU, upfirdn2d


    from torch.nn.utils import spectral_norm

    from clusgan.basic_layers import (Blur, Downsample, EqualConv2d, EqualLinear,
                              ScaledLeakyReLU)

except ImportError as e:
    print(e)
    raise ImportError


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """
    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
            # (Optional)Set the extra information about this module. You can test
            # it by printing an object of this class.
            return 'shape={}'.format(
                self.shape
            )


class Generator_CNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(Generator_CNN, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose

        self.model = nn.Sequential(
            # Fully connected layers
            torch.nn.Linear(self.latent_dim + self.n_c, 1024),
            nn.BatchNorm1d(1024),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),

            # Reshape to 128 x (7x7)
            Reshape(self.ishape),

            # Upconvolution layers
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
        )

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        #z = z.unsqueeze(2).unsqueeze(3)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


class Encoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """
    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()

        self.name = 'encoder'
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose
        
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, latent_dim + n_c)
        )

        initialize_weights(self)
        
        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # Separate continuous and one-hot components
        zn = z[:, 0:self.latent_dim]
        zc_logits = z[:, self.latent_dim:]
        # Softmax on zc component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


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
            sn=False
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

        if sn:
            # Not use equal conv2d when apply SN
            layers.append(
                spectral_norm(nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                ))
            )
        else:
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
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], sn=False):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, sn=sn)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, sn=sn)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


def get_haar_wavelet(in_channels):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh


class HaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', lh)
        self.register_buffer('hl', hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll = upfirdn2d(input, self.ll, down=2)
        lh = upfirdn2d(input, self.lh, down=2)
        hl = upfirdn2d(input, self.hl, down=2)
        hh = upfirdn2d(input, self.hh, down=2)

        return torch.cat((ll, lh, hl, hh), 1)


class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ll, lh, hl, hh = get_haar_wavelet(in_channels)

        self.register_buffer('ll', ll)
        self.register_buffer('lh', -lh)
        self.register_buffer('hl', -hl)
        self.register_buffer('hh', hh)

    def forward(self, input):
        ll, lh, hl, hh = input.chunk(4, 1)
        ll = upfirdn2d(ll, self.ll, up=2, pad=(1, 0, 1, 0))
        lh = upfirdn2d(lh, self.lh, up=2, pad=(1, 0, 1, 0))
        hl = upfirdn2d(hl, self.hl, up=2, pad=(1, 0, 1, 0))
        hh = upfirdn2d(hh, self.hh, up=2, pad=(1, 0, 1, 0))

        return ll + lh + hl + hh


class FromRGB(nn.Module):
    def __init__(self, out_channel, downsample=True, blur_kernel=[1, 3, 3, 1], sn=False):
        super().__init__()

        self.downsample = downsample

        if downsample:
            self.iwt = InverseHaarTransform(3)
            self.downsample = Downsample(blur_kernel)
            self.dwt = HaarTransform(3)
        self.conv = ConvLayer(3 * 4, out_channel, 1, sn=sn)
        self.conv = ConvLayer(4, out_channel, 1, sn=sn)

    def forward(self, input, skip=None):
        if self.downsample:
            input = self.iwt(input)
            input = self.downsample(input)
            input = self.dwt(input)

        out = self.conv(input)

        if skip is not None:
            out = out + skip

        return input, out






class Discriminator_CNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], sn=False, ssd=False,wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()

        self.name = 'discriminator'
        self.channels = 1
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.verbose = verbose
        channels = {
            4: 512,
            8: 512,
            16: 512,
            28: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.dwt = HaarTransform(3)

        self.from_rgbs = nn.ModuleList()
        self.convs = nn.ModuleList()

        log_size = int(math.log(size, 2)) - 1

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            self.from_rgbs.append(FromRGB(in_channel, downsample=i != log_size, sn=sn))
            self.convs.append(ConvBlock(in_channel, out_channel, blur_kernel, sn=sn))

            in_channel = out_channel

        self.from_rgbs.append(FromRGB(channels[4], sn=sn))

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, sn=sn)
        if sn:
            self.final_linear = nn.Sequential(
                spectral_norm(nn.Linear(channels[4] * 4 * 4, channels[4])),
                FusedLeakyReLU(channels[4]),
                spectral_norm(nn.Linear(channels[4], 1)),
            )
        else:
            self.final_linear = nn.Sequential(

                # EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                EqualLinear(4608, channels[4], activation='fused_lrelu'),
                EqualLinear(channels[4], 1),
            )




    def forward(self, input):
        input = self.dwt(input)
        out = None

        for from_rgb, conv in zip(self.from_rgbs, self.convs):
            input, out = from_rgb(input, out)
            out = conv(out)

        _, out = self.from_rgbs[-1](input, out)

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


D = Discriminator_CNN(28)
input = torch.randn(128, 1, 28, 28)

output = D(input)
print(output)
