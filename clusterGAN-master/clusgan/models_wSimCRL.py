from __future__ import print_function
# 引入StyleGAN2解耦思想与对比学习思想 进入Encoder
# 对比学习中的模型作用是编码图片，得到对比的特征，Encoder的作用也是这个，说不定可以加强的
try:
    import numpy as np
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    import math
    from itertools import chain as ichain
    from clusgan.utils import sample_z
    from clusgan.utils import tlog, softmax, initialize_weights, calc_gradient_penalty
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


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


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
            out = F.leaky_relu(out)

        else:
            if self.bias is not None:
                out = F.linear(
                    input, self.weight * self.scale, bias=self.bias * self.lr_mul
                )
            else:
                out = F.linear(
                    input, self.weight * self.scale, bias=None)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
class FC(nn.Module):
    def __init__(
            self,
            size=28,
            style_dim=60,
            n_mlp=3,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            lr_mlp=0.01,
    ):
        super().__init__()
        self.size = size
        self.style_dim = style_dim
        layers = [PixelNorm()]

        for i in range(n_mlp):  # n_mlp=8, style_dim=512， lr_mlp=0.01
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)
    def forward(self,input):
        output = self.style(input)
        return output

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
        self.fc = FC(28,latent_dim)
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
        zn = self.fc(zn)
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
        self.fc = FC(28,latent_dim)
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
        zn = self.fc(zn)
        zc_logits = z[:, self.latent_dim:]
        # Softmax on zc component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class Discriminator_CNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    Input is tuple (X,z) of an image vector and its corresponding
    representation z vector. For example, if X comes from the dataset, corresponding
    z is Encoder(X), and if z is sampled from representation space, X is Generator(z)
    Output is a 1-dimensional value
    """            
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()
        
        self.name = 'discriminator'
        self.channels = 1
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.wass = wass_metric
        self.verbose = verbose
        
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, 1),
        )
        
        # If NOT using Wasserstein metric, final Sigmoid
        if (not self.wass):
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, img):
        # Get output
        validity = self.model(img)
        return validity
# latent_dim =30
# n_c = 10
# zn, zc, zc_idx = sample_z(shape=28,
#                           latent_dim =latent_dim,
#                                       n_c=n_c)
# x_shape = (1, 28, 28)
# sc = Generator_CNN(latent_dim,n_c,x_shape=x_shape)
#
# output = sc(zn,zc)
# print(output)


# encoder = Encoder_CNN(latent_dim,n_c)
# input = torch.randn(1,1,28,28)
# output = encoder(input)
# print(output)