from __future__ import print_function

try:
    import numpy as np
    from torch.autograd import Function
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    
    from itertools import chain as ichain

    from clusgan.utils import tlog, softmax, initialize_weights, calc_gradient_penalty
except ImportError as e:
    print(e)
    raise ImportError
# class SCConv(nn.Module):
#     def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
#         super(SCConv, self).__init__()
#         self.k2 = nn.Sequential(
#                     # 下采样   pooling_r
#                     nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
#                     nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
#                                 padding=padding, dilation=dilation,
#                                 groups=groups, bias=False),
#                     norm_layer(planes),
#                     )
#         # 普通卷积
#         self.k3 = nn.Sequential(
#                     nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
#                                 padding=padding, dilation=dilation,
#                                 groups=groups, bias=False),
#                     norm_layer(planes),
#                     )
#         # 普通卷积
#         self.k4 = nn.Sequential(
#                     nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
#                                 padding=padding, dilation=dilation,
#                                 groups=groups, bias=False),
#                     norm_layer(planes),
#                     )
#
#     def forward(self, x):
#         identity = x
#
#         x_up = F.interpolate(self.k2(x), identity.size()[2:])
#         # sigmoid(identity + k2)
#         out = torch.sigmoid(torch.add(identity,x_up))
#         # k3 * sigmoid(identity + k2)
#         out = torch.mul(self.k3(x), out)
#         out = self.k4(out) # k4
#
#         return out
#
# class SC(nn.Module):
#     """SCNet SCBottleneck
#     """
#     expansion = 4
#     pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.
#
#     def __init__(self, inplanes, planes,kernel_size=1, stride=1, downsample=None,
#                  cardinality=1, bottleneck_width=32,padding=1,bias=False,
#                  avd=False, dilation=1, is_first=False,
#                  norm_layer=nn.BatchNorm2d):
#         super(SC, self).__init__()
#         group_width = int(planes * (bottleneck_width / 64.)) * cardinality
#
#         # 这里是使用 1 x 1卷积进行改变通道数
#         self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=True)
#         self.bn1_a = norm_layer(group_width)
#         self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=True)
#         self.bn1_b = norm_layer(group_width)
#
#         # conv1_a 代表下面  X2
#         # conv1_b 代表上面  X1
#         self.avd = avd and (stride > 1 or is_first)
#
#         if self.avd:
#             self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
#             stride = 1
#
#         self.k1 = nn.Sequential(
#                     nn.Conv2d(
#                         group_width, group_width, kernel_size=3, stride=stride,
#                         padding=dilation, dilation=dilation,
#                         groups=cardinality, bias=True),
#                     norm_layer(group_width),
#                     )
#
#         self.scconv = SCConv(
#             group_width, group_width, stride=stride,
#             padding=dilation, dilation=dilation,
#             groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)
#         # 这个conv3并不在图里，是 Y之后
#         self.conv3 = nn.Conv2d(
#             group_width * 2, planes, kernel_size=kernel_size, padding=padding,bias=True)
#         self.bn3 = norm_layer(planes)
#
#         self.relu = nn.ReLU(inplace=False)
#         self.downsample = downsample
#         self.dilation = dilation
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out_a= self.conv1_a(x)
#         out_a = self.bn1_a(out_a)
#         out_b = self.conv1_b(x)
#         out_b = self.bn1_b(out_b)
#         out_a = self.relu(out_a)
#         out_b = self.relu(out_b)
#
#         out_a = self.k1(out_a)
#
#         out_b = self.scconv(out_b)
#         out_a = self.relu(out_a)
#         out_b = self.relu(out_b)
#
#         if self.avd:
#             out_a = self.avd_layer(out_a)
#             out_b = self.avd_layer(out_b)
#
#         out = self.conv3(torch.cat([out_a, out_b], dim=1))
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         # out += residual
#         out = self.relu(out)
#
#         return out
# class hash(Function):
#     @staticmethod
#     def forward(ctx, input):
#         #ctx.save_for_backward(input)
#         return torch.sign(input)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         #input,  = ctx.saved_tensors
#         #grad_output = grad_output.data
#
#         return grad_output
#
#
# def hash_layer(input):
#     return hash.apply(input)
class SCConv(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
                    # 下采样   pooling_r
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=True),
                    norm_layer(planes),
                    )
        # 普通卷积
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=True),
                    norm_layer(planes),
                    )
        # 普通卷积
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=True),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        x_up = F.interpolate(self.k2(x), identity.size()[2:])
        # sigmoid(identity + k2)
        out = torch.sigmoid(torch.add(identity,x_up))
        # k3 * sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)
        out = self.k4(out) # k4

        return out

class SC(nn.Module):
    """SCNet SCBottleneck
    """
    expansion = 4
    pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, inplanes, planes,kernel_size=1, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32,padding=0,bias=True,
                 avd=False, dilation=1, is_first=False,
                 norm_layer=nn.BatchNorm2d):
        super(SC, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality

        # 这里是使用 1 x 1卷积进行改变通道数
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=True)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=True)
        self.bn1_b = norm_layer(group_width)

        # conv1_a 代表下面  X2
        # conv1_b 代表上面  X1
        # self.avd = avd and (stride > 1 or is_first)
        #
        # if self.avd:
        #     self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
        #     stride = 1

        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=1,
                        padding=dilation, dilation=dilation,
                        groups=cardinality, bias=True),
                    norm_layer(group_width),
                    )

        self.scconv = SCConv(
            group_width, group_width, stride=1,
            padding=dilation, dilation=dilation,
            groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)
        # 这个conv3并不在图里，是 Y之后
        self.conv3 = nn.Conv2d(
            group_width * 2, planes, kernel_size=kernel_size,stride=stride, padding=padding,bias=bias)
        self.bn3 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_a = self.bn1_a(out_a)
        out_b = self.conv1_b(x)
        out_b = self.bn1_b(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        out_a = self.k1(out_a)

        out_b = self.scconv(out_b)
        out_a = self.relu(out_a)
        out_b = self.relu(out_b)

        # if self.avd:
        #     out_a = self.avd_layer(out_a)
        #     out_b = self.avd_layer(out_b)
        out = torch.cat([out_a, out_b] ,dim=1)
        out = self.conv3(out)
        out = self.bn3(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out += residual
        out = self.relu(out)

        return out


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
            nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=False),
        
            # Reshape to 128 x (7x7)
            Reshape(self.ishape),

            # Upconvolution layers
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            #torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=False),
            
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
        self.sc_channels_64=SC(self.channels,64,4,stride=2,bias=True)
        self.sc_64_128 = SC(64, 128, 4, stride=2, bias=True)
        self.model = nn.Sequential(
            # Convolutional layers
            SC(self.channels, 64, 4, stride=2, bias=True),
            # nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            SC(64, 128, 4, stride=2, bias=True),
            # nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, latent_dim + n_c)
        )
        self.bottle = nn.Sequential(
            # # Convolutional layers
            # SC(self.channels, 64, 4, stride=2, bias=True),
            # # nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            # nn.LeakyReLU(0.2, inplace=False),
            # SC(64, 128, 4, stride=2, bias=True),
            # nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),

            # Flatten
            Reshape(self.lshape),

            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, latent_dim + n_c)
        )

        initialize_weights(self)
        
        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, in_feat):
        z_img = self.sc_channels_64(in_feat)
        z_img = F.leaky_relu(z_img,0.2,inplace=False)
        z_img = self.sc_64_128(z_img)
        z_img = self.bottle(z_img)



        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # Separate continuous and one-hot components
        zn = z[:, 0:self.latent_dim]
        # zn_hash = hash_layer(zn)
        zc_logits = z[:, self.latent_dim:]
        # Softmax on zc component
        zc = softmax(zc_logits)
        return zn,zc, zc_logits


class Discriminator_CNN_SC(nn.Module):
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
        self.sc_channels_64 = SC(self.channels, 64, 4, 2, bias=True)
        self.sc_64_128= SC(64,128,4,2,bias=True)
        self.reshape = Reshape(self.lshape)
        self.model = nn.Sequential(
            # Convolutional layers
            # SC(self.channels,64,4,2,bias=True),
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            # SC(64,128,4,2,bias=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            
            # Flatten
            Reshape(self.lshape),
            
            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, 1),
        )
        self.bottle = nn.Sequential(
            # # Convolutional layers
            # # SC(self.channels,64,4,2,bias=True),
            # nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            # nn.LeakyReLU(0.2, inplace=False),
            # # SC(64,128,4,2,bias=True),
            # nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),

            # Flatten
            Reshape(self.lshape),

            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, 1),

        )
        # If NOT using Wasserstein metric, final Sigmoid
        if (not self.wass):
            self.bottle = nn.Sequential(self.bottle, torch.nn.Sigmoid())

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, img):
        # Get output
        img= self.sc_channels_64(img)
        F.leaky_relu(img,0.2, inplace=False)
        img = self.sc_64_128(img)

        img = self.bottle(img)
        # img = validity = self.model(img)
        return img


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
# sc = Encoder_CNN(12,10)
# input = torch.randn(1,1,28,28)
# output = sc(input)
# print(output)
#
#
# d = Discriminator_CNN()
# input = torch.randn(1,1,28,28)
# output = d(input)
# print(output)