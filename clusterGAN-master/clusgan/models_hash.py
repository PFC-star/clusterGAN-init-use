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

def normalize_gradient(net_D, x, **kwargs):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    x.requires_grad_(True)
    f = net_D(x, **kwargs)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), p=2, dim=1)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    f_hat = (f / (grad_norm + torch.abs(f)))
    return f_hat

# torch.backends.cudnn.benchmark = False
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input,  = ctx.saved_tensors
        # grad_output = grad_output.data

        return grad_output


def hash_layer(input):
    return hash.apply(input)


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
            # torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            # torch.nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),

            # Reshape to 128 x (7x7)
            Reshape(self.ishape),

            # Upconvolution layers
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            # torch.nn.ReLU(True),
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
        # z = z.unsqueeze(2).unsqueeze(3)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


# class Encoder_CNN(nn.Module):
#     """
#     CNN to model the encoder of a ClusterGAN
#     Input is vector X from image space if dimension X_dim
#     Output is vector z from representation space of dimension z_dim
#     """
#
#     def __init__(self, latent_dim, n_c, verbose=False, x_shape=(1, 28, 28), encode_length=200):
#         super(Encoder_CNN, self).__init__()
#
#         self.name = 'encoder'
#         self.x_shape = x_shape
#         self.channels = x_shape[0]
#         self.latent_dim = latent_dim
#         self.n_c = n_c
#         self.cshape = (128, self.x_shape[1] // 4 - 2, self.x_shape[2] // 4 - 2)
#         self.iels = int(np.prod(self.cshape))
#         self.lshape = (self.iels,)
#         self.verbose = verbose
#
#         self.model = nn.Sequential(
#             # Convolutional layers
#             nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
#             nn.LeakyReLU(0.2, inplace=False),
#             nn.Conv2d(64, 128, 4, stride=2, bias=True),
#             nn.LeakyReLU(0.2, inplace=False),
#
#             # Flatten
#             Reshape(self.lshape),
#
#             # Fully connected layers
#             torch.nn.Linear(self.iels, 1024),
#             nn.LeakyReLU(0.2, inplace=False),
#             torch.nn.Linear(1024, latent_dim + n_c)
#         )
#         self.fc_encoder = torch.nn.Linear(latent_dim, encode_length)
#         self.fc = torch.nn.Linear(encode_length, latent_dim, bias=False)
#         initialize_weights(self)
#
#         if self.verbose:
#             print("Setting up {}...\n".format(self.name))
#             print(self.model)
#
#     def forward(self, in_feat):
#         z_img = self.model(in_feat)
#         # Reshape for output
#         z = z_img.view(z_img.shape[0], -1)
#         # Separate continuous and one-hot components
#         zn = z[:, 0:self.latent_dim]
#         # zn(latent_dim) --> zn_H(encode_length)
#         zn_H = self.fc_encoder(zn)
#         # zn_H --> zn_B(hash code)
#         zn_B = hash_layer(zn_H)
#         # zn_B = hash_layer(zn)
#         zc_logits = z[:, self.latent_dim:]
#         # Softmax on zc component
#         # zn_B(encode_length) --> output(latent_dim)
#         output_zn = self.fc(zn_B)
#         # 这里的output_zn 与 zn 是同纬度的，output_zn是经过 hash code 映射的，zn是原始的输出
#         # 这里zc 也可以使用 hash方式 zc的逻辑值 与zc_idx进行 xe_loss,就是类似于greedy hash论文里的有监督分类方式
#         zc = softmax(zc_logits)
#
#         return output_zn,zn, zn_H, zn_B, zc, zc_logits

#



# Hash版本


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


class Encoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """

    def __init__(self, latent_dim, n_c, verbose=False, x_shape=(1, 28, 28), encode_length=200):
        super(Encoder_CNN, self).__init__()

        self.name = 'encoder'
        self.x_shape = x_shape
        self.channels = x_shape[0]
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, self.x_shape[1] // 4 - 2, self.x_shape[2] // 4 - 2)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=False),

            # Flatten
            Reshape(self.lshape),

            # Fully connected layers
            torch.nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=False),
            torch.nn.Linear(1024, latent_dim + n_c)
        )
        self.fc_encoder = torch.nn.Linear(latent_dim, encode_length)
        self.fc = torch.nn.Linear(encode_length, latent_dim, bias=False)
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
        # zn(latent_dim) --> zn_H(encode_length)
        zn_H = self.fc_encoder(zn)
        # zn_H --> zn_B(hash code)
        zn_B = hash_layer(zn_H)
        # zn_B = hash_layer(zn)
        zc_logits = z[:, self.latent_dim:]
        # Softmax on zc component
        # zn_B(encode_length) --> output(latent_dim)
        output_zn = self.fc(zn_B)
        # 这里的output_zn 与 zn 是同纬度的，output_zn是经过 hash code 映射的，zn是原始的输出
        # 这里zc 也可以使用 hash方式 zc的逻辑值 与zc_idx进行 xe_loss,就是类似于greedy hash论文里的有监督分类方式
        zc = softmax(zc_logits)
        # output_zn,zn_H,zn_B = (0,0,0)
        return output_zn,zn, zn_H, zn_B, zc, zc_logits

