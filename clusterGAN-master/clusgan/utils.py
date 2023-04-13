from __future__ import print_function

try:
    import os
    import random
    import numpy as np

    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad

    import torch.nn as nn
    import torch.nn.functional as F
    import torch

    from itertools import chain as ichain

except ImportError as e:
    print(e)
    raise ImportError

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')


# Nan-avoiding logarithm
def tlog(x):
    return torch.log(x + 1e-8)


# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)


# Cross Entropy loss with two vector inputs
def cross_entropy(pred, soft_targets):
    log_softmax_pred = torch.nn.functional.log_softmax(pred, dim=1)
    return torch.mean(torch.sum(- soft_targets * log_softmax_pred, 1))


# Save a provided model to file
# Save a provided model to file
# def save_model(models=[], out_dir=''):

#     # Ensure at least one model to save
#     assert len(models) > 0, "Must have at least one model to save."

#     # Save models to directory out_dir
#     for model in models:
#         filename = model.name + '.pth.tar'
#         outfile = os.path.join(out_dir, filename)
#         torch.save(model.state_dict(), outfile)
def save_model(models=[], iteration=0, optimizers=[], out_dir=''):
    # Ensure at least one model to save
    out_dir1 =  out_dir + '\\model_full.pth.tar'
    out_dir2 =  out_dir + '\\model_full_backup.pth.tar'
    assert len(models) > 0, "Must have at least one model to save."
    torch.save({'iteration': iteration,
                'optimizerD_dict': optimizers[0].state_dict(),
                'optimizerGE_dict': optimizers[1].state_dict(),
                'D_dict': models[0].state_dict(),
                'E_dict': models[1].state_dict(),
                'G_dict': models[2].state_dict()},
                out_dir1,_use_new_zipfile_serialization=False)
    # 防覆盖专用
    torch.save({'iteration': iteration,
                'optimizerD_dict': optimizers[0].state_dict(),
                'optimizerGE_dict': optimizers[1].state_dict(),
                'D_dict': models[0].state_dict(),
                'E_dict': models[1].state_dict(),
                'G_dict': models[2].state_dict()},
               out_dir2,_use_new_zipfile_serialization=False)
def load_model(models=[], iteration=0, optimizers=[], out_dir=''):
    out_dir = out_dir + '\\model_full.pth.tar'
    model_data = torch.load(out_dir,map_location=device)

    iteration = model_data['iteration']
    models[0].load_state_dict(model_data['D_dict'])
    models[1].load_state_dict(model_data['E_dict'])
    models[2].load_state_dict(model_data['G_dict'])
    optimizers[0].load_state_dict(model_data['optimizerD_dict'])
    optimizers[1].load_state_dict(model_data['optimizerGE_dict'])

    return iteration



# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        # elif isinstance(m, nn.Linear):
        #     m.weight.data.normal_(0, 0.02)
        #     m.bias.data.zero_()


def seed_torch(seed=0):
    pass
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Sample a random latent space vector
def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):
    assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c)), "Requested class %i outside bounds." % fix_class
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Sample noise as generator input, zn
    zn = Variable(Tensor(0.75 * np.random.normal(0, 1, (shape, latent_dim)), device=device), requires_grad=req_grad)
    # 标准差有问题，不是原论文的标准差
    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    zc_FT = Tensor(shape, n_c).fill_(0)  # 初始化为 0
    zc_idx = torch.empty(shape, dtype=torch.long)

    if (fix_class == -1):
        zc_idx = zc_idx.random_(n_c).to(device)
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.)
        # zc_idx = torch.empty(shape, dtype=torch.long).random_(n_c).cuda()
        # zc_FT = Tensor(shape, n_c).fill_(0).scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

        zc_idx = zc_idx.to(device)
        zc_FT = zc_FT.to(device)

    zc = Variable(zc_FT, requires_grad=req_grad)

    ## Gaussian-noisey vector generation
    # zc = Variable(Tensor(np.random.normal(0, 1, (shape, n_c))), requires_grad=req_grad)
    # zc = softmax(zc)
    # zc_idx = torch.argmax(zc, dim=1)

    # Return components of latent space variable
    return zn, zc, zc_idx


def calc_gradient_penalty(netD, real_data, generated_data):
    seed_torch()
    # GP strength
    LAMBDA = 10
    real_data.to(device)
    # print(real_data.type)
    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.to(device)
    alpha = alpha.expand_as(real_data)

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients1 = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients1 ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


def sign(x):
    return torch.sign(x)
