from __future__ import print_function
# 引入transformer
try:
    import numpy as np
    import math
    import torch
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint as checkpoint
    from torch.nn.utils import spectral_norm

    from itertools import chain as ichain


    from clusgan.utils import tlog, softmax, initialize_weights, calc_gradient_penalty
    from clusgan.basic_layers import (EqualLinear, PixelNorm,
                                      SinusoidalPositionalEmbedding, Upsample)
    from clusgan.basic_layers import (Blur, Downsample, EqualConv2d, EqualLinear,
                                      ScaledLeakyReLU)
    from op import FusedLeakyReLU, upfirdn2d

    from timm.models.layers import to_2tuple, trunc_normal_


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



class ToRGB(nn.Module):
    def __init__(self, in_channel, upsample=True, resolution=None, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.is_upsample = upsample
        self.resolution = resolution

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = nn.Conv2d(in_channel, 1, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, skip=None):
        out = self.conv(input)
        out = out + self.bias

        if skip is not None:
            if self.is_upsample:
                skip = self.upsample(skip)

            out = out + skip
        return out

    def flops(self):
        m = self.conv
        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        bias_ops = 1
        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        flops = 1 * self.resolution * self.resolution * 3 * (m.in_channels // m.groups * kernel_ops + bias_ops)
        if self.is_upsample:
            # there is a conv used in upsample
            w_shape = (1, 1, 4, 4)
            kernel_ops = torch.zeros(w_shape[2:]).numel()  # Kw x Kh
            # N x Cout x H x W x  (Cin x Kw x Kh + bias)
            flops = 1 * 3 * (2 * self.resolution + 3) * (2 * self.resolution + 3) * (3 * kernel_ops)
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: queries with shape of (num_windows*B, N, C)
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).unsqueeze(-1)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta
        return out


class StyleSwinTransformerBlock(nn.Module):
    r""" StyleSwin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        style_dim (int): Dimension of style vector.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, style_dim=512):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = self.window_size // 2
        self.style_dim = style_dim
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = AdaptiveInstanceNorm(dim, style_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn = nn.ModuleList([
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])

        attn_mask1 = None
        attn_mask2 = None
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))

        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)

        self.norm2 = AdaptiveInstanceNorm(dim, style_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, style):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Double Attn
        shortcut = x
        x = self.norm1(x.transpose(-1, -2), style).transpose(-1, -2)

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        if self.shift_size > 0:
            qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(
                3, B, H, W, C // 2)
        else:
            qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)

        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, self.attn_mask1)
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, self.attn_mask2)

        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x2 = x2

        x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)
        x = self.proj(x)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x.transpose(-1, -2), style).transpose(-1, -2))

        return x

    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size,
                                                               C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size,
                                                               C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size,
                                                               C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += 1 * self.style_dim * self.dim * 2
        flops += 2 * (H * W) * self.dim
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        for attn in self.attn:
            flops += nW * (attn.flops(self.window_size * self.window_size))
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += 1 * self.style_dim * self.dim * 2
        flops += 2 * (H * W) * self.dim
        return flops


class StyleBasicLayer(nn.Module):
    """ A basic StyleSwin layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        out_dim (int): Number of output channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        style_dim (int): Dimension of style vector.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, out_dim=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., upsample=None,
                 use_checkpoint=False, style_dim=512):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            StyleSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                      num_heads=num_heads, window_size=window_size,
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop, style_dim=style_dim)
            for _ in range(depth)])

        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim)
        else:
            self.upsample = None

    def forward(self, x, latent1, latent2):
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.blocks[0], x, latent1)
            x = checkpoint.checkpoint(self.blocks[1], x, latent2)
        else:
            x = self.blocks[0](x, latent1)
            x = self.blocks[1](x, latent2)

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.upsample is not None:
            flops += self.upsample.flops()
        return flops


class BilinearUpsample(nn.Module):
    """ BilinearUpsample Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, input_resolution, dim, out_dim=None):
        super().__init__()
        assert dim % 2 == 0, f"x dim are not even."
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, out_dim, bias=False)
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.alpha = nn.Parameter(torch.zeros(1))
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=out_dim // 2, padding_idx=0,
                                                           init_size=out_dim // 2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert C == self.dim, "wrong in PatchMerging"

        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()  # B,C,H,W
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L * 4, C)  # B,H,W,C
        x = self.norm(x)
        x = self.reduction(x)

        # Add SPE
        x = x.reshape(B, H * 2, W * 2, self.out_dim).permute(0, 3, 1, 2)
        x += self.sin_pos_embed.make_grid2d(H * 2, W * 2, B) * self.alpha
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * 2 * W * 2, self.out_dim)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        # LN
        flops = 4 * H * W * self.dim
        # proj
        flops += 4 * H * W * self.dim * (self.out_dim)
        # SPE
        flops += 4 * H * W * 2
        # bilinear
        flops += 4 * self.input_resolution[0] * self.input_resolution[1] * self.dim * 5
        return flops


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class Generator_CNN(nn.Module):
    def __init__(
            self,
            latent_dim=30,
            n_c=10,
            x_shape=(1,16,16),
            size=16,
            style_dim=64,
            n_mlp=8,
            channel_multiplier=2,
            lr_mlp=0.01,
            enable_full_resolution=8,
            mlp_ratio=4,
            use_checkpoint=False,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0,
            attn_drop_rate=0,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.size = size
        self.mlp_ratio = mlp_ratio

        layers = [PixelNorm()]
        for _ in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.style = nn.Sequential(*layers)

        start = 2
        depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        in_channels = [
            512,
            512,
            512,
            512,
            256 * channel_multiplier,
            128 * channel_multiplier,
            64 * channel_multiplier,
            32 * channel_multiplier,
            16 * channel_multiplier
        ]

        end = int(math.log(size, 2))
        num_heads = [max(c // 32, 4) for c in in_channels]
        full_resolution_index = int(math.log(enable_full_resolution, 2))
        window_sizes = [2 ** i if i <= full_resolution_index else 8 for i in range(start, end + 1)]

        self.input = ConstantInput(in_channels[0])
        self.layers = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        num_layers = 0

        for i_layer in range(start, end + 1):
            in_channel = in_channels[i_layer - start]
            layer = StyleBasicLayer(dim=in_channel,
                                    input_resolution=(2 ** i_layer, 2 ** i_layer),  # 这地方可以修改为自己的网络分辨率架构
                                    depth=depths[i_layer - start],
                                    num_heads=num_heads[i_layer - start],
                                    window_size=window_sizes[i_layer - start],
                                    out_dim=in_channels[i_layer - start + 1] if (i_layer < end) else None,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    upsample=BilinearUpsample if (i_layer < end) else None,
                                    use_checkpoint=use_checkpoint, style_dim=style_dim)
            self.layers.append(layer)

            out_dim = in_channels[i_layer - start + 1] if (i_layer < end) else in_channels[i_layer - start]
            upsample = True if (i_layer < end) else False
            to_rgb = ToRGB(out_dim, upsample=upsample, resolution=(2 ** i_layer))
            self.to_rgbs.append(to_rgb)
            num_layers += 2

        self.n_latent = num_layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            noise,
            zc,
            return_latents=False,
            inject_index=None,
            truncation=1,
            truncation_latent=None,
    ):
        noise = torch.cat((noise, zc), 1)
        styles = self.style(noise)  # 这是FC层，特征解耦
        inject_index = self.n_latent  # 有多少个 A ，每一层有 2，一共有 8 个

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = torch.cat(style_t, dim=0)

        if styles.ndim < 3:
            latent = styles.unsqueeze(1).repeat(1, inject_index, 1)
        else:
            latent = styles

        x = self.input(latent)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        count = 0
        skip = None
        for layer, to_rgb in zip(self.layers, self.to_rgbs):
            x = layer(x, latent[:, count, :], latent[:, count + 1, :])
            b, n, c = x.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            skip = to_rgb(x.transpose(-1, -2).reshape(b, c, h, w), skip)
            count = count + 2

        B, L, C = x.shape

        assert L == self.size * self.size
        x = x.reshape(B, self.size, self.size, C).permute(0, 3, 1, 2).contiguous()
        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None

    def flops(self):
        flops = 0
        for _, layer in enumerate(self.layers):
            flops += layer.flops()
        for _, layer in enumerate(self.to_rgbs):
            flops += layer.flops()
        # 8 FC + PixelNorm
        flops += 1 * 10 * self.style_dim * self.style_dim
        return flops


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
    def __init__(self, wass_metric=True,size=16, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], sn=False, ssd=False):
        super().__init__()

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

                EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
                # EqualLinear(4608, channels[4], activation='fused_lrelu'),
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

#
# D = Discriminator(16)
# input = torch.randn(1, 1, 16, 16)
#
# output = D(input)
# print(output)





# noise = torch.randn((1, 64))
# g = Generator_CNN()
# fake_img, _ = g(noise)
# print(fake_img.shape)