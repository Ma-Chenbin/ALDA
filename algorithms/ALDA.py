# Chenbin Ma July 2021
# This is the implementation of ITTR in PyTorch


# !/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
try:
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
# from torch.cuda import amp

class DeformConv2dFunction(Function):
    @staticmethod
    @amp.float_function
    def forward(ctx, input, offset, weight, bias,
                stride, padding, dilation, group, deformable_groups, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.group = group
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        output = deform_conv2d_forward(input, weight, bias,
                                         offset,
                                         ctx.kernel_size[0], ctx.kernel_size[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.group,
                                         ctx.deformable_groups,
                                         ctx.im2col_step)
        ctx.save_for_backward(input, offset, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    @amp.float_function
    def backward(ctx, grad_output):
        input, offset, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_weight, grad_bias = \
            deform_conv2d_backward(input, weight,
                                     bias,
                                     offset,
                                     grad_output,
                                     ctx.kernel_size[0], ctx.kernel_size[1],
                                     ctx.stride[0], ctx.stride[1],
                                     ctx.padding[0], ctx.padding[1],
                                     ctx.dilation[0], ctx.dilation[1],
                                     ctx.group,
                                     ctx.deformable_groups,
                                     ctx.im2col_step)

        return grad_input, grad_offset, grad_weight, grad_bias,\
            None, None, None, None, None, None



class DeformConv2d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=32, bias=True):
        super(DeformConv2d, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False
            self.bias.data.zero_()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
               offset.shape[1]
        return DeformConv2dFunction.apply(input, offset,
                                          self.weight,
                                          self.bias,
                                          self.stride,
                                          self.padding,
                                          self.dilation,
                                          self.groups,
                                          self.deformable_groups,
                                          self.im2col_step)


_DeformConv2d = DeformConv2dFunction.apply


class DeformConv2dPack(DeformConv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=32, bias=True, lr_mult=0.1):
        super(DeformConv2dPack, self).__init__(in_channels, out_channels,
                                               kernel_size, stride, padding, dilation, groups, deformable_groups,
                                               im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels,
                                     out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.conv_offset.inited = True
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input, return_offset=False):
        offset = self.conv_offset(input)
        bs = input.size()[0]
        im2col_step = bs // 2 if bs > 1 else 1
        out = DeformConv2dFunction.apply(input, offset,
                                         self.weight,
                                         self.bias,
                                         self.stride,
                                         self.padding,
                                         self.dilation,
                                         self.groups,
                                         self.deformable_groups,
                                         im2col_step)
        if return_offset:
            return out, offset
        return out, None


class DeformConv2dPackMore(DeformConv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConv2dPackMore, self).__init__(in_channels, out_channels,
                                                   kernel_size, stride, padding, dilation, groups, deformable_groups,
                                                   im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // 4, out_channels, kernel_size=self.kernel_size, stride=self.stride,
                      padding=self.padding, bias=True)
        )
        self.conv_offset[-1].lr_mult = lr_mult
        self.conv_offset[-1].inited = True
        self.init_offset()

    def init_offset(self):
        self.conv_offset[-1].weight.data.zero_()
        self.conv_offset[-1].bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        bs = input.size()[0]
        im2col_step = bs // 2
        return DeformConv2dFunction.apply(input, offset,
                                          self.weight,
                                          self.bias,
                                          self.stride,
                                          self.padding,
                                          self.dilation,
                                          self.groups,
                                          self.deformable_groups,
                                          im2col_step)


def conv_flops(k, c_in, c_out, stride, padding, resolution, bias=True, dialation=1):
    batch_size = 1
    output_dims = math.floor((resolution + 2*padding - dialation*(k - 1) - 1) / stride + 1)
    kernel_dims = k
    in_channels = c_in
    out_channels = c_out
    groups = 1

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(kernel_dims**2) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(output_dims**2)

    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0

    if bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        # _, _, H, W = x.shape
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class DeformablePatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.kernel_size = 2
        self.stride = 2
        self.padding = 0
        self.c_in = dim
        self.c_out = dim*2
        self.dconv = DeformConv2dPack(dim, dim*2, kernel_size=2, stride=2, padding=0)
        self.norm_layer = nn.BatchNorm2d(dim*2)
        self.act_layer = nn.GELU()

    def forward(self, x, return_offset=False):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x, offset = self.dconv(x, return_offset=False)
        x = self.act_layer(self.norm_layer(x)).flatten(2).transpose(1, 2)
        if return_offset:
            return x, offset
        else:
            return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        offset_flops = conv_flops(self.kernel_size, self.c_in, 12, self.stride, self.padding, H)
        dconv_flops = conv_flops(self.kernel_size, self.c_in, self.c_out, self.stride, self.padding, H)
        deformable_flops = offset_flops + dconv_flops
        # norm layer
        norm_flops = (H // 2) * (W // 2) * 2 * self.dim

        return deformable_flops + norm_flops


def conv_flops(k, c_in, c_out, stride, padding, resolution, bias=True, dialation=1):
    batch_size = 1
    output_dims = math.floor((resolution + 2*padding - dialation*(k - 1) - 1) / stride + 1)
    kernel_dims = k
    in_channels = c_in
    out_channels = c_out
    groups = 1

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(kernel_dims**2) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(output_dims**2)

    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0

    if bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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


class RelPosAttention(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_resolution[0] - 1) * (2 * input_resolution[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.input_resolution[0])
        coords_w = torch.arange(self.input_resolution[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.input_resolution[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.input_resolution[1] - 1
        relative_coords[:, :, 0] *= 2 * self.input_resolution[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.input_resolution[0] * self.input_resolution[1], self.input_resolution[0] * self.input_resolution[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}'

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



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
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

class MLPBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.sr_ratio=sr_ratio
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class Block(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        # print('input resolution', input_resolution)
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = RelPosAttention(
            dim, input_resolution=input_resolution, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # attn
        msa_flops = self.attn.flops(H * W)
        # mlp
        mlp_flops = 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W + msa_flops + mlp_flops
        return flops


class LITLayer(nn.Module):
    """ A basic LIT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, has_msa=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        block = Block if has_msa else MLPBlock
        self.blocks = nn.ModuleList([
            block(dim=dim, input_resolution=input_resolution,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop, attn_drop=attn_drop,
                  drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                  norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

def flops_to_string(flops, units='G', precision=2):
    if units.startswith('G'):
        return str(round(flops / 10. ** 9, precision)) + ' ' + units
    elif units.startswith('M'):
        return str(round(flops / 10. ** 6, precision)) + ' ' + units
    elif units.startswith('K'):
        return str(round(flops / 10. ** 3, precision)) + ' ' + units
    else:
        return str(flops) + units


class PDEM(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, has_msa=[0, 0, 1, 1], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.has_msa = has_msa

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = LITLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=DeformablePatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint, has_msa=self.has_msa[i_layer]==1)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            temp = layer.flops()
            flops += temp
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

