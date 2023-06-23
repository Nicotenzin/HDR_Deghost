# -*- coding:utf-8 -*-
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from fightingcv_attention.conv.CondConv import *
import numbers


## Layer Norm

class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class DLA(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, expand_ratio=3):
        super(DLA, self).__init__()
        """
            Distributed Local Attention used for refining the attention map.
        """
        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.inp = inp
        self.oup = oup
        self.high_dim_id = False

        if self.expand_ratio != 1:
            self.conv_exp = Conv2dSamePadding(inp, hidden_dim, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.depth_sep_conv = Conv2dSamePadding(hidden_dim, hidden_dim, (kernel_size, kernel_size), stride, (1, 1),
                                                groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv_pro = Conv2dSamePadding(hidden_dim, oup, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        x = input
        if self.expand_ratio != 1:
            x = self.relu(self.bn1(self.conv_exp(x)))
        x = self.relu(self.bn2(self.depth_sep_conv(x)))
        x = self.bn3(self.conv_pro(x))
        if self.identity:
            return x + input
        else:
            return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.DLA = DLA(self.num_heads, self.num_heads)
        self.adapt_bn = nn.BatchNorm2d(self.num_heads)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)      # (B,dim*3,H,W) -> 3*(B,dim,H,W)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads) # (B, heads, dim//heads, H*W)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)        # (B, heads, dim//heads, dim//heads)

        # refine attention map
        attn = self.adapt_bn(self.DLA(attn))

        out = (attn @ v)          # (B, heads, dim//heads, H*W)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

"""
# Refine Attention Map in Mluti-Head Self-Attention
class Refined_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5
        self.DLA = DLA(self.num_heads, self.num_heads)
        self.adapt_bn = nn.BatchNorm2d(self.num_heads)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]     # (B, heads, N, D//heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # refine attention map
        attn = self.adapt_bn(self.DLA(attn))

        x = (attn @ v).transpose(1, 2).reshape(B, attn.shape[-1], C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
"""


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, complement):
        # x [B, N, C]
        B_x, N_x, C_x = x.shape

        x_copy = x

        complement = torch.cat([x, complement], 1)

        B_c, N_c, C_c = complement.shape

        # q [B, heads, N, C//num_heads]
        q = self.to_q(x).reshape(B_x, N_x, self.num_heads, C_x // self.num_heads).permute(0, 2, 1, 3)
        kv = self.to_kv(complement).reshape(B_c, N_c, 2, self.num_heads, C_c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_x, N_x, C_x)

        x = x + x_copy

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbbedingCrossAttention(nn.Module):
    def __init__(self, dim, patch_size, num_heads, norm_layer=nn.LayerNorm, drop_path=0.,
                 attn_drop=0., proj_drop=0.):
        super(PatchEmbbedingCrossAttention, self).__init__()
        self.patch_size = patch_size
        self.emb_dim = patch_size * patch_size * dim
        self.x_pos_embedding = nn.Parameter(torch.randn(1, self.num_patch, self.emb_dim))
        self.complement_pos_embedding = nn.Parameter(torch.randn(1, self.num_patch, self.emb_dim))
        self.x_norm1 = norm_layer(self.emb_dim)
        self.c_norm1 = norm_layer(self.emb_dim)
        self.attn = MultiHeadCrossAttention(self.emb_dim, num_heads, attn_drop, proj_drop)
        self.drop1 = nn.Dropout(drop_path)

    def forward(self, reference_features, non_reference_features):

        B, C, H, W = reference_features.shape
        num_patch = (H // self.patch_size) * (W // self.patch_size)

        x_pos_embedding = nn.Parameter(torch.randn(1, num_patch, self.emb_dim))
        complement_pos_embedding = nn.Parameter(torch.randn(1, num_patch, self.emb_dim))

        x = rearrange(reference_features, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=self.patch_size, p2=self.patch_size)
        x += x_pos_embedding

        complement = rearrange(non_reference_features, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                               p1=self.patch_size, p2=self.patch_size)
        complement += complement_pos_embedding

        x = self.x_norm1(x)
        complement = self.c_norm1(complement)

        x = x + self.drop1(self.attn(x, complement))

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=H // self.patch_size,
                      w=W // self.patch_size, p1=self.patch_size, p2=self.patch_size)

        return x


class SpatialAttentionModule(nn.Module):

    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map


class AttentionTransHDR(nn.Module):
    def __init__(self,
                 in_channels=6,
                 out_channels=3,
                 dim=64,
                 num_heads=8,
                 patch_size=4,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ffn_expansion_factor=2.66,
                 ):
        super(AttentionTransHDR, self).__init__()

        # Feature Extraction Network
        # coarse feature
        self.conv_f1 = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.conv_f2 = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.conv_f3 = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)

        # spatial attention module
        self.att_module_l = SpatialAttentionModule(n_feats=dim)
        self.att_module_h = SpatialAttentionModule(n_feats=dim)
        self.conv31_1 = nn.Conv2d(dim * 3, dim, kernel_size=3, stride=1, padding=1)

        # cross attention
        self.cross_att_l = PatchEmbbedingCrossAttention(dim, patch_size, num_heads)
        self.cross_att_h = PatchEmbbedingCrossAttention(dim, patch_size, num_heads)
        self.conv31_2 = nn.Conv2d(dim * 3, dim, kernel_size=3, stride=1, padding=1)
        self.conv31_3 = nn.Conv2d(dim * 3, dim, kernel_size=3, stride=1, padding=1)

        self.conv21_1 = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)

        # Feature Fusion Network
        # reconstruction
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.conv_last = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, low, middle, high):
        # feature extraction network
        # coarse feature
        f1 = self.conv_f1(low)
        f2 = self.conv_f2(middle)
        f3 = self.conv_f3(high)

        # spatial feature attention
        f1_att_m = self.att_module_h(f1, f2)
        f1_att = f1 * f1_att_m
        f3_att_m = self.att_module_l(f3, f2)
        f3_att = f3 * f3_att_m
        spa_att = self.conv31_1(torch.cat((f1_att, f2, f3_att), dim=1))

        # cross attention
        cross_att_low = self.cross_att_l(f2, f1)
        cross_att_high = self.cross_att_h(f2, f3)
        cross_att = self.conv31_2(torch.cat((cross_att_low, f2, cross_att_high), dim=1))

        x = torch.sigmoid(self.conv21_1(torch.cat((spa_att, cross_att), dim=1)))
        spa_att = spa_att * x
        cross_att = cross_att * x

        x = self.conv31_3(torch.cat((spa_att, f2, cross_att), dim=1))

        # reconstruction
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = self.conv_last(x + f2)
        x = torch.sigmoid(x)

        return x
