# -*- coding:utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numbers

# get mask
def blur_tensor(tensor):
    """
    对tensor张量进行均值滤波操作
    :param tensor: 输入的0-1范围的tensor张量
    :param kernel_size: 滤波器的大小，默认为3
    :return: 模糊化后的0-1范围的tensor张量
    """
    # 使用平均滤波器对tensor进行模糊化
    blurred = F.avg_pool2d(tensor, kernel_size=5, stride=1, padding=2)

    return blurred


def match_brightness(img1, img2):
    """
    将img1的亮度调整到与img2一致，并保持颜色分布不变。
    """
    # 计算img1和img2的平均亮度
    mean1 = torch.mean(img1)
    mean2 = torch.mean(img2)

    # 计算调整因子
    factor = mean2 / mean1

    # 调整img1的亮度
    img1_matched = img1 * factor

    # 保持颜色分布不变
    img1_matched = torch.clamp(img1_matched, 0, 1)  # 将像素值限制在[0, 1]范围内
    img1_matched = (img1_matched - torch.min(img1_matched)) / (
            torch.max(img1_matched) - torch.min(img1_matched))  # 将像素值归一化到[0, 1]范围内
    img1_matched = img1_matched * (torch.max(img1) - torch.min(img1)) + torch.min(img1)  # 将像素值重新缩放到原始范围内

    return img1_matched


def motion_mask(img1, img2, threshold=0.3):
    """
    将一个BCHW的tensor张量中，如果某个像素一个通道的值大于阈值，则将这个元素所有通道的值设为1，否则设为0

    Args:
        tensor (torch.Tensor): 输入的BCHW的tensor张量
        threshold (float): 阈值，范围为0到1

    Returns:
        torch.Tensor: 输出的BCHW的tensor张量，其中每个像素的值都为0或1
    """
    diff = torch.abs(img1 - img2)

    mask = diff.new_zeros(diff.shape)

    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            mask[i, j, :, :] = diff[i, j, :, :] > threshold
        mask[i, :, :, :] = mask[i, :, :, :].sum(dim=0) > 0
        mask[i, :, :, :] = mask[i, :, :, :].float()
    # input = B,C,H,W 像素值（0，0，0）或（1，1，1）

    return mask


def dilate_mask(mask, kernel_size=3, iterations=1):
    b, c, h, w = mask.shape
    result = torch.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    for i in range(b):
        for j in range(c):
            img = mask[i, j].cpu().numpy()  # 将tensor转换为numpy数组
            img = img.astype('uint8') * 255  # 将0-1范围内的值转换为0-255范围内的值
            img = cv2.dilate(img, kernel, iterations=iterations)  # 进行膨胀操作
            img = img.astype('float32') / 255  # 将0-255范围内的值转换为0-1范围内的值
            result[i, j] = torch.from_numpy(img)  # 将numpy数组转换为tensor
    return result


def erode_mask(mask, kernel_size=11, iterations=1):
    b, c, h, w = mask.shape
    result = torch.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    for i in range(b):
        for j in range(c):
            img = mask[i, j].cpu().numpy()  # 将tensor转换为numpy数组
            img = img.astype('uint8') * 255  # 将0-1范围内的值转换为0-255范围内的值
            img = cv2.erode(img, kernel, iterations=iterations)  # 进行腐蚀操作
            img = img.astype('float32') / 255  # 将0-255范围内的值转换为0-1范围内的值
            result[i, j] = torch.from_numpy(img)  # 将numpy数组转换为tensor
    return result


class GetMask(nn.Module):

    def __init__(self, threshold=0.3, erode_kernel_size=11, dilate_kernel_size=11, literations=1):
        super().__init__()
        self.threshold = threshold
        self.erode_kernel_size = erode_kernel_size
        self.dilate_kernel_size = dilate_kernel_size
        self.literations = literations

    def forward(self, non_refer, refer):
        non_refer = blur_tensor(non_refer)
        refer = blur_tensor(refer)
        non_refer = match_brightness(non_refer, refer)
        mask = motion_mask(non_refer, refer, threshold=self.threshold)
        mask_erode = erode_mask(mask, kernel_size=self.erode_kernel_size, iterations=self.literations)
        ghost_mask = dilate_mask(mask_erode, kernel_size=self.dilate_kernel_size, iterations=self.literations)
        non_ghost_mask = 1 - ghost_mask

        return ghost_mask, non_ghost_mask


# local information
class MultiScaleCNN(nn.Module):

    def __init__(self, in_channels=64):
        super(MultiScaleCNN, self).__init__()

        # 第一个卷积层，卷积核为1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # 第二个卷积层，卷积核为3
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # 第三个卷积层，卷积核为5
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()

        # 第四个卷积层，卷积核为3
        self.conv4 = nn.Conv2d(in_channels=in_channels * 3, out_channels=in_channels, kernel_size=3, stride=1,
                               padding=1)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        # 第一个卷积层
        x1 = self.relu1(self.conv1(x))

        # 第二个卷积层
        x2 = self.relu2(self.conv2(x))

        # 第三个卷积层
        x3 = self.relu3(self.conv3(x))

        # 第四个卷积层
        x = self.relu4(self.conv4(torch.cat((x1, x2, x3), dim=1)))
        return x


class LocalExFeature(nn.Module):
    def __init__(self, in_channels=6, out_channels=64):
        super(LocalExFeature, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.ms_cnn1 = MultiScaleCNN(in_channels=out_channels)
        self.ms_cnn2 = MultiScaleCNN(in_channels=out_channels)
        self.ms_cnn3 = MultiScaleCNN(in_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.ms_cnn1(x)
        x = self.ms_cnn2(x)
        x = self.ms_cnn3(x)
        return x


# global information
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
        # x [B, HW, C]
        B_x, N_x, C_x = x.shape

        x_copy = x

        complement = torch.cat([x, complement], 1)

        B_c, N_c, C_c = complement.shape

        # q [B, heads, HW, C//num_heads]
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
    def __init__(self, dim, patch_size, num_heads, img_high=256, img_wide=256, norm_layer=nn.LayerNorm, drop_path=0.,
                 attn_drop=0., proj_drop=0.):
        super(PatchEmbbedingCrossAttention, self).__init__()
        self.img_high = img_high
        self.img_wide = img_wide
        self.num_patch = (img_high // patch_size) * (img_wide // patch_size)
        self.patch_size = patch_size
        self.emb_dim = patch_size * patch_size * dim
        self.x_pos_embedding = nn.Parameter(torch.randn(1, self.num_patch, self.emb_dim))
        self.complement_pos_embedding = nn.Parameter(torch.randn(1, self.num_patch, self.emb_dim))
        self.x_norm1 = norm_layer(self.emb_dim)
        self.c_norm1 = norm_layer(self.emb_dim)
        self.attn = MultiHeadCrossAttention(self.emb_dim, num_heads, attn_drop, proj_drop)
        self.drop1 = nn.Dropout(drop_path)

    def forward(self, reference_features, non_reference_features):
        x = rearrange(reference_features, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=self.patch_size, p2=self.patch_size)
        x += self.x_pos_embedding

        complement = rearrange(non_reference_features, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                               p1=self.patch_size, p2=self.patch_size)
        complement += self.complement_pos_embedding

        x = self.x_norm1(x)
        complement = self.c_norm1(complement)

        x = x + self.drop1(self.attn(x, complement))

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.img_high // self.patch_size,
                      w=self.img_wide // self.patch_size, p1=self.patch_size, p2=self.patch_size)

        return x


# feature fusion
class DownSampling(nn.Module):

    def __init__(self, in_channels=64):
        super().__init__()
        out_channels = in_channels * 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.ms_cnn1 = MultiScaleCNN(in_channels=out_channels)
        self.ms_cnn2 = MultiScaleCNN(in_channels=out_channels)

    def forward(self, feature):
        output = F.max_pool2d(feature, kernel_size=2, stride=2)
        output = self.relu1(self.conv1(output))
        output = self.ms_cnn1(output)
        output = self.ms_cnn2(output)

        return output


class UpSampling(nn.Module):

    def __init__(self, in_channels=128):
        super().__init__()
        out_channels = in_channels // 2
        self.conv_tr = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                               padding=1)
        self.relu1 = nn.ReLU()
        self.ms_cnn1 = MultiScaleCNN(in_channels=out_channels)
        self.ms_cnn2 = MultiScaleCNN(in_channels=out_channels)

    def forward(self, feature):
        output = self.conv_tr(feature)
        output = self.relu1(self.conv1(output))
        output = self.ms_cnn1(output)
        output = self.ms_cnn2(output)

        return output


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

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class MaskHDR(nn.Module):
    def __init__(self,
                 in_channels=6,
                 out_channels=3,
                 dim=48,
                 num_heads=4,
                 patch_size=4,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ffn_expansion_factor=2.66,
                 ):
        super(MaskHDR, self).__init__()
        self.in_channels = in_channels
        self.dim = dim

        # Feature Extraction Network
        # coarse feature
        self.mask_low = GetMask(threshold=0.3, erode_kernel_size=11, dilate_kernel_size=11, literations=1)
        self.mask_high = GetMask(threshold=0.3, erode_kernel_size=11, dilate_kernel_size=11, literations=1)

        self.conv_f1 = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.conv_f2 = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.conv_f3 = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.conv21_1 = LocalExFeature(in_channels=dim*2, out_channels=dim)
        self.conv21_2 = LocalExFeature(in_channels=dim*2, out_channels=dim)

        # cross attention
        self.cross_att_l = PatchEmbbedingCrossAttention(dim, patch_size, num_heads)
        self.cross_att_h = PatchEmbbedingCrossAttention(dim, patch_size, num_heads)
        self.conv31_1 = nn.Conv2d(dim * 3, dim, kernel_size=3, stride=1, padding=1)
        self.conv31_3 = nn.Conv2d(dim * 3, dim, kernel_size=3, stride=1, padding=1)

        # Feature Fusion Network
        # reconstruction
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.conv_dia1 = nn.Conv2d(in_channels=dim//3, out_channels=dim//3, kernel_size=3, padding=1)
        self.conv_dia2 = nn.Conv2d(in_channels=dim//3, out_channels=dim//3, kernel_size=3, padding=1, dilation=2)
        self.attn1 = Attention(dim//3, num_heads, bias)
        self.attn2 = Attention(dim//3, num_heads, bias)
        self.attn3 = Attention(dim//3, num_heads, bias)
        self.conv_last = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, low, middle, high):
        # feature extraction network
        # get mask
        ghost_low, non_low = self.mask_low(low, middle)
        ghost_high, non_high = self.mask_high(high, middle)

        ghost_low = torch.mean(ghost_low, dim=1, keepdim=True).repeat(1, self.dim, 1, 1)
        non_low = torch.mean(non_low, dim=1, keepdim=True).repeat(1, self.dim, 1, 1)
        ghost_high = torch.mean(ghost_high, dim=1, keepdim=True).repeat(1, self.dim, 1, 1)
        non_high = torch.mean(non_high, dim=1, keepdim=True).repeat(1, self.dim, 1, 1)

        # coarse feature
        f1 = self.conv_f1(low)
        f2 = self.conv_f2(middle)
        f3 = self.conv_f3(high)

        # low image
        f2_ghost_low = f2 * ghost_low
        f2_non_low = f2 * non_low
        cross_att_low = self.cross_att_l(f2_ghost_low, f1)
        local_low = self.conv21_1(torch.cat((f2_non_low, f1), dim=1))
        full_low = cross_att_low * ghost_low + local_low * non_low

        # high image
        f2_ghost_high = f2 * ghost_high
        f2_non_high = f2 * non_high
        cross_att_high = self.cross_att_h(f2_ghost_high, f3)
        local_high = self.conv21_2(torch.cat((f2_non_high, f3), dim=1))
        full_high = cross_att_high * ghost_high + local_high * non_high

        x = self.conv31_1(torch.cat((full_low, f2, full_high), dim=1))

        # reconstruction
        x1, x2, x3 = x.chunk(3, dim=1)
        x1 = x1 + self.attn1(self.norm1(self.conv_dia1(x1)))
        x2 = x2 + self.attn2(self.norm2(self.conv_dia2(x2)))
        x3 = x3 + self.attn3(self.norm3(x3))
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv_last(x + f2)
        x = torch.sigmoid(x)
        return x
