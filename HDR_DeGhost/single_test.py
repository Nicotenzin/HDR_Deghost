# -*- coding:utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from skimage.color import rgb2hsv
from PIL import Image


class RGB_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb


def blur_tensor(tensor):
    """
    对tensor张量进行均值滤波操作
    :param tensor: 输入的0-1范围的tensor张量
    :param kernel_size: 滤波器的大小，默认为3
    :return: 模糊化后的0-1范围的tensor张量
    """
    # 使用平均滤波器对tensor进行模糊化
    blurred = F.avg_pool2d(tensor, kernel_size=11, stride=1, padding=5)

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


def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)
    # 将tensor张量转化为numpy数组
    numpy_array = tensor.cpu().detach().numpy()
    # 将取值范围从[0,1]转化为[0,255]
    numpy_array = np.transpose(numpy_array, (1, 2, 0)) * 255.0
    # 将浮点型数据转化为整型数据
    numpy_array = numpy_array.astype(np.uint8)
    # 输出图像
    plt.imshow(numpy_array)
    plt.show()


def tif_to_tensor(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 读取tif图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色通道顺序
    img = img.astype('float32') / 65535.0  # 归一化到0~1范围
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 将numpy数组转换为tensor张量

    return tensor


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


def main():
    image0 = tif_to_tensor('./data/Test/009/262A3239.tif')
    image1 = tif_to_tensor('./data/Test/009/262A3240.tif')
    image2 = tif_to_tensor('./data/Test/009/262A3241.tif')

    image0 = blur_tensor(image0)
    image1 = blur_tensor(image1)
    image2 = blur_tensor(image2)

    tensor_to_image(image0)
    tensor_to_image(image1)

    # equal the brightness
    image0 = match_brightness(image0, image1)
    image2 = match_brightness(image2, image1)

    tensor_to_image(image0)

    # get mask
    mask = motion_mask(image0, image1, threshold=0.3)
    print(mask.shape)

    tensor_to_image(mask)

    # opening operation
    # erode mask
    mask_erode = erode_mask(mask, kernel_size=7, iterations=1)
    tensor_to_image(mask_erode)

    # dilate mask
    mask_dilate = dilate_mask(mask_erode, kernel_size=10, iterations=1)
    tensor_to_image(mask_dilate)


if __name__ == '__main__':
    main()
