import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import random
from PIL import Image


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 获取原始尺寸
        original_size = image.shape[:2]

        # 调整图像大小
        image = F.resize(image, self.size)

        # 计算缩放因子
        h_scale = self.size[0] / original_size[0]
        w_scale = self.size[1] / original_size[1]

        # 调整边界框坐标
        if target is not None:
            target[:, 0] = target[:, 0] * w_scale
            target[:, 1] = target[:, 1] * h_scale
            target[:, 2] = target[:, 2] * w_scale
            target[:, 3] = target[:, 3] * h_scale

        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[:2]
            image = np.fliplr(image).copy()

            if target is not None:
                target[:, [0, 2]] = width - target[:, [2, 0]]

        return image, target


class ToTensor:
    def __call__(self, image, target):
        # 转换图像为Tensor
        image = F.to_tensor(image)

        # 转换边界框为Tensor
        if target is not None and not isinstance(target, torch.Tensor):
            target = torch.as_tensor(target, dtype=torch.float32)

        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def get_transform(train=True):
    transforms = []

    # 基本变换
    transforms.append(Resize((800, 800)))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # 训练时添加数据增强
    if train:
        transforms.append(RandomHorizontalFlip())

    return Compose(transforms)