import torch
import numpy as np
import math


def calculate_bbox_area(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height


def calculate_bbox_center(bbox):
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return (center_x, center_y)


def coordinate_transform(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox

    # 计算中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # 坐标归一化
    center_x_norm = center_x / image_width
    center_y_norm = center_y / image_height

    # 转换为极坐标
    x_polar = math.cos(2 * math.pi * center_x_norm)
    y_polar = math.sin(2 * math.pi * center_y_norm)

    # 边界框右下角点的极坐标表示
    x2_norm = x2 / image_width
    y2_norm = y2 / image_height

    x2_polar = math.cos(2 * math.pi * x2_norm)
    y2_polar = math.sin(2 * math.pi * y2_norm)

    return torch.tensor([
        center_x_norm, center_y_norm,
        x_polar, y_polar,
        x2_polar, y2_polar
    ])