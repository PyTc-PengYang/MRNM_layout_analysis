import torch
import torch.nn as nn
import math


class SpatialEmbedding(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(SpatialEmbedding, self).__init__()

        # 多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, boxes, image_size):
        batch_size, num_boxes, _ = boxes.shape
        H, W = image_size

        # 提取边界框特征
        x1, y1, x2, y2 = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2], boxes[:, :, 3]

        # 计算中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算宽高
        width = x2 - x1
        height = y2 - y1

        # 将坐标归一化到[0,1]范围
        center_x = center_x / W
        center_y = center_y / H
        width = width / W
        height = height / H

        # 将坐标映射到极坐标系，如论文所述
        x_polar = torch.cos(2 * math.pi * center_x)
        y_polar = torch.sin(2 * math.pi * center_y)
        x2_polar = torch.cos(2 * math.pi * x2 / W)
        y2_polar = torch.sin(2 * math.pi * y2 / H)

        # 拼接特征
        spatial_features = torch.stack([
            center_x, center_y,
            width, height,
            x_polar, y_polar,
            x2_polar, y2_polar
        ], dim=-1)

        # 通过MLP获取嵌入
        spatial_embeds = self.mlp(spatial_features)

        return spatial_embeds