import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))

        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接并处理
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)


class FeatureFusion(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(FeatureFusion, self).__init__()

        # 全连接层用于融合特征
        self.fc = nn.Linear(embedding_dim * 3, embedding_dim)

        # 特征增强模块
        self.channel_attention = ChannelAttention(embedding_dim)
        self.spatial_attention = SpatialAttention()

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 损失函数
        self.contrastive_loss = nn.CosineEmbeddingLoss()

    def forward(self, visual_embeds, text_embeds, spatial_embeds, is_paragraph=False):
        # 拼接特征
        concat_features = torch.cat([visual_embeds, text_embeds, spatial_embeds], dim=-1)

        # 初步融合
        initial_fusion = self.fc(concat_features)

        # 根据是否为段落应用不同的注意力机制
        if is_paragraph:
            # 应用空间注意力 (用于文本段)
            spatial_attention = self.spatial_attention(initial_fusion.unsqueeze(-1).unsqueeze(-1))
            enhanced_features = initial_fusion.unsqueeze(-1).unsqueeze(-1) * spatial_attention
            enhanced_features = enhanced_features.squeeze(-1).squeeze(-1)
        else:
            # 应用通道注意力 (用于文本行)
            channel_attention = self.channel_attention(initial_fusion.unsqueeze(-1).unsqueeze(-1))
            enhanced_features = initial_fusion.unsqueeze(-1).unsqueeze(-1) * channel_attention
            enhanced_features = enhanced_features.squeeze(-1).squeeze(-1)

        # 通过Transformer编码器
        enhanced_features = enhanced_features.unsqueeze(1)  # 添加序列维度
        transformed_features = self.transformer_encoder(enhanced_features)

        return transformed_features.squeeze(1)  # 移除序列维度

    def compute_contrastive_loss(self, visual_embeds, text_embeds, spatial_embeds):
        batch_size = visual_embeds.size(0)
        target = torch.ones(batch_size, device=visual_embeds.device)

        # 视觉-文本对比损失
        vt_loss = self.contrastive_loss(visual_embeds, text_embeds, target)

        # 视觉-空间对比损失
        vs_loss = self.contrastive_loss(visual_embeds, spatial_embeds, target)

        # 文本-空间对比损失
        ts_loss = self.contrastive_loss(text_embeds, spatial_embeds, target)

        return vt_loss + vs_loss + ts_loss