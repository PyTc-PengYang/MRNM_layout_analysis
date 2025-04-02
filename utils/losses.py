import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative=None):
        # 如果没有提供负样本，则只计算正样本对之间的损失
        if negative is None:
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(anchor, positive)
            # 目标是最大化相似度
            return torch.mean(1.0 - cos_sim)

        # 如果提供了负样本，则同时考虑正样本和负样本
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)

        # 目标是最小化正样本距离，最大化负样本距离
        loss = torch.mean(pos_dist + F.relu(self.margin - neg_dist))

        return loss


class RegularizationLoss(nn.Module):
    def __init__(self, weight=0.01):
        super(RegularizationLoss, self).__init__()
        self.weight = weight

    def forward(self, features):
        return self.weight * torch.norm(features, p=2)