import torch
import torch.nn as nn
import torchvision.ops as ops

from .backbone.cnn import CNNBackbone
from .backbone.transformer import TransformerEncoder
from .multimodal.visual import VisualEmbedding
from .multimodal.text import TextEmbedding
from .multimodal.spatial import SpatialEmbedding
from .multimodal.fusion import FeatureFusion
from .heads.rpn import RegionProposalNetwork
from .heads.classifier import ClassificationHead
from .heads.relation_prediction import RelationPredictionNetwork
from ..utils.edge_features import EdgeFeatureExtractor


class MRNM(nn.Module):

    def __init__(self, num_classes=5, pretrained=True):
        super(MRNM, self).__init__()

        # 特征提取骨干网络
        self.backbone = CNNBackbone(pretrained=pretrained)

        # 多模态特征提取
        self.visual_embedding = VisualEmbedding(embedding_dim=1024)
        self.text_embedding = TextEmbedding(embedding_dim=1024)
        self.spatial_embedding = SpatialEmbedding(embedding_dim=1024)

        # 特征融合
        self.feature_fusion = FeatureFusion(embedding_dim=1024)

        # 区域建议网络
        self.rpn = RegionProposalNetwork(
            in_channels=256,
            anchor_sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # 区域分类头
        self.roi_head = ClassificationHead(
            in_channels=1024,
            hidden_channels=512,
            num_classes=num_classes
        )

        # 边特征提取器
        self.edge_feature_extractor = EdgeFeatureExtractor()

        # 关系预测网络
        self.relation_prediction_network = RelationPredictionNetwork(
            node_feature_dim=1024,
            edge_feature_dim=9,
            hidden_dim=512
        )

    def forward(self, images, boxes=None, texts=None, reading_order=None):
        batch_size = images.size(0)

        # 特征提取
        features = self.backbone(images)
        fused_features = features['fused']

        if self.training and boxes is not None:
            # 训练模式：使用标注的边界框
            proposals = boxes
        else:
            # 测试模式：使用RPN生成区域建议
            proposals, _ = self.rpn(images, features, boxes)

        # 多模态特征提取
        visual_embeds = []
        text_embeds = []
        spatial_embeds = []

        for i in range(batch_size):
            # 视觉特征
            visual_embed = self.visual_embedding(fused_features[i:i + 1], proposals[i])
            visual_embeds.append(visual_embed)

            if texts is not None:
                # 文本特征
                batch_texts = texts[i]
                text_embed = self.text_embedding(batch_texts)
                text_embeds.append(text_embed)

                # 空间特征
                spatial_embed = self.spatial_embedding(proposals[i], (images.size(2), images.size(3)))
                spatial_embeds.append(spatial_embed)

        # 特征融合
        fused_embeds = []

        for i in range(batch_size):
            if len(text_embeds) > 0 and len(spatial_embeds) > 0:
                fused_embed = self.feature_fusion(
                    visual_embeds[i],
                    text_embeds[i],
                    spatial_embeds[i],
                    is_paragraph=False  # 可以根据文本长度或类型自动判断
                )
                fused_embeds.append(fused_embed)
            else:
                # 如果没有文本或空间特征，则只使用视觉特征
                fused_embeds.append(visual_embeds[i])

        # 区域分类
        region_scores = []
        for i in range(batch_size):
            if len(fused_embeds[i]) > 0:
                scores = self.roi_head(fused_embeds[i])
                region_scores.append(scores)
            else:
                region_scores.append(None)

        # 关系预测
        edge_features_list = []
        adjacency_matrices = []
        node_embeddings_list = []
        relation_probs_list = []
        reading_orders_list = []

        if texts is not None:
            for i in range(batch_size):
                # 提取边特征和构建邻接矩阵
                edge_features, adjacency_matrix = self.edge_feature_extractor.extract_edge_features(
                    proposals[i], texts[i], (images.size(2), images.size(3))
                )

                # 节点特征 (使用融合的多模态特征)
                node_features = fused_embeds[i]

                # 构建和预测关系
                node_embeddings, relation_probs = self.relation_prediction_network(
                    node_features, adjacency_matrix, edge_features
                )

                # 预测阅读顺序
                reading_order = self.relation_prediction_network.predict_reading_order(relation_probs)

                edge_features_list.append(edge_features)
                adjacency_matrices.append(adjacency_matrix)
                node_embeddings_list.append(node_embeddings)
                relation_probs_list.append(relation_probs)
                reading_orders_list.append(reading_order)

        return {
            'proposals': proposals,
            'region_scores': region_scores,
            'edge_features': edge_features_list,
            'adjacency_matrices': adjacency_matrices,
            'node_embeddings': node_embeddings_list,
            'relation_probs': relation_probs_list,
            'reading_orders': reading_orders_list,
            'fused_features': fused_embeds,
            'visual_embeds': visual_embeds,
            'text_embeds': text_embeds,
            'spatial_embeds': spatial_embeds
        }

    def compute_losses(self, outputs, targets):
        losses = {}

        # 区域分类损失
        region_loss = 0
        for i, scores in enumerate(outputs['region_scores']):
            if scores is not None and targets['labels'][i] is not None:
                region_loss += nn.functional.cross_entropy(scores, targets['labels'][i])

        losses['region_loss'] = region_loss

        # 关系预测损失
        relation_loss = 0
        if 'relation_probs' in outputs and 'reading_order' in targets:
            for i, batch_relation_probs in enumerate(outputs['relation_probs']):
                if batch_relation_probs is not None and targets['reading_order'][i]:
                    # 构建目标矩阵
                    target_matrix = torch.zeros_like(batch_relation_probs)
                    for src, dst in targets['reading_order'][i]:
                        target_matrix[src, dst] = 1.0

                    # 二元交叉熵损失
                    relation_loss += nn.functional.binary_cross_entropy(
                        batch_relation_probs.view(-1),
                        target_matrix.view(-1)
                    )

        losses['relation_loss'] = relation_loss

        # 对比损失
        contrast_loss = 0
        visual_embeds = []
        text_embeds = []
        spatial_embeds = []

        if 'visual_embeds' in outputs and 'text_embeds' in outputs and 'spatial_embeds' in outputs:
            for i in range(len(outputs['visual_embeds'])):
                if i < len(outputs['text_embeds']) and i < len(outputs['spatial_embeds']):
                    visual_embeds.append(outputs['visual_embeds'][i])
                    text_embeds.append(outputs['text_embeds'][i])
                    spatial_embeds.append(outputs['spatial_embeds'][i])

        if visual_embeds and text_embeds and spatial_embeds:
            # 计算对比损失
            contrast_loss = self.feature_fusion.compute_contrastive_loss(
                torch.cat(visual_embeds, dim=0),
                torch.cat(text_embeds, dim=0),
                torch.cat(spatial_embeds, dim=0)
            )

        losses['contrast_loss'] = contrast_loss

        # 总损失
        alpha1, alpha2, alpha3 = 1.0, 0.5, 0.3  # 权重系数
        losses['total_loss'] = alpha1 * region_loss + alpha2 * relation_loss + alpha3 * contrast_loss

        return losses


# 导出MRNM类，使其可被导入
__all__ = ['MRNM']