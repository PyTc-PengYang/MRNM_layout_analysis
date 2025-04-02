import torch
import numpy as np
import math
from transformers import AutoTokenizer, AutoModel


class EdgeFeatureExtractor:
    def __init__(self, text_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)

    def extract_edge_features(self, regions, texts, image_size):
        num_regions = len(regions)
        H, W = image_size

        # 计算文本嵌入
        text_embeddings = self._compute_text_embeddings(texts)

        # 初始化边特征和邻接矩阵
        edge_features = []
        adjacency_matrix = torch.zeros((num_regions, num_regions), dtype=torch.float32)

        for u in range(num_regions):
            for v in range(num_regions):
                if u != v:
                    # 提取空间距离
                    spatial_distance = self._compute_spatial_distance(regions[u], regions[v], (W, H))

                    # 提取语义距离
                    semantic_distance = self._compute_semantic_distance(text_embeddings[u], text_embeddings[v])

                    # 计算相对距离 (公式: e_dis = alpha * log(d_S_{u,v} + 1) + beta * d_T_{u,v})
                    alpha, beta = 0.7, 0.3  # 权重系数
                    relative_distance = alpha * math.log(spatial_distance + 1) + beta * semantic_distance

                    # 计算方向 (D-LoS方法)
                    direction = self._compute_direction(regions[u], regions[v])

                    # 构建边特征
                    edge_feature = torch.tensor([
                        spatial_distance,  # 空间距离
                        semantic_distance,  # 语义距离
                        relative_distance,  # 相对距离
                        direction,  # 方向 (0-7)
                        regions[u][0] / W,  # 源节点x1归一化
                        regions[u][1] / H,  # 源节点y1归一化
                        regions[u][2] / W,  # 源节点x2归一化
                        regions[u][3] / H,  # 源节点y2归一化
                        1.0  # 边存在标志
                    ], dtype=torch.float32)

                    edge_features.append(edge_feature)

                    # 更新邻接矩阵 (公式: A_{u,v} = exp(-e_dis), if u and v are connected, 0 otherwise)
                    adjacency_matrix[u, v] = math.exp(-relative_distance)

        return torch.stack(edge_features), adjacency_matrix

    def _compute_text_embeddings(self, texts):
        if not texts:
            return []

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.text_model(**inputs)

        # 使用平均池化获取文本嵌入
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = outputs.last_hidden_state * attention_mask
        embeddings = embeddings.sum(1) / attention_mask.sum(1)

        return embeddings

    def _compute_spatial_distance(self, box1, box2, image_size):
        W, H = image_size

        # 计算原始坐标中的中心点
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2

        # 计算原始坐标距离
        dist_orig = math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

        # 计算变换坐标 (论文中的坐标变换)
        x1_polar = math.cos(2 * math.pi * center1_x / W)
        y1_polar = math.sin(2 * math.pi * center1_y / H)
        x2_polar = math.cos(2 * math.pi * center2_x / W)
        y2_polar = math.sin(2 * math.pi * center2_y / H)

        # 计算变换坐标距离
        dist_trans = math.sqrt((x1_polar - x2_polar) ** 2 + (y1_polar - y2_polar) ** 2)

        # 结合两种距离
        distance = (dist_orig + dist_trans) / 2

        return distance

    def _compute_semantic_distance(self, embedding1, embedding2):
        if isinstance(embedding1, torch.Tensor) and isinstance(embedding2, torch.Tensor):
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
            # 转换为距离 (1 - 相似度)
            distance = 1.0 - similarity.item()
        else:
            # 默认距离
            distance = 1.0

        return distance

    def _compute_direction(self, box1, box2):
        # 计算中心点
        center1_x = (box1[0] + box1[2]) / 2
        center1_y = (box1[1] + box1[3]) / 2
        center2_x = (box2[0] + box2[2]) / 2
        center2_y = (box2[1] + box2[3]) / 2

        # 计算角度
        dx = center2_x - center1_x
        dy = center2_y - center1_y
        angle = math.atan2(dy, dx)

        # 将角度转换为0-2π范围
        if angle < 0:
            angle += 2 * math.pi

        # 将角度映射到0-7的方向索引
        direction = int((angle / (2 * math.pi) * 8) % 8)

        return direction