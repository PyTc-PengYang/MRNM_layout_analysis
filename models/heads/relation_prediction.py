import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class RelationPredictionNetwork(nn.Module):
    def __init__(self, node_feature_dim=1024, edge_feature_dim=9, hidden_dim=512):
        super(RelationPredictionNetwork, self).__init__()

        # GNN层
        self.gnn_layer1 = nn.Linear(node_feature_dim, hidden_dim)
        self.gnn_layer2 = nn.Linear(hidden_dim, hidden_dim)

        # 距离预测头
        self.distance_head = nn.Linear(hidden_dim * 2, 1)

        # 方向预测头 (8个方向类别)
        self.direction_head = nn.Linear(hidden_dim * 2, 8)

        # 关系概率预测头
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, node_features, adjacency_matrix, edge_features=None):
        # 初始GNN层
        x = F.relu(self.gnn_layer1(node_features))

        # 消息传递 (根据论文公式)
        # ReLU(h_u^(l) + sum_{v in N(u)} A_{u,v} * W^(l) * h_v^(l))
        num_nodes = adjacency_matrix.size(0)
        h = x.clone()

        # 第一层消息传递
        for u in range(num_nodes):
            neighbor_sum = torch.zeros_like(h[u])
            for v in range(num_nodes):
                if adjacency_matrix[u, v] > 0:
                    neighbor_sum += adjacency_matrix[u, v] * h[v]
            h[u] = F.relu(h[u] + neighbor_sum)

        # 第二层GNN
        h = F.relu(self.gnn_layer2(h))

        # 第二层消息传递
        h_final = h.clone()
        for u in range(num_nodes):
            neighbor_sum = torch.zeros_like(h[u])
            for v in range(num_nodes):
                if adjacency_matrix[u, v] > 0:
                    neighbor_sum += adjacency_matrix[u, v] * h[v]
            h_final[u] = F.relu(h_final[u] + neighbor_sum)

        # 计算节点对之间的关系概率
        relation_probs = torch.zeros((num_nodes, num_nodes), device=node_features.device)

        for u in range(num_nodes):
            for v in range(num_nodes):
                if u != v:
                    # 拼接节点表示
                    node_pair = torch.cat([h_final[u], h_final[v]], dim=0)

                    if edge_features is not None:
                        # 如果提供了边特征，则拼接起来
                        edge_idx = u * num_nodes + v  # 假设边特征是按照(u,v)对索引的
                        node_edge_pair = torch.cat([node_pair, edge_features[edge_idx]], dim=0)
                        # 预测关系概率
                        relation_probs[u, v] = self.relation_head(node_edge_pair)
                    else:
                        # 如果没有边特征，则只使用节点特征
                        relation_probs[u, v] = self.relation_head(
                            torch.cat([node_pair, torch.zeros(9, device=node_features.device)])
                        )

        return h_final, relation_probs

    def compute_losses(self, node_embeddings, relation_probs, true_distances, true_directions, true_relations,
                       relative_distances=None):
        num_nodes = node_embeddings.size(0)

        # 预测距离和方向
        pred_distances = []
        pred_directions = []
        num_edges = 0

        for u in range(num_nodes):
            for v in range(num_nodes):
                if u != v:
                    # 拼接节点嵌入
                    node_pair = torch.cat([node_embeddings[u], node_embeddings[v]], dim=0)

                    # 预测距离
                    distance = self.distance_head(node_pair)
                    pred_distances.append(distance)

                    # 预测方向
                    direction = self.direction_head(node_pair)
                    pred_directions.append(direction)

                    num_edges += 1

        if pred_distances:
            pred_distances = torch.cat(pred_distances)
            pred_directions = torch.stack(pred_directions)

            # 计算距离损失
            distance_loss = self.mse_loss(pred_distances, true_distances)

            # 计算方向损失
            direction_loss = self.ce_loss(pred_directions, true_directions)

            # 计算关系概率损失
            relation_loss = self.bce_loss(relation_probs.view(-1), true_relations.view(-1))

            # 计算总损失 (根据论文公式)
            # ROP_loss = (alpha1*loss_dis + alpha2*loss_dir + alpha3*loss_rela) * (1-r_{u,v})
            alpha1, alpha2, alpha3 = 0.3, 0.3, 0.4  # 可调超参数

            if relative_distances is not None:
                # 使用相对距离作为权重
                r = relative_distances.view(-1)
                weight = 1.0 - r

                weighted_loss = (
                                        alpha1 * distance_loss +
                                        alpha2 * direction_loss +
                                        alpha3 * relation_loss
                                ) * weight.mean()
            else:
                weighted_loss = (
                        alpha1 * distance_loss +
                        alpha2 * direction_loss +
                        alpha3 * relation_loss
                )

            loss_components = {
                'distance_loss': distance_loss.item(),
                'direction_loss': direction_loss.item(),
                'relation_loss': relation_loss.item(),
                'weighted_loss': weighted_loss.item()
            }

            return weighted_loss, loss_components
        else:
            # 处理空图的情况
            return torch.tensor(0.0, device=node_embeddings.device), {
                'distance_loss': 0.0,
                'direction_loss': 0.0,
                'relation_loss': 0.0,
                'weighted_loss': 0.0
            }

    def predict_reading_order(self, relation_probs):
        num_nodes = relation_probs.size(0)

        # 找出起始节点 (入度最小的节点)
        in_degrees = relation_probs.sum(dim=0)
        start_node = torch.argmin(in_degrees).item()

        # 使用贪心策略寻找最优路径
        visited = set([start_node])
        reading_order = [start_node]

        while len(visited) < num_nodes:
            current = reading_order[-1]
            next_probs = relation_probs[current].clone()

            # 将已访问节点的概率设为0
            for i in visited:
                next_probs[i] = 0

            # 选择最高概率的下一个节点
            next_node = torch.argmax(next_probs).item()
            reading_order.append(next_node)
            visited.add(next_node)

        return reading_order