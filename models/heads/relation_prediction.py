import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class RelationPredictionNetwork(nn.Module):
    def __init__(self, node_feature_dim=1024, edge_feature_dim=9, hidden_dim=512):
        super(RelationPredictionNetwork, self).__init__()

        self.gnn_layer1 = nn.Linear(node_feature_dim, hidden_dim)
        self.gnn_layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.distance_head = nn.Linear(hidden_dim * 2, 1)

        self.direction_head = nn.Linear(hidden_dim * 2, 8)

        self.relation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, node_features, adjacency_matrix, edge_features=None):
        x = F.relu(self.gnn_layer1(node_features))

        num_nodes = adjacency_matrix.size(0)
        h = x.clone()
        
        for u in range(num_nodes):
            neighbor_sum = torch.zeros_like(h[u])
            for v in range(num_nodes):
                if adjacency_matrix[u, v] > 0:
                    neighbor_sum += adjacency_matrix[u, v] * h[v]
            h[u] = F.relu(h[u] + neighbor_sum)

        h = F.relu(self.gnn_layer2(h))

        h_final = h.clone()
        for u in range(num_nodes):
            neighbor_sum = torch.zeros_like(h[u])
            for v in range(num_nodes):
                if adjacency_matrix[u, v] > 0:
                    neighbor_sum += adjacency_matrix[u, v] * h[v]
            h_final[u] = F.relu(h_final[u] + neighbor_sum)

        relation_probs = torch.zeros((num_nodes, num_nodes), device=node_features.device)

        for u in range(num_nodes):
            for v in range(num_nodes):
                if u != v:
                    node_pair = torch.cat([h_final[u], h_final[v]], dim=0)

                    if edge_features is not None:
                        edge_idx = u * num_nodes + v  
                        node_edge_pair = torch.cat([node_pair, edge_features[edge_idx]], dim=0)
                        relation_probs[u, v] = self.relation_head(node_edge_pair)
                    else:
                        relation_probs[u, v] = self.relation_head(
                            torch.cat([node_pair, torch.zeros(9, device=node_features.device)])
                        )

        return h_final, relation_probs

    def compute_losses(self, node_embeddings, relation_probs, true_distances, true_directions, true_relations,
                       relative_distances=None):
        num_nodes = node_embeddings.size(0)

        pred_distances = []
        pred_directions = []
        num_edges = 0

        for u in range(num_nodes):
            for v in range(num_nodes):
                if u != v:
                    node_pair = torch.cat([node_embeddings[u], node_embeddings[v]], dim=0)

                    distance = self.distance_head(node_pair)
                    pred_distances.append(distance)

                    direction = self.direction_head(node_pair)
                    pred_directions.append(direction)

                    num_edges += 1

        if pred_distances:
            pred_distances = torch.cat(pred_distances)
            pred_directions = torch.stack(pred_directions)

            distance_loss = self.mse_loss(pred_distances, true_distances)

            direction_loss = self.ce_loss(pred_directions, true_directions)

            relation_loss = self.bce_loss(relation_probs.view(-1), true_relations.view(-1))

            alpha1, alpha2, alpha3 = 0.2, 0.2, 0.6  

            if relative_distances is not None:
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
            return torch.tensor(0.0, device=node_embeddings.device), {
                'distance_loss': 0.0,
                'direction_loss': 0.0,
                'relation_loss': 0.0,
                'weighted_loss': 0.0
            }

    def predict_reading_order(self, relation_probs):
        num_nodes = relation_probs.size(0)

        in_degrees = relation_probs.sum(dim=0)
        start_node = torch.argmin(in_degrees).item()

        visited = set([start_node])
        reading_order = [start_node]

        while len(visited) < num_nodes:
            current = reading_order[-1]
            next_probs = relation_probs[current].clone()

            for i in visited:
                next_probs[i] = 0

            next_node = torch.argmax(next_probs).item()
            reading_order.append(next_node)
            visited.add(next_node)

        return reading_order
