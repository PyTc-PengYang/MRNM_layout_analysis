import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(ClassificationHead, self).__init__()

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, num_classes)

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        scores = self.fc2(x)

        return scores

    def predict(self, x):
        scores = self.forward(x)
        pred_scores, pred_classes = torch.max(F.softmax(scores, dim=-1), dim=-1)

        return pred_classes, pred_scores