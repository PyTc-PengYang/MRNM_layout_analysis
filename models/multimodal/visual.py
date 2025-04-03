import torch
import torch.nn as nn
import torchvision.ops as ops


class VisualEmbedding(nn.Module):
    def __init__(self, embedding_dim=1024):
        super(VisualEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        self.fc = nn.Linear(256 * 7 * 7, embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, features, regions):
        batch_size = features.size(0)
        visual_embeds = []

        for i in range(batch_size):
            if len(regions[i]) > 0:
                roi_features = ops.roi_align(
                    features[i:i + 1],
                    [regions[i]],
                    output_size=(7, 7),
                    spatial_scale=1.0 / 16.0,  
                    sampling_ratio=2
                )

                roi_features = roi_features.view(roi_features.size(0), -1)
                region_embeds = self.fc(roi_features)
                region_embeds = self.relu(region_embeds)
                region_embeds = self.layer_norm(region_embeds)

                visual_embeds.append(region_embeds)
            else:
                visual_embeds.append(torch.zeros((0, self.embedding_dim), device=features.device))

        return visual_embeds
