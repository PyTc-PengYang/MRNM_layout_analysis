import torch
import torch.nn as nn
import torchvision.models as models


class CNNBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(CNNBackbone, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.conv2 = resnet.layer1  # C2
        self.conv3 = resnet.layer2  # C3
        self.conv4 = resnet.layer3  # C4
        self.conv5 = resnet.layer4  # C5

        self.fusion_conv = nn.Conv2d(2048 + 1024 + 512, 256, kernel_size=3, padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)  # 256 channels
        c4 = self.conv4(c3)  # 512 channels
        c5 = self.conv5(c4)  # 1024 channels

        c4_upsampled = nn.functional.interpolate(c4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        c5_upsampled = nn.functional.interpolate(c5, size=c3.shape[2:], mode='bilinear', align_corners=False)

        fused_features = torch.cat([c3, c4_upsampled, c5_upsampled], dim=1)
        fused_features = self.fusion_conv(fused_features)  # 256 channels

        return {
            'c3': c3,
            'c4': c4,
            'c5': c5,
            'fused': fused_features
        }
