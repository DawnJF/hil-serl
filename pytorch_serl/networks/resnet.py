"""ResNet编码器的PyTorch实现"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """ResNet-10编码器"""

    def __init__(
        self,
        pretrained=True,
        pooling_method="spatial_learned_embeddings",
        num_spatial_blocks=8,
        bottleneck_dim=256,
    ):
        super().__init__()
        self.pooling_method = pooling_method
        self.num_spatial_blocks = num_spatial_blocks
        self.bottleneck_dim = bottleneck_dim

        # 使用ResNet18的前几层模拟ResNet-10
        resnet = models.resnet18(pretrained=pretrained)
        # 去掉最后的全连接层和平均池化层
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 channels
            resnet.layer2,  # 128 channels
        )

        # 冻结预训练权重
        if pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 空间学习嵌入
        if pooling_method == "spatial_learned_embeddings":
            self.spatial_embeddings = SpatialLearnedEmbeddings(num_spatial_blocks)

        # 瓶颈层
        if bottleneck_dim:
            self.bottleneck = nn.Sequential(
                nn.Linear(num_spatial_blocks, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.Tanh(),
            )

    def forward(self, x):
        # 输入: (B, C, H, W)
        x = self.backbone(x)  # (B, 128, H/8, W/8)

        if self.pooling_method == "spatial_learned_embeddings":
            x = self.spatial_embeddings(x)
        elif self.pooling_method == "avg":
            x = torch.mean(x, dim=(-2, -1))
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")

        if hasattr(self, "bottleneck"):
            x = self.bottleneck(x)

        return x


class SpatialLearnedEmbeddings(nn.Module):
    """空间学习嵌入层"""

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.conv = nn.Conv2d(128, num_features, 1)  # 1x1卷积降维到num_features
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (B, 128, H, W)
        x = self.conv(x)  # (B, num_features, H, W)
        # 全局平均池化
        x = torch.mean(x, dim=(-2, -1))  # (B, num_features)
        x = self.dropout(x)
        return x
