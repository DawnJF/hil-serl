import sys
import os
import torch.nn as nn
from torchvision import models


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, n_image=1):
        super().__init__()
        self.n_image = n_image

        self.model = models.resnet18(pretrained=True)

        # 替换掉最后的分类层
        features = self.model.fc.in_features
        self.model.fc = nn.Linear(features, features)

        classifier_features = features * n_image

        self.classifier = nn.Sequential(
            nn.Linear(classifier_features, 256),  # 两张图片的特征拼接
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        x: Tensor, shape [B, N, C, H, W]
        """
        B, N, C, H, W = x.shape
        assert N == self.n_image, f"Expected {self.n_image} images, but got {N}"

        x = x.view(B * N, C, H, W)  # 合并 batch 和 N
        feats = self.model(x)  # [B*N, features]
        feats = feats.view(B, N * feats.shape[1])  # [B, N*features]
        out = self.classifier(feats)  # [B, num_classes]
        return out
