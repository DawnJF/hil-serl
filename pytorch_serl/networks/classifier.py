"""奖励分类器的PyTorch实现"""

import torch
import torch.nn as nn
from networks.resnet import ResNetEncoder
from networks.mlp import MLP


class RewardClassifier(nn.Module):
    """二分类奖励分类器"""

    def __init__(self, image_keys, hidden_dim=256):
        super().__init__()
        self.image_keys = image_keys

        # 为每个图像输入创建编码器
        self.encoders = nn.ModuleDict()
        for key in image_keys:
            self.encoders[key] = ResNetEncoder(
                pretrained=True,
                pooling_method="spatial_learned_embeddings",
                num_spatial_blocks=8,
                bottleneck_dim=256,
            )

        # 分类头
        encoder_dim = 256 * len(image_keys)  # 所有编码器输出拼接
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 二分类输出logit
        )

    def forward(self, obs):
        """前向传播

        Args:
            obs: 观测字典，包含图像

        Returns:
            logits: 分类logits (batch_size, 1)
        """
        encoded_features = []

        for key in self.image_keys:
            if key in obs:
                encoded = self.encoders[key](obs[key])
                encoded_features.append(encoded)

        # 拼接所有编码特征
        if encoded_features:
            x = torch.cat(encoded_features, dim=-1)
        else:
            raise ValueError("观测中没有找到有效的图像键")

        logits = self.classifier(x)
        return logits

    def predict_success(self, obs, threshold=0.5):
        """预测成功概率

        Args:
            obs: 观测
            threshold: 阈值

        Returns:
            success_prob: 成功概率
            is_success: 是否成功（基于阈值）
        """
        with torch.no_grad():
            logits = self.forward(obs)
            probs = torch.sigmoid(logits)
            is_success = probs > threshold

        return probs, is_success
