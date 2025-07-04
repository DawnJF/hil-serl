"""Actor-Critic网络的PyTorch实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from networks.resnet import ResNetEncoder
from networks.mlp import MLP, EnsembleMLP


class Actor(nn.Module):
    """SAC的演员网络"""

    def __init__(self, encoder, hidden_dims, action_dim, input_dim=None):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim

        # 获取编码器输出维度
        if encoder is not None:
            encoder_dim = 256  # ResNet编码器的瓶颈维度
        else:
            encoder_dim = input_dim if input_dim is not None else 256  # 默认输入维度

        # 策略网络
        self.policy_net = MLP(
            encoder_dim, hidden_dims, hidden_dims[-1], activate_final=True
        )

        # 均值和标准差头
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        # 动作范围限制
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, obs):
        """前向传播

        Args:
            obs: 观测，可以是图像或状态特征

        Returns:
            action: 动作样本
            log_prob: 对数概率
            mean: 动作均值（用于确定性策略）
        """
        if self.encoder is not None:
            # 处理图像观测
            if isinstance(obs, dict):
                # 多摄像头情况，取第一个图像
                image_key = list(obs.keys())[0]
                x = obs[image_key]
            else:
                x = obs
            x = self.encoder(x)
        else:
            x = obs

        x = self.policy_net(x)

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 限制标准差范围

        std = torch.exp(log_std)

        # 重参数化技巧
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 可微分采样
        action = torch.tanh(x_t)

        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        # 对tanh变换进行校正
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 缩放动作
        action = action * self.action_scale + self.action_bias
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


class Critic(nn.Module):
    """SAC的评论家网络（集成版本）"""

    def __init__(
        self, encoder, hidden_dims, action_dim, ensemble_size=2, input_dim=None
    ):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # 获取编码器输出维度
        if encoder is not None:
            encoder_dim = 256
        elif input_dim is not None:
            encoder_dim = input_dim
        else:
            encoder_dim = 0

        # Q网络集成
        input_dim = encoder_dim + action_dim
        self.q_networks = EnsembleMLP(input_dim, hidden_dims, 1, ensemble_size)

    def forward(self, obs, action):
        """前向传播

        Args:
            obs: 观测
            action: 动作

        Returns:
            q_values: Q值 (ensemble_size, batch_size, 1)
        """
        if self.encoder is not None:
            if isinstance(obs, dict):
                image_key = list(obs.keys())[0]
                x = obs[image_key]
            else:
                x = obs
            obs_encoded = self.encoder(x)
        else:
            obs_encoded = obs

        # 拼接观测和动作
        x = torch.cat([obs_encoded, action], dim=-1)
        q_values = self.q_networks(x)

        return q_values.squeeze(-1)  # (ensemble_size, batch_size)


class GraspCritic(nn.Module):
    """抓取评论家网络（用于混合策略）"""

    def __init__(self, encoder, hidden_dims, output_dim=3, input_dim=None):
        super().__init__()
        self.encoder = encoder
        self.output_dim = output_dim

        if encoder is not None:
            encoder_dim = 256
        elif input_dim is not None:
            encoder_dim = input_dim
        else:
            encoder_dim = 0

        self.q_network = MLP(encoder_dim, hidden_dims, output_dim, activate_final=False)

    def forward(self, obs):
        """前向传播

        Args:
            obs: 观测

        Returns:
            q_values: 抓取Q值 (batch_size, output_dim)
        """
        if self.encoder is not None:
            if isinstance(obs, dict):
                image_key = list(obs.keys())[0]
                x = obs[image_key]
            else:
                x = obs
            obs_encoded = self.encoder(x)
        else:
            obs_encoded = obs

        q_values = self.q_network(obs_encoded)
        return q_values
