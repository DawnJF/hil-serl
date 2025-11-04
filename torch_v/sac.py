import logging
from torch.distributions import (
    Normal,
    Independent,
    TransformedDistribution,
    TanhTransform,
)
import torch
from torch import nn

from torch.encoder import EncoderWrapper


class TanhMultivariateNormalDiag(TransformedDistribution):
    def __init__(self, loc: torch.Tensor, scale_diag: torch.Tensor):
        # base distribution: diagonal Gaussian
        base_dist = Independent(Normal(loc, scale_diag), 1)

        # bijector: tanh
        transforms = [TanhTransform(cache_size=1)]

        super().__init__(base_dist, transforms)

    def mode(self) -> torch.Tensor:
        mode = self.base_dist.mode
        for transform in self.transforms:
            mode = transform(mode)
        return mode


class Actor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        image_keys: list[str],
        std_min: float = 1e-05,
        std_max: float = 5,
    ):
        super().__init__()

        self.encoder = EncoderWrapper(image_keys, proprio_dim=7)
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max

        self.network = nn.Sequential(
            nn.Linear(self.encoder.get_out_shape(), 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

        # Mean layer
        self.mean_layer = nn.Linear(256, action_dim)
        # torch.nn.init.xavier_uniform_
        torch.nn.init.xavier_uniform_(self.mean_layer.weight)

        # Standard deviation layer or parameter
        self.std_layer = nn.Linear(256, action_dim)
        torch.nn.init.xavier_uniform_(self.std_layer.weight)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> TanhMultivariateNormalDiag:
        # 提取观测信息
        obs_enc = self.encoder(observations)

        # Get network outputs
        outputs = self.network(obs_enc)

        means = self.mean_layer(outputs)

        # Compute standard deviations. Match JAX "exp"
        log_std = self.std_layer(outputs)
        std = torch.exp(log_std)

        std = torch.clamp(std, self.std_min, self.std_max)  # Match JAX default clip

        # Build transformed distribution
        dist = TanhMultivariateNormalDiag(loc=means, scale_diag=std)

        return dist

    def freeze_bc_params(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.mean_layer.parameters():
            param.requires_grad = False

        self.encoder.eval()
        self.network.eval()
        self.mean_layer.eval()


class CriticHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, action_dim=7):
        super().__init__()
        self.encoder = EncoderWrapper(image_num=2, proprio_dim=7)

        input_dim = self.encoder.get_out_shape() + action_dim

        self.critics = nn.ModuleList([CriticHead(input_dim) for _ in range(2)])

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # 提取观测信息

        obs_enc = self.encoder(observations)

        # 拼接动作到编码后的观测
        critic_input = torch.cat([obs_enc, actions], dim=-1)

        q_values = []
        for critic in self.critics:
            q_values.append(critic(critic_input))
        return torch.stack(q_values, dim=-1)  # (B, 2)


class DiscreteQCritic(nn.Module):
    """Discrete Q-value critic for discrete actions (like grasp/no-grasp)"""

    def __init__(self, num_discrete_actions, image_keys: list[str]):
        super().__init__()
        self.encoder = EncoderWrapper(image_keys, proprio_dim=7)
        self.num_discrete_actions = num_discrete_actions

        # 使用简化的Dueling网络架构
        encoder_dim = self.encoder.get_out_shape()

        # 状态值流
        self.value_stream = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # 动作优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_discrete_actions),
        )

    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # 提取观测信息
        obs_enc = self.encoder(observations)

        # Dueling架构: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        value = self.value_stream(obs_enc)  # (B, 1)
        advantage = self.advantage_stream(obs_enc)  # (B, num_actions)

        # 减去平均优势以保证唯一性
        advantage_mean = advantage.mean(dim=-1, keepdim=True)
        q_values = value + advantage - advantage_mean

        return q_values
