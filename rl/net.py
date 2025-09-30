from torch.distributions import (
    TanhTransform,
    TransformedDistribution,
)
from torch.distributions import (
    Normal,
    Independent,
    TransformedDistribution,
    TanhTransform,
)
import torchvision.models as models
import torch
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ImageEncoder(nn.Module):
    """Smart pretrained encoder with efficient parameter usage"""

    def __init__(self, bottleneck_dim=256, freeze_backbone=True):
        super().__init__()
        # Use EfficientNet-B0 as backbone (much lighter than ResNet18 but better performance)

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        backbone_out_dim = 1280  # EfficientNet-B0 output

        # Remove final classifier
        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()

        # Freeze backbone parameters if requested (similar to JAX frozen encoder)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(
                f"Froze {sum(p.numel() for p in self.backbone.parameters())} backbone parameters"
            )

        # Efficient feature projection with residual connection
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, bottleneck_dim),
        )

    def forward(self, x):
        # 输入: (B, C, H, W)
        backbone_features = self.backbone(x)
        projected = self.feature_proj(backbone_features)
        return projected


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, state):
        return self.encoder(state)


class EncoderWrapper(nn.Module):
    def __init__(self, image_num, proprio_dim=7):
        super().__init__()
        self.image_num = image_num
        self.proprio_dim = proprio_dim

        self.image_encoder = ImageEncoder(bottleneck_dim=256)
        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim, hidden_dim=64, output_dim=64
        )

    def forward(self, observations):
        state = observations["state"]  # 本体感受信息
        image_rgb = observations["rgb"]
        image_wrist = observations["wrist"]
        images = torch.stack([image_rgb, image_wrist], dim=1)  # (B, N, C, H, W)

        B, N, C, H, W = images.shape
        assert N == self.image_num, f"Expected {self.image_num} images, but got {N}"

        image_features = []

        # Extract features from all images
        for i in range(N):
            img = images[:, i, :, :, :]  # Shape: (B, C, H, W)
            img_features = self.image_encoder(img)  # (B, 256)
            image_features.append(img_features)

        image_features = torch.cat(image_features, dim=1)  # Shape: (B, N * image_dim)

        state_features = self.proprio_encoder(state)
        return torch.cat([image_features, state_features], dim=-1)

    def get_out_shape(self, image_shape=128):
        """获取编码器输出的形状"""

        image1 = torch.zeros(1, self.image_num, 3, image_shape, image_shape)
        state = torch.zeros(1, self.proprio_dim)

        observations = {"state": state, "rgb": image1[:, 0], "wrist": image1[:, 1]}
        return self.forward(observations).shape[1]


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
        std_min: float = 1e-05,
        std_max: float = 5,
    ):
        super().__init__()

        self.encoder = EncoderWrapper(image_num=2, proprio_dim=7)
        self.action_dim = action_dim
        self.std_min = std_min
        self.std_max = std_max

        self.network = nn.Sequential(
            nn.Linear(self.encoder.get_out_shape(), 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
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
        return torch.stack(q_values, dim=0)  # (2, B)


class DiscreteQCritic(nn.Module):
    """Discrete Q-value critic for discrete actions (like grasp/no-grasp)"""

    def __init__(self, num_discrete_actions=2):
        super().__init__()
        self.encoder = EncoderWrapper(image_num=2, proprio_dim=7)
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
