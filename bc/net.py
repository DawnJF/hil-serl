import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):

    def __init__(self, bottleneck_dim=256):
        super().__init__()
        # ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            *list(self.backbone.children())[:-2]
        )  # 去掉最后的FC层和平均池化层

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(512, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # 输入: (B, C, H, W)
        x = self.backbone(x)  # (B, 512, H/32, W/32)
        x = self.global_pool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        x = self.mlp(x)  # (B, bottleneck_dim)
        return x


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.encoder(state)


class EncoderWrapper(nn.Module):
    def __init__(self, image_keys, image_dim=512, proprio_dim=16, hidden_dim=256):
        super().__init__()
        self.image_keys = image_keys
        self.proprio_dim = proprio_dim
        self.image_encoder = ImageEncoder(bottleneck_dim=image_dim)
        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim, hidden_dim=64, output_dim=64
        )

    def forward(self, obs):
        assert isinstance(obs, dict), "Input must be a dictionary"
        image_list = []
        for key in obs:
            image_list.append(obs[key])
        state = obs["state"]

        image_features = [self.image_encoder(img) for img in image_list]

        state_features = self.proprio_encoder(state)
        return torch.cat([*image_features, state_features], dim=-1)

    def get_out_shape(self, image_shape=(128, 128, 3)):
        """获取编码器输出的形状"""

        image_shape = image_shape[:2]  # (H, W)
        fake_obs = {
            "image": torch.zeros(1, 3, *image_shape),
            "state": torch.zeros(1, self.proprio_dim),
        }
        return self.forward(fake_obs).shape[1]


class Actor(nn.Module):
    def __init__(self, image_keys, image_shape, action_dim):
        super().__init__()
        self.encoder = EncoderWrapper(image_keys=image_keys)

        encode_dim = self.encoder.get_out_shape(image_shape)

        self.actor = nn.Sequential(
            nn.Linear(encode_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs):
        x = self.encoder(obs)
        return self.actor(x)
