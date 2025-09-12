import logging
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
    def __init__(self, image_num, proprio_dim=16):
        super().__init__()
        self.image_num = image_num
        self.proprio_dim = proprio_dim
        self.image_encoder = ImageEncoder(bottleneck_dim=512)
        self.proprio_encoder = ProprioEncoder(
            input_dim=proprio_dim, hidden_dim=64, output_dim=64
        )

    def forward(self, state, imgs):
        B, N, C, H, W = imgs.shape
        assert N == self.image_num, f"Expected {self.image_num} images, but got {N}"

        image_features = []

        # Iterate over the N images for each batch in imgs
        for i in range(N):
            # Extract each image corresponding to the i-th image in the sequence
            img = imgs[:, i, :, :, :]  # Shape: (B, C, H, W)
            img_features = self.image_encoder(img)  # Pass through the image encoder
            image_features.append(img_features)

        image_features = torch.cat(image_features, dim=1)  # Shape: (B, N * image_dim)

        state_features = self.proprio_encoder(state)
        return torch.cat([image_features, state_features], dim=-1)

    def get_out_shape(self, image_shape=128):
        """获取编码器输出的形状"""

        image1 = torch.zeros(1, self.image_num, 3, image_shape, image_shape)
        state = torch.zeros(1, self.proprio_dim)
        return self.forward(state, image1).shape[1]


class BCActor(nn.Module):
    def __init__(self, args):
        super().__init__()
        image_num = args.image_num
        image_shape = args.image_shape
        state_dim = args.state_dim
        action_dim = args.action_dim

        self.encoder = EncoderWrapper(image_num=image_num, proprio_dim=state_dim)

        encode_dim = self.encoder.get_out_shape(image_shape)
        logging.info(f"Encoder output dim: {encode_dim}")

        self.actor = nn.Sequential(
            nn.Linear(encode_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state, imgs):
        x = self.encoder(state, imgs)
        return self.actor(x)
