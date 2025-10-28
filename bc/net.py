import logging
import torch
import torch.nn as nn
import torchvision.models as models

from rl.net import Actor, DiscreteQCritic, ImageEncoder


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


class BCActor(nn.Module):
    def __init__(self, args):
        super().__init__()
        image_num = args.get("image_num", 2)
        state_dim = args.get("state_dim", 7)
        action_continue_dim = args.get("action_continue_dim", 3)
        action_discrete_dim = args.get("action_discrete_dim", 3)

        self.encoder = EncoderWrapper(image_num=image_num, proprio_dim=state_dim)

        encode_dim = self.encoder.get_out_shape()
        logging.info(f"Encoder output dim: {encode_dim}")

        self.actor = nn.Sequential(
            nn.Linear(encode_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.continue_head = nn.Linear(256, action_continue_dim)
        self.discrete_head = nn.Linear(256, action_discrete_dim)

    def forward(self, observations: dict[str, torch.Tensor]):
        x = self.encoder(observations)
        features = self.actor(x)
        continue_actions = self.continue_head(features)
        discrete_logits = self.discrete_head(features)
        return continue_actions, discrete_logits

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)


class RLActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_actor = Actor(3)
        self.d_actor = DiscreteQCritic(3)

    def forward(self, batch):
        # batch should be a dict with observations format
        dist = self.c_actor(batch)

        discrete_actions = self.d_actor(batch)

        return dist, discrete_actions

    def save_checkpoint(self, path):
        torch.save(
            {
                "continue_actor": self.c_actor.state_dict(),
                "discrete_actor": self.d_actor.state_dict(),
            },
            path,
        )

    def READ_CHECKPOINT(path):
        checkpoint_dict = torch.load(path)
        return checkpoint_dict["continue_actor"], checkpoint_dict["discrete_actor"]

    def load_checkpoint(self, path):
        c_dict, d_dict = RLActor.READ_CHECKPOINT(path)
        self.c_actor.load_state_dict(c_dict)
        self.d_actor.load_state_dict(d_dict)
