import logging
import torch
import torch.nn as nn
import torchvision.models as models

from bc.encoder import EncoderWrapper
from rl.net import Actor, DiscreteQCritic


class BCActor(nn.Module):
    def __init__(self, args):
        super().__init__()
        state_dim = args.get("state_dim", 7)
        action_continue_dim = args.get("action_continue_dim", 3)
        action_discrete_dim = args.get("action_discrete_dim", 3)
        image_keys = args.get("image_keys", ["image1", "image2"])

        self.encoder = EncoderWrapper(image_keys=image_keys, proprio_dim=state_dim)

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

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))


class RLActor(nn.Module):
    def __init__(self, args):
        super().__init__()
        action_continue_dim = args.get("action_continue_dim", 3)
        action_discrete_dim = args.get("action_discrete_dim", 3)
        image_keys = args.get("image_keys", ["image1", "image2"])
        self.c_actor = Actor(action_continue_dim, image_keys)
        self.d_actor = DiscreteQCritic(action_discrete_dim, image_keys)

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
