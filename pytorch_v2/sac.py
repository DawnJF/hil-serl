# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
import pickle
import logging
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

# Import local modules
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pytorch_serl.data.replay_buffer import ReplayBuffer as CustomReplayBuffer
from pytorch_serl.utils.device import get_device
from pytorch_serl.utils.logger_utils import setup_logging


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Data specific arguments
    dataset_path: str = "dataset/success_demo.pkl"
    """path to the demonstration dataset"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = int(5e3)
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    save_model: bool = True
    """whether to save model checkpoints"""
    save_frequency: int = 4000
    """frequency to save model checkpoints"""
    run_path: str = "runs"
    """directory to save model checkpoints"""


# ALGO LOGIC: initialize agent here:
class ImageEncoder(nn.Module):
    """图像编码器，使用ResNet18 backbone + 少量MLP层"""

    def __init__(self, bottleneck_dim=256):
        super().__init__()
        # ResNet18 backbone
        resnet = models.resnet18(pretrained=True)
        # 去掉最后的全连接层和平均池化层
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 channels
            resnet.layer2,  # 128 channels
            resnet.layer3,  # 256 channels
            resnet.layer4,  # 512 channels
        )
        # freeze self.backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 少量MLP层
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


class SoftQNetwork(nn.Module):
    def __init__(self, image_dim=256, action_dim=16, hidden_dim=256):
        super().__init__()
        self.image_encoder = ImageEncoder(bottleneck_dim=image_dim)

        # Q网络
        self.fc1 = nn.Linear(image_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, a):
        # 处理图像观测
        if isinstance(obs, dict):
            image = obs["image"]
        else:
            image = obs

        # 确保图像在正确的设备上并且格式正确
        if len(image.shape) == 4:  # (B, H, W, C) -> (B, C, H, W)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)

        # 归一化到[0,1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        image_features = self.image_encoder(image)
        x = torch.cat([image_features, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(
        self,
        image_dim=256,
        action_dim=16,
        hidden_dim=256,
        action_low=None,
        action_high=None,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(bottleneck_dim=image_dim)

        # Actor网络
        self.fc1 = nn.Linear(image_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

        # action rescaling - 从实际数据范围计算
        if action_low is not None and action_high is not None:
            self.register_buffer(
                "action_scale",
                torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
            )
            self.register_buffer(
                "action_bias",
                torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
            )
        else:
            # 默认假设动作范围是[-1, 1]
            self.register_buffer("action_scale", torch.ones(action_dim))
            self.register_buffer("action_bias", torch.zeros(action_dim))

    def forward(self, obs):
        # 处理图像观测
        if isinstance(obs, dict):
            image = obs["image"]
        else:
            image = obs

        # 确保图像在正确的设备上并且格式正确
        if len(image.shape) == 4:  # (B, H, W, C) -> (B, C, H, W)
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)

        # 归一化到[0,1]
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        image_features = self.image_encoder(image)
        x = F.relu(self.fc1(image_features))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def load_dataset(dataset_path):
    """加载演示数据集"""
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_replay_buffer_from_dataset(dataset, image_keys=["image"]):
    """从数据集创建重放缓冲区"""
    rb = CustomReplayBuffer(len(dataset), image_keys=image_keys)

    for transition in dataset:
        rb.insert(transition)

    logging.info(f"Loaded {len(rb)} transitions from dataset")
    return rb


def extract_action_range(dataset):
    """从数据集中提取动作的范围"""
    all_actions = []
    for transition in dataset:
        action = transition["actions"]
        if hasattr(action, "numpy"):
            action = action.numpy()
        elif isinstance(action, list):
            action = np.array(action)
        all_actions.append(action)

    all_actions = np.array(all_actions)
    action_low = np.min(all_actions, axis=0)
    action_high = np.max(all_actions, axis=0)

    logging.info(f"Action range: low={action_low}, high={action_high}")
    return action_low, action_high


"""
cd /Users/majianfei/Projects/Github/ML/hil-serl/pytorch_v2 && python sac.py --total_timesteps 300 --batch_size 8 --dataset_path ../dataset/success_demo.pkl
"""
if __name__ == "__main__":

    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"

    # Create run directory
    run_dir = f"{args.run_path}/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Setup logging
    setup_logging(run_dir)

    logging.info("Starting SAC training with demonstration data")
    logging.info(f"Run name: {run_name}")
    logging.info(f"Args: {vars(args)}")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    logging.info(f"Setting seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = get_device()
    logging.info(f"Using device: {device}")

    # Load dataset instead of creating environment
    logging.info("Loading dataset...")
    dataset = load_dataset(args.dataset_path)

    # Create replay buffer from dataset
    rb = create_replay_buffer_from_dataset(dataset, image_keys=["image"])
    logging.info(f"Created replay buffer with {len(rb)} transitions")

    # 从数据集中推断动作维度和范围
    sample_action = dataset[0]["actions"]
    action_dim = (
        sample_action.shape[0]
        if hasattr(sample_action, "shape")
        else len(sample_action)
    )

    # 提取动作范围
    action_low, action_high = extract_action_range(dataset)

    # 从数据集中推断图像维度
    sample_obs = dataset[0]["observations"]
    if "image" in sample_obs:
        image_shape = sample_obs["image"].shape
        logging.info(f"Image shape: {image_shape}")

    logging.info(f"Action dimension: {action_dim}")

    # Initialize networks
    logging.info("Initializing networks...")
    actor = Actor(
        image_dim=256,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
    ).to(device)
    qf1 = SoftQNetwork(image_dim=256, action_dim=action_dim).to(device)
    qf2 = SoftQNetwork(image_dim=256, action_dim=action_dim).to(device)
    qf1_target = SoftQNetwork(image_dim=256, action_dim=action_dim).to(device)
    qf2_target = SoftQNetwork(image_dim=256, action_dim=action_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    logging.info("Initializing optimizers...")
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.tensor(
            action_dim, dtype=torch.float32, device=device
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        logging.info(f"Auto-tuning entropy with target: {target_entropy}")
    else:
        alpha = args.alpha
        logging.info(f"Using fixed alpha: {alpha}")

    start_time = time.time()
    logging.info(f"Starting training for {args.total_timesteps} steps...")

    # Create model save directory
    if args.save_model:
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"Model checkpoints will be saved to: {run_dir}")

    # Training loop - sample from dataset instead of environment interaction
    for global_step in tqdm(range(args.total_timesteps)):
        # ALGO LOGIC: training only (no environment interaction)
        if len(rb) >= args.batch_size:
            data = rb.sample(args.batch_size, device)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data["next_observations"]
                )
                qf1_next_target = qf1_target(
                    data["next_observations"], next_state_actions
                )
                qf2_next_target = qf2_target(
                    data["next_observations"], next_state_actions
                )
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data["rewards"].flatten() + (
                    1 - (1 - data["masks"]).flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data["observations"], data["actions"]).view(-1)
            qf2_a_values = qf2(data["observations"], data["actions"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data["observations"])
                    qf1_pi = qf1(data["observations"], pi)
                    qf2_pi = qf2(data["observations"], pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data["observations"])
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            # Log to tensorboard
            writer.add_scalar(
                "losses/qf1_values", qf1_a_values.mean().item(), global_step
            )
            writer.add_scalar(
                "losses/qf2_values", qf2_a_values.mean().item(), global_step
            )
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)

            if global_step % 50 == 0:
                # Log to logging
                logging.info(
                    f"Step {global_step}: "
                    f"QF1_loss={qf1_loss.item():.4f}, "
                    f"QF2_loss={qf2_loss.item():.4f}, "
                    f"Actor_loss={actor_loss.item():.4f}, "
                    f"Alpha={alpha:.4f}, "
                )

                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
                    logging.info(f"Alpha_loss={alpha_loss.item():.4f}")

            # Save model checkpoints
            if (
                args.save_model
                and global_step % args.save_frequency == 0
                and global_step > 0
            ):
                checkpoint = {
                    "global_step": global_step,
                    "actor_state_dict": actor.state_dict(),
                    "qf1_state_dict": qf1.state_dict(),
                    "qf2_state_dict": qf2.state_dict(),
                    "qf1_target_state_dict": qf1_target.state_dict(),
                    "qf2_target_state_dict": qf2_target.state_dict(),
                    "actor_optimizer_state_dict": actor_optimizer.state_dict(),
                    "q_optimizer_state_dict": q_optimizer.state_dict(),
                    "action_low": action_low,
                    "action_high": action_high,
                    "action_dim": action_dim,
                    "args": vars(args),
                }
                if args.autotune:
                    checkpoint["log_alpha"] = log_alpha
                    checkpoint["a_optimizer_state_dict"] = a_optimizer.state_dict()
                    checkpoint["alpha"] = alpha

                checkpoint_path = os.path.join(run_dir, f"checkpoint_{global_step}.pt")
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    if args.save_model:
        final_checkpoint = {
            "global_step": args.total_timesteps,
            "actor_state_dict": actor.state_dict(),
            "qf1_state_dict": qf1.state_dict(),
            "qf2_state_dict": qf2.state_dict(),
            "qf1_target_state_dict": qf1_target.state_dict(),
            "qf2_target_state_dict": qf2_target.state_dict(),
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "action_low": action_low,
            "action_high": action_high,
            "action_dim": action_dim,
            "args": vars(args),
        }
        if args.autotune:
            final_checkpoint["log_alpha"] = log_alpha
            final_checkpoint["a_optimizer_state_dict"] = a_optimizer.state_dict()
            final_checkpoint["alpha"] = alpha

        final_model_path = os.path.join(run_dir, "final_model.pt")
        torch.save(final_checkpoint, final_model_path)
        logging.info(f"Saved final model to {final_model_path}")

    logging.info("Training completed!")
    logging.info(f"Total training time: {time.time() - start_time:.2f} seconds")
    writer.close()
