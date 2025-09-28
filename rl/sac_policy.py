import os
import sys
import pickle as pkl
import einops
import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from dataclasses import dataclass, field
import tyro

sys.path.append(os.getcwd())
from rl.net import Actor, Critic, DiscreteQCritic
from rl.replay_buffer_data_store import ReplayBufferDataStore
from utils.tools import get_device


@dataclass
class SACConfig:
    demo_path: list[str] = field(
        default_factory=lambda: [
            "/Users/majianfei/Downloads/usb_pickup_insertion_5_11-05-02.pkl"
        ]
    )
    replay_buffer_capacity: int = 200000

    action_dim: int = 4  # 连续动作维度
    learning_rate: float = 3e-4
    discount: float = 0.99
    soft_target_update_rate: float = 0.005
    target_entropy: float = -action_dim / 2
    num_discrete_actions: int = 3  # 改为3，对应 {0, 1, 2}

    temperature: float = 1.0


class SACPolicy:

    name = "sac"

    def __init__(
        self,
        config: SACConfig,
    ):
        self.config = config

        # 初始化所有组件
        self._init()

    def _init(self):
        continue_action_dim = self.config.action_dim - 1
        # 初始化网络
        self.actor = Actor(action_dim=continue_action_dim)
        self.critic_ensemble = Critic(action_dim=continue_action_dim)

        # 创建目标网络
        self.critic_target = Critic(action_dim=continue_action_dim)
        # 复制参数到目标网络
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        # 如果有离散动作，初始化离散critic
        if self.config.num_discrete_actions is not None:
            self.discrete_critic = DiscreteQCritic(
                num_discrete_actions=self.config.num_discrete_actions
            )
            self.discrete_critic_target = DiscreteQCritic(
                num_discrete_actions=self.config.num_discrete_actions
            )
            self.discrete_critic_target.load_state_dict(
                self.discrete_critic.state_dict()
            )

        # 初始化温度参数
        self.log_alpha = torch.nn.Parameter(
            torch.log(torch.tensor(self.config.temperature))
        )

        self.discount = torch.tensor(self.config.discount)

        # 初始化优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_ensemble.parameters(), lr=self.config.learning_rate
        )

        if self.config.num_discrete_actions is not None:
            self.discrete_critic_optimizer = torch.optim.Adam(
                self.discrete_critic.parameters(), lr=self.config.learning_rate
            )

        self.temperature_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.learning_rate
        )

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], argmax) -> Tensor:
        """Select action for inference/evaluation"""

        dist = self.actor(batch)
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.rsample()

        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            # {0, 1, 2} -> [-1, 1]
            discrete_action = discrete_action.float() - 1.0
            actions = torch.cat([actions, discrete_action], dim=-1)

        return actions

    def update_critic(self, batch):
        actions: Tensor = batch["actions"]
        observations: dict[str, Tensor] = batch["observations"]

        rewards: Tensor = batch["rewards"]
        next_observations: dict[str, Tensor] = batch["next_observations"]
        done: Tensor = batch["dones"]

        loss_critic = self.compute_loss_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
        )

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        return loss_critic

    def update_grasp_critic(self, batch):
        actions: Tensor = batch["actions"]
        observations: dict[str, Tensor] = batch["observations"]

        # Extract critic-specific components
        rewards: Tensor = batch["rewards"]
        next_observations: dict[str, Tensor] = batch["next_observations"]
        done: Tensor = batch["dones"]
        grasp_penalty = batch.get("grasp_penalty")
        loss_discrete_critic = self.compute_loss_discrete_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            done=done,
            grasp_penalty=grasp_penalty,
        )

        self.discrete_critic_optimizer.zero_grad()
        loss_discrete_critic.backward()
        self.discrete_critic_optimizer.step()

        return loss_discrete_critic

    def update_actor(self, batch):
        observations: dict[str, Tensor] = batch["observations"]

        loss_actor = self.compute_loss_actor(observations=observations)

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        return loss_actor

    def update_temperature(self, batch):
        observations: dict[str, Tensor] = batch["observations"]

        loss_temperature = self.compute_loss_temperature(observations=observations)

        self.temperature_optimizer.zero_grad()
        loss_temperature.backward()
        self.temperature_optimizer.step()

        return loss_temperature

    def update_target_networks(self):
        """Update target networks with exponential moving average"""
        for target_param, param in zip(
            self.critic_target.parameters(),
            self.critic_ensemble.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                param.data * self.config.soft_target_update_rate
                + target_param.data * (1.0 - self.config.soft_target_update_rate)
            )
        if self.config.num_discrete_actions is not None:
            for target_param, param in zip(
                self.discrete_critic_target.parameters(),
                self.discrete_critic.parameters(),
                strict=True,
            ):
                target_param.data.copy_(
                    param.data * self.config.soft_target_update_rate
                    + target_param.data * (1.0 - self.config.soft_target_update_rate)
                )

    def compute_loss_actor(
        self,
        observations: dict[str, Tensor],
    ) -> Tensor:
        """Compute the actor loss"""
        # 从actor获取动作分布
        dist = self.actor(observations)

        # 采样动作
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)

        # 计算Q值（只用连续动作部分）
        q_values = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
        )

        # 取最小的Q值
        min_q, _ = q_values.min(dim=0)

        alpha = self.log_alpha.exp()

        # Actor loss = E[alpha * log_prob - Q(s,a)]
        actor_loss = (alpha * log_probs - min_q).mean()

        return actor_loss

    def compute_loss_temperature(
        self,
        observations: dict[str, Tensor],
    ) -> Tensor:
        """Compute the temperature loss"""

        with torch.no_grad():
            dist = self.actor(observations)
            actions = dist.rsample()
            log_probs = dist.log_prob(actions)

        temperature_loss = (
            -self.log_alpha * (log_probs + self.config.target_entropy)
        ).mean()
        return temperature_loss

    def compute_loss_discrete_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        grasp_penalty=None,
    ):
        # NOTE: We only want to keep the discrete action part
        # In the buffer we have the full action space (continuous + discrete)
        # We need to split them before concatenating them in the critic forward
        actions_discrete: Tensor = actions[:, -1:].clone()  # 固定取最后1维
        # [-1,1] → {0, 1, 2}
        actions_discrete = torch.round(actions_discrete).long() + 1

        discrete_penalties = grasp_penalty

        with torch.no_grad():
            # For DQN, select actions using online network, evaluate with target network
            next_discrete_qs = self.discrete_critic_forward(
                next_observations,
                use_target=False,
            )
            best_next_discrete_action = torch.argmax(
                next_discrete_qs, dim=-1, keepdim=True
            )

            # Get target Q-values from target network
            target_next_discrete_qs = self.discrete_critic_forward(
                observations=next_observations,
                use_target=True,
            )

            # Use gather to select Q-values for best actions
            target_next_discrete_q = torch.gather(
                target_next_discrete_qs, dim=1, index=best_next_discrete_action
            ).squeeze(-1)

            # Compute target Q-value with Bellman equation
            rewards_discrete = rewards
            if discrete_penalties is not None:
                rewards_discrete = rewards + discrete_penalties
            target_discrete_q = (
                rewards_discrete
                + (1 - done.float()) * self.discount * target_next_discrete_q
            )

        # Get predicted Q-values for current observations
        predicted_discrete_qs = self.discrete_critic_forward(
            observations=observations,
            use_target=False,
        )

        # Use gather to select Q-values for taken actions
        predicted_discrete_q = torch.gather(
            predicted_discrete_qs, dim=1, index=actions_discrete
        ).squeeze(-1)

        # Compute MSE loss between predicted and target Q-values
        discrete_critic_loss = F.mse_loss(
            input=predicted_discrete_q, target=target_discrete_q
        )
        return discrete_critic_loss

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
    ) -> Tensor:
        with torch.no_grad():
            # 获取下一个状态的动作分布并采样
            next_dist = self.actor(next_observations)
            next_action_preds = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_action_preds)

            # 计算目标Q值
            q_targets = self.critic_forward(
                observations=next_observations,
                actions=next_action_preds,
                use_target=True,
            )

            # 取最小Q值
            min_q, _ = q_targets.min(dim=0)

            # Bellman方程目标：r + gamma * (Q(s',a') - alpha * log_pi(a'|s'))
            td_target = rewards + (1 - done.float()) * self.discount * min_q

        # 计算当前Q值预测
        if self.config.num_discrete_actions is not None:
            # 只保留连续动作部分
            actions = actions[:, :-1]

        q_preds = self.critic_forward(
            observations=observations,
            actions=actions,
            use_target=False,
        )

        # 计算损失
        # 复制目标值以匹配ensemble的维度
        td_target_duplicate = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])

        # 计算MSE损失
        critics_loss = (
            F.mse_loss(
                input=q_preds, target=td_target_duplicate, reduction="none"
            ).mean(dim=1)
        ).sum()
        return critics_loss

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
    ) -> Tensor:
        """Forward pass through a critic network ensemble

        Args:
            observations: Dictionary of observations
            actions: Action tensor
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from all critics
        """

        critics = self.critic_target if use_target else self.critic_ensemble
        q_values = critics(observations, actions)
        return q_values

    def discrete_critic_forward(self, observations, use_target=False) -> torch.Tensor:
        """Forward pass through a discrete critic network

        Args:
            observations: Dictionary of observations
            use_target: If True, use target critics, otherwise use ensemble critics

        Returns:
            Tensor of Q-values from the discrete critic network
        """
        discrete_critic = (
            self.discrete_critic_target if use_target else self.discrete_critic
        )
        q_values = discrete_critic(observations)
        return q_values

    def get_action(
        self, observations: dict[str, Tensor], deterministic: bool = False
    ) -> Tensor:
        """Get action for single observation (for evaluation)"""
        with torch.no_grad():
            # Add batch dimension if needed
            if len(list(observations.values())[0].shape) == 1:
                observations = {k: v.unsqueeze(0) for k, v in observations.items()}

            actions = self.sample_actions({"state": observations}, argmax=deterministic)
            return actions.squeeze(0)  # Remove batch dimension

    def forward_critic_eval(self, obs, actions):
        """Evaluate Q-values for action selection (used in select_action_v2)"""
        with torch.no_grad():
            # Process observations if they're not tensors
            if isinstance(obs, dict):
                obs_processed = {}
                for k, v in obs.items():
                    if isinstance(v, np.ndarray):
                        obs_processed[k] = torch.from_numpy(v)
                        # Add batch dimension if missing
                        if len(obs_processed[k].shape) == 1:
                            obs_processed[k] = obs_processed[k].unsqueeze(0)
                    elif isinstance(v, torch.Tensor):
                        obs_processed[k] = v
                        if len(v.shape) == 1:
                            obs_processed[k] = v.unsqueeze(0)
                    else:
                        obs_processed[k] = v
                obs = obs_processed

            # Process actions
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            elif not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions).float()

            # Add batch dimension if missing
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(0)

            # Only use continuous action part for critic evaluation
            if self.config.num_discrete_actions is not None and actions.shape[-1] > 3:
                actions = actions[..., :-1]  # Remove discrete action

            # Forward through critic
            q_values = self.critic_forward(obs, actions, use_target=False)

            return q_values

    def train_step(
        self, batch: dict[str, Tensor], critic_only=False
    ) -> dict[str, float]:
        """Complete training step for SAC"""
        metrics = {}

        # Update critics
        critic_loss = self.update_critic(batch)
        metrics["critic_loss"] = critic_loss.item()

        # Update discrete critic if exists
        if self.config.num_discrete_actions is not None:
            discrete_critic_loss = self.update_grasp_critic(batch)
            metrics["discrete_critic_loss"] = discrete_critic_loss.item()

        if not critic_only:

            # Update actor
            actor_loss = self.update_actor(batch)
            metrics["actor_loss"] = actor_loss.item()

            # Update temperature if using automatic entropy tuning

            temperature_loss = self.update_temperature(batch)
            metrics["temperature_loss"] = temperature_loss.item()
            metrics["alpha"] = self.log_alpha.exp().item()

        # Update target networks
        self.update_target_networks()

        return metrics

    def get_params(self) -> dict[str, torch.Tensor]:
        """
        导出所有网络参数为字典格式，用于checkpoint保存和进程间通信

        Returns:
            包含所有网络参数的字典
        """
        params = {}

        # Actor网络参数
        for name, param in self.actor.named_parameters():
            params[f"actor.{name}"] = param.data.clone()

        # Critic网络参数
        for name, param in self.critic_ensemble.named_parameters():
            params[f"critic_ensemble.{name}"] = param.data.clone()

        # Critic目标网络参数
        for name, param in self.critic_target.named_parameters():
            params[f"critic_target.{name}"] = param.data.clone()

        # 离散Critic网络参数（如果存在）
        if self.config.num_discrete_actions is not None:
            for name, param in self.discrete_critic.named_parameters():
                params[f"discrete_critic.{name}"] = param.data.clone()

            for name, param in self.discrete_critic_target.named_parameters():
                params[f"discrete_critic_target.{name}"] = param.data.clone()

        # 温度参数
        params["log_alpha"] = self.log_alpha.data.clone()

        return params

    def load_params(self, params: dict[str, torch.Tensor]) -> None:
        """
        从字典格式加载网络参数

        Args:
            params: 包含网络参数的字典
        """
        # 加载Actor网络参数
        for name, param in self.actor.named_parameters():
            param_key = f"actor.{name}"
            if param_key in params:
                param.data.copy_(params[param_key])
            else:
                print(f"Warning: {param_key} not found in params during load.")

        # 加载Critic网络参数
        for name, param in self.critic_ensemble.named_parameters():
            param_key = f"critic_ensemble.{name}"
            if param_key in params:
                param.data.copy_(params[param_key])
            else:
                print(f"Warning: {param_key} not found in params during load.")

        # 加载Critic目标网络参数
        for name, param in self.critic_target.named_parameters():
            param_key = f"critic_target.{name}"
            if param_key in params:
                param.data.copy_(params[param_key])

        # 加载离散Critic网络参数（如果存在）
        if self.config.num_discrete_actions is not None:
            for name, param in self.discrete_critic.named_parameters():
                param_key = f"discrete_critic.{name}"
                if param_key in params:
                    param.data.copy_(params[param_key])
                else:
                    print(f"Warning: {param_key} not found in params during load.")

            for name, param in self.discrete_critic_target.named_parameters():
                param_key = f"discrete_critic_target.{name}"
                if param_key in params:
                    param.data.copy_(params[param_key])
                else:
                    print(f"Warning: {param_key} not found in params during load.")

        # 加载温度参数
        if "log_alpha" in params:
            self.log_alpha.data.copy_(params["log_alpha"])
        else:
            print(f"Warning: log_alpha not found in params during load.")

    def save_checkpoint(
        self, filepath: str, step: int = 0, additional_info: dict = None
    ) -> None:
        """
        保存完整的checkpoint到文件，包含所有训练状态

        Args:
            filepath: checkpoint文件路径
            step: 当前训练步数
            additional_info: 额外信息字典
        """
        checkpoint = {
            "params": self.get_params(),
            "config": self.config,
            "step": step,
            "optimizer_states": {
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "temperature_optimizer": self.temperature_optimizer.state_dict(),
            },
        }

        # 添加离散critic优化器状态（如果存在）
        if hasattr(self, "discrete_critic_optimizer"):
            checkpoint["optimizer_states"][
                "discrete_critic_optimizer"
            ] = self.discrete_critic_optimizer.state_dict()

        # 添加额外信息
        if additional_info:
            checkpoint["additional_info"] = additional_info

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> dict:
        """
        从文件加载完整的checkpoint，恢复所有训练状态

        Args:
            filepath: checkpoint文件路径

        Returns:
            包含训练元数据的字典（步数、额外信息等）
        """
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        # 加载网络参数
        self.load_params(checkpoint["params"])

        # 加载优化器状态
        if "optimizer_states" in checkpoint:
            optimizer_states = checkpoint["optimizer_states"]

            if "actor_optimizer" in optimizer_states:
                self.actor_optimizer.load_state_dict(
                    optimizer_states["actor_optimizer"]
                )

            if "critic_optimizer" in optimizer_states:
                self.critic_optimizer.load_state_dict(
                    optimizer_states["critic_optimizer"]
                )

            if "temperature_optimizer" in optimizer_states:
                self.temperature_optimizer.load_state_dict(
                    optimizer_states["temperature_optimizer"]
                )

            # 加载离散critic优化器状态（如果存在）
            if (
                hasattr(self, "discrete_critic_optimizer")
                and "discrete_critic_optimizer" in optimizer_states
            ):
                self.discrete_critic_optimizer.load_state_dict(
                    optimizer_states["discrete_critic_optimizer"]
                )

        # 返回训练元数据
        metadata = {
            "step": checkpoint.get("step", 0),
            "additional_info": checkpoint.get("additional_info", {}),
        }

        return metadata


def get_train_transform():
    """need CxHxW input"""
    return transforms.Compose(
        [
            # pre-process
            transforms.Lambda(lambda img: img.squeeze()),
            transforms.ToTensor(),
            # data augmentations
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            # post-process
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform():
    return transforms.Compose(
        [
            # pre-process
            transforms.Lambda(lambda img: img.squeeze()),
            transforms.ToTensor(),
            # post-process
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def dict_data_to_torch(obj, image_transform=None):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    elif isinstance(obj, dict):
        d = {}
        for k, v in obj.items():
            if k in ["rgb", "wrist"]:
                if len(v.shape) == 3:
                    v = v[np.newaxis, ...]  # Add batch dimension if missing

                assert v.shape[-1] == 3, f"Expected C=3, got {v.shape}"
                images = []
                for image in v:
                    images.append(image_transform(image).unsqueeze(0))
                v = torch.cat(images, dim=0)

            elif k in ["state"]:
                if len(v.shape) == 3:
                    v = v[:, 0, :]  # Remove extra dimension if present

            d[k] = dict_data_to_torch(v, image_transform=image_transform)

        return d

    elif isinstance(obj, (list, tuple)):
        t = [dict_data_to_torch(x, image_transform=image_transform) for x in obj]
        return tuple(t) if isinstance(obj, tuple) else t
    else:
        return obj


def test_learner(config: SACConfig):

    observation_space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
            "rgb": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
            "wrist": gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
        }
    )
    action_space = gym.spaces.Box(
        np.ones((4,), dtype=np.float32) * -1,
        np.ones((4,), dtype=np.float32),
    )

    demo_buffer = ReplayBufferDataStore(
        observation_space,
        action_space,
        capacity=config.replay_buffer_capacity,
        include_grasp_penalty=True,
    )
    for path in config.demo_path:
        with open(path, "rb") as f:
            transitions = pkl.load(f)
            for transition in transitions:
                if "infos" in transition and "grasp_penalty" in transition["infos"]:
                    transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
                demo_buffer.insert(transition)
    print(f"demo buffer size: {len(demo_buffer)}")

    demo_iterator = demo_buffer.get_iterator(sample_args={"batch_size": 4})
    demo_batch = next(demo_iterator)

    demo_batch = dict_data_to_torch(demo_batch, image_transform=get_train_transform())

    print("Creating SAC agent...")
    agent = SACPolicy(config)

    print("Running training step...")
    update_info = agent.train_step(demo_batch)
    print("Training step completed!")
    print(f"Update info: {update_info}")

    return agent, update_info


if __name__ == "__main__":
    # 测试配置
    config = tyro.cli(SACConfig)

    print("Starting test_learner...")
    try:
        agent, update_info = test_learner(config)
        print("Test completed successfully!")
        print(f"Final update info: {update_info}")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
