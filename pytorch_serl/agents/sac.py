"""SAC智能体的PyTorch实现"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional
import copy

from networks.actor_critic import Actor, Critic, GraspCritic
from networks.resnet import ResNetEncoder
from utils.device import DEVICE, to_device


class SACAgent:
    """标准SAC智能体（固定抓取）"""

    def __init__(
        self,
        image_keys: list,
        action_dim: int,
        lr: float = 3e-4,
        discount: float = 0.95,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        critic_ensemble_size: int = 2,
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.image_keys = image_keys

        # 自动目标熵
        if target_entropy is None:
            target_entropy = -action_dim / 2
        self.target_entropy = target_entropy

        # 创建编码器
        self.encoder = ResNetEncoder(pretrained=True).to(device)

        # 创建网络
        self.actor = Actor(
            encoder=copy.deepcopy(self.encoder),
            hidden_dims=[256, 256],
            action_dim=action_dim,
        ).to(device)

        self.critic = Critic(
            encoder=copy.deepcopy(self.encoder),
            hidden_dims=[256, 256],
            action_dim=action_dim,
            ensemble_size=critic_ensemble_size,
        ).to(device)

        # 目标网络
        self.target_critic = copy.deepcopy(self.critic)
        # 冻结目标网络
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # 温度参数
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_actions(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> torch.Tensor:
        """采样动作"""
        with torch.no_grad():
            obs = to_device(obs, self.device)
            action, _, mean = self.actor(obs)

            if deterministic:
                return mean
            else:
                return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新网络参数"""
        batch = to_device(batch, self.device)

        # 更新Critic
        critic_loss = self._update_critic(batch)

        # 更新Actor和温度参数
        actor_loss = self._update_actor(batch)
        alpha_loss = self._update_alpha(batch)

        # 软更新目标网络
        self._soft_update_target()

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha.item(),
        }

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """更新Critic网络"""
        with torch.no_grad():
            # 计算目标Q值
            next_actions, next_log_probs, _ = self.actor(batch["next_observations"])
            target_q_values = self.target_critic(
                batch["next_observations"], next_actions
            )
            target_q = target_q_values.min(dim=0)[0]  # 取最小值

            # SAC目标公式
            target_q = batch["rewards"] + self.discount * batch["masks"] * (
                target_q - self.alpha * next_log_probs.squeeze()
            )

        # 当前Q值
        current_q_values = self.critic(batch["observations"], batch["actions"])

        # Critic损失
        critic_loss = 0
        for i in range(current_q_values.shape[0]):
            critic_loss += F.mse_loss(current_q_values[i], target_q)
        critic_loss /= current_q_values.shape[0]

        # 更新
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> float:
        """更新Actor网络"""
        actions, log_probs, _ = self.actor(batch["observations"])
        q_values = self.critic(batch["observations"], actions)
        q_value = q_values.mean(dim=0)  # 集成平均

        # Actor损失
        actor_loss = (self.alpha * log_probs.squeeze() - q_value).mean()

        # 更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, batch: Dict[str, torch.Tensor]) -> float:
        """更新温度参数"""
        with torch.no_grad():
            _, log_probs, _ = self.actor(batch["observations"])

        # 温度损失
        alpha_loss = -(
            self.log_alpha * (log_probs.squeeze() + self.target_entropy).detach()
        ).mean()

        # 更新
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def _soft_update_target(self):
        """软更新目标网络"""
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "log_alpha": self.log_alpha,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.log_alpha.data = checkpoint["log_alpha"]
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


class SACHybridAgent:
    """混合SAC智能体（学习抓取）"""

    def __init__(
        self,
        image_keys: list,
        continuous_action_dim: int,  # 连续动作维度（不包括抓取）
        grasp_action_dim: int = 3,  # 抓取动作维度 {关闭, 不动, 张开}
        lr: float = 3e-4,
        discount: float = 0.95,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        critic_ensemble_size: int = 2,
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.continuous_action_dim = continuous_action_dim
        self.grasp_action_dim = grasp_action_dim
        self.discount = discount
        self.tau = tau
        self.image_keys = image_keys

        # 自动目标熵（仅针对连续动作）
        if target_entropy is None:
            target_entropy = -continuous_action_dim / 2
        self.target_entropy = target_entropy

        # 创建编码器
        self.encoder = ResNetEncoder(pretrained=True).to(device)

        # 创建网络
        self.actor = Actor(
            encoder=copy.deepcopy(self.encoder),
            hidden_dims=[256, 256],
            action_dim=continuous_action_dim,  # 只输出连续动作
        ).to(device)

        self.critic = Critic(
            encoder=copy.deepcopy(self.encoder),
            hidden_dims=[256, 256],
            action_dim=continuous_action_dim,  # 只评估连续动作
            ensemble_size=critic_ensemble_size,
        ).to(device)

        self.grasp_critic = GraspCritic(
            encoder=copy.deepcopy(self.encoder),
            hidden_dims=[128, 128],
            output_dim=grasp_action_dim,
        ).to(device)

        # 目标网络
        self.target_critic = copy.deepcopy(self.critic)
        self.target_grasp_critic = copy.deepcopy(self.grasp_critic)

        # 冻结目标网络
        for param in self.target_critic.parameters():
            param.requires_grad = False
        for param in self.target_grasp_critic.parameters():
            param.requires_grad = False

        # 温度参数
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.grasp_critic_optimizer = optim.Adam(self.grasp_critic.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def sample_actions(
        self, obs: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> torch.Tensor:
        """采样动作（连续动作 + 抓取动作）"""
        with torch.no_grad():
            obs = to_device(obs, self.device)

            # 连续动作
            continuous_action, _, continuous_mean = self.actor(obs)
            if deterministic:
                continuous_action = continuous_mean

            # 抓取动作（使用Q值选择）
            grasp_q_values = self.grasp_critic(obs)
            grasp_action = grasp_q_values.argmax(dim=-1, keepdim=True)
            # 转换到 {-1, 0, 1}
            grasp_action = grasp_action.float() - 1.0

            # 拼接动作
            full_action = torch.cat([continuous_action, grasp_action], dim=-1)
            return full_action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新网络参数"""
        batch = to_device(batch, self.device)

        # 分离连续动作和抓取动作
        continuous_actions = batch["actions"][..., :-1]
        grasp_actions = batch["actions"][..., -1:]

        # 更新连续动作的Critic
        critic_loss = self._update_critic(batch, continuous_actions)

        # 更新抓取Critic
        grasp_critic_loss = self._update_grasp_critic(batch, grasp_actions)

        # 更新Actor和温度参数
        actor_loss = self._update_actor(batch)
        alpha_loss = self._update_alpha(batch)

        # 软更新目标网络
        self._soft_update_target()

        return {
            "critic_loss": critic_loss,
            "grasp_critic_loss": grasp_critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha.item(),
        }

    def _update_critic(
        self, batch: Dict[str, torch.Tensor], continuous_actions: torch.Tensor
    ) -> float:
        """更新连续动作Critic"""
        with torch.no_grad():
            next_continuous_actions, next_log_probs, _ = self.actor(
                batch["next_observations"]
            )
            target_q_values = self.target_critic(
                batch["next_observations"], next_continuous_actions
            )
            target_q = target_q_values.min(dim=0)[0]

            target_q = batch["rewards"] + self.discount * batch["masks"] * (
                target_q - self.alpha * next_log_probs.squeeze()
            )

        current_q_values = self.critic(batch["observations"], continuous_actions)

        critic_loss = 0
        for i in range(current_q_values.shape[0]):
            critic_loss += F.mse_loss(current_q_values[i], target_q)
        critic_loss /= current_q_values.shape[0]

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss.item()

    def _update_grasp_critic(
        self, batch: Dict[str, torch.Tensor], grasp_actions: torch.Tensor
    ) -> float:
        """更新抓取Critic（DQN风格）"""
        with torch.no_grad():
            # 目标Q值
            target_next_grasp_q = self.target_grasp_critic(batch["next_observations"])
            next_grasp_q = self.grasp_critic(batch["next_observations"])
            best_next_grasp_actions = next_grasp_q.argmax(dim=-1)
            target_next_q = target_next_grasp_q.gather(
                1, best_next_grasp_actions.unsqueeze(1)
            ).squeeze(1)

            # 包含抓取惩罚的奖励
            grasp_rewards = batch["rewards"]
            if "grasp_penalty" in batch:
                grasp_rewards = grasp_rewards + batch["grasp_penalty"]

            target_q = grasp_rewards + self.discount * batch["masks"] * target_next_q

        # 当前Q值
        current_grasp_q = self.grasp_critic(batch["observations"])
        # 转换抓取动作到索引 {-1, 0, 1} -> {0, 1, 2}
        grasp_action_indices = (grasp_actions.squeeze() + 1).long()
        current_q = current_grasp_q.gather(
            1, grasp_action_indices.unsqueeze(1)
        ).squeeze(1)

        # 损失
        grasp_critic_loss = F.mse_loss(current_q, target_q)

        self.grasp_critic_optimizer.zero_grad()
        grasp_critic_loss.backward()
        self.grasp_critic_optimizer.step()

        return grasp_critic_loss.item()

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> float:
        """更新Actor网络"""
        actions, log_probs, _ = self.actor(batch["observations"])
        q_values = self.critic(batch["observations"], actions)
        q_value = q_values.mean(dim=0)

        actor_loss = (self.alpha * log_probs.squeeze() - q_value).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, batch: Dict[str, torch.Tensor]) -> float:
        """更新温度参数"""
        with torch.no_grad():
            _, log_probs, _ = self.actor(batch["observations"])

        alpha_loss = -(
            self.log_alpha * (log_probs.squeeze() + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def _soft_update_target(self):
        """软更新目标网络"""
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.grasp_critic.parameters(), self.target_grasp_critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "grasp_critic": self.grasp_critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "target_grasp_critic": self.target_grasp_critic.state_dict(),
                "log_alpha": self.log_alpha,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "grasp_critic_optimizer": self.grasp_critic_optimizer.state_dict(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.grasp_critic.load_state_dict(checkpoint["grasp_critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.target_grasp_critic.load_state_dict(checkpoint["target_grasp_critic"])
        self.log_alpha.data = checkpoint["log_alpha"]
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.grasp_critic_optimizer.load_state_dict(
            checkpoint["grasp_critic_optimizer"]
        )
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
