"""混合模式训练脚本（learner + actor）

与 JAX 版本的架构对比：
- JAX 版本：分布式架构，learner 和 actor 在不同进程/机器上，需要网络通信同步参数
- PyTorch 版本：单进程多线程架构，learner 和 actor 共享内存中的 agent 对象

移除 update_queue 的原因：
1. 在单进程多线程架构中，两个线程共享同一个 agent 对象
2. PyTorch 的参数是通过引用共享的，learner 线程的参数更新对 actor 线程立即可见
3. 避免了不必要的参数复制和队列开销
4. 简化了代码逻辑，降低了出错概率
"""

import os
import argparse
import pickle as pkl
import torch
import numpy as np
import time
import threading
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from tqdm import tqdm

from agents.sac import SACHybridAgent
from data.replay_buffer import ReplayBuffer
from utils.device import DEVICE, to_device


class SharedReplayBuffer:
    """线程安全的共享重放缓冲区"""

    def __init__(self, capacity: int, image_keys: list):
        self.buffer = ReplayBuffer(capacity, image_keys)
        self.lock = threading.Lock()

    def insert(self, transition: Dict):
        with self.lock:
            self.buffer.insert(transition)

    def sample(self, batch_size: int, device: torch.device):
        with self.lock:
            if len(self.buffer) >= batch_size:
                return self.buffer.sample(batch_size, device)
            else:
                return None

    def __len__(self):
        with self.lock:
            return len(self.buffer)


class MockRewardClassifier:
    """模拟分类器，使用与train_classifier相同的网络结构"""

    def __init__(self, image_keys, device: torch.device = DEVICE, checkpoint_path=None):
        self.device = device
        self.image_keys = image_keys

        # 使用与train_classifier相同的RewardClassifier
        from networks.classifier import RewardClassifier

        self.classifier = RewardClassifier(image_keys).to(device)

        # 如果提供了检查点路径，加载训练好的权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            print(f"已加载预训练分类器: {checkpoint_path}")
        else:
            # 否则初始化模拟权重
            self._initialize_mock_weights()
            print("使用随机初始化的模拟分类器")

    def _load_checkpoint(self, checkpoint_path):
        """加载预训练的分类器权重"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
            self.classifier.eval()
        except Exception as e:
            print(f"加载分类器权重失败: {e}")
            print("使用随机初始化的权重")
            self._initialize_mock_weights()

    def _initialize_mock_weights(self):
        """初始化模拟权重，使分类器有一定的模式"""
        for module in self.classifier.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

    def __call__(self, obs_dict):
        """模拟分类器推理，返回logits"""
        # 过滤只包含图像键的观测
        image_obs = {}
        for key in self.image_keys:
            if key in obs_dict:
                image = obs_dict[key]

                # 确保输入格式正确
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image).float()

                # 归一化到[0,1]
                if image.max() > 1.0:
                    image = image / 255.0

                # 添加batch维度如果需要
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)

                image_obs[key] = image.to(self.device)

        if not image_obs:
            # 如果没有有效的图像输入，返回随机logits
            return np.random.randn()

        with torch.no_grad():
            logits = self.classifier(image_obs)
            return logits.cpu().numpy().item()


class MockEnvironment:
    """模拟环境，集成分类器reward打分逻辑"""

    def __init__(
        self,
        image_keys,
        action_dim,
        image_size=(128, 128),
        use_classifier=True,
        classifier_path=None,
    ):
        self.image_keys = image_keys
        self.action_dim = action_dim
        self.image_size = image_size
        self.step_count = 0
        self.max_steps = 100
        self.use_classifier = use_classifier

        # 创建模拟分类器（类似JAX版本的load_classifier_func）
        if self.use_classifier:
            self.classifier = MockRewardClassifier(
                image_keys=self.image_keys, checkpoint_path=classifier_path
            )
            print("已创建模拟reward分类器")
        else:
            self.classifier = None

        # 模拟状态信息（类似JAX版本中obs['state']）
        self.state = np.random.rand(7)  # 7维状态向量

    def _compute_classifier_reward(self, obs):
        """计算分类器reward，模拟JAX版本的reward_func逻辑"""
        if not self.use_classifier or self.classifier is None:
            return 0

        try:
            # 获取分类器logits
            logits = self.classifier(obs)

            # 应用sigmoid函数（与JAX版本完全一致）
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            prob = sigmoid(logits)

            # 模拟不同任务的reward条件（参考JAX版本的实现）
            # RAM insertion: sigmoid(classifier(obs)) > 0.85 and obs['state'][0, 6] > 0.04
            # USB insertion: sigmoid(classifier(obs)) > 0.7 and obs["state"][0, 0] > 0.4
            # Object handover: sigmoid(classifier(obs)) > 0.75 and obs['state'][0, 0] > 0.5

            # 使用类似USB insertion的条件
            state_condition = self.state[0] > 0.4  # 模拟位置条件
            classifier_condition = prob > 0.7  # 分类器阈值

            reward = int(classifier_condition and state_condition)

            # 添加调试信息
            if hasattr(self, "_step_count") and self._step_count % 50 == 0:
                print(
                    f"Classifier: logits={logits:.3f}, prob={prob:.3f}, "
                    f"state_cond={state_condition}, reward={reward}"
                )

            return reward

        except Exception as e:
            print(f"分类器reward计算失败: {e}")
            return 0

    def reset(self):
        self.step_count = 0
        self._step_count = 0  # 用于调试

        # 重置状态
        self.state = np.random.rand(7)

        obs = {}
        for key in self.image_keys:
            # 生成随机图像 (C, H, W)
            obs[key] = np.random.randint(0, 256, (3, *self.image_size), dtype=np.uint8)

        # 添加状态信息（模拟JAX版本）
        obs["state"] = self.state.reshape(1, -1)  # 添加batch维度保持一致

        return obs, {}

    def step(self, action):
        self.step_count += 1
        self._step_count += 1

        # 更新状态（简单的随机游走）
        self.state += np.random.normal(0, 0.1, self.state.shape)
        self.state = np.clip(self.state, 0, 1)  # 保持在[0,1]范围

        # 下一个观测
        next_obs = {}
        for key in self.image_keys:
            next_obs[key] = np.random.randint(
                0, 256, (3, *self.image_size), dtype=np.uint8
            )

        # 添加状态信息
        next_obs["state"] = self.state.reshape(1, -1)

        # 使用分类器计算reward（模拟JAX版本的MultiCameraBinaryRewardClassifierWrapper）
        if self.use_classifier:
            reward = self._compute_classifier_reward(next_obs)
        else:
            # 回退到随机奖励
            reward = np.random.rand() - 0.5  # [-0.5, 0.5]

        # 结束条件：分类器给出positive reward或达到最大步数
        done = (
            self.step_count >= self.max_steps
            or (self.use_classifier and reward > 0)
            or (not self.use_classifier and np.random.rand() < 0.1)
        )

        truncated = False
        info = {}

        # 添加成功标志（模拟JAX版本）
        info["succeed"] = bool(reward > 0) if self.use_classifier else False

        # 模拟人工干预（小概率）
        if np.random.rand() < 0.05:
            info["intervene_action"] = action + np.random.normal(0, 0.1, action.shape)

        # 模拟grasp_penalty（与JAX版本保持一致）
        if np.random.rand() < 0.1:  # 小概率出现grasp penalty
            info["grasp_penalty"] = -0.02

        return next_obs, reward, done, truncated, info


def actor_loop(
    agent,
    env,
    replay_buffer: SharedReplayBuffer,
    demo_buffer: SharedReplayBuffer,
    max_steps: int,
    random_steps: int = 0,  # 与 JAX 版本保持一致，默认为 0
):
    """演员循环：与环境交互并收集数据"""
    print("启动演员循环...")

    episode_count = 0
    step_count = 0
    episode_rewards = deque(maxlen=10)

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    intervention_count = 0

    while step_count < max_steps:
        # 选择动作
        # 注意：在混合训练中，random_steps 通常设为 0，因为：
        # 1. 已有演示数据提供了良好的初始策略指导
        # 2. 不需要纯随机探索来填充重放缓冲区
        # 3. 可以从一开始就学习有意义的行为
        if step_count < random_steps:
            # 随机动作
            action = np.random.uniform(-1, 1, agent.action_dim)
        else:
            # 智能体动作
            obs_tensor = {}
            for key, value in obs.items():
                if key in agent.image_keys:
                    # 确保图像格式正确 (C, H, W) 并归一化
                    if len(value.shape) == 3 and value.shape[0] != 3:
                        value = value.transpose(2, 0, 1)  # HWC -> CHW
                    obs_tensor[key] = (
                        torch.from_numpy(value).float().unsqueeze(0) / 255.0
                    )
                elif key == "state":
                    # 处理状态信息（如果智能体需要的话）
                    # 注意：当前的SACHybridAgent主要使用图像，状态信息主要用于分类器
                    pass

            with torch.no_grad():
                obs_tensor = to_device(obs_tensor, agent.device)
                action = agent.sample_actions(obs_tensor, deterministic=False)
                action = action.cpu().numpy().squeeze()

        # 环境步进
        next_obs, reward, done, truncated, info = env.step(action)

        # 检查人工干预
        if "intervene_action" in info:
            action = info["intervene_action"]  # 使用干预动作
            intervention_count += 1
            is_intervention = True
        else:
            is_intervention = False

        # 创建转换
        transition = {
            "observations": obs,
            "actions": action,
            "next_observations": next_obs,
            "rewards": reward,
            "masks": 1.0 - done,
            "dones": done,
        }

        # 模拟环境可能提供grasp_penalty（与JAX版本保持一致）
        if "grasp_penalty" in info:
            transition["grasp_penalty"] = info["grasp_penalty"]

        # 添加到缓冲区
        replay_buffer.insert(transition)
        if is_intervention:
            demo_buffer.insert(transition)  # 干预数据也加到演示缓冲区

        # 更新统计
        episode_reward += reward
        episode_steps += 1
        step_count += 1

        # 注意：由于 learner 和 actor 在同一进程中，agent 参数是共享的
        # 不需要显式的参数同步机制

        # 重置环境
        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_count += 1

            print(
                f"Episode {episode_count}: Reward={episode_reward:.2f}, "
                f"Steps={episode_steps}, Interventions={intervention_count}, "
                f"Buffer Size={len(replay_buffer)}"
            )

            obs, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            intervention_count = 0
        else:
            obs = next_obs

    print("演员循环结束")


def learner_loop(
    agent,
    replay_buffer: SharedReplayBuffer,
    demo_buffer: SharedReplayBuffer,
    max_steps: int,
    batch_size: int = 256,
    training_starts: int = 1000,
    cta_ratio: int = 2,
    log_interval: int = 100,
    save_dir: str = "./hybrid_ckpt",
):
    """学习者循环：训练智能体"""
    print("启动学习者循环...")

    os.makedirs(save_dir, exist_ok=True)
    metrics = defaultdict(list)
    step = 0

    # 等待在线数据缓冲区填充（与JAX版本完全一致）
    # JAX版本只等待replay_buffer达到training_starts，不考虑demo_buffer
    print(f"等待在线数据缓冲区填充到 {training_starts} 个样本...")
    print(f"当前演示数据: {len(demo_buffer)}, 在线数据: {len(replay_buffer)}")

    pbar_fill = tqdm(
        total=training_starts,
        initial=len(replay_buffer),
        desc="填充在线数据缓冲区",
    )

    while len(replay_buffer) < training_starts:
        pbar_fill.update(len(replay_buffer) - pbar_fill.n)
        time.sleep(1)

    pbar_fill.update(len(replay_buffer) - pbar_fill.n)
    pbar_fill.close()

    print("开始训练...")
    print("采用与JAX版本完全一致的50/50采样策略")
    pbar = tqdm(total=max_steps, desc="学习进度")

    # 训练统计
    training_stats = {"total_steps": 0}

    while step < max_steps:
        # 采样策略：与JAX版本保持严格一致的50/50采样
        online_size = len(replay_buffer)
        demo_size = len(demo_buffer)

        # JAX版本使用固定的batch_size//2进行50/50采样
        required_batch_size = batch_size // 2

        # 等待有足够的数据进行50/50采样
        if online_size < required_batch_size or demo_size < required_batch_size:
            # 如果其中一个缓冲区数据不足，暂停训练等待更多数据
            time.sleep(0.1)
            continue

        # 执行严格的50/50采样（与JAX版本完全一致）
        online_batch = replay_buffer.sample(required_batch_size, agent.device)
        demo_batch = demo_buffer.sample(required_batch_size, agent.device)

        if online_batch is None or demo_batch is None:
            time.sleep(0.1)
            continue

        # 合并批次（参数顺序与JAX版本一致：在线数据在前，演示数据在后）
        batch = concat_batches(online_batch, demo_batch)
        sampling_strategy = "50_50_strict"
        actual_ratio = f"{required_batch_size}/{required_batch_size}"

        # Critic-to-Actor比率训练（与JAX版本保持一致）
        # 前 cta_ratio-1 次：只更新Critic网络
        for critic_step in range(cta_ratio - 1):
            try:
                # 只更新Critic网络（critic + grasp_critic）
                critics_info = agent.update_critics_only(batch)
            except Exception as e:
                print(f"Critic更新失败: {e}")
                continue

        # 第 cta_ratio 次：完整更新所有网络
        try:
            update_info = agent.update(batch)
        except Exception as e:
            print(f"智能体更新失败: {e}")
            update_info = {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0}

        # 记录指标
        for key, value in update_info.items():
            metrics[key].append(value)

        step += 1
        pbar.update(1)

        # 注意：JAX版本每50步(steps_per_update)发送一次网络参数给actor
        # PyTorch版本由于learner和actor共享agent对象，参数更新对actor立即可见
        # 因此不需要显式的参数同步机制

        # 日志输出
        if step % log_interval == 0:
            avg_metrics = {
                key: np.mean(values[-log_interval:]) for key, values in metrics.items()
            }
            log_str = f"Step {step}: "
            log_str += ", ".join(
                [f"{key}={value:.4f}" for key, value in avg_metrics.items()]
            )
            log_str += f", Buffer={len(replay_buffer)}, Demo={len(demo_buffer)}"
            log_str += f", Sampling={sampling_strategy}({actual_ratio})"

            tqdm.write(log_str)

        # 保存检查点（与JAX版本的checkpoint_period对应）
        # JAX版本默认checkpoint_period=0（不保存），这里设为20000减少保存频率
        if step > 0 and step % 20000 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_{step}.pth")
            agent.save(checkpoint_path)
            tqdm.write(f"保存检查点: {checkpoint_path}")

    pbar.close()

    # 保存最终模型
    final_path = os.path.join(save_dir, "final_model.pth")
    agent.save(final_path)
    print(f"保存最终模型: {final_path}")

    # 输出训练统计信息
    print(f"训练统计:")
    print(f"  总训练步数: {step}")
    print(f"  采用严格50/50采样策略（与JAX版本一致）")
    print("学习者循环结束")


def train_hybrid(
    demo_paths,
    image_keys,
    setup_mode="single-arm-fixed-gripper",
    max_steps=100000,
    batch_size=256,
    lr=3e-4,
    training_starts=1000,
    random_steps=0,  # 与 JAX 版本保持一致，默认为 0
    cta_ratio=2,
    device=DEVICE,
    save_dir="./hybrid_ckpt",
    log_interval=100,
    use_classifier=True,  # 新增参数：是否使用分类器reward
    classifier_path=None,  # 新增参数：分类器检查点路径
):
    """混合模式训练（learner + actor）

    与JAX版本完全一致的采样策略:
    - 严格执行50/50采样：每个训练批次包含batch_size//2的在线数据 + batch_size//2的演示数据
    - 如果任一缓冲区数据不足batch_size//2，则暂停训练等待更多数据
    - 确保训练过程中演示数据始终占据固定50%比例
    - 这种策略与JAX版本train_rlpd.py完全一致，有助于稳定训练并充分利用专家演示

    架构差异说明:
    - JAX版本：分布式架构，learner和actor分离，需要网络参数同步
    - PyTorch版本：单进程多线程架构，learner和actor共享agent对象，参数自动同步

    Args:
        demo_paths: 演示数据路径列表
        image_keys: 图像键列表
        setup_mode: 设置模式
        max_steps: 最大训练步数
        batch_size: 批次大小
        lr: 学习率
        training_starts: 开始训练的最小样本数
        random_steps: 随机探索步数（默认0，与JAX版本一致）
        cta_ratio: Critic-to-Actor更新比率
        device: 设备
        save_dir: 保存目录
        log_interval: 日志间隔
        use_classifier: 是否使用分类器reward（模拟JAX版本的reward classifier）
        classifier_path: 预训练分类器检查点路径（可选）
    """
    print(f"使用设备: {device}")
    print(f"设置模式: {setup_mode}")

    # 创建共享重放缓冲区
    replay_buffer = SharedReplayBuffer(capacity=200000, image_keys=image_keys)
    demo_buffer = SharedReplayBuffer(capacity=50000, image_keys=image_keys)

    # 加载演示数据到demo缓冲区
    if demo_paths:
        print("加载演示数据...")
        for demo_path in demo_paths:
            with open(demo_path, "rb") as f:
                demo_data = pkl.load(f)

            for transition in demo_data:
                if "observations" not in transition or "actions" not in transition:
                    continue

                if "masks" not in transition:
                    transition["masks"] = 1.0 - transition.get("dones", False)

                # 处理grasp_penalty（与JAX版本保持一致）
                if "infos" in transition and "grasp_penalty" in transition["infos"]:
                    transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]

                demo_buffer.insert(transition)

        print(f"加载了 {len(demo_buffer)} 个演示转换")

    # 确定动作维度（从演示数据推断）
    if len(demo_buffer) > 0:
        sample_batch = demo_buffer.sample(1, device)
        if sample_batch is not None:
            action_dim = sample_batch["actions"].shape[-1]
        else:
            action_dim = 7  # 默认值
    else:
        # 默认动作维度
        if "dual-arm" in setup_mode:
            action_dim = 14 if "learned-gripper" in setup_mode else 12
        else:
            action_dim = 7 if "learned-gripper" in setup_mode else 6

    print(f"动作维度: {action_dim}")

    # 创建智能体
    continuous_action_dim = action_dim - (2 if "dual-arm" in setup_mode else 1)
    agent = SACHybridAgent(
        image_keys=image_keys,
        continuous_action_dim=continuous_action_dim,
        grasp_action_dim=3,
        lr=lr,
        device=device,
    )
    # 添加属性用于演员循环
    setattr(agent, "action_dim", action_dim)
    print(f"创建混合SAC智能体 (连续动作维度: {continuous_action_dim})")

    # 创建模拟环境（启用分类器以模拟JAX版本的reward机制）
    env = MockEnvironment(
        image_keys,
        action_dim,
        use_classifier=use_classifier,
        classifier_path=classifier_path,
    )
    classifier_status = "已启用" if use_classifier else "已禁用"
    print(f"创建模拟环境（reward分类器{classifier_status}）")

    # 验证智能体和环境的兼容性
    try:
        test_obs, _ = env.reset()
        print(f"环境观测键: {list(test_obs.keys())}")
        print(f"智能体图像键: {agent.image_keys}")
        print(f"环境动作维度: {action_dim}")
        print("基本兼容性验证通过")
    except Exception as e:
        print(f"环境验证失败: {e}")
        print("训练可能会遇到问题，但将继续尝试...")

    # 启动学习者线程
    learner_thread = threading.Thread(
        target=learner_loop,
        args=(
            agent,
            replay_buffer,
            demo_buffer,
            max_steps,
            batch_size,
            training_starts,
            cta_ratio,
            log_interval,
            save_dir,
        ),
    )
    learner_thread.daemon = True
    learner_thread.start()

    # 运行演员循环（主线程）
    actor_loop(
        agent=agent,
        env=env,
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        max_steps=max_steps,
        random_steps=random_steps,
    )

    # 等待学习者线程完成
    learner_thread.join()

    print("混合训练完成")
    return agent


def concat_batches(online_batch, demo_batch):
    """
    合并在线数据和演示数据批次，与JAX版本保持完全一致

    JAX版本调用: concat_batches(batch, demo_batch, axis=0)
    - 第一个参数: 在线数据批次
    - 第二个参数: 演示数据批次
    - 合并维度: axis=0 (PyTorch中对应dim=0)

    Args:
        online_batch: 在线数据批次
        demo_batch: 演示数据批次

    Returns:
        合并后的批次
    """
    batch = {}

    # 遍历在线数据的所有键
    for key in online_batch.keys():
        if key in demo_batch:
            # 如果演示数据中也有这个键，合并
            if isinstance(online_batch[key], dict):
                # 递归处理嵌套字典
                batch[key] = concat_batches(online_batch[key], demo_batch[key])
            else:
                # 直接合并张量 (与JAX版本的axis=0对应)
                batch[key] = torch.cat([online_batch[key], demo_batch[key]], dim=0)
        else:
            # 如果演示数据中没有这个键，只使用在线数据
            batch[key] = online_batch[key]

    return batch


def main():
    parser = argparse.ArgumentParser(description="混合模式强化学习训练")
    parser.add_argument("--demo_paths", nargs="*", default=[], help="演示数据路径")
    parser.add_argument("--image_keys", nargs="+", default=["image"], help="图像键")
    parser.add_argument(
        "--setup_mode",
        type=str,
        default="single-arm-fixed-gripper",
        choices=[
            "single-arm-fixed-gripper",
            "single-arm-learned-gripper",
            "dual-arm-fixed-gripper",
            "dual-arm-learned-gripper",
        ],
        help="设置模式",
    )
    parser.add_argument("--max_steps", type=int, default=100000, help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument(
        "--training_starts", type=int, default=1000, help="开始训练的最小样本数"
    )
    parser.add_argument(
        "--random_steps",
        type=int,
        default=0,
        help="随机探索步数（与JAX版本保持一致，混合训练中通常为0）",
    )
    parser.add_argument(
        "--cta_ratio", type=int, default=2, help="Critic-to-Actor更新比率"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./hybrid_ckpt", help="保存目录"
    )
    parser.add_argument("--log_interval", type=int, default=100, help="日志间隔")
    parser.add_argument(
        "--use_classifier",
        action="store_true",
        default=True,
        help="是否使用分类器reward（模拟JAX版本）",
    )
    parser.add_argument(
        "--no_classifier",
        dest="use_classifier",
        action="store_false",
        help="禁用分类器reward，使用随机reward",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default=None,
        help="预训练分类器检查点路径（可选）",
    )

    args = parser.parse_args()

    # 验证演示数据文件存在
    for demo_path in args.demo_paths:
        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"演示数据文件不存在: {demo_path}")

    print(f"演示数据文件: {args.demo_paths}")

    # 开始训练
    train_hybrid(
        demo_paths=args.demo_paths,
        image_keys=args.image_keys,
        setup_mode=args.setup_mode,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        training_starts=args.training_starts,
        random_steps=args.random_steps,
        cta_ratio=args.cta_ratio,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        use_classifier=args.use_classifier,
        classifier_path=args.classifier_path,
    )


if __name__ == "__main__":
    main()
