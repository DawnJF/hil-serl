"""混合模式训练脚本（learner + actor）"""

import os
import argparse
import pickle as pkl
import torch
import numpy as np
import time
import threading
from queue import Queue
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


class MockEnvironment:
    """模拟环境，用于测试"""

    def __init__(self, image_keys, action_dim, image_size=(128, 128)):
        self.image_keys = image_keys
        self.action_dim = action_dim
        self.image_size = image_size
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        self.step_count = 0
        obs = {}
        for key in self.image_keys:
            # 生成随机图像 (C, H, W)
            obs[key] = np.random.randint(0, 256, (3, *self.image_size), dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self.step_count += 1

        # 下一个观测
        next_obs = {}
        for key in self.image_keys:
            next_obs[key] = np.random.randint(
                0, 256, (3, *self.image_size), dtype=np.uint8
            )

        # 随机奖励和结束标志
        reward = np.random.rand() - 0.5  # [-0.5, 0.5]
        done = self.step_count >= self.max_steps or np.random.rand() < 0.1
        truncated = False
        info = {}

        # 模拟人工干预（小概率）
        if np.random.rand() < 0.05:
            info["intervene_action"] = action + np.random.normal(0, 0.1, action.shape)

        return next_obs, reward, done, truncated, info


def actor_loop(
    agent,
    env,
    replay_buffer: SharedReplayBuffer,
    demo_buffer: SharedReplayBuffer,
    max_steps: int,
    random_steps: int = 1000,
    update_queue: Optional[Queue] = None,
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

        # 添加到缓冲区
        replay_buffer.insert(transition)
        if is_intervention:
            demo_buffer.insert(transition)  # 干预数据也加到演示缓冲区

        # 更新统计
        episode_reward += reward
        episode_steps += 1
        step_count += 1

        # 检查网络更新
        if update_queue and not update_queue.empty():
            try:
                new_state_dict = update_queue.get_nowait()
                # 这里应该更新智能体的网络参数
                # 但由于我们的设计，暂时跳过
                pass
            except:
                pass

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
    update_queue: Optional[Queue] = None,
):
    """学习者循环：训练智能体"""
    print("启动学习者循环...")

    os.makedirs(save_dir, exist_ok=True)
    metrics = defaultdict(list)
    step = 0

    # 等待足够的数据
    print(f"等待足够的数据 (需要 {training_starts} 个样本)...")
    while len(replay_buffer) < training_starts:
        time.sleep(1)
        if len(replay_buffer) % 100 == 0:
            print(f"当前缓冲区大小: {len(replay_buffer)}")

    print("开始训练...")
    pbar = tqdm(total=max_steps, desc="学习进度")

    while step < max_steps:
        # 采样数据
        online_batch = replay_buffer.sample(batch_size // 2, agent.device)
        demo_batch = (
            demo_buffer.sample(batch_size // 2, agent.device)
            if len(demo_buffer) >= batch_size // 2
            else None
        )

        if online_batch is None:
            time.sleep(0.1)
            continue

        # 合并批次（如果有演示数据）
        if demo_batch is not None:
            # 合并两个批次
            batch = {}
            for key in online_batch.keys():
                if key in demo_batch:
                    batch[key] = torch.cat([online_batch[key], demo_batch[key]], dim=0)
                else:
                    batch[key] = online_batch[key]
        else:
            batch = online_batch

        # Critic-to-Actor比率训练
        for critic_step in range(cta_ratio - 1):
            # 只更新Critic
            if hasattr(agent, "_update_critic") and hasattr(
                agent, "_update_grasp_critic"
            ):
                # 混合智能体
                continuous_actions = (
                    batch["actions"][..., :-1] if "actions" in batch else None
                )
                grasp_actions = (
                    batch["actions"][..., -1:] if "actions" in batch else None
                )

                if continuous_actions is not None:
                    agent._update_critic(batch, continuous_actions)
                if grasp_actions is not None:
                    agent._update_grasp_critic(batch, grasp_actions)
            else:
                # 标准智能体
                agent._update_critic(batch)

        # 完整更新
        update_info = agent.update(batch)

        # 记录指标
        for key, value in update_info.items():
            metrics[key].append(value)

        step += 1
        pbar.update(1)

        # 发送网络更新
        if update_queue and step % 50 == 0:
            try:
                update_queue.put_nowait(agent.actor.state_dict())
            except:
                pass  # 队列满了，跳过

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
            tqdm.write(log_str)

        # 保存检查点
        if step % 10000 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_{step}.pth")
            agent.save(checkpoint_path)
            tqdm.write(f"保存检查点: {checkpoint_path}")

    pbar.close()

    # 保存最终模型
    final_path = os.path.join(save_dir, "final_model.pth")
    agent.save(final_path)
    print(f"保存最终模型: {final_path}")


def train_hybrid(
    demo_paths,
    image_keys,
    setup_mode="single-arm-fixed-gripper",
    max_steps=100000,
    batch_size=256,
    lr=3e-4,
    training_starts=1000,
    random_steps=1000,
    cta_ratio=2,
    device=DEVICE,
    save_dir="./hybrid_ckpt",
    log_interval=100,
):
    """混合模式训练（learner + actor）

    Args:
        demo_paths: 演示数据路径列表
        image_keys: 图像键列表
        setup_mode: 设置模式
        max_steps: 最大训练步数
        batch_size: 批次大小
        lr: 学习率
        training_starts: 开始训练的最小样本数
        random_steps: 随机探索步数
        cta_ratio: Critic-to-Actor更新比率
        device: 设备
        save_dir: 保存目录
        log_interval: 日志间隔
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

    # 创建模拟环境
    env = MockEnvironment(image_keys, action_dim)
    print("创建模拟环境")

    # 创建更新队列
    update_queue = Queue(maxsize=10)

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
            update_queue,
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
        update_queue=update_queue,
    )

    # 等待学习者线程完成
    learner_thread.join()

    print("混合训练完成")
    return agent


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
    parser.add_argument("--random_steps", type=int, default=1000, help="随机探索步数")
    parser.add_argument(
        "--cta_ratio", type=int, default=2, help="Critic-to-Actor更新比率"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./hybrid_ckpt", help="保存目录"
    )
    parser.add_argument("--log_interval", type=int, default=100, help="日志间隔")

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
    )


if __name__ == "__main__":
    main()
