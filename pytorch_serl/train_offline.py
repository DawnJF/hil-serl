"""纯离线训练脚本（仅learner）"""

import logging
import os
import argparse
import pickle as pkl
import torch
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from agents.sac import SACHybridAgent
from data.replay_buffer import ReplayBuffer
from utils import logger_utils
from utils.device import DEVICE, to_device


def load_demo_data(demo_paths, replay_buffer):
    """加载演示数据到重放缓冲区"""
    total_transitions = 0

    for demo_path in demo_paths:
        logging.info(f"加载演示数据: {demo_path}")
        with open(demo_path, "rb") as f:
            demo_data = pkl.load(f)

        for transition in demo_data:
            # 确保转换格式正确
            if "observations" not in transition or "actions" not in transition:
                continue

            # 添加必要字段
            if "masks" not in transition:
                transition["masks"] = 1.0 - transition.get("dones", False)

            # 处理抓取惩罚（如果存在）
            if "infos" in transition and "grasp_penalty" in transition["infos"]:
                transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]

            replay_buffer.insert(transition)
            total_transitions += 1

    logging.info(f"总共加载了 {total_transitions} 个转换")
    return total_transitions


def train_offline(
    demo_paths,
    image_keys,
    batch_size=256,
    max_steps=50000,
    lr=3e-4,
    device=DEVICE,
    save_dir="./offline_ckpt",
    log_interval=100,
):
    """纯离线训练

    Args:
        demo_paths: 演示数据路径列表
        image_keys: 图像键列表
        batch_size: 批次大小
        max_steps: 最大训练步数
        lr: 学习率
        device: 设备
        save_dir: 保存目录
        log_interval: 日志间隔
    """
    logging.info(f"使用设备: {device}")

    # 创建重放缓冲区
    replay_buffer = ReplayBuffer(capacity=200000, image_keys=image_keys)

    # 加载演示数据
    logging.info("加载演示数据...")
    load_demo_data(demo_paths, replay_buffer)

    if len(replay_buffer) < batch_size:
        raise ValueError(
            f"数据不足，需要至少 {batch_size} 个样本，但只有 {len(replay_buffer)} 个"
        )

    # 确定动作维度
    sample_batch = replay_buffer.sample(1, device)
    action_dim = sample_batch["actions"].shape[-1]
    logging.info(f"动作维度: {action_dim}")

    # 创建智能体
    # continuous_action_dim = action_dim - 1  # 最后一维是抓取动作
    continuous_action_dim = action_dim
    agent = SACHybridAgent(
        image_keys=image_keys,
        continuous_action_dim=continuous_action_dim,
        grasp_action_dim=3,
        lr=lr,
        device=device,
        # 离线训练需要更保守的设置
        discount=0.99,  # 更高的折扣因子
        tau=0.005,  # 更慢的目标网络更新
        target_entropy_scale=0.1,  # 修复：更低的熵正则化，防止alpha爆炸
    )
    logging.info(f"创建混合SAC智能体 (连续动作维度: {continuous_action_dim})")

    # 训练循环
    logging.info("开始离线训练...")

    metrics = defaultdict(list)

    for step in tqdm(range(max_steps), desc="训练进度"):
        # 采样批次数据
        batch = replay_buffer.sample(batch_size, device)

        # 更新智能体
        update_info = agent.update(batch)

        # 记录指标
        for key, value in update_info.items():
            metrics[key].append(value)

        # 日志输出
        if step % log_interval == 0 and step > 0:
            avg_metrics = {
                key: np.mean(values[-log_interval:]) for key, values in metrics.items()
            }
            log_str = f"Step {step}: "
            log_str += ", ".join(
                [f"{key}={value:.4f}" for key, value in avg_metrics.items()]
            )
            logging.info(log_str)

            # 额外的调试信息
            if step % (log_interval * 10) == 0:
                sample_batch = replay_buffer.sample(32, device)
                with torch.no_grad():
                    actions, log_probs, _ = agent.actor(sample_batch["observations"])
                    q_values = agent.critic(sample_batch["observations"], actions)
                    logging.info(
                        f"  Debug - Q值范围: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]"
                    )
                    logging.info(
                        f"  Debug - 奖励范围: [{sample_batch['rewards'].min().item():.3f}, {sample_batch['rewards'].max().item():.3f}]"
                    )
                    logging.info(f"  Debug - Alpha: {agent.alpha.item():.4f}")
                    logging.info(
                        f"  Debug - 对数概率范围: [{log_probs.min().item():.3f}, {log_probs.max().item():.3f}]"
                    )

        # 保存检查点
        if step % 10000 == 0 and step > 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_{step}.pth")
            agent.save(checkpoint_path)
            logging.info(f"保存检查点: {checkpoint_path}")

    # 保存最终模型
    final_path = os.path.join(save_dir, "final_model.pth")
    agent.save(final_path)
    logging.info(f"保存最终模型: {final_path}")

    return agent


def main():
    parser = argparse.ArgumentParser(description="纯离线强化学习训练")
    parser.add_argument("--demo_paths", nargs="+", required=True, help="演示数据路径")
    parser.add_argument("--image_keys", nargs="+", default=["image"], help="图像键")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--max_steps", type=int, default=1000000, help="最大训练步数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/liujinxin/mjf/hil-serl/log/offline",
        help="保存目录",
    )
    parser.add_argument("--log_interval", type=int, default=100, help="日志间隔")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    logger_utils.setup_logging(args.save_dir)

    # 验证演示数据文件存在
    for demo_path in args.demo_paths:
        if not os.path.exists(demo_path):
            logging.error(f"演示数据文件不存在: {demo_path}")
            raise FileNotFoundError(f"演示数据文件不存在: {demo_path}")

    logging.info(f"演示数据文件: {args.demo_paths}")

    # 开始训练
    train_offline(
        demo_paths=args.demo_paths,
        image_keys=args.image_keys,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.lr,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
