"""简单测试脚本，验证PyTorch SERL实现"""

import os
import torch
import numpy as np
import sys

sys.path.append(".")

from utils.device import DEVICE
from networks.resnet import ResNetEncoder
from networks.actor_critic import Actor, Critic, GraspCritic
from networks.classifier import RewardClassifier
from agents.sac import SACAgent, SACHybridAgent
from data.replay_buffer import ReplayBuffer


def test_device():
    """测试设备配置"""
    print(f"检测到设备: {DEVICE}")

    # 测试基本tensor操作
    x = torch.randn(2, 3).to(DEVICE)
    y = torch.randn(2, 3).to(DEVICE)
    z = x + y
    print(f"设备测试通过: {z.device}")


def test_networks():
    """测试网络架构"""
    print("\n=== 测试网络架构 ===")

    batch_size = 4
    image_size = (128, 128)
    image_keys = ["image"]

    # 测试ResNet编码器
    print("测试ResNet编码器...")
    encoder = ResNetEncoder().to(DEVICE)
    test_image = torch.randn(batch_size, 3, *image_size).to(DEVICE)
    encoded = encoder(test_image)
    print(f"编码器输出形状: {encoded.shape}")

    # 测试Actor
    print("测试Actor网络...")
    actor = Actor(encoder=None, hidden_dims=[256, 256], action_dim=6, input_dim=256).to(
        DEVICE
    )
    test_obs = torch.randn(batch_size, 256).to(DEVICE)  # 假设已编码特征
    action, log_prob, mean = actor(test_obs)
    print(f"动作形状: {action.shape}, 对数概率形状: {log_prob.shape}")

    # 测试Critic
    print("测试Critic网络...")
    critic = Critic(
        encoder=None, hidden_dims=[256, 256], action_dim=6, input_dim=256
    ).to(DEVICE)
    q_values = critic(test_obs, action)
    print(f"Q值形状: {q_values.shape}")

    # 测试GraspCritic
    print("测试GraspCritic网络...")
    grasp_critic = GraspCritic(encoder=None, hidden_dims=[128, 128], input_dim=256).to(
        DEVICE
    )
    grasp_q = grasp_critic(test_obs)
    print(f"抓取Q值形状: {grasp_q.shape}")

    # 测试分类器
    print("测试奖励分类器...")
    classifier = RewardClassifier(image_keys).to(DEVICE)
    obs_dict = {"image": test_image}
    logits = classifier(obs_dict)
    probs, is_success = classifier.predict_success(obs_dict)
    print(f"分类器logits形状: {logits.shape}, 概率形状: {probs.shape}")


def test_agents():
    """测试SAC智能体"""
    print("\n=== 测试SAC智能体 ===")

    image_keys = ["image"]
    batch_size = 4

    # 测试标准SAC智能体
    print("测试标准SAC智能体...")
    agent = SACAgent(image_keys=image_keys, action_dim=6, device=DEVICE)

    # 创建测试观测
    test_obs = {"image": torch.randn(batch_size, 3, 128, 128).to(DEVICE)}

    # 测试动作采样
    actions = agent.sample_actions(test_obs, deterministic=False)
    print(f"采样动作形状: {actions.shape}")

    # 测试混合SAC智能体
    print("测试混合SAC智能体...")
    hybrid_agent = SACHybridAgent(
        image_keys=image_keys,
        continuous_action_dim=6,
        grasp_action_dim=3,
        device=DEVICE,
    )

    hybrid_actions = hybrid_agent.sample_actions(test_obs, deterministic=False)
    print(f"混合动作形状: {hybrid_actions.shape}")


def test_replay_buffer():
    """测试重放缓冲区"""
    print("\n=== 测试重放缓冲区 ===")

    image_keys = ["image"]
    buffer = ReplayBuffer(capacity=1000, image_keys=image_keys)

    # 添加一些测试数据
    for i in range(10):
        transition = {
            "observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                "state": np.random.randn(10),
            },
            "actions": np.random.randn(7),
            "next_observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
                "state": np.random.randn(10),
            },
            "rewards": np.random.randn(),
            "masks": 1.0,
            "dones": False,
        }
        buffer.insert(transition)

    print(f"缓冲区大小: {len(buffer)}")

    # 测试采样
    batch = buffer.sample(batch_size=4, device=DEVICE)
    print(f"采样批次键: {list(batch.keys())}")
    print(f"观测图像形状: {batch['observations']['image'].shape}")
    print(f"动作形状: {batch['actions'].shape}")


def test_training_data():
    """测试生成训练数据"""
    print("\n=== 生成测试训练数据 ===")

    # 创建测试目录
    os.makedirs("test_data", exist_ok=True)

    # 生成模拟成功数据
    success_data = []
    for i in range(20):
        transition = {
            "observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "actions": np.random.randn(7),
            "next_observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "rewards": 1.0,  # 成功
            "masks": 1.0,
            "dones": True,
        }
        success_data.append(transition)

    # 保存成功数据
    import pickle as pkl

    with open("test_data/success_demo.pkl", "wb") as f:
        pkl.dump(success_data, f)

    # 生成模拟失败数据
    failure_data = []
    for i in range(20):
        transition = {
            "observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "actions": np.random.randn(7),
            "next_observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "rewards": 0.0,  # 失败
            "masks": 1.0,
            "dones": True,
        }
        failure_data.append(transition)

    # 保存失败数据
    with open("test_data/failure_demo.pkl", "wb") as f:
        pkl.dump(failure_data, f)

    print("生成测试数据:")
    print("- test_data/success_demo.pkl (20个成功样本)")
    print("- test_data/failure_demo.pkl (20个失败样本)")


def main():
    """主测试函数"""
    print("=== PyTorch SERL 实现测试 ===")

    try:
        test_device()
        test_networks()
        test_agents()
        test_replay_buffer()
        test_training_data()

        print("\n✅ 所有测试通过！")
        print("\n可以尝试运行以下命令:")
        print("1. 训练分类器:")
        print(
            "   python train_classifier.py --success_dir test_data --failure_dir test_data --output_dir test_classifier"
        )
        print("\n2. 离线训练:")
        print(
            "   python train_offline.py --demo_paths test_data/success_demo.pkl --save_dir test_offline"
        )
        print("\n3. 混合训练:")
        print(
            "   python train_hybrid.py --demo_paths test_data/success_demo.pkl --save_dir test_hybrid"
        )

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
