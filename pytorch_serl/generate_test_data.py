"""生成更多的测试数据"""

import os
import numpy as np
import pickle as pkl
from tqdm import tqdm


def generate_test_data(num_samples=1000):
    """生成测试数据"""
    print(f"=== 生成 {num_samples} 个测试样本 ===")

    # 创建测试目录
    os.makedirs("log/test_data", exist_ok=True)

    # 生成成功数据
    print("生成成功数据...")
    success_data = []
    for i in tqdm(range(num_samples)):
        # 生成随机图像数据，使用一些简单的模式来模拟成功场景
        success_pattern = np.random.randint(
            100, 200, (128, 128, 3), dtype=np.uint8
        )  # 较亮的图像
        success_pattern[:, :, 0] += 20  # 稍微增加红色通道

        transition = {
            "observations": {
                "image": success_pattern,
                "state": np.random.randn(10),  # 添加状态信息
            },
            "actions": np.random.randn(7),  # 7维动作 (6D连续 + 1D抓取)
            "next_observations": {
                "image": np.random.randint(
                    150, 255, (128, 128, 3), dtype=np.uint8
                ),  # 更亮的下一帧
                "state": np.random.randn(10),
            },
            "rewards": 1.0,  # 成功奖励
            "masks": 1.0,
            "dones": True,
        }
        success_data.append(transition)

    # 保存成功数据
    with open("log/test_data/success_demo.pkl", "wb") as f:
        pkl.dump(success_data, f)

    # 生成失败数据
    print("生成失败数据...")
    failure_data = []
    for i in tqdm(range(num_samples)):
        # 生成随机图像数据，使用一些简单的模式来模拟失败场景
        failure_pattern = np.random.randint(
            20, 120, (128, 128, 3), dtype=np.uint8
        )  # 较暗的图像
        failure_pattern[:, :, 2] += 15  # 稍微增加蓝色通道

        transition = {
            "observations": {
                "image": failure_pattern,
                "state": np.random.randn(10),
            },
            "actions": np.random.randn(7),
            "next_observations": {
                "image": np.random.randint(
                    0, 100, (128, 128, 3), dtype=np.uint8
                ),  # 更暗的下一帧
                "state": np.random.randn(10),
            },
            "rewards": 0.0,  # 失败奖励
            "masks": 1.0,
            "dones": True,
        }
        failure_data.append(transition)

    # 保存失败数据
    with open("log/test_data/failure_demo.pkl", "wb") as f:
        pkl.dump(failure_data, f)

    print(f"✅ 生成完成!")
    print(f"- log/test_data/success_demo.pkl ({num_samples} 个成功样本)")
    print(f"- log/test_data/failure_demo.pkl ({num_samples} 个失败样本)")

    # 验证数据格式
    print("\n=== 验证数据格式 ===")
    with open("log/test_data/success_demo.pkl", "rb") as f:
        success_data = pkl.load(f)

    print(f"成功数据样本数: {len(success_data)}")
    sample = success_data[0]
    print(f"样本键: {list(sample.keys())}")
    print(f"观测键: {list(sample['observations'].keys())}")
    print(f"图像形状: {sample['observations']['image'].shape}")
    print(f"状态形状: {sample['observations']['state'].shape}")
    print(f"动作形状: {sample['actions'].shape}")
    print(f"奖励: {sample['rewards']}")

    return success_data, failure_data


if __name__ == "__main__":
    generate_test_data(num_samples=1000)
