"""简单的混合训练测试"""

import sys
import os
import torch
import numpy as np
import pickle as pkl
import tempfile
import shutil

sys.path.append(".")

from train_hybrid import train_hybrid
from utils.device import DEVICE


def create_simple_test():
    """创建一个简单的测试"""
    print("创建简单测试...")

    # 创建临时目录
    test_dir = tempfile.mkdtemp(prefix="simple_test_")
    print(f"测试目录: {test_dir}")

    # 生成简单的演示数据
    demo_data = []
    for i in range(10):
        transition = {
            "observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "actions": np.random.uniform(-1, 1, 7),
            "next_observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "rewards": np.random.uniform(0.5, 1.0),
            "masks": 1.0,
            "dones": np.random.choice([True, False], p=[0.2, 0.8]),
        }
        demo_data.append(transition)

    # 保存演示数据
    demo_path = os.path.join(test_dir, "demo.pkl")
    with open(demo_path, "wb") as f:
        pkl.dump(demo_data, f)

    print(f"创建演示数据: {len(demo_data)} 个样本")

    # 运行训练
    try:
        print("开始训练...")
        agent = train_hybrid(
            demo_paths=[demo_path],
            image_keys=["image"],
            setup_mode="single-arm-fixed-gripper",
            max_steps=50,  # 非常短的训练
            batch_size=4,  # 非常小的批次
            lr=3e-4,
            training_starts=5,  # 很低的启动阈值
            random_steps=10,
            cta_ratio=1,  # 减少critic更新
            save_dir=os.path.join(test_dir, "ckpt"),
            log_interval=10,
        )

        print("✅ 训练成功完成")
        success = True

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback

        traceback.print_exc()
        success = False

    finally:
        # 清理
        shutil.rmtree(test_dir)
        print(f"清理测试目录: {test_dir}")

    return success


if __name__ == "__main__":
    print("=== 简单混合训练测试 ===")
    print(f"设备: {DEVICE}")

    success = create_simple_test()

    if success:
        print("\n🎉 简单测试通过！")
    else:
        print("\n❌ 简单测试失败！")
        exit(1)
