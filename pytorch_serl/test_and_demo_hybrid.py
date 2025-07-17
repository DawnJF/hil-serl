"""完整的混合训练测试和演示脚本"""

import sys
import os
import argparse
import torch
import numpy as np
import pickle as pkl
import shutil

sys.path.append(".")

from train_hybrid import train_hybrid
from utils.device import DEVICE


def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")

    # 创建测试目录
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)

    # 生成成功演示数据
    success_data = []
    for i in range(30):
        transition = {
            "observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "actions": np.random.uniform(-1, 1, 7),
            "next_observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "rewards": np.random.uniform(0.7, 1.0),  # 高奖励
            "masks": 1.0,
            "dones": np.random.choice([True, False], p=[0.1, 0.9]),
        }
        success_data.append(transition)

    # 保存成功数据
    success_path = os.path.join(test_dir, "success_demo.pkl")
    with open(success_path, "wb") as f:
        pkl.dump(success_data, f)

    # 生成失败演示数据
    failure_data = []
    for i in range(20):
        transition = {
            "observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "actions": np.random.uniform(-1, 1, 7),
            "next_observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "rewards": np.random.uniform(-0.5, 0.3),  # 低奖励
            "masks": 1.0,
            "dones": np.random.choice([True, False], p=[0.3, 0.7]),
        }
        failure_data.append(transition)

    # 保存失败数据
    failure_path = os.path.join(test_dir, "failure_demo.pkl")
    with open(failure_path, "wb") as f:
        pkl.dump(failure_data, f)

    print(f"创建测试数据:")
    print(f"  成功演示: {success_path} ({len(success_data)} 样本)")
    print(f"  失败演示: {failure_path} ({len(failure_data)} 样本)")

    return [success_path, failure_path]


def run_quick_test():
    """运行快速测试"""
    print("\n=== 快速测试 ===")

    # 创建临时测试数据
    test_dir = os.path.join("log", "hybrid_test")
    os.makedirs(test_dir, exist_ok=True)

    try:
        # 生成少量演示数据
        demo_data = []
        for i in range(5):
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
                "dones": np.random.choice([True, False]),
            }
            demo_data.append(transition)

        demo_path = os.path.join(test_dir, "demo.pkl")
        with open(demo_path, "wb") as f:
            pkl.dump(demo_data, f)

        # 运行训练
        print("开始快速训练...")
        agent = train_hybrid(
            demo_paths=[demo_path],
            image_keys=["image"],
            setup_mode="single-arm-fixed-gripper",
            max_steps=20,
            batch_size=2,
            lr=3e-4,
            training_starts=3,
            random_steps=5,
            cta_ratio=1,
            save_dir=os.path.join(test_dir, "ckpt"),
            log_interval=5,
        )

        print("✅ 快速测试成功")
        return True

    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        shutil.rmtree(test_dir)


def run_full_test(demo_paths, max_steps=1000):
    """运行完整测试"""
    print(f"\n=== 完整测试 (max_steps={max_steps}) ===")

    save_dir = "hybrid_test_ckpt"
    os.makedirs(save_dir, exist_ok=True)

    try:
        print("开始完整训练...")
        agent = train_hybrid(
            demo_paths=demo_paths,
            image_keys=["image"],
            setup_mode="single-arm-fixed-gripper",
            max_steps=max_steps,
            batch_size=16,
            lr=3e-4,
            training_starts=50,
            random_steps=100,
            cta_ratio=2,
            save_dir=save_dir,
            log_interval=100,
        )

        print("✅ 完整测试成功")

        # 检查保存的文件
        if os.path.exists(save_dir):
            files = os.listdir(save_dir)
            print(f"保存的文件: {files}")

            # 检查最终模型
            final_model_path = os.path.join(save_dir, "final_model.pth")
            if os.path.exists(final_model_path):
                print(f"✅ 最终模型已保存: {final_model_path}")

                # 检查模型大小
                size = os.path.getsize(final_model_path)
                print(f"模型大小: {size / 1024 / 1024:.2f} MB")

                # 尝试加载模型
                try:
                    checkpoint = torch.load(final_model_path, map_location=DEVICE)
                    print(f"✅ 模型加载成功，包含键: {list(checkpoint.keys())}")
                except Exception as e:
                    print(f"❌ 模型加载失败: {e}")
            else:
                print("❌ 最终模型文件不存在")

        return True

    except Exception as e:
        print(f"❌ 完整测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="混合训练测试和演示")
    parser.add_argument(
        "--test", choices=["quick", "full", "both"], default="both", help="测试类型"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="完整测试的最大步数"
    )
    parser.add_argument("--demo_paths", nargs="*", default=[], help="演示数据路径")
    parser.add_argument("--create_data", action="store_true", help="创建测试数据")

    args = parser.parse_args()

    print("=== 混合训练测试和演示 ===")
    print(f"设备: {DEVICE}")
    print(f"PyTorch版本: {torch.__version__}")

    results = []

    # 创建测试数据
    if args.create_data or not args.demo_paths:
        demo_paths = create_test_data()
    else:
        demo_paths = args.demo_paths

    # 验证演示数据存在
    for path in demo_paths:
        if not os.path.exists(path):
            print(f"❌ 演示数据文件不存在: {path}")
            return 1

    print(f"使用演示数据: {demo_paths}")

    # 运行测试
    if args.test in ["quick", "both"]:
        results.append(("快速测试", run_quick_test()))

    if args.test in ["full", "both"]:
        results.append(("完整测试", run_full_test(demo_paths, args.max_steps)))

    # 输出结果
    print("\n=== 测试结果汇总 ===")
    all_passed = True
    for test_name, success in results:
        if success:
            print(f"✅ {test_name}: 通过")
        else:
            print(f"❌ {test_name}: 失败")
            all_passed = False

    if all_passed:
        print("\n🎉 所有测试通过！")
        print("\n可以使用以下命令运行更长时间的训练:")
        print(f"python {__file__} --test full --max_steps 10000")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
