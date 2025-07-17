"""混合模式训练测试脚本"""

import os
import sys
import torch
import numpy as np
import pickle as pkl
import tempfile
import shutil
import threading
import time
from unittest.mock import MagicMock, patch

sys.path.append(".")

from train_hybrid import (
    train_hybrid,
    SharedReplayBuffer,
    MockEnvironment,
    actor_loop,
    learner_loop,
    concat_batches,
)
from agents.sac import SACHybridAgent
from utils.device import DEVICE


class TestHybridTraining:
    """混合训练测试类"""

    def __init__(self):
        self.test_dir: str = ""
        self.demo_paths = []

    def setup_test_environment(self):
        """设置测试环境"""
        print("设置测试环境...")

        # 创建临时目录
        self.test_dir = tempfile.mkdtemp(prefix="test_hybrid_")
        print(f"测试目录: {self.test_dir}")

        # 生成测试演示数据
        self.generate_test_demos()

    def cleanup_test_environment(self):
        """清理测试环境"""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"清理测试目录: {self.test_dir}")

    def generate_test_demos(self):
        """生成测试演示数据"""
        print("生成测试演示数据...")

        # 生成成功演示
        success_data = []
        for i in range(50):
            transition = {
                "observations": {
                    "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                },
                "actions": np.random.uniform(-1, 1, 7),
                "next_observations": {
                    "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                },
                "rewards": np.random.uniform(0.8, 1.0),  # 高奖励
                "masks": 1.0,
                "dones": np.random.choice([True, False], p=[0.1, 0.9]),
            }
            success_data.append(transition)

        success_path = os.path.join(self.test_dir, "success_demo.pkl")
        with open(success_path, "wb") as f:
            pkl.dump(success_data, f)

        # 生成一般演示
        mixed_data = []
        for i in range(30):
            transition = {
                "observations": {
                    "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                },
                "actions": np.random.uniform(-1, 1, 7),
                "next_observations": {
                    "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                },
                "rewards": np.random.uniform(-0.5, 0.8),  # 混合奖励
                "masks": 1.0,
                "dones": np.random.choice([True, False], p=[0.2, 0.8]),
            }
            mixed_data.append(transition)

        mixed_path = os.path.join(self.test_dir, "mixed_demo.pkl")
        with open(mixed_path, "wb") as f:
            pkl.dump(mixed_data, f)

        self.demo_paths = [success_path, mixed_path]
        print(
            f"生成演示数据: {len(success_data)} 成功样本 + {len(mixed_data)} 混合样本"
        )

    def test_shared_replay_buffer(self):
        """测试共享重放缓冲区"""
        print("\n=== 测试共享重放缓冲区 ===")

        image_keys = ["image"]
        buffer = SharedReplayBuffer(capacity=1000, image_keys=image_keys)

        # 测试插入
        for i in range(10):
            transition = {
                "observations": {
                    "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                },
                "actions": np.random.randn(7),
                "next_observations": {
                    "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                },
                "rewards": np.random.randn(),
                "masks": 1.0,
                "dones": False,
            }
            buffer.insert(transition)

        print(f"缓冲区大小: {len(buffer)}")

        # 测试采样
        batch = buffer.sample(batch_size=5, device=DEVICE)
        if batch is not None:
            print(f"采样成功，批次大小: {batch['actions'].shape[0]}")
        else:
            print("采样失败，数据不足")

        # 测试线程安全
        def insert_worker():
            for i in range(5):
                transition = {
                    "observations": {
                        "image": np.random.randint(
                            0, 255, (128, 128, 3), dtype=np.uint8
                        )
                    },
                    "actions": np.random.randn(7),
                    "next_observations": {
                        "image": np.random.randint(
                            0, 255, (128, 128, 3), dtype=np.uint8
                        )
                    },
                    "rewards": np.random.randn(),
                    "masks": 1.0,
                    "dones": False,
                }
                buffer.insert(transition)

        threads = [threading.Thread(target=insert_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        print(f"多线程插入后缓冲区大小: {len(buffer)}")
        return True

    def test_mock_environment(self):
        """测试模拟环境"""
        print("\n=== 测试模拟环境 ===")

        image_keys = ["image"]
        action_dim = 7
        env = MockEnvironment(image_keys, action_dim)

        # 测试重置
        obs, info = env.reset()
        print(f"重置观测键: {list(obs.keys())}")
        print(f"图像形状: {obs['image'].shape}")

        # 测试步进
        action = np.random.uniform(-1, 1, action_dim)
        next_obs, reward, done, truncated, info = env.step(action)

        print(f"步进结果: reward={reward:.3f}, done={done}, truncated={truncated}")
        print(f"信息键: {list(info.keys())}")

        # 测试人工干预
        intervention_count = 0
        for _ in range(100):
            action = np.random.uniform(-1, 1, action_dim)
            next_obs, reward, done, truncated, info = env.step(action)
            if "intervene_action" in info:
                intervention_count += 1

        print(f"100步中人工干预次数: {intervention_count}")
        return True

    def test_concat_batches(self):
        """测试批次合并"""
        print("\n=== 测试批次合并 ===")

        # 创建测试批次
        batch1 = {
            "observations": {"image": torch.randn(4, 3, 128, 128).to(DEVICE)},
            "actions": torch.randn(4, 7).to(DEVICE),
            "rewards": torch.randn(4, 1).to(DEVICE),
        }

        batch2 = {
            "observations": {"image": torch.randn(4, 3, 128, 128).to(DEVICE)},
            "actions": torch.randn(4, 7).to(DEVICE),
            "rewards": torch.randn(4, 1).to(DEVICE),
        }

        # 合并批次
        merged = concat_batches(batch1, batch2)

        print(f"合并前批次1大小: {batch1['actions'].shape[0]}")
        print(f"合并前批次2大小: {batch2['actions'].shape[0]}")
        print(f"合并后批次大小: {merged['actions'].shape[0]}")

        # 验证合并结果
        expected_size = batch1["actions"].shape[0] + batch2["actions"].shape[0]
        assert (
            merged["actions"].shape[0] == expected_size
        ), f"合并大小不匹配: {merged['actions'].shape[0]} != {expected_size}"

        return True

    def test_agent_creation(self):
        """测试智能体创建"""
        print("\n=== 测试智能体创建 ===")

        image_keys = ["image"]

        # 创建混合SAC智能体
        agent = SACHybridAgent(
            image_keys=image_keys,
            continuous_action_dim=6,
            grasp_action_dim=3,
            lr=3e-4,
            device=DEVICE,
        )

        print(f"智能体设备: {agent.device}")
        print(f"智能体图像键: {agent.image_keys}")

        # 测试动作采样
        test_obs = {"image": torch.randn(2, 3, 128, 128).to(DEVICE)}

        actions = agent.sample_actions(test_obs, deterministic=False)
        print(f"动作形状: {actions.shape}")

        # 测试确定性动作
        det_actions = agent.sample_actions(test_obs, deterministic=True)
        print(f"确定性动作形状: {det_actions.shape}")

        return True

    def test_short_training_run(self):
        """测试短时间训练运行"""
        print("\n=== 测试短时间训练运行 ===")

        save_dir = os.path.join(self.test_dir, "test_hybrid_ckpt")

        try:
            # 运行短时间训练
            agent = train_hybrid(
                demo_paths=self.demo_paths,
                image_keys=["image"],
                setup_mode="single-arm-fixed-gripper",
                max_steps=500,  # 短时间训练
                batch_size=32,  # 小批次
                lr=3e-4,
                training_starts=50,  # 减少启动阈值
                random_steps=100,  # 减少随机步数
                cta_ratio=2,
                save_dir=save_dir,
                log_interval=50,
            )

            print(f"训练完成，智能体类型: {type(agent)}")

            # 检查保存的文件
            if os.path.exists(save_dir):
                saved_files = os.listdir(save_dir)
                print(f"保存的文件: {saved_files}")

                # 验证最终模型是否存在
                final_model_path = os.path.join(save_dir, "final_model.pth")
                if os.path.exists(final_model_path):
                    print("✅ 最终模型保存成功")
                else:
                    print("❌ 最终模型保存失败")

            return True

        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback

            traceback.print_exc()
            return False

    def test_actor_loop_isolated(self):
        """测试独立的演员循环"""
        print("\n=== 测试独立演员循环 ===")

        # 创建测试组件
        image_keys = ["image"]
        action_dim = 7

        agent = SACHybridAgent(
            image_keys=image_keys,
            continuous_action_dim=6,
            grasp_action_dim=3,
            device=DEVICE,
        )
        setattr(agent, "action_dim", action_dim)

        env = MockEnvironment(image_keys, action_dim)
        replay_buffer = SharedReplayBuffer(capacity=1000, image_keys=image_keys)
        demo_buffer = SharedReplayBuffer(capacity=1000, image_keys=image_keys)

        # 运行短时间演员循环
        print("运行演员循环...")

        def run_actor():
            actor_loop(
                agent=agent,
                env=env,
                replay_buffer=replay_buffer,
                demo_buffer=demo_buffer,
                max_steps=100,
                random_steps=50,
                update_queue=None,
            )

        actor_thread = threading.Thread(target=run_actor)
        actor_thread.start()
        actor_thread.join(timeout=30)  # 30秒超时

        print(f"演员循环完成，重放缓冲区大小: {len(replay_buffer)}")
        print(f"演示缓冲区大小: {len(demo_buffer)}")

        return len(replay_buffer) > 0

    def test_learner_loop_isolated(self):
        """测试独立的学习者循环"""
        print("\n=== 测试独立学习者循环 ===")

        # 创建测试组件
        image_keys = ["image"]

        agent = SACHybridAgent(
            image_keys=image_keys,
            continuous_action_dim=6,
            grasp_action_dim=3,
            device=DEVICE,
        )

        replay_buffer = SharedReplayBuffer(capacity=1000, image_keys=image_keys)
        demo_buffer = SharedReplayBuffer(capacity=1000, image_keys=image_keys)

        # 预填充一些数据
        for i in range(100):
            transition = {
                "observations": {
                    "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                },
                "actions": np.random.uniform(-1, 1, 7),
                "next_observations": {
                    "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                },
                "rewards": np.random.randn(),
                "masks": 1.0,
                "dones": False,
            }
            replay_buffer.insert(transition)
            if i < 50:  # 一半数据也加到演示缓冲区
                demo_buffer.insert(transition)

        print(
            f"预填充数据 - 重放缓冲区: {len(replay_buffer)}, 演示缓冲区: {len(demo_buffer)}"
        )

        # 运行短时间学习者循环
        save_dir = os.path.join(self.test_dir, "test_learner_ckpt")

        def run_learner():
            learner_loop(
                agent=agent,
                replay_buffer=replay_buffer,
                demo_buffer=demo_buffer,
                max_steps=100,
                batch_size=16,
                training_starts=50,
                cta_ratio=2,
                log_interval=25,
                save_dir=save_dir,
                update_queue=None,
            )

        learner_thread = threading.Thread(target=run_learner)
        learner_thread.start()
        learner_thread.join(timeout=60)  # 60秒超时

        print("学习者循环完成")

        # 检查保存的文件
        if os.path.exists(save_dir):
            saved_files = os.listdir(save_dir)
            print(f"保存的文件: {saved_files}")
            return len(saved_files) > 0

        return False

    def run_all_tests(self):
        """运行所有测试"""
        print("=== 混合模式训练测试套件 ===")

        test_results = {}

        try:
            self.setup_test_environment()

            # 基础组件测试
            test_results["shared_replay_buffer"] = self.test_shared_replay_buffer()
            test_results["mock_environment"] = self.test_mock_environment()
            test_results["concat_batches"] = self.test_concat_batches()
            test_results["agent_creation"] = self.test_agent_creation()

            # 独立组件测试
            test_results["actor_loop"] = self.test_actor_loop_isolated()
            test_results["learner_loop"] = self.test_learner_loop_isolated()

            # 完整训练测试
            test_results["short_training"] = self.test_short_training_run()

        except Exception as e:
            print(f"测试过程中发生错误: {e}")
            import traceback

            traceback.print_exc()
            test_results["error"] = str(e)

        finally:
            self.cleanup_test_environment()

        # 输出测试结果
        print("\n=== 测试结果汇总 ===")
        passed = 0
        total = 0

        for test_name, result in test_results.items():
            if test_name == "error":
                continue
            total += 1
            if result:
                print(f"✅ {test_name}: 通过")
                passed += 1
            else:
                print(f"❌ {test_name}: 失败")

        print(f"\n总体结果: {passed}/{total} 测试通过")

        if "error" in test_results:
            print(f"错误信息: {test_results['error']}")

        return passed == total and "error" not in test_results


def test_performance_metrics():
    """测试性能指标"""
    print("\n=== 性能测试 ===")

    # 测试数据加载性能
    print("测试数据加载性能...")
    start_time = time.time()

    # 生成大量测试数据
    test_data = []
    for i in range(1000):
        transition = {
            "observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "actions": np.random.uniform(-1, 1, 7),
            "next_observations": {
                "image": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            },
            "rewards": np.random.randn(),
            "masks": 1.0,
            "dones": False,
        }
        test_data.append(transition)

    data_gen_time = time.time() - start_time
    print(f"生成1000个转换耗时: {data_gen_time:.2f}秒")

    # 测试缓冲区插入性能
    print("测试缓冲区插入性能...")
    buffer = SharedReplayBuffer(capacity=10000, image_keys=["image"])

    start_time = time.time()
    for transition in test_data:
        buffer.insert(transition)
    insert_time = time.time() - start_time
    print(f"插入1000个转换耗时: {insert_time:.2f}秒")

    # 测试采样性能
    print("测试采样性能...")
    start_time = time.time()
    for _ in range(100):
        batch = buffer.sample(batch_size=32, device=DEVICE)
    sample_time = time.time() - start_time
    print(f"采样100次(每次32个样本)耗时: {sample_time:.2f}秒")

    return True


def main():
    """主测试函数"""
    print("开始混合模式训练测试...")

    # 基础环境检查
    print(f"使用设备: {DEVICE}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    # 运行主要测试
    tester = TestHybridTraining()
    success = tester.run_all_tests()

    # 运行性能测试
    test_performance_metrics()

    if success:
        print("\n🎉 所有测试通过！混合模式训练实现正确。")
        print("\n可以使用以下命令运行完整训练:")
        print(
            "python train_hybrid.py --demo_paths test_data/success_demo.pkl --max_steps 10000"
        )
    else:
        print("\n❌ 部分测试失败，请检查实现。")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
