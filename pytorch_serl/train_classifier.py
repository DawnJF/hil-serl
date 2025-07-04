"""奖励分类器训练脚本"""

import os
import argparse
import pickle as pkl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from networks.classifier import RewardClassifier
from utils.device import DEVICE, to_device


class ClassifierDataset(Dataset):
    """分类器数据集"""

    def __init__(self, data_paths, image_keys, label):
        self.image_keys = image_keys
        self.label = label
        self.data = []

        # 加载数据
        for path in data_paths:
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for transition in transitions:
                    obs = transition["observations"]
                    # 检查是否包含图像数据
                    if all(key in obs for key in image_keys):
                        self.data.append(obs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs = self.data[idx]

        # 转换图像格式 (HWC -> CHW)
        processed_obs = {}
        for key in self.image_keys:
            if key in obs:
                image = obs[key]
                if len(image.shape) == 3 and image.shape[-1] == 3:  # HWC
                    image = image.transpose(2, 0, 1)  # CHW
                processed_obs[key] = image.astype(np.float32) / 255.0  # 归一化到[0,1]

        return processed_obs, self.label


def collate_fn(batch):
    """自定义数据整理函数"""
    obs_list, labels = zip(*batch)

    # 合并观测
    batch_obs = {}
    for key in obs_list[0].keys():
        images = [obs[key] for obs in obs_list]
        batch_obs[key] = torch.from_numpy(np.stack(images))

    batch_labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    return batch_obs, batch_labels


def train_classifier(
    success_paths,
    failure_paths,
    image_keys,
    output_dir,
    batch_size=64,
    num_epochs=150,
    lr=1e-4,
    device=DEVICE,
):
    """训练奖励分类器

    Args:
        success_paths: 成功数据路径列表
        failure_paths: 失败数据路径列表
        image_keys: 图像键列表
        output_dir: 输出目录
        batch_size: 批次大小
        num_epochs: 训练轮数
        lr: 学习率
        device: 设备
    """
    print(f"使用设备: {device}")

    # 创建数据集
    print("加载数据...")
    success_dataset = ClassifierDataset(success_paths, image_keys, label=1.0)
    failure_dataset = ClassifierDataset(failure_paths, image_keys, label=0.0)

    print(f"成功样本数: {len(success_dataset)}")
    print(f"失败样本数: {len(failure_dataset)}")

    # 合并数据集
    full_dataset = torch.utils.data.ConcatDataset([success_dataset, failure_dataset])

    # 创建数据加载器
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 设为0避免多进程问题
    )

    # 创建模型
    print("创建模型...")
    model = RewardClassifier(image_keys).to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 训练循环
    print("开始训练...")
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_obs, batch_labels in pbar:
            # 移动到设备
            batch_obs = to_device(batch_obs, device)
            batch_labels = batch_labels.to(device)

            # 前向传播
            logits = model(batch_obs)
            loss = criterion(logits, batch_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            predictions = (torch.sigmoid(logits) > 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)

            # 更新进度条
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.4f}"}
            )

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "classifier.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_keys": image_keys,
            "model_config": {
                "image_keys": image_keys,
            },
        },
        model_path,
    )

    print(f"模型已保存到: {model_path}")

    return model


def load_classifier(model_path, device=DEVICE):
    """加载已训练的分类器"""
    checkpoint = torch.load(model_path, map_location=device)
    image_keys = checkpoint["image_keys"]

    model = RewardClassifier(image_keys).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, image_keys


def main():
    parser = argparse.ArgumentParser(description="训练奖励分类器")
    parser.add_argument("--success_dir", type=str, required=True, help="成功数据目录")
    parser.add_argument("--failure_dir", type=str, required=True, help="失败数据目录")
    parser.add_argument("--image_keys", nargs="+", default=["image"], help="图像键")
    parser.add_argument(
        "--output_dir", type=str, default="./classifier_ckpt", help="输出目录"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--num_epochs", type=int, default=150, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")

    args = parser.parse_args()

    # 获取数据文件
    success_paths = [
        os.path.join(args.success_dir, f)
        for f in os.listdir(args.success_dir)
        if f.endswith(".pkl") and "success" in f
    ]
    failure_paths = [
        os.path.join(args.failure_dir, f)
        for f in os.listdir(args.failure_dir)
        if f.endswith(".pkl") and "failure" in f
    ]

    print(f"找到 {len(success_paths)} 个成功文件")
    print(f"找到 {len(failure_paths)} 个失败文件")

    # 训练分类器
    train_classifier(
        success_paths=success_paths,
        failure_paths=failure_paths,
        image_keys=args.image_keys,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
