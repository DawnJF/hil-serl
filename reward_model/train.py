import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image

from reward_model.net import ResNetClassifier
from reward_model.pkl_utils import load_pkl_to_reward_model
from utils.image_augmentations import get_eval_transform, get_train_transform


# 示例数据集，假设每个样本有n张图片和一个标签
class ImageSequenceDataset(Dataset):
    # Image.open(p).convert("RGB")
    def __init__(
        self,
        images_list,
        class_list,
        transform=None,
    ):

        self.transform = transform
        self.images_list = images_list
        self.class_list = class_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        label = self.class_list[idx]
        imgs = []
        for i in self.images_list[idx]:
            if self.transform:
                i = self.transform(i)
            imgs.append(torch.tensor(i, dtype=torch.float32))

        imgs = torch.stack(imgs, dim=0)  # [N, C, H, W]
        label = torch.tensor(label, dtype=torch.long)  # 转换为张量
        return imgs, label


def save_checkpoint(model, crop_dict, path):
    torch.save({"model_state_dict": model.state_dict(), "crop_dict": crop_dict}, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded from {path}")
    return checkpoint.get("crop_dict", None)


def load_and_split_data(test_size=0.2, random_state=42):
    """
    加载数据并分割为训练集和测试集

    Returns:
        train_images, test_images, train_labels, test_labels
    """
    images_list, class_list, _, _ = load_pkl_to_reward_model()

    train_images, test_images, train_labels, test_labels = train_test_split(
        images_list,
        class_list,
        test_size=test_size,
        random_state=random_state,
        stratify=class_list,  # 保持各类别比例
    )

    return train_images, test_images, train_labels, test_labels


def test(model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_images, test_labels = load_to_reward_model()
    # _, _, test_images, test_labels = jax_load_to_reward_model()
    dataset = ImageSequenceDataset(
        test_images, test_labels, transform=image_normalization
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = ResNetClassifier().to(device)
    load_checkpoint(model, path=model_path)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")


def train(save_path="reward_model", model_path=None, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_images, test_images, train_labels, test_labels = load_and_split_data()

    print(f"Training samples: {len(train_images)}")
    print(f"Testing samples: {len(test_images)}")

    crop_dict = {"top": 250, "left": 380, "height": 200, "width": 300}

    train_dataset = ImageSequenceDataset(
        train_images, train_labels, transform=get_train_transform(crop_dict)
    )
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = ImageSequenceDataset(
        test_images, test_labels, transform=get_eval_transform(crop_dict)
    )
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = ResNetClassifier().to(device)
    if model_path:
        load_checkpoint(model, path=model_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)  # [B, num_classes]
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        # 计算训练集平均损失和成功率
        avg_loss = total_loss / len(train_dataloader)
        train_accuracy = 100 * correct / total

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in test_dataloader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.2f}% ({correct}/{total})")
        print(f"  Val Accuracy: {val_accuracy:.2f}% ({val_correct}/{val_total})")
        print("-" * 40)

        os.makedirs(save_path, exist_ok=True)
        save_checkpoint(model, crop_dict, f"{save_path}/checkpoint-{epoch+1}.pth")


class RewardModelInferencer:
    def __init__(self, model_path, threshold=0.7):
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ResNetClassifier().to(self.device)
        self.crop_dict = load_checkpoint(self.model, path=model_path)
        print(f"Using crop: {self.crop_dict}")
        self.model.eval()

    def predict(self, images_list):
        """
        np_imgs: numpy.ndarray, shape [N, C, H, W] or list of np.ndarray [C, H, W]
        返回: int, 预测类别
        """

        t = get_eval_transform(self.crop_dict)

        for i in images_list:
            i = t(i)
            imgs.append(torch.tensor(i, dtype=torch.float32))
        imgs = torch.stack(imgs, dim=0)  # [N, C, H, W]

        with torch.no_grad():
            outputs = self.model(imgs)  # [1, num_classes]
            probs = torch.softmax(outputs, dim=1)  # 转成概率
            p1 = probs[0, 1].item()  # 类别 1 的概率

            print(f"predict: {p1:.4f}: {p1 > self.threshold}")
            return 1 if p1 > self.threshold else 0


if __name__ == "__main__":
    # train("/home/robot/code/UR_Robot_Arm_Show/reward_model/checkpoint-4.pth")
    # test("/home/robot/code/debug_UR_Robot_Arm_Show/reward_model/checkpoint_by_jax_data/checkpoint-9.pth")
    # test("/home/robot/code/debug_UR_Robot_Arm_Show/reward_model/checkpoint-4.pth")
    # test("/home/facelesswei/code/debug_UR_Robot_Arm_Show/reward_model/checkpoint_strip")
    train()
