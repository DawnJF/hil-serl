import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import tyro
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from bc.net import BCActor
from reward_model.net import ResNetClassifier
from reward_model.pkl_utils import load_pkl, load_pkl_to_reward_model
from utils.image_augmentations import get_eval_transform, get_train_transform
from utils.tools import get_device, setup_logging


@dataclass
class Args:
    output_dir: str = "outputs/bc"
    dataset_path: str = None
    model_path: str = None

    image_shape: int = 128
    action_dim: int = 4
    image_num: int = 2  # 输入图像数量
    state_dim: int = 7

    batch_size: int = 128

    log_interval: int = 10  # 每多少个batch记录一次详细指标
    print_action_interval: int = 50

    use_gripper_mapping: bool = True  # 控制是否使用抓爪映射


def get_train_transform():
    """need CxHxW input"""
    return transforms.Compose(
        [
            # pre-process
            transforms.Lambda(lambda img: img.squeeze()),
            transforms.ToTensor(),
            # data augmentations
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
            # post-process
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform():
    return transforms.Compose(
        [
            # pre-process
            transforms.Lambda(lambda img: img.squeeze()),
            transforms.ToTensor(),
            # post-process
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ImagesActionDataset(Dataset):

    def __init__(
        self,
        data_list,
        transform,
        use_gripper_mapping: bool = True,  # 控制是否使用抓爪映射
    ):

        self.transform = transform
        self.data = data_list
        self.use_gripper_mapping = use_gripper_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        imgs = []
        for key in ["image1", "image2"]:
            img = self.data[idx][key]
            img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs, dim=0)  # [N, C, H, W]

        state = self.data[idx]["state"]
        state = state.squeeze()
        state = torch.tensor(state, dtype=torch.float32)

        original_label = self.data[idx]["actions"]
        original_label = torch.tensor(original_label, dtype=torch.float32)

        # 创建映射后的label，不修改原始label
        if self.use_gripper_mapping:
            label = original_label.clone()  # 创建副本

            if label[-1] <= -0.5:
                label[-1] = torch.tensor(-1.0)  # 闭合
            elif label[-1] >= 0.5:
                label[-1] = torch.tensor(1.0)  # 打开
            else:
                label[-1] = state[0]
        else:
            label = original_label

        return imgs, state, label, original_label


def save_checkpoint(model, path):
    torch.save({"model_state_dict": model.state_dict()}, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Checkpoint loaded from {path}")
    return checkpoint.get("crop_dict", None)


# def test(model_path=None):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     _, test_images, _, test_labels = load_and_split_data()
#     # _, _, test_images, test_labels = jax_load_to_reward_model()
#     dataset = ImageSequenceDataset(
#         test_images, test_labels, transform=image_normalization
#     )
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
#     model = ResNetClassifier().to(device)
#     load_checkpoint(model, path=model_path)
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for imgs, labels in dataloader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             outputs = model(imgs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     accuracy = 100 * correct / total
#     logging.info(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")


def load_and_split_data(args):
    use_gripper_mapping = args.use_gripper_mapping
    mapping = {
        "observations:rgb": "image1",
        "observations:wrist": "image2",
        "observations:state": "state",
        "actions": "actions",
    }

    data_list = []

    # 定义要加载的文件列表
    data_files = [
        "/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-09-09/usb_pickup_insertion_30_11-50-21.pkl",
        # classifier_data 子目录中的文件
        "/home/facelesswei/code/hil-serl/outputs/classifier_data/2025-09-12/*.pkl",
        "/home/facelesswei/code/hil-serl/outputs/classifier_data/2025-09-12-13/*.pkl",
        "/home/facelesswei/code/hil-serl/outputs/classifier_data/2025-09-15/*.pkl",
    ]

    total_added = 0

    for file_pattern in data_files:
        # 处理通配符模式
        if "*" in file_pattern:
            # 使用 glob 查找匹配的文件
            import glob

            matched_files = glob.glob(file_pattern)

            if not matched_files:
                logging.warning(f"没有找到匹配的文件: {file_pattern}")
                continue

            for file_path in matched_files:
                try:
                    file_data = load_pkl(file_path, mapping)
                    data_list.extend(file_data)
                    total_added += len(file_data)
                    logging.info(f"加载 {file_path}: {len(file_data)} 个样本")
                except Exception as e:
                    logging.warning(f"跳过 {file_path}: {e}")
        else:
            # 处理单个文件
            if not os.path.exists(file_pattern):
                logging.warning(f"文件不存在，跳过: {file_pattern}")
                continue

            try:
                file_data = load_pkl(file_pattern, mapping)
                data_list.extend(file_data)
                total_added += len(file_data)
                logging.info(f"加载 {file_pattern}: {len(file_data)} 个样本")
            except Exception as e:
                logging.warning(f"跳过 {file_pattern}: {e}")

    train_data, test_data = train_test_split(data_list)

    logging.info(f"Training samples: {len(train_data)}")
    logging.info(f"Testing samples: {len(test_data)}")

    train_dataset = ImagesActionDataset(
        train_data,
        transform=get_train_transform(),
        use_gripper_mapping=use_gripper_mapping,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    test_dataset = ImagesActionDataset(
        test_data,
        transform=get_eval_transform(),
        use_gripper_mapping=use_gripper_mapping,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader


def train(args: Args, epochs=200):
    device = get_device()

    args.output_dir = f"{args.output_dir}_" + time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    train_dataloader, test_dataloader = load_and_split_data(args)

    model = BCActor(args).to(device)

    if args.model_path is not None:
        load_checkpoint(model, path=args.model_path)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    global_step = 0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0

        for batch_idx, (imgs, state, mapped_action, original_action) in tqdm(
            enumerate(train_dataloader)
        ):
            imgs, state, mapped_action = (
                imgs.to(device),
                state.to(device),
                mapped_action.to(device),
            )
            predicted = model(state, imgs)  # [B, num_classes]action
            loss = criterion(predicted, mapped_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # 记录每步的损失和epoch信息
            if batch_idx % args.log_interval == 0:
                writer.add_scalar("Loss/Step", loss.item(), global_step)
                writer.add_scalar("Epoch/Step", epoch, global_step)

            # 打印预测的action
            if batch_idx % args.print_action_interval == 0:
                # 打印第一个样本的预测结果和真实值
                pred_action = predicted[0].detach().cpu().numpy()
                true_mapped_action = mapped_action[0].detach().cpu().numpy()
                true_original_action = original_action[0].detach().cpu().numpy()
                state_values = state[0].detach().cpu().numpy()

                logging.info(f"\nBatch {batch_idx}, Step {global_step}:")
                logging.info(f"State:              {state_values}")
                logging.info(f"Predicted action:    {pred_action}")
                logging.info(f"Mapped true action:  {true_mapped_action}")
                logging.info(f"Original true action:{true_original_action}")
                logging.info(f"Loss: {loss.item():.6f}")

        # 计算训练集平均损失和成功率
        avg_loss = total_loss / len(train_dataloader)

        # 验证阶段
        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for imgs, state, mapped_action, original_action in test_dataloader:
                imgs, state, mapped_action = (
                    imgs.to(device),
                    state.to(device),
                    mapped_action.to(device),
                )
                outputs = model(state, imgs)
                val_loss += criterion(outputs, mapped_action).item()

        val_loss /= len(test_dataloader)

        writer.add_scalar("Loss/Train_Epoch", avg_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", val_loss, epoch)

        logging.info(f"Epoch {epoch+1}/{epochs}:")
        logging.info(f"  Train Loss: {avg_loss:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}")
        logging.info("-" * 40)

        cp_name = f"{args.output_dir}/checkpoint-{epoch+1}.pth"
        save_checkpoint(model, path=cp_name)

    writer.close()
    logging.info("Tensorboard logging completed.")


class ActorWrapper:
    def __init__(self, model_path):
        self.device = get_device()
        args = Args()
        self.model = BCActor(args).to(self.device)
        load_checkpoint(self.model, path=model_path)
        self.model.eval()
        self.last_predicted_gripper = 0
        self.use_gripper_mapping = args.use_gripper_mapping

    def predict(self, obs):
        state = obs["state"]
        rgb = obs["rgb"]
        wrist = obs["wrist"]

        imgs = []

        img_transform = get_eval_transform()

        for img in [rgb, wrist]:
            img = img_transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)  # [N, C, H, W]
        imgs = imgs.unsqueeze(0)  # [1, N, C, H, W]

        imgs = imgs.to(self.device)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.model(state, imgs)

        np_action = action.cpu().numpy().squeeze()

        """
        np_action[-1] in [0,1]
        """

        # 只有在启用抓爪映射时才进行后处理
        if self.use_gripper_mapping:
            predicted_gripper = 0 if np_action[-1] > 0.5 else 1
            print(f"predicted_gripper:{predicted_gripper}")
            if (
                predicted_gripper == 0
                and predicted_gripper != self.last_predicted_gripper
            ):
                np_action[-1] = 1.0  # 打开
            elif (
                predicted_gripper == 1
                and predicted_gripper != self.last_predicted_gripper
            ):
                np_action[-1] = -1.0  # 闭合
            else:
                np_action[-1] = 0

            self.last_predicted_gripper = predicted_gripper

        print(f"Predicted action: {np_action}")

        return np_action


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
