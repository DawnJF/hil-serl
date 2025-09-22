import sys
import os
import time
import jax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import tyro
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from bc.net import BCActor, BCPolicyWithDiscrete
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
    action_dim: int = 3 + 3  # xyz + gripper(open/close/keep)
    image_num: int = 2  # 输入图像数量
    state_dim: int = 7

    batch_size: int = 128

    log_interval: int = 10  # 每多少个batch记录一次详细指标
    print_action_interval: int = 500
    dis_create_weight: list = field(
        default_factory=lambda: [20.0, 1.0, 20.0]
    )  # gripper open/keep/close 权重


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
    ):

        self.transform = transform
        self.data = data_list

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

        action = self.data[idx]["actions"]

        continue_label = torch.tensor(action[:3], dtype=torch.float32)

        """
        -1, 0, 1 -> 0, 1, 2
        """
        if action[-1] <= -0.5:
            gripper_label = torch.tensor(0, dtype=torch.long)
        elif action[-1] < 0.5:
            gripper_label = torch.tensor(1, dtype=torch.long)
        else:
            gripper_label = torch.tensor(2, dtype=torch.long)

        return imgs, state, continue_label, gripper_label


def save_checkpoint(model, path, args):
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, path)
    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, path):
    checkpoint_dict = torch.load(path)
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    logging.info(f"Checkpoint loaded from {path}")


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
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    test_dataset = ImagesActionDataset(
        test_data,
        transform=get_eval_transform(),
    )
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader


def train(args: Args, epochs=100):
    device = get_device()

    args.output_dir = f"{args.output_dir}_" + time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    logging.info(f"Args: {vars(args)}")

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    train_dataloader, test_dataloader = load_and_split_data(args)

    model = BCPolicyWithDiscrete(args).to(device)

    if args.model_path is not None:
        load_checkpoint(model, path=args.model_path)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-5)
    weights = torch.tensor(list(args.dis_create_weight), dtype=torch.float32).to(device)
    discrete_criterion = nn.CrossEntropyLoss(weight=weights)

    global_step = 0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0

        for batch_idx, (imgs, state, continue_label, discrete_label) in tqdm(
            enumerate(train_dataloader)
        ):
            imgs = imgs.to(device)
            state = state.to(device)
            continue_label = continue_label.to(device)
            discrete_label = discrete_label.to(device)

            pred_continue, pred_discrete = model(state, imgs)  # [B, num_classes]action
            continue_loss = criterion(pred_continue, continue_label)
            discrete_loss = discrete_criterion(pred_discrete, discrete_label)  # gripper

            loss = continue_loss + discrete_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # 记录每步的损失和epoch信息
            writer.add_scalar("Loss/Step", loss.item(), global_step)
            writer.add_scalar("Loss/Continue_Step", continue_loss.item(), global_step)
            writer.add_scalar("Loss/Discrete_Step", discrete_loss.item(), global_step)
            writer.add_scalar("Epoch_Step", epoch, global_step)

            # 打印预测的action
            # if batch_idx % args.print_action_interval == 0:
            #     # 打印第一个样本的预测结果和真实值
            #     pred_action = predicted[0].detach().cpu().numpy()
            #     label = label[0].detach().cpu().numpy()
            #     original_label = original_action[0].detach().cpu().numpy()
            #     state_values = state[0].detach().cpu().numpy()

            #     logging.info(f"\nBatch {batch_idx}, Step {global_step}:")
            #     logging.info(f"State:              {state_values}")
            #     logging.info(f"Predicted action:    {pred_action}")
            #     logging.info(f"Mapped true action:  {label}")
            #     logging.info(f"Original true action:{original_label}")
            #     logging.info(f"Loss: {loss.item():.6f}")

        # 计算训练集平均损失和成功率
        avg_loss = total_loss / len(train_dataloader)

        # 验证阶段
        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for imgs, state, continue_label, discrete_label in test_dataloader:
                imgs, state, continue_label, discrete_label = (
                    imgs.to(device),
                    state.to(device),
                    continue_label.to(device),
                    discrete_label.to(device),
                )
                pred_continue, pred_discrete = model(state, imgs)
                continue_loss = criterion(pred_continue, continue_label)
                discrete_loss = discrete_criterion(pred_discrete, discrete_label)
                loss = continue_loss + discrete_loss
                val_loss += loss.item()

        val_loss /= len(test_dataloader)

        writer.add_scalar("Loss/Epoch", avg_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", val_loss, epoch)

        logging.info(f"Epoch {epoch+1}/{epochs}:")
        logging.info(f"  Train Loss: {avg_loss:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}")
        logging.info("-" * 40)

        cp_name = f"{args.output_dir}/checkpoint-{epoch+1}.pth"
        save_checkpoint(model, path=cp_name, args=args)

    writer.close()
    logging.info("Tensorboard logging completed.")


class ActorWrapper:
    def __init__(self, model_path):
        self.device = get_device()
        args = Args()
        self.model = BCPolicyWithDiscrete(args).to(self.device)
        load_checkpoint(self.model, path=model_path)
        self.model.eval()

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
            action, gripper = self.model(state, imgs)

        np_action = action.cpu().numpy().squeeze()
        np_gripper = gripper.cpu().numpy().squeeze()

        gripper_action = np.argmax(np_gripper) - 1  # 0, 1, 2 -> -1, 0, 1
        np_action = np.concatenate([np_action, [gripper_action]], axis=0)

        print(f"Predicted action: {np_action}")

        return np_action


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
