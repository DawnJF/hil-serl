import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
import numpy as np
import logging
from tqdm import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from reward_model.pkl_utils import load_pkl
from utils.tools import get_device, setup_logging
from rl.net import Actor, DiscreteQCritic
from rl.sac_policy import get_eval_transform, get_train_transform, dict_data_to_torch


@dataclass
class Args:
    output_dir: str = "outputs/bc2rl"
    dataset_path: str = None
    model_path: str = None

    image_shape: int = 128
    action_dim: int = 3 + 3  # xyz + gripper(open/close/keep)
    image_num: int = 2  # 输入图像数量
    state_dim: int = 7

    batch_size: int = 128

    log_interval: int = 10  # 每多少个batch记录一次详细指标
    print_action_interval: int = 500
    discrete_weight: list = field(
        default_factory=lambda: [4.0, 1.0, 4.0]
    )  # gripper open/keep/close 权重


class RLActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_actor = Actor(3)
        self.d_actor = DiscreteQCritic(3)

    def forward(self, batch):
        # batch should be a dict with observations format
        dist = self.c_actor(batch)
        actions = dist.mode()

        discrete_actions = self.d_actor(batch)

        return actions, discrete_actions

    def save_checkpoint(self, path):
        torch.save(
            {
                "continue_actor": self.c_actor.state_dict(),
                "discrete_actor": self.d_actor.state_dict(),
            },
            path,
        )

    def READ_CHECKPOINT(path):
        checkpoint_dict = torch.load(path)
        return checkpoint_dict["continue_actor"], checkpoint_dict["discrete_actor"]

    def load_checkpoint(self, path):
        c_dict, d_dict = RLActor.READ_CHECKPOINT(path)
        self.c_actor.load_state_dict(c_dict)
        self.d_actor.load_state_dict(d_dict)


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
        # 构造observations字典，匹配模型期望的输入格式
        rgb_img = self.data[idx]["image1"]
        wrist_img = self.data[idx]["image2"]

        state = self.data[idx]["state"]

        # 构造observations字典
        observations = {"rgb": rgb_img, "wrist": wrist_img, "state": state}

        action = self.data[idx]["actions"]
        continue_label = torch.tensor(action[:3], dtype=torch.float32)

        # gripper action: -1, 0, 1 -> 0, 1, 2
        if action[-1] <= -0.5:
            gripper_label = torch.tensor(0, dtype=torch.long)
        elif action[-1] < 0.5:
            gripper_label = torch.tensor(1, dtype=torch.long)
        else:
            gripper_label = torch.tensor(2, dtype=torch.long)

        observations = dict_data_to_torch(observations, self.transform)

        return observations, continue_label, gripper_label


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

    train_dataset = ImagesActionDataset(train_data, transform=get_train_transform())
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    test_dataset = ImagesActionDataset(test_data, transform=get_eval_transform())
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

    model = RLActor().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=4e-5)
    weights = torch.tensor(list(args.discrete_weight), dtype=torch.float32).to(device)
    discrete_criterion = nn.CrossEntropyLoss(weight=weights)

    global_step = 0
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0

        for batch_idx, (observations, continue_label, discrete_label) in tqdm(
            enumerate(train_dataloader)
        ):

            # 移动到设备
            for key in observations:
                observations[key] = observations[key].squeeze().to(device)
            continue_label = continue_label.to(device).squeeze()
            discrete_label = discrete_label.to(device).squeeze()

            pred_continue, pred_discrete = model(observations)  # 传入observations字典
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

        # 计算训练集平均损失和成功率
        avg_loss = total_loss / len(train_dataloader)

        # 验证阶段
        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for observations, continue_label, discrete_label in test_dataloader:
                # 使用dict_data_to_torch处理观测数据

                # 移动到设备
                for key in observations:
                    observations[key] = observations[key].squeeze().to(device)
                continue_label = continue_label.to(device).squeeze()
                discrete_label = discrete_label.to(device).squeeze()

                pred_continue, pred_discrete = model(observations)
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
        model.save_checkpoint(cp_name)
        logging.info(f"Checkpoint saved to {cp_name}")

    writer.close()
    logging.info("Tensorboard logging completed.")


class ActorWrapper:
    def __init__(self, model_path):
        self.device = get_device()
        self.model = RLActor(self.device)
        self.model.load_checkpoint(model_path)
        self.model.eval()

    def predict(self, obs):
        # 构造observations字典，匹配训练时的格式
        observations = {
            "state": torch.tensor(obs["state"], dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device),
            "rgb": obs["rgb"],
            "wrist": obs["wrist"],
        }

        # 使用dict_data_to_torch处理观测数据
        from rl.sac_policy import dict_data_to_torch, get_eval_transform

        observations = dict_data_to_torch(
            observations, image_transform=get_eval_transform()
        )

        # 移动到设备
        for key in observations:
            observations[key] = observations[key].to(self.device)

        with torch.no_grad():
            action, gripper = self.model(observations)

        np_action = action.cpu().numpy().squeeze()
        np_gripper = gripper.cpu().numpy().squeeze()

        gripper_action = np.argmax(np_gripper) - 1  # 0, 1, 2 -> -1, 0, 1
        np_action = np.concatenate([np_action, [gripper_action]], axis=0)

        print(f"Predicted action: {np_action}")

        return np_action


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
