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
import glob
from tqdm import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.distribution import Distribution

sys.path.append(os.getcwd())
from bc.net import RLActor, BCActor
from reward_model.pkl_utils import load_pkl
from utils.tools import get_device, logging_args, setup_logging

from rl.sac_policy import get_eval_transform, get_train_transform, dict_data_to_torch


@dataclass
class Args:
    output_dir: str = "outputs/bc2rl"
    dataset_path: str = None
    model_path: str = None

    image_shape: int = 128
    action_continue_dim: int = 3  # xyz
    action_discrete_dim: int = 3  # gripper(open/close/keep)
    image_num: int = 2  # 输入图像数量
    state_dim: int = 28

    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-4
    save_interval: int = 4
    resume_checkpoint: str = None

    discrete_weight: list = field(
        default_factory=lambda: [4.0, 1.0, 4.0]
    )  # gripper open/keep/close 权重


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


def process_trajectory_history(transitions, key, history_length):
    shape = transitions[0][key].shape
    # [history_length, ...]
    history = np.zeros((history_length, *shape), dtype=transitions[0][key].dtype)
    for transition in transitions:
        # history[:history_len-1] + new
        history = np.roll(history, shift=-1, axis=0)
        history[-1] = transition[key]
        transition[key] = history.flatten()
    return transitions


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
        #     "/home/facelesswei/code/Jax_Hil_Serl_Dataset/2025-09-09/usb_pickup_insertion_30_11-50-21.pkl",
        #     # classifier_data 子目录中的文件
        #     "/home/facelesswei/code/hil-serl/outputs/classifier_data/2025-09-12/*.pkl",
        #     "/home/facelesswei/code/hil-serl/outputs/classifier_data/2025-09-12-13/*.pkl",
        "datasets/trajectories/2025-10-27/*.pkl",
    ]
    # data_files = ["/Users/majianfei/Downloads/usb_pickup_insertion_5_11-05-02.pkl"]

    total_added = 0

    def load_traj_file(file_path):
        nonlocal total_added, data_list, mapping, args

        file_data = load_pkl(file_path, mapping)
        process_trajectory_history(file_data, "state", 4)
        data_list.extend(file_data)
        total_added += len(file_data)
        logging.info(f"加载 {file_path}: {len(file_data)} 个样本")

    for file_pattern in data_files:
        # 处理通配符模式
        if "*" in file_pattern:
            # 使用 glob 查找匹配的文件

            matched_files = glob.glob(file_pattern)
            if not matched_files:
                logging.warning(f"没有找到匹配的文件: {file_pattern}")
                continue

            for file_path in matched_files:
                load_traj_file(file_path)

        else:
            # 处理单个文件
            if not os.path.exists(file_pattern):
                logging.warning(f"文件不存在，跳过: {file_pattern}")
                continue

            load_traj_file(file_pattern)

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


def train(args: Args):
    device = get_device()

    args.output_dir = os.path.join(args.output_dir, time.strftime("%Y%m%d_%H%M%S"))

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)

    logging_args(args)

    tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    writer = SummaryWriter(tensorboard_dir)

    train_dataloader, test_dataloader = load_and_split_data(args)

    # model = RLActor().to(device)
    model = BCActor(args.__dict__).to(device)

    if args.resume_checkpoint:
        model.load_checkpoint(args.resume_checkpoint)
        logging.info(f"Resumed training from checkpoint: {args.resume_checkpoint}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    weights = torch.tensor(list(args.discrete_weight), dtype=torch.float32).to(device)
    discrete_criterion = nn.CrossEntropyLoss(weight=weights)

    def compute_loss(observations, continue_label, discrete_label):
        pred_continue, pred_discrete = model(observations)  # 传入observations字典
        if isinstance(pred_continue, Distribution):
            # 将连续动作剪切到[-1,1]范围内，确保与TanhMultivariateNormalDiag分布兼容
            continue_label_clipped = torch.clamp(
                continue_label, -1.0 + 1e-6, 1.0 - 1e-6
            )

            # 计算log_prob并使用mean而不是sum，避免梯度爆炸
            log_probs = pred_continue.log_prob(continue_label_clipped)

            # 概率密度可以 > 1，因此其对数可能 > 0
            # log_probs = torch.clamp(log_probs, min=-50.0, max=0.0)

            continue_loss = -log_probs.mean()
        else:
            continue_loss = criterion(pred_continue, continue_label)

        discrete_loss = discrete_criterion(pred_discrete, discrete_label)  # gripper

        loss = continue_loss + discrete_loss
        return loss, continue_loss, discrete_loss

    global_step = 0
    for epoch in range(args.epochs):
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

            loss, continue_loss, discrete_loss = compute_loss(
                observations, continue_label, discrete_label
            )

            optimizer.zero_grad()
            loss.backward()

            # 添加梯度剪切防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
        val_loss_continue = 0.0
        val_loss_discrete = 0.0

        with torch.no_grad():
            for observations, continue_label, discrete_label in test_dataloader:
                # 使用dict_data_to_torch处理观测数据

                # 移动到设备
                for key in observations:
                    observations[key] = observations[key].squeeze().to(device)
                continue_label = continue_label.to(device).squeeze()
                discrete_label = discrete_label.to(device).squeeze()

                loss, continue_loss, discrete_loss = compute_loss(
                    observations, continue_label, discrete_label
                )
                val_loss += loss.item()
                val_loss_continue += continue_loss.item()
                val_loss_discrete += discrete_loss.item()

        val_loss /= len(test_dataloader)
        val_loss_continue /= len(test_dataloader)
        val_loss_discrete /= len(test_dataloader)

        writer.add_scalar("Loss/Epoch", avg_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", val_loss, epoch)
        writer.add_scalar("Loss/Val_Continue_Epoch", val_loss_continue, epoch)
        writer.add_scalar("Loss/Val_Discrete_Epoch", val_loss_discrete, epoch)

        logging.info(f"Epoch {epoch+1}/{args.epochs}, Step {global_step}:")
        logging.info(f"  Train Loss: {avg_loss:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}")
        logging.info(f"  Val Continue Loss: {val_loss_continue:.4f}")
        logging.info(f"  Val Discrete Loss: {val_loss_discrete:.4f}")
        logging.info("-" * 40)

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            cp_name = f"{args.output_dir}/checkpoint-{epoch+1}.pth"
            model.save_checkpoint(cp_name)
            logging.info(f"Checkpoint saved to {cp_name}")

    writer.close()
    logging.info("Tensorboard logging completed.")


class ActorWrapper:
    def __init__(self, model_path):
        args = Args()
        self.device = get_device()
        self.model = BCActor(args.__dict__).to(self.device)
        # self.model = RLActor().to(self.device)
        self.model.load_checkpoint(model_path)
        self.model.eval()
        self.image_transform = get_eval_transform()

        self.history_state = np.zeros((4, 7), dtype=np.float32)

    def predict(self, obs, argmax=True):
        # 构造observations字典，匹配训练时的格式
        observations = {
            "state": obs["state"],
            "rgb": obs["rgb"],
            "wrist": obs["wrist"],
        }
        self.history_state = np.roll(self.history_state, shift=-1, axis=0)
        self.history_state[-1] = obs["state"]
        observations["state"] = self.history_state.flatten()

        observations = dict_data_to_torch(observations, self.image_transform)

        # 移动到设备
        for key in observations:
            observations[key] = observations[key].to(self.device)

        with torch.no_grad():
            action_continue, gripper = self.model(observations)

        if isinstance(action_continue, Distribution):
            if argmax:
                np_action = action_continue.mode().cpu().numpy().squeeze()
            else:
                np_action = action_continue.sample().cpu().numpy().squeeze()
        else:
            np_action = action_continue.cpu().numpy().squeeze()

        np_gripper = gripper.cpu().numpy().squeeze()

        gripper_action = np.argmax(np_gripper) - 1  # 0, 1, 2 -> -1, 0, 1
        np_action = np.concatenate([np_action, [gripper_action]], axis=0)

        print(f"Predicted action: {np_action}")

        return np_action

    def reset(self):
        self.history_state = np.zeros((4, 7), dtype=np.float32)


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
